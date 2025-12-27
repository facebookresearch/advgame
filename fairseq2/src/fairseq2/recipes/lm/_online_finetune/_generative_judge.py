import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import Any, List, Literal, Tuple

from typing_extensions import override

from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.recipes.lm._online_finetune._common import (
    generate_rewards_generative,
    xError,
    yError,
)
from fairseq2.recipes.lm._online_finetune._remote_model import RemoteVllmModel
from fairseq2.recipes.lm._online_finetune._templates_format import (
    format_instructions_prompt_judge_pointwise,
    format_instructions_response_judge_pairwise,
    format_instructions_response_judge_pointwise,
)
from fairseq2.recipes.lm._online_finetune._templates_task import (
    task_instructions_prompt_judge_pointwise,
    task_instructions_response_judge_pairwise,
    task_instructions_response_judge_pointwise,
)
from fairseq2.utils.list import lget_shape, linit


class JudgmentExtractorHandler(ABC):
    @abstractmethod
    def create(self): ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


"""
All judgment extractors are expected to:
(1) define their judgment prompt
(2) implement their judgment (i.e., scores, preferences, etc) extraction logic from the CoTs
(3) implement their aggregation logic over judgments, if sampling multiple CoTs
"""


class JudgmentExtractor(ABC):
    """
    This class defines the interface for extracting judgments from generative models,
    including formatting prompts for the reward model, extracting scalar scores from
    model responses, and aggregating multiple judgments into a single value.
    """

    @abstractmethod
    def prompt(self) -> str: ...

    @abstractmethod
    def format_prompt(self, prompt_text, **kwargs: Any) -> str: ...

    """
    Format the prompt text and additional arguments into a string suitable for input to the reward model.
    This method is responsible for formatting the question and responses as input to the reward model.
    Args:
        prompt_text (str): The main prompt or question text to be formatted.
        **kwargs (Any): Additional keyword arguments that may be required for formatting such as rollout_text, reference_answer, etc.
    Returns:
        str: The formatted prompt string ready for the reward model.
    """

    @abstractmethod
    def extract(self, generation) -> float | str: ...

    """
    Extract the final scalar reward score from the model's response.
    This should be implemented to process the given `generation`
    and return either a float representing the reward score or a string with
    additional information.

    Args:
        generation: The model's generated response to be evaluated.

    Returns:
        float | str: The extracted scalar reward score or a string with details.

    Note:
        This method is intended for extracting the final scalar reward score from the model's response.
    """

    @abstractmethod
    def aggregate(self, judgments) -> float | str: ...

    """
    Aggregate multiple responses (judgments) from the reward model into a single value.
    This should combine the results of several model outputs (e.g., scores or preferences)
    into a final scalar or summary value, such as an average score or majority preference.

    Args:
        judgments: A list of individual judgments (e.g., scores or preferences) to aggregate.

    Returns:
        float | str: The aggregated result, such as an average score or consensus preference.
    """


class GeneralVerifierExtractorHandler(JudgmentExtractorHandler):
    def __init__(self):
        pass

    @override
    def create(self):
        return GeneralVerifierExtractor()

    @property
    @override
    def name(self):
        return "general_verifier_extractor"

    @property
    @override
    def config_kls(self):
        return None


class GeneralVerifierExtractor(JudgmentExtractor):
    def __init__(self):
        try:
            from math_verify import parse
            from math_verify.parser import (
                ExprExtractionConfig,
                LatexExtractionConfig,
                NormalizationConfig,
            )
        except ImportError:
            raise ImportError(
                "install mathverify from https://github.com/huggingface/Math-Verify"
            )

        self.student_extraction_config = (
            LatexExtractionConfig(boxed_match_priority=0),
        )
        self.parse = parse

    @override
    def prompt(self):
        raise NotImplementedError(
            "Using the string provided by the general verifier code in format_prompt instead"
        )

    def get_preferred_index(self, lst):
        """
        math_verify parse returns a list of parsed answers, we want want the item at idex 1, which is a string
        """
        if len(lst) > 1:
            return lst[1]
        elif len(lst) == 1:
            return lst[0]
        else:
            return "None"

    @override
    def format_prompt(self, prompt_text, rollout_text, reference_answer):
        student_answer_list = self.parse(rollout_text, self.student_extraction_config)
        student_answer = self.get_preferred_index(student_answer_list)

        prompt = (
            f"User: ### Question: {prompt_text}\n\n"
            f"### Ground Truth Answer: {reference_answer}\n\n"
            f"### Student Answer: {student_answer}\n\n"
            "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
            "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
            'If the student\'s answer is correct, output "Final Decision: Yes". If the student\'s answer is incorrect, output "Final Decision: No". Assistant:'
        )

        return prompt

    @override
    def extract(self, generation):
        if "Final Decision: Yes" in generation:
            return 1.0
        else:
            return 0.0

    @override
    def aggregate(self, judgments):
        avg_score = 0.0
        for score in judgments:
            avg_score += score

        return round(avg_score / len(judgments), 4)


class J1PointwiseExtractorHandler(JudgmentExtractorHandler):
    def __init__(self):
        pass

    @override
    def create(self):
        return J1PointwiseExtractor()

    @property
    @override
    def name(self):
        return "j1_pointwise_extractor"

    @property
    @override
    def config_kls(self):
        return None


POINTWISE_J1_PROMPT = """
You are given a user question and a response from an AI assistant. Your task is to act as an impartial judge and evaluate how well the response fulfills the user's instructions. You will be shown multiple responses to the same prompt, but only one at a time. Evaluate each response independently. 

Think carefully about how to assess the quality of the response, and enclose your reasoning within <think> and </think> tags. Your reasoning should include your evaluation criteria, a clear understanding of what an ideal response would look like for this particular question, and a concrete example of such an ideal or reference answer if possible. Then compare the assistant's response to your ideal or reference answer, explaining how it aligns with or deviates from your expectations. Be specific and avoid vague or overly general judgments. Remain as objective as possible. 

Finally, assign the assistant's response a score from 0 to 10, using either an integer or a decimal with up to 0.1 precision. A higher score should indicate a higher-quality response. Enclose the score within <score> and </score> tags.

Format your output like this: 
<think> your_thinking_process </think> 
<score> your_score </score>

Below are the user's question and the assistant's response:

[User Question]
{instruction}

[The Start of the Assistant's Answer]
{response}
[The End of the Assistant's Answer]
"""

PAIRWISE_J1_PROMPT = """
You are given a user question and two responses from two AI assistants. Your task is to act as an impartial judge and evaluate which response better follows the user's instructions and provides a higher-quality answer.

First, provide your reasoning within <think> and </think> tags. This should include your evaluation criteria for a high-quality response, a detailed comparison of the two responses, and when helpful, a reference answer as part of your evaluation. Be explicit in your thought process, referencing your criteria and explaining how each response aligns with or deviates from them.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.

Finally, provide your verdict within <answer> and </answer> tags, strictly following this format:
- <answer> [[A]] </answer> if Assistant A is better
- <answer> [[B]] </answer> if Assistant B is better

Below are the user's question and the two responses:

[User Question]
{instruction}

<response_A>
{response_A}
</response_A>

<response_B>
{response_B}
</response_B>
"""

PAIRWISE_WITH_SCORES_J1_PROMPT = """
You are given a user question and two responses from two AI assistants. Your task is to act as an impartial judge and evaluate which response better follows the user's instructions and provides a higher-quality answer.

First, provide your reasoning within <think> and </think> tags. This should include your evaluation criteria for a high-quality response, a detailed comparison of the two responses, and when helpful, a reference answer as part of your evaluation. Be explicit in your thought process, referencing your criteria and explaining how each response aligns with or deviates from them.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.

Finally, assign the assistant's response a score from 0 to 10, using either an integer or a decimal with up to 0.1 precision, with a higher score indicating a higher-quality response that better satisfies the criteria. Enclose the scores within the tags <score_A> </score_A>, and <score_B> </score_B>.

Format your output like this:
<think> your_thinking_process </think>
<score_A> your_score_a </score_A> <score_B> your_score_b </score_B>

Below are the user's question and the two responses:

[User Question]
{instruction}

<response_A>
{response_A}
</response_A>

<response_B>
{response_B}
</response_B>
"""


class J1PointwiseExtractor(JudgmentExtractor):
    def __init__(self):
        pass

    @override
    def prompt(self):
        return POINTWISE_J1_PROMPT

    @override
    def format_prompt(self, prompt_text, rollout_text, reference_answer):
        content = self.prompt().format(instruction=prompt_text, response=rollout_text)
        wrapped_text = [{"role": "user", "content": content}]
        chat_str = self.tokenizer.apply_chat_template(
            wrapped_text, tokenize=False, add_generation_prompt=True
        )
        return chat_str

    @override
    def extract(self, generation):
        matches = re.findall(
            r"<score>\s*([0-9]+(?:\.[0-9])?)\s*(?:/10)?\s*</score>", generation
        )
        if matches and float(matches[-1].strip()) > 10.0:
            log.info(f"Judge output = {generation}")
        return float(matches[-1].strip()) if matches else 0.0

    @override
    def aggregate(self, judgments):
        avg_score = 0.0
        for score in judgments:
            avg_score += score

        return round(avg_score / len(judgments), 4)


class J1PairwiseScoreExtractorHandler(JudgmentExtractorHandler):
    def __init__(self):
        pass

    @override
    def create(self):
        return J1PairwiseScoreExtractor()

    @property
    @override
    def name(self):
        return "j1_pairwise_score_extractor"

    @property
    @override
    def config_kls(self):
        return None


class J1PairwiseScoreExtractor(JudgmentExtractor):
    def __init__(self):
        pass

    @override
    def prompt(self):
        return PAIRWISE_WITH_SCORES_J1_PROMPT

    @override
    def format_prompt(self, prompt_text, rollout_A_text, rollout_B_text):
        content = self.prompt().format(
            instruction=prompt_text,
            response_A=rollout_A_text,
            response_B=rollout_B_text,
        )
        wrapped_text = [{"role": "user", "content": content}]
        chat_str = self.tokenizer.apply_chat_template(
            wrapped_text, tokenize=False, add_generation_prompt=True
        )
        return chat_str

    @override
    def extract(self, generation):
        score_a_matches = re.findall(
            r"<score_A>\s*([0-9]+(?:\.[0-9])?)\s*(?:/10)?\s*</score_A>", generation
        )
        score_b_matches = re.findall(
            r"<score_B>\s*([0-9]+(?:\.[0-9])?)\s*(?:/10)?\s*</score_B>", generation
        )

        if score_a_matches and score_b_matches:
            score_a = score_a_matches[-1]
            score_b = score_b_matches[-1]
            if float(score_a.strip()) > 10.0 or float(score_b.strip()) > 10.0:
                log.info(f"Judge output = {generation}")
            return (float(score_a.strip()), float(score_b.strip()))
        else:
            return (0.0, 0.0)

    @override
    def aggregate(self, judgments):
        avg_score = (0.0, 0.0)
        for score in judgments:
            avg_score = (avg_score[0] + score[0], avg_score[1] + score[1])

        return (
            round(avg_score[0] / len(judgments), 4),
            round(avg_score[1] / len(judgments), 4),
        )


class J1PairwisePreferenceExtractorHandler(JudgmentExtractorHandler):
    def __init__(self):
        pass

    @override
    def create(self):
        return J1PairwisePreferenceExtractor()

    @property
    @override
    def name(self):
        return "j1_pairwise_preference_extractor"

    @property
    @override
    def config_kls(self):
        return None


class J1PairwisePreferenceExtractor(JudgmentExtractor):
    def __init__(self):
        pass

    @override
    def prompt(self):
        return PAIRWISE_J1_PROMPT

    @override
    def extract(self, generation):
        matches = list(
            re.findall(r"<answer>\s*\[\[(A|B)\]\]\s*</answer>", generation.strip())
        )

        return matches[-1].strip() if matches else None

    @override
    def aggregate(self, judgments):
        pass


class JudgmentExtractorGame(ABC):
    def __init__(
        self,
        gangs: Gangs,
        reward_model: RemoteVllmModel,
        tokenizer: Any,
        config: object,
    ):
        self._gangs = gangs
        self._reward_model = reward_model
        self._tokenizer = tokenizer
        self._config = config

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def format_prompt(self, *args, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract(self, generation: str, prefix: str):
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, judgments):
        raise NotImplementedError

    def generate_judgments(
        self,
        vllm_inputs_flat: List[str],
        prefix: str = "",
    ) -> List[Any]:
        time_before = time.time()
        judgments_flat = generate_rewards_generative(
            vllm_inputs_flat,
            dp_gang=self._gangs.dp,
            vllm_model=self._reward_model,
        )
        time_after = time.time()

        F = len(judgments_flat)

        if F > 0 and self._config.log_rollouts:
            inputs = vllm_inputs_flat[0]
            outputs = judgments_flat[0].outputs[0].text
            judging = self.extract(outputs, prefix=prefix)
            log.info(
                f"[{prefix}{self.name}] Num inputs: {F}, Input: {inputs}, Output: {outputs}, Parsed result: {judging}, time taken: {time_after - time_before} seconds"
            )

        judgments_agg_flat = []
        for judgments in judgments_flat:
            judgments_ext = [
                self.extract(judgment.text, prefix=prefix)
                for judgment in judgments.outputs
            ]
            judgments_agg = self.aggregate(judgments_ext)
            judgments_agg_flat.append(judgments_agg)
        assert len(judgments_agg_flat) == len(vllm_inputs_flat)
        return judgments_agg_flat


class ResponseJudgePairwiseScoreExtractor(JudgmentExtractorGame):
    @override
    def format_prompt(
        self,
        s_category: Literal["harmful", "benign"],
        query: str,
        response_A: str,
        response_B: str,
    ) -> str:
        assert response_A is not None
        assert response_B is not None
        if s_category == "benign":
            task_instruction_template = task_instructions_response_judge_pairwise[
                self._config.task_instruction_benign_name
            ]
        elif s_category == "harmful":
            task_instruction_template = task_instructions_response_judge_pairwise[
                self._config.task_instruction_harmful_name
            ]
        else:
            raise ValueError(f"Unknown category {s_category}.")

        task_instruction = task_instruction_template.format(
            query=query, response_A=response_A, response_B=response_B
        )
        format_instruction = format_instructions_response_judge_pairwise[
            self._config.format_instruction_name
        ]
        content = task_instruction + format_instruction
        wrapped_text = [{"role": "user", "content": content}]
        chat_str = self._tokenizer.apply_chat_template(
            wrapped_text, tokenize=False, add_generation_prompt=True
        )
        return chat_str

    @override
    def extract(
        self, generation: str, prefix: str
    ) -> Tuple[float | None, float | None]:
        score_a_matches = re.findall(
            r"<score_A>\s*([0-9]+(?:\.[0-9])?)\s*(?:/10)?\s*</score_A>", generation
        )
        score_b_matches = re.findall(
            r"<score_B>\s*([0-9]+(?:\.[0-9])?)\s*(?:/10)?\s*</score_B>", generation
        )

        if score_a_matches and score_b_matches:
            score_a = float(score_a_matches[-1].strip())
            score_b = float(score_b_matches[-1].strip())
            score_clipped_a = min(10.0, max(0.0, score_a))
            score_clipped_b = min(10.0, max(0.0, score_b))

            if score_a > 10.0 or score_b > 10.0 or score_a < 0.0 or score_b < 0.0:
                log.info(
                    f"[{prefix}{self.name}] Warning: Bad judge output, scores ('{score_a}', '{score_b}') out of range, clipped to ('{score_clipped_a}', '{score_clipped_b}')"
                )

            return (score_clipped_a, score_clipped_b)
        else:
            log.info(
                f"[{prefix}{self.name}] Warning: Bad judge output, no match, returning (None, None)"
            )
            return (None, None)

    @override
    def aggregate(
        self, judgments: List[Tuple[float | None, float | None]]
    ) -> Tuple[float | None, float | None]:
        scores_1 = [sc1 for sc1, _ in judgments if sc1 is not None]
        scores_2 = [sc2 for _, sc2 in judgments if sc2 is not None]
        if len(scores_1) > 0:
            avg_score_1 = round(sum(scores_1) / len(scores_1), 4)
        else:
            avg_score_1 = None
        if len(scores_2) > 0:
            avg_score_2 = round(sum(scores_2) / len(scores_2), 4)
        else:
            avg_score_2 = None
        return (avg_score_1, avg_score_2)


class ResponseJudgePointwiseScoreExtractor(JudgmentExtractorGame):
    @override
    def format_prompt(
        self,
        s_category: Literal["harmful", "benign"],
        query: str,
        response: str,
    ) -> str:
        if s_category == "benign":
            task_instruction_template = task_instructions_response_judge_pointwise[
                self._config.task_instruction_benign_name
            ]
        elif s_category == "harmful":
            task_instruction_template = task_instructions_response_judge_pointwise[
                self._config.task_instruction_harmful_name
            ]
        else:
            raise ValueError(f"Unknown category {s_category}.")

        task_instruction = task_instruction_template.format(
            query=query, response=response
        )
        format_instruction = format_instructions_response_judge_pointwise[
            self._config.format_instruction_name
        ]
        content = task_instruction + format_instruction
        wrapped_text = [{"role": "user", "content": content}]
        chat_str = self._tokenizer.apply_chat_template(
            wrapped_text, tokenize=False, add_generation_prompt=True
        )
        return chat_str

    @override
    def extract(self, generation: str, prefix: str) -> float:
        score_matches = re.findall(
            r"<score>\s*([0-9]+(?:\.[0-9])?)\s*(?:/10)?\s*</score>", generation
        )
        if score_matches:
            score = float(score_matches[-1].strip())
            score_clipped = min(10.0, max(0.0, score))

            if score > 10.0 or score < 0.0:
                log.info(
                    f"[{prefix}{self.name}] Warning: Bad judge output, score '{score}' out of range, clipped to '{score_clipped}'"
                )

            return score_clipped
        else:
            log.info(
                f"[{prefix}{self.name}] Warning: Bad judge output, no match, returning None"
            )
            return None

    def aggregate(self, judgments: List[float | None]) -> float | None:
        valid_judgments = [j for j in judgments if j is not None]
        if len(valid_judgments) > 0:
            avg_score = sum(valid_judgments) / len(valid_judgments)
            return round(avg_score, 4)
        else:
            return None


class PromptMaskExtractor(ABC):
    @abstractmethod
    def __call__(
        self,
        s_category_B: List[str],
        s_text_B: List[str],
        x_text_BA: List[List[str | None]],
        x_error_BA: List[List[xError]],
        prefix: str = "",
    ) -> Tuple[List[List[bool | None]], List[List[xError]]]:
        raise NotImplementedError


@dataclass(kw_only=True)
class PromptMaskExtractorConfig:
    log_rollouts: bool = False
    task_instruction_benign_name: str = "dummy"
    task_instruction_harmful_name: str = "dummy"
    format_instruction_name: str = "dummy"


@dataclass(kw_only=True)
class PromptMaskExtractorSection:
    name: str = "dummy"
    config: PromptMaskExtractorConfig | None = field(
        default_factory=lambda: PromptMaskExtractorConfig()
    )


class PromptMaskPointwiseExtractorHandler(JudgmentExtractorHandler):
    @override
    def create(
        self,
        gangs: Gangs,
        reward_model: RemoteVllmModel,
        tokenizer: Any,
        judgment_extractor_config: PromptMaskExtractorConfig,
    ):
        return PromptMaskPointwiseExtractor(
            gangs=gangs,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=judgment_extractor_config,
        )

    @property
    @override
    def name(self):
        return "prompt_mask_pointwise_extractor"

    @property
    @override
    def config_kls(self):
        return PromptMaskExtractorConfig


class PromptMaskPointwiseExtractor(PromptMaskExtractor, JudgmentExtractorGame):
    @property
    def name(self):
        return "prompt_mask_pointwise_extractor"

    def __call__(
        self,
        s_category_B: List[str],
        s_text_B: List[str],
        x_text_BA: List[List[str | None]],
        x_error_BA: List[List[xError]],
        prefix: str = "",
    ) -> Tuple[List[List[bool | None]], List[List[xError]]]:
        B, A = lget_shape(x_text_BA, depth=2)

        vllm_inputs_flat = []
        ba_indices_flat = []
        for b, a in product(range(B), range(A)):
            if x_error_BA[b][a] == xError.NO_ERROR:
                vllm_input = self.format_prompt(
                    s_category=s_category_B[b],
                    query=s_text_B[b],
                    prompt=x_text_BA[b][a],
                )
                vllm_inputs_flat.append(vllm_input)
                ba_indices_flat.append((b, a))

        judgments_agg_flat = self.generate_judgments(
            vllm_inputs_flat=vllm_inputs_flat,
            prefix=prefix,
        )
        x_mask_BA = linit([B, A])
        for judgments_agg, (b, a) in zip(judgments_agg_flat, ba_indices_flat):
            if judgments_agg is None:
                x_error_BA[b][a] = xError.BAD_FAITH_JUDGE
            else:
                x_mask_BA[b][a] = judgments_agg

        log.info(f"[{prefix}{self.name}] Pointwise mask: {x_mask_BA}")
        return x_mask_BA, x_error_BA

    @override
    def format_prompt(
        self,
        s_category: Literal["harmful", "benign"],
        query: str,
        prompt: str,
    ) -> str:
        assert query is not None
        assert prompt is not None
        if s_category == "benign":
            task_instruction_template = task_instructions_prompt_judge_pointwise[
                self._config.task_instruction_benign_name
            ]
        elif s_category == "harmful":
            task_instruction_template = task_instructions_prompt_judge_pointwise[
                self._config.task_instruction_harmful_name
            ]
        else:
            raise ValueError(f"Unknown category {s_category}.")

        task_instruction = task_instruction_template.format(query=query, prompt=prompt)
        format_instruction = format_instructions_prompt_judge_pointwise[
            self._config.format_instruction_name
        ]
        content = task_instruction + format_instruction
        wrapped_text = [{"role": "user", "content": content}]
        chat_str = self._tokenizer.apply_chat_template(
            wrapped_text, tokenize=False, add_generation_prompt=True
        )
        return chat_str

    @override
    def extract(self, generation: str, prefix: str) -> bool | None:
        match = re.search(r"<answer>\s*(True|False)\s*</answer>", generation)
        if match:
            answer = match.group(1).strip()
            if answer == "True":
                return True
            elif answer == "False":
                return False
            else:
                log.info(
                    f"[{prefix}{self.name}] Warning: Bad judge output answer '{answer}' not recognized, returning None"
                )
                return None
        else:
            log.info(
                f"[{prefix}{self.name}] Warning: Bad judge output, no match, returning None"
            )
            return None

    @override
    def aggregate(self, judgments: List[bool | None]) -> bool | None:
        judgments_filtered = [j for j in judgments if j is not None]
        if not judgments_filtered:
            return None
        else:
            num_true = sum(judgments_filtered)
            num_false = len(judgments_filtered) - num_true
            return num_true > num_false


@dataclass(kw_only=True)
class AttackerScoreExtractorConfig:
    log_rollouts: bool = False
    task_instruction_benign_name: str = "dummy"
    task_instruction_harmful_name: str = "dummy"
    format_instruction_name: str = "dummy"


@dataclass(kw_only=True)
class AttackerScoreExtractorSection:
    name: str = "dummy"
    config: AttackerScoreExtractorConfig = field(
        default_factory=lambda: AttackerScoreExtractorConfig()
    )


class AttackerScoreExtractor(ABC):
    @abstractmethod
    def __call__(
        self,
        s_category_B: List[str],
        s_text_B: List[str],
        x_error_BA: List[List[xError]],
        y_text_BA: List[List[str | None]],
        y_error_BA: List[List[yError]],
        prefix: str = "",
    ) -> Tuple[List[List[float | None]], List[List[xError]]]:
        raise NotImplementedError


class AttackerPairwiseScoreExtractorHandler(JudgmentExtractorHandler):
    @override
    def create(
        self,
        gangs: Gangs,
        reward_model: RemoteVllmModel,
        tokenizer: Any,
        judgment_extractor_config: AttackerScoreExtractorConfig,
    ):
        return AttackerPairwiseScoreExtractor(
            gangs=gangs,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=judgment_extractor_config,
        )

    @property
    @override
    def name(self):
        return "attacker_pairwise_score_extractor"

    @property
    @override
    def config_kls(self):
        return AttackerScoreExtractorConfig


class AttackerPairwiseScoreExtractor(
    AttackerScoreExtractor, ResponseJudgePairwiseScoreExtractor
):
    @property
    def name(self):
        return "attacker_pairwise_score_extractor"

    @override
    def __call__(
        self,
        s_category_B: List[str],
        s_text_B: List[str],
        x_error_BA: List[List[xError]],
        y_text_BA: List[List[str | None]],
        y_error_BA: List[List[yError]],
        prefix: str = "",
    ) -> Tuple[List[List[float | None]], List[List[xError]]]:
        B, A = lget_shape(y_text_BA, depth=2)

        vllm_inputs_flat = []
        baa_indices_flat = []
        for b in range(B):
            valid_indices = [
                a
                for a, (xerr, yerr) in enumerate(zip(x_error_BA[b], y_error_BA[b]))
                if xerr == xError.NO_ERROR and yerr == yError.NO_ERROR
            ]
            if len(valid_indices) == 0:
                continue
            elif len(valid_indices) == 1:
                a = valid_indices[0]
                vllm_input = self.format_prompt(
                    s_category=s_category_B[b],
                    query=s_text_B[b],
                    response_A=y_text_BA[b][a],
                    response_B=y_text_BA[b][a],
                )
                vllm_inputs_flat.append(vllm_input)
                baa_indices_flat.append((b, a, a))
            else:
                for a1, a2 in product(valid_indices, valid_indices):
                    if a1 < a2:
                        vllm_input = self.format_prompt(
                            s_category=s_category_B[b],
                            query=s_text_B[b],
                            response_A=y_text_BA[b][a1],
                            response_B=y_text_BA[b][a2],
                        )
                        vllm_inputs_flat.append(vllm_input)
                        baa_indices_flat.append((b, a1, a2))

        log.info(
            f"[{prefix}{self.name}] Pairwise indices (b, a1, a2): {baa_indices_flat}"
        )
        judgments_agg_flat = self.generate_judgments(
            vllm_inputs_flat=vllm_inputs_flat, prefix=prefix
        )
        log.info(f"[{prefix}{self.name}] Pairwise rewards: {judgments_agg_flat}")

        rewards_list_BA = linit([B, A])
        for (b, a1, a2), (R1, R2) in zip(baa_indices_flat, judgments_agg_flat):
            if rewards_list_BA[b][a1] is None:
                rewards_list_BA[b][a1] = [R1]
            else:
                rewards_list_BA[b][a1].append(R1)
            if rewards_list_BA[b][a2] is None:
                rewards_list_BA[b][a2] = [R2]
            else:
                rewards_list_BA[b][a2].append(R2)

        rewards_BA = linit([B, A])
        for b, a in product(range(B), range(A)):
            if rewards_list_BA[b][a] is not None:
                rewards_list_filtered = [
                    r for r in rewards_list_BA[b][a] if r is not None
                ]
                if len(rewards_list_filtered) == 0:
                    # this means we called judge but got no valid output
                    x_error_BA[b][a] = xError.BAD_ATT_JUDGE
                else:
                    rewards_BA[b][a] = round(
                        sum(rewards_list_filtered) / len(rewards_list_filtered), 4
                    )

        log.info(f"[{prefix}{self.name}] Pointwise rewards: {rewards_BA}")
        return rewards_BA, x_error_BA


class AttackerPointwiseScoreExtractorHandler(JudgmentExtractorHandler):
    @override
    def create(
        self,
        gangs: Gangs,
        reward_model: RemoteVllmModel,
        tokenizer: Any,
        judgment_extractor_config: AttackerScoreExtractorConfig,
    ):
        return AttackerPointwiseScoreExtractor(
            gangs=gangs,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=judgment_extractor_config,
        )

    @property
    @override
    def name(self):
        return "attacker_pointwise_score_extractor"

    @property
    @override
    def config_kls(self):
        return AttackerScoreExtractorConfig


class AttackerPointwiseScoreExtractor(
    AttackerScoreExtractor, ResponseJudgePointwiseScoreExtractor
):
    @property
    def name(self):
        return "attacker_pointwise_score_extractor"

    def __call__(
        self,
        s_category_B: List[str],
        s_text_B: List[str],
        x_error_BA: List[List[xError]],
        y_text_BA: List[List[str | None]],
        y_error_BA: List[List[yError]],
        prefix: str = "",
    ) -> Tuple[List[List[float | None]], List[List[xError]]]:
        B, A = lget_shape(y_text_BA, depth=2)

        vllm_inputs_flat = []
        ba_indices_flat = []
        for b, a in product(range(B), range(A)):
            if (
                x_error_BA[b][a] == xError.NO_ERROR
                and y_error_BA[b][a] == yError.NO_ERROR
            ):
                vllm_input = self.format_prompt(
                    s_category=s_category_B[b],
                    query=s_text_B[b],
                    response=y_text_BA[b][a],
                )
                vllm_inputs_flat.append(vllm_input)
                ba_indices_flat.append((b, a))

        judgments_agg_flat = self.generate_judgments(
            vllm_inputs_flat=vllm_inputs_flat,
            prefix=prefix,
        )

        rewards_BA = linit([B, A])
        for judgments_agg, (b, a) in zip(judgments_agg_flat, ba_indices_flat):
            if judgments_agg is None:
                x_error_BA[b][a] = xError.BAD_ATT_JUDGE
            else:
                rewards_BA[b][a] = judgments_agg

        log.info(f"[{prefix}{self.name}] Pointwise rewards: {rewards_BA}")
        return rewards_BA, x_error_BA


@dataclass(kw_only=True)
class DefenderScoreExtractorConfig:
    log_rollouts: bool = False
    task_instruction_benign_name: str = "dummy"
    task_instruction_harmful_name: str = "dummy"
    format_instruction_name: str = "dummy"
    condition_on: Literal["seed", "prompt"] = "seed"


@dataclass(kw_only=True)
class DefenderScoreExtractorSection:
    name: str = "dummy"
    config: DefenderScoreExtractorConfig = field(
        default_factory=lambda: DefenderScoreExtractorConfig()
    )


class DefenderScoreExtractor(ABC):
    @abstractmethod
    def __call__(
        self,
        s_category_B: List[str],
        s_text_B: List[str],
        x_text_BA: List[List[str | None]],
        y_text_BAD: List[List[List[str | None]]],
        y_error_BAD: List[List[List[yError]]],
        prefix: str = "",
    ) -> Tuple[List[List[List[float | None]]], List[List[List[yError]]]]:
        raise NotImplementedError


class DefenderPairwiseScoreExtractorHandler(JudgmentExtractorHandler):
    @override
    def create(
        self,
        gangs: Gangs,
        reward_model: RemoteVllmModel,
        tokenizer: Any,
        judgment_extractor_config: DefenderScoreExtractorConfig,
    ):
        return DefenderPairwiseScoreExtractor(
            gangs=gangs,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=judgment_extractor_config,
        )

    @property
    @override
    def name(self):
        return "defender_pairwise_score_extractor"

    @property
    @override
    def config_kls(self):
        return DefenderScoreExtractorConfig


class DefenderPairwiseScoreExtractor(
    DefenderScoreExtractor, ResponseJudgePairwiseScoreExtractor
):
    @property
    def name(self):
        return "defender_pairwise_score_extractor"

    @override
    def __call__(
        self,
        s_category_B,
        s_text_B: List[str],
        x_text_BA: List[List[str | None]],
        y_text_BAD: List[List[List[str | None]]],
        y_error_BAD: List[List[List[yError]]],
        prefix: str = "",
    ) -> Tuple[List[List[List[float | None]]], List[List[List[yError]]]]:
        B, A, D = lget_shape(y_text_BAD, depth=3)

        vllm_inputs_flat = []
        badd_indices_flat = []
        for b, a in product(range(B), range(A)):
            if self._config.condition_on == "seed":
                query = s_text_B[b]
            elif self._config.condition_on == "prompt":
                query = x_text_BA[b][a]
            else:
                raise ValueError(f"Unknown conditioning {self._config.condition_on}")
            valid_indices = [
                d for d, yerr in enumerate(y_error_BAD[b][a]) if yerr == yError.NO_ERROR
            ]
            if len(valid_indices) == 0:
                continue
            elif len(valid_indices) == 1:
                d = valid_indices[0]
                vllm_input = self.format_prompt(
                    s_category=s_category_B[b],
                    query=query,
                    response_A=y_text_BAD[b][a][d],
                    response_B=y_text_BAD[b][a][d],
                )
                vllm_inputs_flat.append(vllm_input)
                badd_indices_flat.append((b, a, d, d))
            else:
                for d1, d2 in product(valid_indices, valid_indices):
                    if d1 < d2:
                        vllm_input = self.format_prompt(
                            s_category=s_category_B[b],
                            query=query,
                            response_A=y_text_BAD[b][a][d1],
                            response_B=y_text_BAD[b][a][d2],
                        )
                        vllm_inputs_flat.append(vllm_input)
                        badd_indices_flat.append((b, a, d1, d2))

        log.info(
            f"[{prefix}{self.name}] Pairwise indices (b, a, d1, d2): {badd_indices_flat}"
        )
        judgments_agg_flat = self.generate_judgments(
            vllm_inputs_flat=vllm_inputs_flat, prefix=prefix
        )
        log.info(f"[{prefix}{self.name}] Pairwise rewards: {judgments_agg_flat}")

        rewards_list_BAD = linit([B, A, D])
        for (b, a, d1, d2), (R1, R2) in zip(badd_indices_flat, judgments_agg_flat):
            if rewards_list_BAD[b][a][d1] is None:
                rewards_list_BAD[b][a][d1] = [R1]
            else:
                rewards_list_BAD[b][a][d1].append(R1)
            if rewards_list_BAD[b][a][d2] is None:
                rewards_list_BAD[b][a][d2] = [R2]
            else:
                rewards_list_BAD[b][a][d2].append(R2)

        rewards_BAD = linit([B, A, D])
        for b, a, d in product(range(B), range(A), range(D)):
            if rewards_list_BAD[b][a][d] is not None:
                rewards_filtered = [
                    r for r in rewards_list_BAD[b][a][d] if r is not None
                ]
                if len(rewards_filtered) == 0:
                    # this means we called judge but got no valid output
                    y_error_BAD[b][a][d] = yError.BAD_DEF_JUDGE
                else:
                    rewards_BAD[b][a][d] = round(
                        sum(rewards_filtered) / len(rewards_filtered),
                        4,
                    )

        log.info(f"[{prefix}{self.name}] Pointwise rewards: {rewards_BAD}")
        return rewards_BAD, y_error_BAD


class DefenderPointwiseScoreExtractorHandler(JudgmentExtractorHandler):
    @override
    def create(
        self,
        gangs: Gangs,
        reward_model: RemoteVllmModel,
        tokenizer: Any,
        judgment_extractor_config: DefenderScoreExtractorConfig,
    ):
        return DefenderPointwiseScoreExtractor(
            gangs=gangs,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=judgment_extractor_config,
        )

    @property
    @override
    def name(self):
        return "defender_pointwise_score_extractor"

    @property
    @override
    def config_kls(self):
        return DefenderScoreExtractorConfig


class DefenderPointwiseScoreExtractor(
    DefenderScoreExtractor, ResponseJudgePointwiseScoreExtractor
):
    @property
    def name(self):
        return "defender_pointwise_score_extractor"

    @override
    def __call__(
        self,
        s_category_B: List[str],
        s_text_B: List[str],
        x_text_BA: List[List[str]],
        y_text_BAD: List[List[List[str | None]]],
        y_error_BAD: List[List[List[yError]]],
        prefix: str = "",
    ) -> Tuple[List[List[List[float | None]]], List[List[List[yError]]]]:
        B, A, D = lget_shape(y_text_BAD, depth=3)

        vllm_inputs_flat = []
        bad_indices_flat = []
        for b, a, d in product(range(B), range(A), range(D)):
            if y_error_BAD[b][a][d] == yError.NO_ERROR:
                if self._config.condition_on == "seed":
                    query = s_text_B[b]
                elif self._config.condition_on == "prompt":
                    query = x_text_BA[b][a]
                else:
                    raise ValueError(
                        f"Unknown conditioning {self._config.condition_on}"
                    )
                vllm_input = self.format_prompt(
                    s_category=s_category_B[b],
                    query=query,
                    response=y_text_BAD[b][a][d],
                )
                vllm_inputs_flat.append(vllm_input)
                bad_indices_flat.append((b, a, d))

        judgments_agg_flat = self.generate_judgments(
            vllm_inputs_flat=vllm_inputs_flat,
            prefix=prefix,
        )

        rewards_BAD = linit([B, A, D])
        for judgments_agg, (b, a, d) in zip(judgments_agg_flat, bad_indices_flat):
            if judgments_agg is None:
                y_error_BAD[b][a][d] = yError.BAD_DEF_JUDGE
            else:
                rewards_BAD[b][a][d] = judgments_agg

        log.info(f"[{prefix}{self.name}] Pointwise rewards: {rewards_BAD}")
        return rewards_BAD, y_error_BAD
