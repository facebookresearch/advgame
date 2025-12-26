# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer
from typing_extensions import override
from vllm import RequestOutput

from fairseq2.context import RuntimeContext
from fairseq2.datasets import SequenceBatch
from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.datasets.prompt import PromptBatch
from fairseq2.gang import Gangs
from fairseq2.logging import log
from fairseq2.metrics import MetricBag
from fairseq2.recipes.lm._online_finetune._common import (
    RolloutTree,
    RolloutTreeDirect,
    ScoredRolloutTree,
    ScoredRolloutTreeDirect,
    _mute_output,
    collate_with_target_mask,
    generate_rewards,
    generate_rewards_generative,
    generate_rollouts,
    log_rollout_tree,
    log_rollout_tree_direct,
    prepare_preference_batch_random_pair,
    update_avg_length_filtered,
    update_avg_value_filtered,
    update_std_value_filtered,
    xError,
    yError,
)
from fairseq2.recipes.lm._online_finetune._generative_judge import (
    AttackerScoreExtractor,
    AttackerScoreExtractorSection,
    DefenderScoreExtractor,
    DefenderScoreExtractorSection,
    JudgmentExtractorHandler,
    PromptMaskExtractor,
    PromptMaskExtractorSection,
)
from fairseq2.recipes.lm._online_finetune._remote_model import RemoteVllmModel
from fairseq2.recipes.lm._online_finetune._templates_format import (
    format_instructions_attacker,
    format_instructions_defender,
)
from fairseq2.recipes.lm._online_finetune._templates_task import (
    task_instructions_attacker,
    task_instructions_defender,
)
from fairseq2.utils.list import (
    largmax_filtered,
    lflatten,
    lget_shape,
    linit,
    lunflatten,
)
from fairseq2.utils.validation import ValidationError, ValidationResult


@dataclass(kw_only=True)
class RewardModelConfig:
    answer_key: str = "answer"
    prompt_key: str = "prompt"
    tokenizer: str | None = None
    judgment_extractor: str | None = None


@dataclass(kw_only=True)
class RewardSection:
    name: str = "dummy"
    config: RewardModelConfig = field(default_factory=lambda: RewardModelConfig())


class VLLMOutputRewardHandler(ABC):
    @abstractmethod
    def create(
        self, reward_model: Any, gangs: Gangs, reward_config: object
    ) -> VLLMOutputReward: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


class VLLMOutputReward(ABC):
    @abstractmethod
    def process_rollouts(self, vllm_outputs: List[RequestOutput]): ...

    @abstractmethod
    def prepare_preference_batch(self, prompt_batch: PromptBatch, rollouts): ...


class GSM8kVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, reward_model, reward_name, reward_config, gangs, context):
        return GSM8kVerifier(
            answer_key=reward_config.answer_key,
            prompt_key=reward_config.prompt_key,
            reward_name=reward_name,
            gangs=gangs,
            context=context,
        )

    @property
    @override
    def name(self):
        return "gsm8k_verifier"

    @property
    @override
    def config_kls(self):
        return None


class GSM8kVerifier(VLLMOutputReward):
    def __init__(self, answer_key, prompt_key, reward_name, gangs, context):
        self.answer_re = re.compile(
            r"#### (\-?[0-9\.\,]+)"
        )  # regexp from original gsm8k to extract formatted answer
        self.invalid_answer = "[invalid]"
        self._gangs = gangs
        self._context = context
        self.answer_key = answer_key
        self.reward_name = reward_name
        self.prompt_key = prompt_key

    def extract_answer(self, completion: str):
        match = self.answer_re.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return self.invalid_answer

    @override
    def process_rollouts(
        self,
        vllm_outputs: List[RequestOutput],
        prompt_batch: PromptBatch,
    ):
        batch_text = []
        batch_tokens = []
        batch_rewards = []

        reference_answers = prompt_batch.meta_info.get(self.answer_key)

        for i, i_batch_request_output in enumerate(vllm_outputs):
            rollouts_text = []
            rollouts_tokens = []
            i_reference_answer = reference_answers[i]
            rollouts_rewards = []
            for rollout_output in i_batch_request_output.outputs:
                rollouts_text.append(rollout_output.text)
                rollouts_tokens.append(rollout_output.token_ids)
                predicted_answer = self.extract_answer(rollout_output.text)
                predicted_reward = 1 if predicted_answer == i_reference_answer else 0
                rollouts_rewards.append(predicted_reward)
            batch_text.append(rollouts_text)
            batch_tokens.append(rollouts_tokens)
            batch_rewards.append(rollouts_rewards)

        return {"text": batch_text, "tokens": batch_tokens, "rewards": batch_rewards}

    def prepare_preference_batch(
        self, prompt_batch: PromptBatch, rollouts
    ) -> PreferenceBatch:
        reward_output = self.process_rollouts(rollouts, prompt_batch)

        batch, is_bad_batch = prepare_preference_batch_random_pair(
            prompt_batch=prompt_batch, reward_output=reward_output, gangs=self._gangs
        )

        return batch, is_bad_batch, reward_output


class MathVerifyHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, reward_model, reward_name, reward_config, gangs, context):
        return MathVerifyVerifier(
            answer_key=reward_config.answer_key,
            prompt_key=reward_config.prompt_key,
            reward_name=reward_name,
            gangs=gangs,
            context=context,
        )

    @property
    @override
    def name(self):
        return "math_verify"

    @property
    @override
    def config_kls(self):
        return None


class MathVerifyVerifier(VLLMOutputReward):
    def __init__(self, answer_key, prompt_key, reward_name, gangs, context):
        try:
            from math_verify.metric import math_metric
            from math_verify.parser import (
                ExprExtractionConfig,
                LatexExtractionConfig,
                NormalizationConfig,
            )
        except ImportError:
            raise ImportError(
                "install mathverify from https://github.com/huggingface/Math-Verify"
            )

        self._gangs = gangs
        self._context = context
        self.answer_key = answer_key
        self.prompt_key = prompt_key
        self.reward_name = reward_name

        label_normalizer = NormalizationConfig(
            basic_latex=True,
            units=True,
            malformed_operators=True,
            nits=True,
            boxed="none",
            equations=False,
        )
        self.verify_func = math_metric(
            gold_extraction_target=(
                LatexExtractionConfig(normalization_config=label_normalizer),
            ),
            pred_extraction_target=(LatexExtractionConfig(boxed_match_priority=0),),
            aggregation_function=max,
            precision=6,
        )

    def verify_answer(self, completion: str, answer: str):
        # here we add extra $$ to label so that LatexExtractor works as expected
        if not answer.startswith("$"):
            answer = f"${answer}$"
        try:
            with _mute_output():
                grade, extracted_answers = self.verify_func([answer], [completion])
        except:
            grade = 0
            extracted_answers = None
        reward = 1.0 if grade == 1 else 0.0

        return reward, extracted_answers

    @override
    def process_rollouts(
        self,
        vllm_outputs: List[RequestOutput],
        prompt_batch: PromptBatch,
    ):
        batch_text = []
        batch_tokens = []
        batch_rewards = []

        reference_answers = prompt_batch.meta_info.get(self.answer_key)

        for i, i_batch_request_output in enumerate(vllm_outputs):
            rollouts_text = []
            rollouts_tokens = []
            i_reference_answer = reference_answers[i]
            rollouts_rewards = []
            for rollout_output in i_batch_request_output.outputs:
                rollouts_text.append(rollout_output.text)
                rollouts_tokens.append(rollout_output.token_ids)
                predicted_reward, predicted_answer = self.verify_answer(
                    rollout_output.text, i_reference_answer
                )
                rollouts_rewards.append(predicted_reward)
            batch_text.append(rollouts_text)
            batch_tokens.append(rollouts_tokens)
            batch_rewards.append(rollouts_rewards)

        return {"text": batch_text, "tokens": batch_tokens, "rewards": batch_rewards}

    def prepare_preference_batch(
        self, prompt_batch: PromptBatch, rollouts
    ) -> PreferenceBatch:
        reward_output = self.process_rollouts(rollouts, prompt_batch)

        batch, is_bad_batch = prepare_preference_batch_random_pair(
            prompt_batch=prompt_batch, reward_output=reward_output, gangs=self._gangs
        )

        return batch, is_bad_batch, reward_output


class AtheneVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, reward_model, reward_name, reward_config, gangs, context):
        if reward_config.tokenizer is not None:
            tokenizer = reward_config.tokenizer
        else:
            tokenizer = "Nexusflow/Athene-RM-8B"

        return AtheneVerifier(
            gangs,
            context,
            reward_model,
            reward_name=reward_name,
            answer_key=reward_config.answer_key,
            prompt_key=reward_config.prompt_key,
            tokenizer=tokenizer,
        )

    @property
    @override
    def name(self):
        return "athene_verifier"

    @property
    @override
    def config_kls(self):
        return None


class AtheneVerifier(VLLMOutputReward):
    """
    A reward model verifier that processes rollouts using the Athene reward model.

    This class evaluates rollouts generated by vLLM by wrapping the prompt and rollout text into a specific format and passing it through the Athene reward model.

    Note: this relies on modified Athene-RM-8B code to ensure compatibility with vLLM.
    """

    def __init__(
        self,
        gangs,
        context,
        reward_model,
        reward_name,
        answer_key,
        prompt_key,
        tokenizer,
    ):
        self.answer_key = answer_key
        self.prompt_key = prompt_key
        self._gangs = gangs
        self._context = context
        self.reward_model = reward_model
        self.reward_name = reward_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def format_prompt(self, prompt_text, rollout_text):
        messages = [
            {
                "role": "user",
                "content": prompt_text,
            },
            {
                "role": "assistant",
                "content": rollout_text,
            },
        ]

        return messages

    @override
    def process_rollouts(
        self, vllm_outputs: List[RequestOutput], prompt_batch: PromptBatch
    ):
        vllm_inputs = []
        batch_text = []
        batch_tokens = []

        if vllm_outputs is None:
            vllm_outputs = [None] * len(prompt_batch.prompts)

        text_prompts = prompt_batch.meta_info.get(self.prompt_key)
        for i, (i_batch_request_output, prompt_text) in enumerate(
            zip(vllm_outputs, text_prompts)
        ):
            rollouts_text = []
            rollouts_tokens = []
            for rollout_output in i_batch_request_output.outputs:
                rollout_text = rollout_output.text
                vllm_input = self.format_prompt(prompt_text, rollout_text)
                vllm_inputs.append(vllm_input)
                rollouts_text.append(rollout_output.text)
                rollouts_tokens.append(rollout_output.token_ids)

            batch_text.append(rollouts_text)
            batch_tokens.append(rollouts_tokens)

        batch_rewards = generate_rewards(
            vllm_inputs, dp_gang=self._gangs.dp, vllm_model=self.reward_model
        )

        # reshape batch_rewards to [Batch, Rollouts]
        B, R = len(batch_text), len(batch_text[0])  # batch size, rollouts
        batch_rewards = [batch_rewards[i * R : (i + 1) * R] for i in range(B)]

        return {"text": batch_text, "tokens": batch_tokens, "rewards": batch_rewards}

    def prepare_preference_batch(
        self, prompt_batch: PromptBatch, rollouts
    ) -> PreferenceBatch:
        reward_output = self.process_rollouts(rollouts, prompt_batch)

        chosen_batch = []
        rejected_batch = []
        prompt_lens = []
        dummy_batch_ids = []  # keep posiitons of dummy pairs here

        # choosing first rollouts with reward 1 as chosen and 0 as rejected (sort of random given that we sample rollouts randomly)
        for i_batch, (i_batch_rewards, i_batch_tokens) in enumerate(
            zip(reward_output["rewards"], reward_output["tokens"])
        ):
            chosen_rollout_position = i_batch_rewards.index(max(i_batch_rewards))
            rejected_rollout_position = i_batch_rewards.index(min(i_batch_rewards))

            if chosen_rollout_position == rejected_rollout_position:
                # cant form preference pair when we dont have such rollouts
                # this will be dummy batch and we zero out loss
                dummy_batch_ids.append(i_batch)

            chosen_rollout_tokens = list(i_batch_tokens[chosen_rollout_position])
            rejected_rollout_tokens = list(i_batch_tokens[rejected_rollout_position])
            prompt_tokens = prompt_batch.prompts[i_batch]

            chosen_tokens = prompt_tokens + chosen_rollout_tokens
            chosen_batch.append(chosen_tokens)

            rejected_tokens = prompt_tokens + rejected_rollout_tokens
            rejected_batch.append(rejected_tokens)

            prompt_lens.append(len(prompt_tokens))

        filter_batch = lambda batch: [
            item for index, item in enumerate(batch) if index not in dummy_batch_ids
        ]

        if len(dummy_batch_ids) == len(reward_output["tokens"]):
            # entire batch does not have a valid preference pair
            # we use it as dummy batch and zero the loss in the end
            is_bad_batch = True
        else:
            # removing dummy pairs from the batch
            chosen_batch = filter_batch(chosen_batch)
            rejected_batch = filter_batch(rejected_batch)
            prompt_lens = filter_batch(prompt_lens)
            is_bad_batch = False

        prompt_lens = torch.tensor(prompt_lens)

        chosen_batch = [
            torch.tensor(sequence, device=self._gangs.dp.device)
            for sequence in chosen_batch
        ]
        chosen_batch = collate_with_target_mask(
            chosen_batch, prompt_lens, device=self._gangs.dp.device
        )

        rejected_batch = [
            torch.tensor(sequence, device=self._gangs.dp.device)
            for sequence in rejected_batch
        ]
        rejected_batch = collate_with_target_mask(
            rejected_batch, prompt_lens, device=self._gangs.dp.device
        )

        batch = PreferenceBatch(
            chosen=chosen_batch,
            rejected=rejected_batch,
            reference_score_chosen=None,
            reference_score_rejected=None,
        )

        return batch, is_bad_batch, reward_output


class GenerativePointwiseVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, reward_model, reward_name, reward_config, gangs, context):
        if reward_config.tokenizer is None:
            raise RuntimeError("Generative judges require tokenizer")

        if reward_config.judgment_extractor is None:
            raise RuntimeError(
                "Generative judges require implementing and specifying a judgment extractor"
            )

        return GenerativePointwiseVerifier(
            gangs,
            context,
            reward_model,
            reward_name,
            judgment_extractor=reward_config.judgment_extractor,
            answer_key=reward_config.answer_key,
            prompt_key=reward_config.prompt_key,
            tokenizer=reward_config.tokenizer,
        )

    @property
    @override
    def name(self):
        return "generative_pointwise_verifier"

    @property
    @override
    def config_kls(self):
        return None


class GenerativePointwiseVerifier(VLLMOutputReward):
    def __init__(
        self,
        gangs,
        context,
        reward_model,
        reward_name,
        judgment_extractor,
        answer_key,
        prompt_key,
        tokenizer,
    ):
        self.answer_key = answer_key
        self.prompt_key = prompt_key
        self._gangs = gangs
        self._context = context
        self.reward_model = reward_model
        self.reward_name = reward_name
        self.judgment_extractor = judgment_extractor
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        judgment_extractor_registry = self._context.get_registry(
            JudgmentExtractorHandler
        )
        judgment_extractor_handler = judgment_extractor_registry.get(judgment_extractor)
        self.judgment_extractor = judgment_extractor_handler.create()

    @override
    def process_rollouts(
        self, vllm_outputs: List[RequestOutput], prompt_batch: PromptBatch
    ):
        vllm_inputs = []
        batch_text = []
        batch_tokens = []

        if vllm_outputs is None:
            vllm_outputs = [None] * len(prompt_batch.prompts)

        text_prompts = prompt_batch.meta_info.get(self.prompt_key)
        reference_answers = prompt_batch.meta_info.get(self.answer_key)
        for i, (i_batch_request_output, prompt_text) in enumerate(
            zip(vllm_outputs, text_prompts)
        ):
            rollouts_text = []
            rollouts_tokens = []
            i_reference_answer = reference_answers[i]
            for rollout_output in i_batch_request_output.outputs:
                rollout_text = rollout_output.text
                vllm_input = self.judgment_extractor.format_prompt(
                    prompt_text, rollout_text, i_reference_answer
                )
                vllm_inputs.append(vllm_input)
                rollouts_text.append(rollout_output.text)
                rollouts_tokens.append(rollout_output.token_ids)

            batch_text.append(rollouts_text)
            batch_tokens.append(rollouts_tokens)

        batch_judgments = generate_rewards_generative(
            vllm_inputs, dp_gang=self._gangs.dp, vllm_model=self.reward_model
        )

        batch_rewards = []
        for per_rollout_judgments in batch_judgments:
            per_rollout_rewards = [
                self.judgment_extractor.extract(judgment.text)
                for judgment in per_rollout_judgments.outputs
            ]
            batch_rewards.append(self.judgment_extractor.aggregate(per_rollout_rewards))

        # reshape batch_rewards to [Batch, Rollouts]
        B, R = len(batch_text), len(batch_text[0])  # batch size, rollouts
        batch_rewards = [batch_rewards[i * R : (i + 1) * R] for i in range(B)]

        return {"text": batch_text, "tokens": batch_tokens, "rewards": batch_rewards}

    def prepare_preference_batch(
        self, prompt_batch: PromptBatch, rollouts
    ) -> PreferenceBatch:
        reward_output = self.process_rollouts(rollouts, prompt_batch)

        chosen_batch = []
        rejected_batch = []
        prompt_lens = []
        dummy_batch_ids = []  # keep posiitons of dummy pairs here

        # choosing first rollouts with reward 1 as chosen and 0 as rejected (sort of random given that we sample rollouts randomly)
        for i_batch, (i_batch_rewards, i_batch_tokens) in enumerate(
            zip(reward_output["rewards"], reward_output["tokens"])
        ):
            chosen_rollout_position = i_batch_rewards.index(max(i_batch_rewards))
            rejected_rollout_position = i_batch_rewards.index(min(i_batch_rewards))

            if chosen_rollout_position == rejected_rollout_position:
                # cant form preference pair when we dont have such rollouts
                # this will be dummy batch and we zero out loss
                dummy_batch_ids.append(i_batch)

            chosen_rollout_tokens = list(i_batch_tokens[chosen_rollout_position])
            rejected_rollout_tokens = list(i_batch_tokens[rejected_rollout_position])
            prompt_tokens = prompt_batch.prompts[i_batch]

            chosen_tokens = prompt_tokens + chosen_rollout_tokens
            chosen_batch.append(chosen_tokens)

            rejected_tokens = prompt_tokens + rejected_rollout_tokens
            rejected_batch.append(rejected_tokens)

            prompt_lens.append(len(prompt_tokens))

        filter_batch = lambda batch: [
            item for index, item in enumerate(batch) if index not in dummy_batch_ids
        ]

        if len(dummy_batch_ids) == len(reward_output["tokens"]):
            # entire batch does not have a valid preference pair
            # we use it as dummy batch and zero the loss in the end
            is_bad_batch = True
        else:
            # removing dummy pairs from the batch
            chosen_batch = filter_batch(chosen_batch)
            rejected_batch = filter_batch(rejected_batch)
            prompt_lens = filter_batch(prompt_lens)
            is_bad_batch = False

        prompt_lens = torch.tensor(prompt_lens)

        chosen_batch = [
            torch.tensor(sequence, device=self._gangs.dp.device)
            for sequence in chosen_batch
        ]
        chosen_batch = collate_with_target_mask(
            chosen_batch, prompt_lens, device=self._gangs.dp.device
        )

        rejected_batch = [
            torch.tensor(sequence, device=self._gangs.dp.device)
            for sequence in rejected_batch
        ]
        rejected_batch = collate_with_target_mask(
            rejected_batch, prompt_lens, device=self._gangs.dp.device
        )

        batch = PreferenceBatch(
            chosen=chosen_batch,
            rejected=rejected_batch,
            reference_score_chosen=None,
            reference_score_rejected=None,
        )

        return batch, is_bad_batch, reward_output


class GenerativePairwiseVerifierHandler(VLLMOutputRewardHandler):
    def __init__(self):
        pass

    @override
    def create(self, reward_model, reward_name, reward_config, gangs, context):
        if reward_config.tokenizer is None:
            raise RuntimeError("Generative judges require tokenizer")

        if reward_config.judgment_extractor is None:
            raise RuntimeError(
                "Generative judges require implementing and specifying a judgment extractor"
            )

        return GenerativePairwiseVerifier(
            gangs,
            context,
            reward_model,
            reward_name,
            judgment_extractor=reward_config.judgment_extractor,
            answer_key=reward_config.answer_key,
            prompt_key=reward_config.prompt_key,
            tokenizer=reward_config.tokenizer,
        )

    @property
    @override
    def name(self):
        return "generative_pairwise_verifier"

    @property
    @override
    def config_kls(self):
        return None


class GenerativePairwiseVerifier(VLLMOutputReward):
    def __init__(
        self,
        gangs,
        context,
        reward_model,
        reward_name,
        judgment_extractor,
        answer_key,
        prompt_key,
        tokenizer,
    ):
        self.answer_key = answer_key
        self.prompt_key = prompt_key
        self._gangs = gangs
        self._context = context
        self.reward_model = reward_model
        self.reward_name = reward_name
        self.judgment_extractor = judgment_extractor
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        judgment_extractor_registry = self._context.get_registry(
            JudgmentExtractorHandler
        )
        judgment_extractor_handler = judgment_extractor_registry.get(judgment_extractor)
        self.judgment_extractor = judgment_extractor_handler.create()

    @override
    def process_rollouts(
        self, vllm_outputs: List[RequestOutput], prompt_batch: PromptBatch
    ):
        vllm_inputs = []
        batch_text = []
        batch_tokens = []
        batch_pairwise_indices = []

        if vllm_outputs is None:
            vllm_outputs = [None] * len(prompt_batch.prompts)

        text_prompts = prompt_batch.meta_info.get(self.prompt_key)
        for i, (i_batch_request_output, prompt_text) in enumerate(
            zip(vllm_outputs, text_prompts)
        ):
            rollouts_text = [
                rollout_output.text for rollout_output in i_batch_request_output.outputs
            ]
            rollouts_tokens = [
                rollout_output.token_ids
                for rollout_output in i_batch_request_output.outputs
            ]
            batch_text.append(rollouts_text)
            batch_tokens.append(rollouts_tokens)

            prompt_pairwise_indices = []
            for a in range(len(i_batch_request_output.outputs)):
                for b in range(len(i_batch_request_output.outputs)):
                    if a != b:
                        rollout_A_text = i_batch_request_output.outputs[a].text
                        rollout_B_text = i_batch_request_output.outputs[b].text
                        vllm_input = self.judgment_extractor.format_prompt(
                            prompt_text, rollout_A_text, rollout_B_text
                        )
                        vllm_inputs.append(vllm_input)
                        prompt_pairwise_indices.append((a, b))

            batch_pairwise_indices.append(prompt_pairwise_indices)

        batch_pairwise_judgments = generate_rewards_generative(
            vllm_inputs,
            dp_gang=self._gangs.dp,
            vllm_model=self.reward_model,
        )

        batch_pairwise_rewards = []
        for per_rollout_judgments in batch_pairwise_judgments:
            per_rollout_rewards = [
                self.judgment_extractor.extract(judgment.text)
                for judgment in per_rollout_judgments.outputs
            ]
            batch_pairwise_rewards.append(
                self.judgment_extractor.aggregate(per_rollout_rewards)
            )

        B, R = len(batch_text), len(batch_text[0])  # batch size, rollouts

        # Logic to convert pairwise scores into pointwise rewards
        # Can be done differently too
        batch_rewards = []
        for i in range(B):
            prompt_pairwise_rewards = batch_pairwise_rewards[
                i * R * (R - 1) : (i + 1) * R * (R - 1)
            ]
            prompt_pairwise_indices = batch_pairwise_indices[i]
            prompt_rewards = [0.0] * R
            for index, rewards in zip(prompt_pairwise_indices, prompt_pairwise_rewards):
                prompt_rewards[index[0]] += rewards[0]
                prompt_rewards[index[1]] += rewards[1]

            # Average score over 2*(R-1) pairwise comparisons
            if (R - 1) > 0:
                prompt_rewards = [
                    round(prompt_reward / (2 * (R - 1)), 4)
                    for prompt_reward in prompt_rewards
                ]

            batch_rewards.append(prompt_rewards)

        return {"text": batch_text, "tokens": batch_tokens, "rewards": batch_rewards}

    def prepare_preference_batch(
        self, prompt_batch: PromptBatch, rollouts
    ) -> PreferenceBatch:
        reward_output = self.process_rollouts(rollouts, prompt_batch)

        chosen_batch = []
        rejected_batch = []
        prompt_lens = []
        dummy_batch_ids = []  # keep posiitons of dummy pairs here

        # choosing first rollouts with reward 1 as chosen and 0 as rejected (sort of random given that we sample rollouts randomly)
        for i_batch, (i_batch_rewards, i_batch_tokens) in enumerate(
            zip(reward_output["rewards"], reward_output["tokens"])
        ):
            chosen_rollout_position = i_batch_rewards.index(max(i_batch_rewards))
            rejected_rollout_position = i_batch_rewards.index(min(i_batch_rewards))

            if chosen_rollout_position == rejected_rollout_position:
                # cant form preference pair when we dont have such rollouts
                # this will be dummy batch and we zero out loss
                dummy_batch_ids.append(i_batch)

            chosen_rollout_tokens = list(i_batch_tokens[chosen_rollout_position])
            rejected_rollout_tokens = list(i_batch_tokens[rejected_rollout_position])
            prompt_tokens = prompt_batch.prompts[i_batch]

            chosen_tokens = prompt_tokens + chosen_rollout_tokens
            chosen_batch.append(chosen_tokens)

            rejected_tokens = prompt_tokens + rejected_rollout_tokens
            rejected_batch.append(rejected_tokens)

            prompt_lens.append(len(prompt_tokens))

        filter_batch = lambda batch: [
            item for index, item in enumerate(batch) if index not in dummy_batch_ids
        ]

        if len(dummy_batch_ids) == len(reward_output["tokens"]):
            # entire batch does not have a valid preference pair
            # we use it as dummy batch and zero the loss in the end
            is_bad_batch = True
        else:
            # removing dummy pairs from the batch
            chosen_batch = filter_batch(chosen_batch)
            rejected_batch = filter_batch(rejected_batch)
            prompt_lens = filter_batch(prompt_lens)
            is_bad_batch = False

        prompt_lens = torch.tensor(prompt_lens)

        chosen_batch = [
            torch.tensor(sequence, device=self._gangs.dp.device)
            for sequence in chosen_batch
        ]
        chosen_batch = collate_with_target_mask(
            chosen_batch, prompt_lens, device=self._gangs.dp.device
        )

        rejected_batch = [
            torch.tensor(sequence, device=self._gangs.dp.device)
            for sequence in rejected_batch
        ]
        rejected_batch = collate_with_target_mask(
            rejected_batch, prompt_lens, device=self._gangs.dp.device
        )

        batch = PreferenceBatch(
            chosen=chosen_batch,
            rejected=rejected_batch,
            reference_score_chosen=None,
            reference_score_rejected=None,
        )

        return batch, is_bad_batch, reward_output


@dataclass(kw_only=True)
class RolloutTreeSection:
    log_rollouts: bool = True
    optimistic_attacker: bool = True
    num_attacker_samples: int = 2
    num_defender_samples: int = 2
    validation_num_attacker_samples: int = 2
    validation_num_defender_samples: int = 2
    reward_normalization_mode: str = "across_seeds"
    attacker_task_instruction_benign_name: str = "dummy"
    attacker_task_instruction_harmful_name: str = "dummy"
    attacker_format_instruction_name: str = "dummy"
    defender_task_instruction_name: str = "dummy"
    defender_format_instruction_name: str = "dummy"

    def validate(self) -> None:
        result = ValidationResult()

        if self.num_attacker_samples <= 0:
            result.add_error("`num_attacker_samples` must be > 0.")

        if self.num_defender_samples <= 0:
            result.add_error("`num_defender_samples` must be > 0.")

        if self.validation_num_attacker_samples <= 0:
            result.add_error("`validation_num_attacker_samples` must be > 0.")

        if self.validation_num_defender_samples <= 0:
            result.add_error("`validation_num_defender_samples` must be > 0.")

        task_instructions_attacker_keys = list(task_instructions_attacker.keys())
        if (
            self.attacker_task_instruction_benign_name
            not in task_instructions_attacker_keys
        ):
            result.add_error(
                f"`attacker_task_instruction_benign_name` must be one of {task_instructions_attacker_keys}."
            )

        if (
            self.attacker_task_instruction_harmful_name
            not in task_instructions_attacker_keys
        ):
            result.add_error(
                f"`attacker_task_instruction_harmful_name` must be one of {task_instructions_attacker_keys}."
            )

        format_instructions_attacker_keys = list(format_instructions_attacker.keys())
        if (
            self.attacker_format_instruction_name
            not in format_instructions_attacker_keys
        ):
            result.add_error(
                f"`attacker_format_instruction_name` must be one of {format_instructions_attacker_keys}."
            )

        task_instructions_defender_keys = list(task_instructions_defender.keys())
        if self.defender_task_instruction_name not in task_instructions_defender_keys:
            result.add_error(
                f"`defender_task_instruction_name` must be one of {task_instructions_defender_keys}."
            )

        format_instructions_defender_keys = list(format_instructions_defender.keys())
        if (
            self.defender_format_instruction_name
            not in format_instructions_defender_keys
        ):
            result.add_error(
                f"`defender_format_instruction_name` must be one of {format_instructions_defender_keys}."
            )

        if result.has_error:
            raise ValidationError(
                "The GRPO finetuning configuration section has one or more validation errors:",
                result,  # fmt: skip
            )

        if self.reward_normalization_mode not in ["across_seeds", "per_seed"]:
            result.add_error(
                "`reward_normalization_mode` must be one of ['across_seeds', 'per_seed']."
            )


@dataclass(kw_only=True)
class RewardGameModelConfig:
    validation_vllm_attacker_sampling_params: Dict[str, Any] = field(
        default_factory=lambda: {}
    )
    """VLLM sampling params for attacker validation. If not set, the same params as training will be used."""

    validation_vllm_defender_sampling_params: Dict[str, Any] = field(
        default_factory=lambda: {}
    )
    """VLLM sampling params for defender validation. If not set, the same params as training will be used."""

    vllm_model_attacker_actor_name: str = "vllm_model_attacker"
    """Name of the Ray vLLM actor used to generate attacker policy rollouts."""

    vllm_model_defender_actor_name: str = "vllm_model_defender"
    """Name of the Ray vLLM actor used to generate defender policy rollouts."""

    vllm_reward_model_actor_name: str = "vllm_reward"
    """Optional name of the Ray vLLM actor used as a reward model."""

    tokenizer_attacker_name: str = "dummy"
    tokenizer_defender_name: str = "dummy"
    tokenizer_reward_name: str = "dummy"

    rollout_tree: RolloutTreeSection = field(
        default_factory=lambda: RolloutTreeSection()
    )

    judgment_extractor_prompt_mask: PromptMaskExtractorSection | None = field(
        default_factory=lambda: PromptMaskExtractorSection(
            name="prompt_mask_pointwise_extractor"
        )
    )
    judgment_extractor_attacker: AttackerScoreExtractorSection | None = field(
        default_factory=lambda: AttackerScoreExtractorSection(
            name="attacker_pairwise_score_extractor"
        )
    )
    judgment_extractor_defender: DefenderScoreExtractorSection | None = field(
        default_factory=lambda: DefenderScoreExtractorSection(
            name="defender_pairwise_score_extractor"
        )
    )


@dataclass(kw_only=True)
class RewardGameSection:
    name: str = "dummy"
    config: RewardGameModelConfig = field(
        default_factory=lambda: RewardGameModelConfig()
    )


class VLLMOutputRewardGameHandler(ABC):
    @abstractmethod
    def create(
        self,
        reward_model: RemoteVllmModel,
        tokenizer: Any,
        gangs: Gangs,
        reward_config: RewardGameModelConfig,
        context: RuntimeContext,
    ) -> VLLMOutputRewardGame: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def config_kls(self) -> type[object]: ...


class VLLMOutputRewardGame(ABC):
    @abstractmethod
    def generate_scored_rollout_tree(
        self, s_prompt_batch: PromptBatch, **kwargs
    ) -> ScoredRolloutTree | ScoredRolloutTreeDirect: ...

    @abstractmethod
    def generate_rollout_tree(
        self, s_prompt_batch: PromptBatch, **kwargs
    ) -> RolloutTree: ...

    @abstractmethod
    def score_rollout_tree(
        self, rt: RolloutTree, metric_bag: MetricBag, prefix: str
    ) -> ScoredRolloutTree: ...

    @abstractmethod
    def generate_rollout_tree_direct(
        self, s_prompt_batch: PromptBatch, **kwargs
    ) -> RolloutTreeDirect: ...

    @abstractmethod
    def score_rollout_tree_direct(
        self, rtd: RolloutTreeDirect, metric_bag: MetricBag
    ) -> ScoredRolloutTreeDirect: ...


class GenerativeGameVerifierHandler(VLLMOutputRewardGameHandler):
    def __init__(self):
        pass

    @override
    def create(
        self,
        vllm_actors: Dict[str, RemoteVllmModel],
        reward_config: RewardGameModelConfig,
        gangs: Gangs,
        context: RuntimeContext,
    ) -> GenerativeGameVerifier:
        vllm_model_attacker = vllm_actors[reward_config.vllm_model_attacker_actor_name]
        vllm_model_defender = vllm_actors[reward_config.vllm_model_defender_actor_name]
        vllm_reward_model = vllm_actors[reward_config.vllm_reward_model_actor_name]

        tokenizer_attacker = AutoTokenizer.from_pretrained(
            reward_config.tokenizer_attacker_name
        )
        tokenizer_defender = AutoTokenizer.from_pretrained(
            reward_config.tokenizer_defender_name
        )
        tokenizer_reward = AutoTokenizer.from_pretrained(
            reward_config.tokenizer_reward_name
        )

        judgment_extractor_registry = context.get_registry(JudgmentExtractorHandler)

        if reward_config.judgment_extractor_prompt_mask is None:
            judgment_extractor_prompt_mask = None
        else:
            judgment_extractor_handler_prompt_mask = judgment_extractor_registry.get(
                reward_config.judgment_extractor_prompt_mask.name
            )
            judgment_extractor_prompt_mask = judgment_extractor_handler_prompt_mask.create(
                gangs=gangs,
                reward_model=vllm_reward_model,
                tokenizer=tokenizer_reward,
                judgment_extractor_config=reward_config.judgment_extractor_prompt_mask.config,
            )

        if reward_config.judgment_extractor_attacker is None:
            judgment_extractor_attacker = None
        else:
            judgment_extractor_handler_attacker = judgment_extractor_registry.get(
                reward_config.judgment_extractor_attacker.name
            )
            judgment_extractor_attacker = judgment_extractor_handler_attacker.create(
                gangs=gangs,
                reward_model=vllm_reward_model,
                tokenizer=tokenizer_reward,
                judgment_extractor_config=reward_config.judgment_extractor_attacker.config,
            )

        if reward_config.judgment_extractor_defender is None:
            judgment_extractor_defender = None
        else:
            judgment_extractor_handler_defender = judgment_extractor_registry.get(
                reward_config.judgment_extractor_defender.name
            )
            judgment_extractor_defender = judgment_extractor_handler_defender.create(
                gangs=gangs,
                reward_model=vllm_reward_model,
                tokenizer=tokenizer_reward,
                judgment_extractor_config=reward_config.judgment_extractor_defender.config,
            )

        return GenerativeGameVerifier(
            rollout_tree_config=reward_config.rollout_tree,
            validation_vllm_attacker_sampling_params=reward_config.validation_vllm_attacker_sampling_params,
            validation_vllm_defender_sampling_params=reward_config.validation_vllm_defender_sampling_params,
            vllm_model_attacker=vllm_model_attacker,
            vllm_model_defender=vllm_model_defender,
            vllm_reward_model=vllm_reward_model,
            tokenizer_attacker=tokenizer_attacker,
            tokenizer_defender=tokenizer_defender,
            judgment_extractor_prompt_mask=judgment_extractor_prompt_mask,
            judgment_extractor_attacker=judgment_extractor_attacker,
            judgment_extractor_defender=judgment_extractor_defender,
            gangs=gangs,
            context=context,
        )

    @property
    @override
    def name(self):
        return "generative_game_verifier"

    @property
    @override
    def config_kls(self):
        return None


@dataclass
class GRPOBatch:
    """Represents a GRPO dataset batch."""

    prompt_rollouts: SequenceBatch
    prompt_lengths: List[int]
    rewards: torch.Tensor
    dummy_mask: torch.Tensor
    vllm_logps: torch.Tensor


class GenerativeGameVerifier(VLLMOutputRewardGame):
    def __init__(
        self,
        rollout_tree_config: RolloutTreeSection,
        validation_vllm_attacker_sampling_params: Dict[str, Any],
        validation_vllm_defender_sampling_params: Dict[str, Any],
        vllm_model_attacker: RemoteVllmModel,
        vllm_model_defender: RemoteVllmModel,
        vllm_reward_model: RemoteVllmModel,
        tokenizer_attacker: Any,
        tokenizer_defender: Any,
        judgment_extractor_prompt_mask: PromptMaskExtractor | None,
        judgment_extractor_attacker: AttackerScoreExtractor | None,
        judgment_extractor_defender: DefenderScoreExtractor | None,
        gangs: Gangs,
        context: RuntimeContext,
    ):
        self._rollout_tree_config = rollout_tree_config
        self._vllm_model_attacker = vllm_model_attacker
        self._vllm_model_defender = vllm_model_defender
        self._vllm_reward_model = vllm_reward_model
        self._tokenizer_attacker = tokenizer_attacker
        self._tokenizer_defender = tokenizer_defender
        self._judgment_extractor_prompt_mask = judgment_extractor_prompt_mask
        self._judgment_extractor_attacker = judgment_extractor_attacker
        self._judgment_extractor_defender = judgment_extractor_defender
        self._gangs = gangs
        self._context = context

        self._attacker_format_instruction, attacker_regex_pattern = (
            format_instructions_attacker[
                self._rollout_tree_config.attacker_format_instruction_name
            ]
        )
        self._attacker_regex_pattern = re.compile(
            attacker_regex_pattern, re.IGNORECASE | re.DOTALL
        )

        self._defender_format_instruction, defender_regex_pattern = (
            format_instructions_defender[
                self._rollout_tree_config.defender_format_instruction_name
            ]
        )
        self._defender_regex_pattern = re.compile(
            defender_regex_pattern, re.IGNORECASE | re.DOTALL
        )

        if self._gangs.dp.rank == 0:
            self.train_vllm_attacker_sampling_params = copy(
                self._vllm_model_attacker.sampling_params
            )
            self.validation_vllm_attacker_sampling_params = copy(
                self._vllm_model_attacker.sampling_params
            )
            for (
                k,
                v,
            ) in validation_vllm_attacker_sampling_params.items():
                self.validation_vllm_attacker_sampling_params.__setattr__(k, v)

            self.train_vllm_defender_sampling_params = copy(
                self._vllm_model_defender.sampling_params
            )
            self.validation_vllm_defender_sampling_params = copy(
                self._vllm_model_defender.sampling_params
            )
            for (
                k,
                v,
            ) in validation_vllm_defender_sampling_params.items():
                self.validation_vllm_defender_sampling_params.__setattr__(k, v)
        else:
            self.train_vllm_attacker_sampling_params = None
            self.validation_vllm_attacker_sampling_params = None
            self.train_vllm_defender_sampling_params = None
            self.validation_vllm_defender_sampling_params = None

    @property
    @override
    def vllm_model_attacker(self) -> RemoteVllmModel:
        return self._vllm_model_attacker

    @property
    @override
    def vllm_model_defender(self) -> RemoteVllmModel:
        return self._vllm_model_defender

    def generate_scored_rollout_tree(
        self,
        s_prompt_batch: PromptBatch,
        train: bool,
        metric_bag: MetricBag,
        prefix: str,
        skip_attacker: bool,
    ) -> ScoredRolloutTree | ScoredRolloutTreeDirect:
        if skip_attacker:
            rtd = self.generate_rollout_tree_direct(
                s_prompt_batch=s_prompt_batch,
                train=train,
                metric_bag=metric_bag,
                prefix=prefix,
            )
            srtd = self.score_rollout_tree_direct(
                rtd, metric_bag=metric_bag, prefix=prefix
            )
            return srtd
        else:
            rt = self.generate_rollout_tree(
                s_prompt_batch=s_prompt_batch,
                train=train,
                metric_bag=metric_bag,
                prefix=prefix,
            )
            srt = self.score_rollout_tree(rt, metric_bag=metric_bag, prefix=prefix)
            return srt

    def generate_rollout_tree(
        self,
        s_prompt_batch: PromptBatch,
        train: bool,
        metric_bag: MetricBag,
        prefix: str,
    ) -> RolloutTree:
        log.info(f"[{prefix}] Start generating rollout tree")
        time_before_generate = time.time()

        s_category_B = s_prompt_batch.meta_info.get("category")
        s_text_B = s_prompt_batch.meta_info.get("prompt_text")

        if train:
            vllm_attacker_sampling_params = self.train_vllm_attacker_sampling_params
            vllm_defender_sampling_params = self.train_vllm_defender_sampling_params
        else:
            vllm_attacker_sampling_params = (
                self.validation_vllm_attacker_sampling_params
            )
            vllm_defender_sampling_params = (
                self.validation_vllm_defender_sampling_params
            )

        B = len(s_text_B)
        if train:
            A = self._rollout_tree_config.num_attacker_samples
            D = self._rollout_tree_config.num_defender_samples
        else:
            A = self._rollout_tree_config.validation_num_attacker_samples
            D = self._rollout_tree_config.validation_num_defender_samples

        s_emb_tokens_B = linit([B])
        s_emb_text_B = linit([B])
        for b in range(B):
            if s_category_B[b] == "benign":
                task_instruction_template = task_instructions_attacker[
                    self._rollout_tree_config.attacker_task_instruction_benign_name
                ]
            elif s_category_B[b] == "harmful":
                task_instruction_template = task_instructions_attacker[
                    self._rollout_tree_config.attacker_task_instruction_harmful_name
                ]
            else:
                raise ValueError(f"Unknown category {s_category_B[b]}.")

            task_instruction = task_instruction_template.format(query=s_text_B[b])

            s_formatted_text = task_instruction + self._attacker_format_instruction

            s_wrapped_text = [{"role": "user", "content": s_formatted_text}]
            s_emb_text_B[b] = self._tokenizer_attacker.apply_chat_template(
                s_wrapped_text, tokenize=False, add_generation_prompt=True
            )
            s_emb_tokens_B[b] = self._tokenizer_attacker.apply_chat_template(
                s_wrapped_text, tokenize=True, add_generation_prompt=True
            )

        s_emb_tokens_BA = linit([B, A])
        for b, a in product(range(B), range(A)):
            s_emb_tokens_BA[b][a] = s_emb_tokens_B[b]

        # K = B * A
        s_emb_tokens_K = lflatten(s_emb_tokens_BA, depth=1)

        x_raw_rollout_K = generate_rollouts(
            s_emb_tokens_K,
            dp_gang=self._gangs.dp,
            vllm_model=self._vllm_model_attacker,
            sampling_params=vllm_attacker_sampling_params,
        )
        x_rollout_BA = lunflatten(x_raw_rollout_K, [B, A])

        x_raw_text_BA = linit([B, A])
        x_raw_tokens_BA = linit([B, A])
        x_text_BA = linit([B, A])
        x_tokens_BA = linit([B, A])
        x_cot_text_BA = linit([B, A])
        x_cot_tokens_BA = linit([B, A])
        x_emb_text_BA = linit([B, A])
        x_emb_tokens_BA = linit([B, A])
        x_error_BA = linit([B, A])
        for b, a in product(range(B), range(A)):
            x_raw_outputs = x_rollout_BA[b][a].outputs
            x_raw_output = x_raw_outputs[0]
            if len(x_raw_outputs) != 1:
                raise ValueError(
                    f"Expected exactly one output per rollout, got {len(x_raw_outputs)}. Instead of adjusting n in vllm_sampling_params, change num_attacker_samples in the config."
                )

            x_raw_text_BA[b][a] = x_raw_output.text
            x_raw_tokens_BA[b][a] = x_raw_output.token_ids

            if x_raw_output.finish_reason != "stop":
                x_error_BA[b][a] = xError.MAX_LENGTH
            else:
                match = self._attacker_regex_pattern.match(x_raw_output.text)
                if match:
                    x_cot_text = match.group(1).strip()
                    x_cot_text_BA[b][a] = x_cot_text
                    x_cot_tokens_BA[b][a] = self._tokenizer_defender.tokenize(
                        x_cot_text
                    )

                    x_text = match.group(2).strip()
                    x_text_BA[b][a] = x_text
                    x_tokens_BA[b][a] = self._tokenizer_defender.tokenize(x_text)

                    task_instruction_template = task_instructions_defender[
                        self._rollout_tree_config.defender_task_instruction_name
                    ]
                    task_instruction = task_instruction_template.format(prompt=x_text)
                    x_formatted_text = (
                        task_instruction + self._defender_format_instruction
                    )
                    x_wrapped_text = [{"role": "user", "content": x_formatted_text}]
                    x_emb_text_BA[b][a] = self._tokenizer_defender.apply_chat_template(
                        x_wrapped_text, tokenize=False, add_generation_prompt=True
                    )
                    x_emb_tokens_BA[b][a] = (
                        self._tokenizer_defender.apply_chat_template(
                            x_wrapped_text, tokenize=True, add_generation_prompt=True
                        )
                    )

                    x_error_BA[b][a] = xError.NO_ERROR
                else:
                    x_error_BA[b][a] = xError.PARSING

        if self._judgment_extractor_prompt_mask is not None:
            x_is_faithful_BA, x_error_BA = self._judgment_extractor_prompt_mask(
                s_category_B=s_category_B,
                s_text_B=s_text_B,
                x_text_BA=x_text_BA,
                x_error_BA=x_error_BA,
                prefix=prefix,
            )
            for b, a in product(range(B), range(A)):
                if x_error_BA[b][a] == xError.NO_ERROR and not x_is_faithful_BA[b][a]:
                    x_error_BA[b][a] = xError.NOT_FAITHFUL

        bad_indices_flat = []
        x_emb_tokens_flat = []
        for b, a, d in product(range(B), range(A), range(D)):
            if x_error_BA[b][a] == xError.NO_ERROR:
                bad_indices_flat.append((b, a, d))
                x_emb_tokens_flat.append(x_emb_tokens_BA[b][a])

        y_raw_rollouts_flat = generate_rollouts(
            x_emb_tokens_flat,
            dp_gang=self._gangs.dp,
            vllm_model=self._vllm_model_defender,
            sampling_params=vllm_defender_sampling_params,
        )

        y_rollout_BAD = linit([B, A, D])
        y_raw_text_BAD = linit([B, A, D])
        y_raw_tokens_BAD = linit([B, A, D])
        y_text_BAD = linit([B, A, D])
        y_tokens_BAD = linit([B, A, D])
        y_cot_text_BAD = linit([B, A, D])
        y_cot_tokens_BAD = linit([B, A, D])
        y_error_BAD = linit([B, A, D], value=yError.PROMPT_NOT_VALID)
        for (b, a, d), y_rollout in zip(bad_indices_flat, y_raw_rollouts_flat):
            y_rollout_BAD[b][a][d] = y_rollout
            y_raw_outputs = y_rollout.outputs
            assert len(y_raw_outputs) == 1
            y_raw_output = y_raw_outputs[0]

            y_raw_text_BAD[b][a][d] = y_raw_output.text
            y_raw_tokens_BAD[b][a][d] = y_raw_output.token_ids

            if y_raw_output.finish_reason != "stop":
                y_error_BAD[b][a][d] = yError.MAX_LENGTH
            else:
                match = self._defender_regex_pattern.match(y_raw_output.text)
                if match:
                    y_cot = match.group(1).strip()
                    y_cot_text_BAD[b][a][d] = y_cot
                    y_cot_tokens_BAD[b][a][d] = self._tokenizer_defender.tokenize(y_cot)

                    y_text = match.group(2).strip()
                    y_text_BAD[b][a][d] = y_text
                    y_tokens_BAD[b][a][d] = self._tokenizer_defender.tokenize(y_text)

                    y_error_BAD[b][a][d] = yError.NO_ERROR
                else:
                    y_error_BAD[b][a][d] = yError.PARSING

        for b, a in product(range(B), range(A)):
            if x_error_BA[b][a] == xError.NO_ERROR:
                if all([y_error_BAD[b][a][d] != yError.NO_ERROR for d in range(D)]):
                    x_error_BA[b][a] = xError.NO_VALID_RESPONSE

        rollout_tree = RolloutTree(
            s_category_B=s_category_B,
            s_text_B=s_text_B,
            s_emb_text_B=s_emb_text_B,
            s_emb_tokens_B=s_emb_tokens_B,
            x_rollout_BA=x_rollout_BA,
            x_raw_text_BA=x_raw_text_BA,
            x_raw_tokens_BA=x_raw_tokens_BA,
            x_text_BA=x_text_BA,
            x_tokens_BA=x_emb_tokens_BA,
            x_cot_text_BA=x_cot_text_BA,
            x_cot_tokens_BA=x_cot_tokens_BA,
            x_emb_text_BA=x_emb_text_BA,
            x_emb_tokens_BA=x_emb_tokens_BA,
            x_error_BA=x_error_BA,
            y_rollout_BAD=y_rollout_BAD,
            y_raw_text_BAD=y_raw_text_BAD,
            y_raw_tokens_BAD=y_raw_tokens_BAD,
            y_text_BAD=y_text_BAD,
            y_tokens_BAD=y_tokens_BAD,
            y_cot_text_BAD=y_cot_text_BAD,
            y_cot_tokens_BAD=y_cot_tokens_BAD,
            y_error_BAD=y_error_BAD,
            B=B,
            A=A,
            D=D,
        )

        time_after_generate = time.time()
        log.info(
            f"[{prefix}] Finished generating rollout tree, took {time_after_generate - time_before_generate} seconds"
        )

        x_error_flat = lflatten(x_error_BA, depth=1)
        x_max_length_error_flat = [e == xError.MAX_LENGTH for e in x_error_flat]
        update_avg_value_filtered(
            metric_bag,
            x_max_length_error_flat,
            name=prefix + "attacker/x_avg_max_length_error_1",
        )

        x_parsing_error_flat = [e == xError.PARSING for e in x_error_flat]
        update_avg_value_filtered(
            metric_bag,
            x_parsing_error_flat,
            name=prefix + "attacker/x_avg_parsing_error_2",
        )

        x_bad_faithful_judge_error_flat = [
            e == xError.BAD_FAITH_JUDGE for e in x_error_flat
        ]
        update_avg_value_filtered(
            metric_bag,
            x_bad_faithful_judge_error_flat,
            name=prefix + "attacker/x_avg_bad_faithful_judge_error_3",
        )

        x_not_faithful_error_flat = [e == xError.NOT_FAITHFUL for e in x_error_flat]
        update_avg_value_filtered(
            metric_bag,
            x_not_faithful_error_flat,
            name=prefix + "attacker/x_avg_not_faithful_error_4",
        )

        x_no_valid_response_error_flat = [
            e == xError.NO_VALID_RESPONSE for e in x_error_flat
        ]
        update_avg_value_filtered(
            metric_bag,
            x_no_valid_response_error_flat,
            name=prefix + "attacker/x_avg_no_valid_response_error_5",
        )

        update_avg_length_filtered(
            metric_bag,
            lflatten(x_raw_tokens_BA, depth=1),
            name=prefix + "attacker/x_raw_avg_length",
        )

        update_avg_length_filtered(
            metric_bag,
            lflatten(x_tokens_BA, depth=1),
            name=prefix + "attacker/x_parsed_avg_length",
        )

        update_avg_length_filtered(
            metric_bag,
            lflatten(x_cot_tokens_BA, depth=1),
            name=prefix + "attacker/x_cot_avg_length",
        )

        y_error_flat = lflatten(y_error_BAD, depth=2)
        y_prompt_not_valid_error_flat = [
            e == yError.PROMPT_NOT_VALID for e in y_error_flat
        ]
        update_avg_value_filtered(
            metric_bag,
            y_prompt_not_valid_error_flat,
            name=prefix + "defender/y_avg_prompt_not_valid_error_1",
        )

        y_max_length_error_flat = [e == yError.MAX_LENGTH for e in y_error_flat]
        update_avg_value_filtered(
            metric_bag,
            y_max_length_error_flat,
            name=prefix + "defender/y_avg_max_length_error_2",
        )

        y_parsing_error_flat = [e == yError.PARSING for e in y_error_flat]
        update_avg_value_filtered(
            metric_bag,
            y_parsing_error_flat,
            name=prefix + "defender/y_avg_parsing_error_3",
        )

        update_avg_length_filtered(
            metric_bag,
            lflatten(y_raw_tokens_BAD, depth=2),
            name=prefix + "defender/y_raw_avg_length",
        )

        update_avg_length_filtered(
            metric_bag,
            lflatten(y_tokens_BAD, depth=2),
            name=prefix + "defender/y_parsed_avg_length",
        )

        update_avg_length_filtered(
            metric_bag,
            lflatten(y_cot_tokens_BAD, depth=2),
            name=prefix + "defender/y_cot_avg_length",
        )

        return rollout_tree

    @override
    def score_rollout_tree(
        self, rt: RolloutTree, metric_bag: MetricBag, prefix: str
    ) -> ScoredRolloutTree:
        log.info(f"[{prefix}] Start scoring rollout tree")
        time_before = time.time()
        B, A, D = (rt.B, rt.A, rt.D)
        x_error_BA = rt.x_error_BA
        y_error_BAD = rt.y_error_BAD

        if self._judgment_extractor_defender is None:
            y_reward_BAD = linit([B, A, D], value=None)
        else:
            y_reward_BAD, y_error_BAD = self._judgment_extractor_defender(
                s_category_B=rt.s_category_B,
                s_text_B=rt.s_text_B,
                x_text_BA=rt.x_text_BA,
                y_text_BAD=rt.y_text_BAD,
                y_error_BAD=y_error_BAD,
                prefix=prefix,
            )

        # optimistic play of attacker: assume attacker knows best response of defender
        y_selected_text_BA = linit([B, A])
        y_selected_error_BA = linit([B, A])
        for b, a in product(range(B), range(A)):
            if x_error_BA[b][a] == xError.NO_ERROR:
                if self._rollout_tree_config.optimistic_attacker:
                    d = largmax_filtered(y_reward_BAD[b][a])
                    if d is None:
                        # no xError -> at least one y without yError, so there was a problem with judge (i.e. BAD_DEF_JUDGE)
                        # in this case we use the first response that had the BAD_DEF_ERROR
                        d = y_error_BAD[b][a].index(yError.BAD_DEF_JUDGE)
                else:
                    # select first no-error response, if none exists, there must be a BAD_DEF_JUDGE error
                    try:
                        d = y_error_BAD[b][a].index(yError.NO_ERROR)
                    except ValueError:
                        d = y_error_BAD[b][a].index(yError.BAD_DEF_JUDGE)

                y_selected_text_BA[b][a] = rt.y_text_BAD[b][a][d]
                y_selected_error_BA[b][a] = y_error_BAD[b][a][d]

        if self._judgment_extractor_attacker is None:
            x_reward_BA = linit([B, A], value=None)
        else:
            x_reward_BA, x_error_BA = self._judgment_extractor_attacker(
                s_category_B=rt.s_category_B,
                s_text_B=rt.s_text_B,
                x_error_BA=x_error_BA,
                y_text_BA=y_selected_text_BA,
                y_error_BA=y_selected_error_BA,
                prefix=prefix,
            )

        y_reward_patched_BAD = linit([B, A, D])
        for b, a, d in product(range(B), range(A), range(D)):
            if y_error_BAD[b][a][d] == yError.NO_ERROR:
                if y_reward_BAD[b][a][d] is None:
                    # this happens only when we are not using a defender judge, in that case all no-error responses are equally good and get reward 0
                    y_reward_patched_BAD[b][a][d] = 0.0
                else:
                    y_reward_patched_BAD[b][a][d] = y_reward_BAD[b][a][d]
            elif y_error_BAD[b][a][d] == yError.PROMPT_NOT_VALID:
                y_reward_patched_BAD[b][a][d] = None
            elif y_error_BAD[b][a][d] == yError.MAX_LENGTH:
                y_reward_patched_BAD[b][a][d] = -1.0
            elif y_error_BAD[b][a][d] == yError.PARSING:
                y_reward_patched_BAD[b][a][d] = -1.0
            elif y_error_BAD[b][a][d] == yError.BAD_DEF_JUDGE:
                y_reward_patched_BAD[b][a][d] = None
            else:
                raise RuntimeError(f"Unknown y_error: {y_error_BAD[b][a][d]}.")
        log.info(f"[{prefix}] Defender rewards (patched): {y_reward_patched_BAD}")

        x_reward_patched_BA = linit([B, A])
        for b, a in product(range(B), range(A)):
            if x_error_BA[b][a] == xError.NO_ERROR:
                if x_reward_BA[b][a] is None:
                    # this happens only when we are not using an attacker judge, in that case all no-error prompts are equally good and get reward 0
                    x_reward_patched_BA[b][a] = 0.0
                else:
                    x_reward_patched_BA[b][a] = x_reward_BA[b][a]
            elif x_error_BA[b][a] == xError.MAX_LENGTH:
                x_reward_patched_BA[b][a] = -1.0
            elif x_error_BA[b][a] == xError.PARSING:
                x_reward_patched_BA[b][a] = -1.0
            elif x_error_BA[b][a] == xError.BAD_FAITH_JUDGE:
                x_reward_patched_BA[b][a] = None
            elif x_error_BA[b][a] == xError.NOT_FAITHFUL:
                x_reward_patched_BA[b][a] = -1.0
            elif x_error_BA[b][a] == xError.NO_VALID_RESPONSE:
                x_reward_patched_BA[b][a] = None
            elif x_error_BA[b][a] == xError.BAD_ATT_JUDGE:
                x_reward_patched_BA[b][a] = None
            else:
                raise RuntimeError(f"Unknown x_error: {x_error_BA[b][a]}.")
        log.info(f"[{prefix}] Attacker rewards (patched): {x_reward_patched_BA}")

        x_advantage_BA, y_advantage_BAD = self._compute_advantage(
            x_reward_patched_BA, y_reward_patched_BAD
        )

        scored_rollout_tree = ScoredRolloutTree(
            s_category_B=rt.s_category_B,
            s_text_B=rt.s_text_B,
            s_emb_text_B=rt.s_emb_text_B,
            s_emb_tokens_B=rt.s_emb_tokens_B,
            x_rollout_BA=rt.x_rollout_BA,
            x_raw_text_BA=rt.x_raw_text_BA,
            x_raw_tokens_BA=rt.x_raw_tokens_BA,
            x_text_BA=rt.x_text_BA,
            x_tokens_BA=rt.x_tokens_BA,
            x_cot_text_BA=rt.x_cot_text_BA,
            x_cot_tokens_BA=rt.x_cot_tokens_BA,
            x_emb_text_BA=rt.x_emb_text_BA,
            x_emb_tokens_BA=rt.x_emb_tokens_BA,
            x_error_BA=x_error_BA,
            x_reward_BA=x_reward_BA,
            x_reward_patched_BA=x_reward_patched_BA,
            x_advantage_BA=x_advantage_BA,
            y_rollout_BAD=rt.y_rollout_BAD,
            y_raw_text_BAD=rt.y_raw_text_BAD,
            y_raw_tokens_BAD=rt.y_raw_tokens_BAD,
            y_text_BAD=rt.y_text_BAD,
            y_tokens_BAD=rt.y_tokens_BAD,
            y_cot_text_BAD=rt.y_cot_text_BAD,
            y_cot_tokens_BAD=rt.y_cot_tokens_BAD,
            y_error_BAD=y_error_BAD,
            y_reward_BAD=y_reward_BAD,
            y_reward_patched_BAD=y_reward_patched_BAD,
            y_advantage_BAD=y_advantage_BAD,
            B=B,
            A=A,
            D=D,
        )

        time_after = time.time()
        log.info(
            f"[{prefix}] Finished scoring rollout tree, took {time_after - time_before} seconds"
        )

        if self._rollout_tree_config.log_rollouts:
            log_rollout_tree(scored_rollout_tree, prefix=prefix)

        update_avg_value_filtered(
            metric_bag,
            lflatten(y_reward_BAD, depth=2),
            name=prefix + "defender/avg_reward",
        )
        update_std_value_filtered(
            metric_bag,
            lflatten(y_reward_BAD, depth=2),
            name=prefix + "defender/std_reward",
        )

        update_avg_value_filtered(
            metric_bag,
            lflatten(y_reward_patched_BAD, depth=2),
            name=prefix + "defender/avg_reward_patched",
        )
        update_std_value_filtered(
            metric_bag,
            lflatten(y_reward_patched_BAD, depth=2),
            name=prefix + "defender/std_reward_patched",
        )

        update_avg_value_filtered(
            metric_bag,
            lflatten(x_reward_BA, depth=1),
            name=prefix + "attacker/avg_reward",
        )
        update_std_value_filtered(
            metric_bag,
            lflatten(x_reward_BA, depth=1),
            name=prefix + "attacker/std_reward",
        )

        update_avg_value_filtered(
            metric_bag,
            lflatten(x_reward_patched_BA, depth=1),
            name=prefix + "attacker/avg_reward_patched",
        )
        update_std_value_filtered(
            metric_bag,
            lflatten(x_reward_patched_BA, depth=1),
            name=prefix + "attacker/std_reward_patched",
        )

        x_error_flat = lflatten(x_error_BA, depth=1)
        x_bad_att_judge_error_flat = [e == xError.BAD_ATT_JUDGE for e in x_error_flat]
        update_avg_value_filtered(
            metric_bag,
            x_bad_att_judge_error_flat,
            name=prefix + "attacker/x_avg_bad_att_judge_error_6",
        )

        x_no_error_flat = [e == xError.NO_ERROR for e in x_error_flat]
        update_avg_value_filtered(
            metric_bag,
            x_no_error_flat,
            name=prefix + "attacker/x_avg_no_error_7",
        )

        y_error_flat = lflatten(y_error_BAD, depth=2)
        y_bad_def_judge_error_flat = [e == yError.BAD_DEF_JUDGE for e in y_error_flat]
        update_avg_value_filtered(
            metric_bag,
            y_bad_def_judge_error_flat,
            name=prefix + "defender/y_avg_bad_def_judge_error_4",
        )

        y_no_error_flat = [e == yError.NO_ERROR for e in y_error_flat]
        update_avg_value_filtered(
            metric_bag,
            y_no_error_flat,
            name=prefix + "defender/y_avg_no_error_5",
        )
        return scored_rollout_tree

    def _compute_advantage(self, x_reward_patched_BA, y_reward_patched_BAD):
        B, A, D = lget_shape(y_reward_patched_BAD, depth=3)

        x_reward_full_patched_BA = linit([B, A], value=0.0)
        x_reward_valid_mask_BA = linit([B, A], value=False)
        for b, a in product(range(B), range(A)):
            if x_reward_patched_BA[b][a] is not None:
                x_reward_full_patched_BA[b][a] = x_reward_patched_BA[b][a]
                x_reward_valid_mask_BA[b][a] = True

        y_reward_full_patched_BAD = linit([B, A, D], value=0.0)
        y_reward_valid_mask_BAD = linit([B, A, D], value=False)
        for b, a, d in product(range(B), range(A), range(D)):
            if y_reward_patched_BAD[b][a][d] is not None:
                y_reward_full_patched_BAD[b][a][d] = y_reward_patched_BAD[b][a][d]
                y_reward_valid_mask_BAD[b][a][d] = True

        x_reward_full_patched_tensor_BA = torch.tensor(
            x_reward_full_patched_BA, device=self._gangs.dp.device
        )
        x_reward_valid_mask_tensor_BA = torch.tensor(
            x_reward_valid_mask_BA, device=self._gangs.dp.device
        ).float()

        y_reward_full_patched_tensor_BAD = torch.tensor(
            y_reward_full_patched_BAD, device=self._gangs.dp.device
        )
        y_reward_valid_mask_tensor_BAD = torch.tensor(
            y_reward_valid_mask_BAD, device=self._gangs.dp.device
        ).float()

        R = self._gangs.dp.size

        x_rewards_BA = [None] * R
        y_rewards_BAD = [None] * R
        x_valid_masks = [None] * R
        y_valid_masks = [None] * R
        if self._gangs.dp.rank == 0:
            self._gangs.dp.gather_object(
                x_reward_full_patched_tensor_BA, x_rewards_BA, 0
            )
            self._gangs.dp.gather_object(
                y_reward_full_patched_tensor_BAD, y_rewards_BAD, 0
            )
            self._gangs.dp.gather_object(
                x_reward_valid_mask_tensor_BA, x_valid_masks, 0
            )
            self._gangs.dp.gather_object(
                y_reward_valid_mask_tensor_BAD, y_valid_masks, 0
            )
        else:
            self._gangs.dp.gather_object(x_reward_full_patched_tensor_BA, None, 0)
            self._gangs.dp.gather_object(y_reward_full_patched_tensor_BAD, None, 0)
            self._gangs.dp.gather_object(x_reward_valid_mask_tensor_BA, None, 0)
            self._gangs.dp.gather_object(y_reward_valid_mask_tensor_BAD, None, 0)

        if self._gangs.dp.rank == 0:
            x_reward_RBA = torch.stack(x_rewards_BA, dim=0)
            y_reward_RBAD = torch.stack(y_rewards_BAD, dim=0)
            x_valid_mask_RBA = torch.stack(x_valid_masks, dim=0)
            y_valid_mask_RBAD = torch.stack(y_valid_masks, dim=0)
            if self._rollout_tree_config.reward_normalization_mode == "per_seed":
                x_baseline_RBA = torch.sum(
                    x_reward_RBA * x_valid_mask_RBA, dim=(1, 2), keepdim=True
                )
                y_baseline_RBAD = torch.sum(
                    y_reward_RBAD * y_valid_mask_RBAD, dim=(1, 2, 3), keepdim=True
                )

                x_num_valid_RBA = torch.sum(x_valid_mask_RBA, dim=(1, 2), keepdim=True)
                y_num_valid_RBAD = torch.sum(
                    y_valid_mask_RBAD, dim=(1, 2, 3), keepdim=True
                )

                x_baseline_RBA = x_baseline_RBA / torch.clamp(x_num_valid_RBA, min=1.0)
                y_baseline_RBAD = y_baseline_RBAD / torch.clamp(
                    y_num_valid_RBAD, min=1.0
                )

                x_advantage_RBA = x_reward_RBA - x_baseline_RBA
                y_advantage_RBAD = y_reward_RBAD - y_baseline_RBAD
            elif self._rollout_tree_config.reward_normalization_mode == "across_seeds":
                x_baseline = torch.sum(x_reward_RBA * x_valid_mask_RBA)
                y_baseline = torch.sum(y_reward_RBAD * y_valid_mask_RBAD)

                x_num_valid = torch.sum(x_valid_mask_RBA)
                y_num_valid = torch.sum(y_valid_mask_RBAD)

                x_baseline = x_baseline / torch.clamp(x_num_valid, min=1.0)
                y_baseline = y_baseline / torch.clamp(y_num_valid, min=1.0)

                x_advantage_RBA = x_reward_RBA - x_baseline
                y_advantage_RBAD = y_reward_RBAD - y_baseline
            else:
                raise RuntimeError(
                    f"Unknown normalization mode: {self._rollout_tree_config.reward_normalization_mode}."
                )

            x_advantages_BA = [x_advantage_RBA[r] for r in range(R)]
            y_advantages_BAD = [y_advantage_RBAD[r] for r in range(R)]
            _x_advantage_BA = [None]
            _y_advantage_BAD = [None]
            self._gangs.dp.scatter_object_list(
                _x_advantage_BA, x_advantages_BA, source_rank=0
            )
            self._gangs.dp.scatter_object_list(
                _y_advantage_BAD, y_advantages_BAD, source_rank=0
            )
        else:
            _x_advantage_BA = [None]
            _y_advantage_BAD = [None]
            self._gangs.dp.scatter_object_list(_x_advantage_BA, None, source_rank=0)
            self._gangs.dp.scatter_object_list(_y_advantage_BAD, None, source_rank=0)

        x_advantage_tensor_BA = _x_advantage_BA[0]
        y_advantage_tensor_BAD = _y_advantage_BAD[0]

        y_advantage_BAD = y_advantage_tensor_BAD.cpu().tolist()
        x_advantage_BA = x_advantage_tensor_BA.cpu().tolist()
        return x_advantage_BA, y_advantage_BAD

    def generate_rollout_tree_direct(
        self,
        s_prompt_batch: PromptBatch,
        train: bool,
        metric_bag: MetricBag,
        prefix: str,
    ) -> RolloutTreeDirect:
        log.info(f"[{prefix}] Start generating rollout tree direct")
        time_before_generate = time.time()

        s_category_B = s_prompt_batch.meta_info.get("category")
        s_text_B = s_prompt_batch.meta_info.get("prompt_text")

        if train:
            vllm_defender_sampling_params = self.train_vllm_defender_sampling_params
        else:
            vllm_defender_sampling_params = (
                self.validation_vllm_defender_sampling_params
            )

        B = len(s_text_B)
        if train:
            D = self._rollout_tree_config.num_defender_samples
        else:
            D = self._rollout_tree_config.validation_num_defender_samples

        s_emb_text_B = linit([B])
        s_emb_tokens_B = linit([B])
        for b in range(B):
            task_instruction_template = task_instructions_defender[
                self._rollout_tree_config.defender_task_instruction_name
            ]
            task_instruction = task_instruction_template.format(prompt=s_text_B[b])
            s_formatted_text = task_instruction + self._defender_format_instruction
            s_wrapped_text = [{"role": "user", "content": s_formatted_text}]

            s_emb_text_B[b] = self._tokenizer_defender.apply_chat_template(
                s_wrapped_text, tokenize=False, add_generation_prompt=True
            )
            s_emb_tokens_B[b] = self._tokenizer_defender.apply_chat_template(
                s_wrapped_text, tokenize=True, add_generation_prompt=True
            )

        s_emb_tokens_BD = linit([B, D])
        for b, d in product(range(B), range(D)):
            s_emb_tokens_BD[b][d] = s_emb_tokens_B[b]

        # F = B * D
        s_emb_tokens_F = lflatten(s_emb_tokens_BD, depth=1)

        y_raw_rollouts_F = generate_rollouts(
            s_emb_tokens_F,
            dp_gang=self._gangs.dp,
            vllm_model=self._vllm_model_defender,
            sampling_params=vllm_defender_sampling_params,
        )
        y_rollout_BD = lunflatten(y_raw_rollouts_F, [B, D])

        y_raw_text_BD = linit([B, D])
        y_raw_tokens_BD = linit([B, D])
        y_text_BD = linit([B, D])
        y_tokens_BD = linit([B, D])
        y_cot_text_BD = linit([B, D])
        y_cot_tokens_BD = linit([B, D])
        y_error_BD = linit([B, D])
        for b, d in product(range(B), range(D)):
            y_raw_outputs = y_rollout_BD[b][d].outputs
            assert len(y_raw_outputs) == 1
            y_raw_output = y_raw_outputs[0]

            y_raw_text_BD[b][d] = y_raw_output.text
            y_raw_tokens_BD[b][d] = y_raw_output.token_ids

            if y_raw_output.finish_reason != "stop":
                y_error_BD[b][d] = yError.MAX_LENGTH
            else:
                match = self._defender_regex_pattern.match(y_raw_output.text)
                if match:
                    y_cot = match.group(1).strip()
                    y_cot_text_BD[b][d] = y_cot
                    y_cot_tokens_BD[b][d] = self._tokenizer_defender.tokenize(y_cot)

                    y_text = match.group(2).strip()
                    y_text_BD[b][d] = y_text
                    y_tokens_BD[b][d] = self._tokenizer_defender.tokenize(y_text)

                    y_error_BD[b][d] = yError.NO_ERROR
                else:
                    y_error_BD[b][d] = yError.PARSING

        rollout_tree = RolloutTreeDirect(
            s_category_B=s_category_B,
            s_text_B=s_text_B,
            s_emb_text_B=s_emb_text_B,
            s_emb_tokens_B=s_emb_tokens_B,
            y_rollout_BD=y_rollout_BD,
            y_raw_text_BD=y_raw_text_BD,
            y_raw_tokens_BD=y_raw_tokens_BD,
            y_text_BD=y_text_BD,
            y_tokens_BD=y_tokens_BD,
            y_cot_text_BD=y_cot_text_BD,
            y_cot_tokens_BD=y_cot_tokens_BD,
            y_error_BD=y_error_BD,
            B=B,
            D=D,
        )

        time_after_generate = time.time()
        log.info(
            f"[{prefix}] Finished generating rollout tree direct, took {time_after_generate - time_before_generate} seconds"
        )

        y_error_flat = lflatten(y_error_BD, depth=1)
        y_prompt_not_valid_error_flat = [
            e == yError.PROMPT_NOT_VALID for e in y_error_flat
        ]
        update_avg_value_filtered(
            metric_bag,
            y_prompt_not_valid_error_flat,
            name=prefix + "defender/y_avg_prompt_not_valid_error_1",
        )

        y_max_length_error_flat = [e == yError.MAX_LENGTH for e in y_error_flat]
        update_avg_value_filtered(
            metric_bag,
            y_max_length_error_flat,
            name=prefix + "defender/y_avg_max_length_error_2",
        )

        y_parsing_error_flat = [e == yError.PARSING for e in y_error_flat]
        update_avg_value_filtered(
            metric_bag,
            y_parsing_error_flat,
            name=prefix + "defender/y_avg_parsing_error_3",
        )

        update_avg_length_filtered(
            metric_bag,
            lflatten(y_raw_tokens_BD, depth=1),
            name=prefix + "defender/y_raw_avg_length",
        )

        update_avg_length_filtered(
            metric_bag,
            lflatten(y_tokens_BD, depth=1),
            name=prefix + "defender/y_parsed_avg_length",
        )

        update_avg_length_filtered(
            metric_bag,
            lflatten(y_cot_tokens_BD, depth=1),
            name=prefix + "defender/y_cot_avg_length",
        )

        return rollout_tree

    @override
    def score_rollout_tree_direct(
        self, rtd: RolloutTreeDirect, metric_bag: MetricBag, prefix: str
    ) -> ScoredRolloutTreeDirect:
        log.info(f"[{prefix}] Start scoring rollout tree direct")
        time_before = time.time()
        B, D = (rtd.B, rtd.D)

        x_text_BA = [[s_text] for s_text in rtd.s_text_B]
        y_text_BAD = [[y_text_D] for y_text_D in rtd.y_text_BD]
        y_error_BAD = [[y_error_D] for y_error_D in rtd.y_error_BD]

        y_reward_BAD, y_error_BAD = self._judgment_extractor_defender(
            s_category_B=rtd.s_category_B,
            s_text_B=rtd.s_text_B,
            x_text_BA=x_text_BA,
            y_text_BAD=y_text_BAD,
            y_error_BAD=y_error_BAD,
            prefix=prefix,
        )

        y_reward_BD = [r[0] for r in y_reward_BAD]
        y_error_BD = [e[0] for e in y_error_BAD]

        y_reward_patched_BD = linit([B, D])
        for b, d in product(range(B), range(D)):
            if y_error_BD[b][d] == yError.NO_ERROR:
                assert y_reward_BD[b][d] is not None
                y_reward_patched_BD[b][d] = y_reward_BD[b][d]
            elif y_error_BD[b][d] == yError.PROMPT_NOT_VALID:
                y_reward_patched_BD[b][d] = None
            elif y_error_BD[b][d] == yError.MAX_LENGTH:
                y_reward_patched_BD[b][d] = -1.0
            elif y_error_BD[b][d] == yError.PARSING:
                y_reward_patched_BD[b][d] = -1.0
            elif y_error_BD[b][d] == yError.BAD_DEF_JUDGE:
                y_reward_patched_BD[b][d] = None
            else:
                raise RuntimeError(f"Unknown y_error: {y_error_BD[b][d]}.")

        log.info(f"[{prefix}] Defender rewards (patched): {y_reward_patched_BD}")

        y_advantage_BD = self._compute_advantage_direct(y_reward_patched_BD)

        scored_rollout_tree_direct = ScoredRolloutTreeDirect(
            s_category_B=rtd.s_category_B,
            s_text_B=rtd.s_text_B,
            s_emb_text_B=rtd.s_emb_text_B,
            s_emb_tokens_B=rtd.s_emb_tokens_B,
            y_rollout_BD=rtd.y_rollout_BD,
            y_raw_text_BD=rtd.y_raw_text_BD,
            y_raw_tokens_BD=rtd.y_raw_tokens_BD,
            y_text_BD=rtd.y_text_BD,
            y_tokens_BD=rtd.y_tokens_BD,
            y_cot_text_BD=rtd.y_cot_text_BD,
            y_cot_tokens_BD=rtd.y_cot_tokens_BD,
            y_error_BD=y_error_BD,
            y_reward_BD=y_reward_BD,
            y_reward_patched_BD=y_reward_patched_BD,
            y_advantage_BD=y_advantage_BD,
            B=B,
            D=D,
        )

        time_after = time.time()
        log.info(
            f"[{prefix}] Finished scoring rollout tree direct, took {time_after - time_before} seconds"
        )

        if self._rollout_tree_config.log_rollouts:
            log_rollout_tree_direct(scored_rollout_tree_direct, prefix=prefix)

        update_avg_value_filtered(
            metric_bag,
            lflatten(y_reward_BD, depth=1),
            name=prefix + "defender/avg_reward",
        )
        update_std_value_filtered(
            metric_bag,
            lflatten(y_reward_BD, depth=1),
            name=prefix + "defender/std_reward",
        )

        update_avg_value_filtered(
            metric_bag,
            lflatten(y_reward_patched_BD, depth=1),
            name=prefix + "defender/avg_reward_patched",
        )
        update_std_value_filtered(
            metric_bag,
            lflatten(y_reward_patched_BD, depth=1),
            name=prefix + "defender/std_reward_patched",
        )

        y_error_flat = lflatten(y_error_BD, depth=1)
        y_bad_def_judge_error_flat = [e == yError.BAD_DEF_JUDGE for e in y_error_flat]
        update_avg_value_filtered(
            metric_bag,
            y_bad_def_judge_error_flat,
            name=prefix + "defender/y_avg_bad_def_judge_error_4",
        )

        y_no_error_flat = [e == yError.NO_ERROR for e in y_error_flat]
        update_avg_value_filtered(
            metric_bag,
            y_no_error_flat,
            name=prefix + "defender/y_avg_no_error_5",
        )
        return scored_rollout_tree_direct

    def _compute_advantage_direct(self, y_reward_patched_BD):
        B, D = lget_shape(y_reward_patched_BD, depth=2)

        y_reward_full_patched_BD = linit([B, D], value=0.0)
        y_reward_valid_mask_BD = linit([B, D], value=False)
        for b, d in product(range(B), range(D)):
            if y_reward_patched_BD[b][d] is not None:
                y_reward_full_patched_BD[b][d] = y_reward_patched_BD[b][d]
                y_reward_valid_mask_BD[b][d] = True

        y_reward_full_patched_tensor_BD = torch.tensor(
            y_reward_full_patched_BD, device=self._gangs.dp.device
        )
        y_reward_valid_mask_tensor_BD = torch.tensor(
            y_reward_valid_mask_BD, device=self._gangs.dp.device
        ).float()

        R = self._gangs.dp.size

        y_rewards_BD = [None] * R
        y_valid_masks = [None] * R
        if self._gangs.dp.rank == 0:
            self._gangs.dp.gather_object(
                y_reward_full_patched_tensor_BD, y_rewards_BD, 0
            )
            self._gangs.dp.gather_object(
                y_reward_valid_mask_tensor_BD, y_valid_masks, 0
            )
        else:
            self._gangs.dp.gather_object(y_reward_full_patched_tensor_BD, None, 0)
            self._gangs.dp.gather_object(y_reward_valid_mask_tensor_BD, None, 0)

        if self._gangs.dp.rank == 0:
            y_reward_RBD = torch.stack(y_rewards_BD, dim=0)
            y_valid_mask_RBD = torch.stack(y_valid_masks, dim=0)
            if self._rollout_tree_config.reward_normalization_mode == "per_seed":
                y_baseline_RBD = torch.sum(
                    y_reward_RBD * y_valid_mask_RBD, dim=(1, 2), keepdim=True
                )
                y_num_valid_RBD = torch.sum(y_valid_mask_RBD, dim=(1, 2), keepdim=True)
                y_baseline_RBD = y_baseline_RBD / torch.clamp(y_num_valid_RBD, min=1.0)
                y_advantage_RBD = y_reward_RBD - y_baseline_RBD
            elif self._rollout_tree_config.reward_normalization_mode == "across_seeds":
                y_baseline = torch.sum(y_reward_RBD * y_valid_mask_RBD)
                y_num_valid = torch.sum(y_valid_mask_RBD)
                y_baseline = y_baseline / torch.clamp(y_num_valid, min=1.0)
                y_advantage_RBD = y_reward_RBD - y_baseline
            else:
                raise RuntimeError(
                    f"Unknown normalization mode: {self._rollout_tree_config.reward_normalization_mode}."
                )

            y_advantages_BD = [y_advantage_RBD[r] for r in range(R)]
            _y_advantage_BD = [None]
            self._gangs.dp.scatter_object_list(
                _y_advantage_BD, y_advantages_BD, source_rank=0
            )
        else:
            _y_advantage_BD = [None]
            self._gangs.dp.scatter_object_list(_y_advantage_BD, None, source_rank=0)

        y_advantage_tensor_BD = _y_advantage_BD[0]

        y_advantage_BD = y_advantage_tensor_BD.cpu().tolist()
        return y_advantage_BD
