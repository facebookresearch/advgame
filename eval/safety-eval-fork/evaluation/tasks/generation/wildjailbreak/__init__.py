import os

from src.data_utils import load_standard_data
from evaluation.tasks.generation.base import AdversarialTaskBase
from src.classifier_models.base import SafetyClassifierBase, ResponseRefusal, ResponseHarmfulness
from src.classifier_models.loader import load_classifier_model


class WildJailbreakValidation(AdversarialTaskBase):

    def prepare_evaluator(self) -> tuple[SafetyClassifierBase, str, ResponseHarmfulness]:
        classifier: SafetyClassifierBase = load_classifier_model(self.classifier_model_name,
                                                                 local_model_path=self.local_classifier_model_path)
        assert "prompt" in classifier.get_possible_input_fields() and "response" in classifier.get_possible_input_fields(), \
            "The classifier model must have input fields 'prompt' and 'response'."
        if "response_harmfulness" in classifier.get_output_fields():
            classifier_output_field = "response_harmfulness"
            desired_outputs = ResponseHarmfulness.HARMFUL
        else:
            classifier_output_field = "response_refusal"
            desired_outputs = ResponseRefusal.COMPLIANCE

        return classifier, classifier_output_field, desired_outputs

    def load(self) -> list[dict]:
        current_dir = os.path.abspath(os.path.dirname(__file__))
        task_file = os.path.join(current_dir, f"harmful.jsonl")
        data = load_standard_data(task_file)
        for item in data:
            item['instruction'] = item['adversarial']
        return data


class WildJailbreakValidationBenign(AdversarialTaskBase):
    def load(self) -> list[dict]:
        current_dir = os.path.abspath(os.path.dirname(__file__))
        task_file = os.path.join(current_dir, f"benign.jsonl")
        data = load_standard_data(task_file)
        for item in data:
            item['instruction'] = item['adversarial']
            # Delete 'vanilla' field, so we use the 'instruction' field for classifier judge input
            del item['vanilla']
        return data
