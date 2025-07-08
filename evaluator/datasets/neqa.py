from typing import Dict, Any, List
from ..eval import Evaluator

class NeQA(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "inverse-scaling/NeQA", split: str = "train", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, **kwargs)
        self.choices = ["A", "B"]
        self.input_keys = ["prompt"]
        self.output_key = "answer_index"
        
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        prompt = "The following are multiple choice questions (with answers) about common sense.
        Question: Tectonic plates aren't like a large
        A. jenga
        B. candy cane
        Answer: "
        classes = [ " A", " B" ]
        """
        prompts = []
        for example in batch_examples:
            # Filter to only include input keys
            prompt = example["prompt"]
            prompt += "\n"
            prompt += "Choose the correct option from the following options (for final answer, please only output the letter choice): \n"
            prompts.append(prompt)
            
        return prompts

    def get_ground_truth_answers(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        Get the ground truth answer from the example.
        Override this method for different datasets.
        """
        answer_maps = {0: "A", 1: "B"}
        return [answer_maps[example["answer_index"]] for example in batch_examples]
