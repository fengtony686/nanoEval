from typing import Dict, Any, List
from ..eval import Evaluator

class PIQA(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "ybisk/piqa", split: str = "train", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, **kwargs)
        self.choices = ["A", "B"]
        self.input_keys = ["goal"]
        self.output_key = "label"
        
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        "goal": "How do I ready a guinea pig cage for it's new occupants?",
        "sol1": "Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, 
                 you will also need to supply it with a water bottle and a food dish.",
        "sol2": "Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, 
                 you will also need to supply it with a water bottle and a food dish.",
        "label": 0,
        """
        prompts = []
        for example in batch_examples:
            # Filter to only include input keys
            prompt = example["goal"]
            prompt += "\n"
            prompt += "Choose the correct option from the following options (for final answer, please only output the letter choice): \n"
            prompt += f"A: {example['sol1']}\n"
            prompt += f"B: {example['sol2']}\n"
            prompts.append(prompt)
            
        return prompts

    def get_ground_truth_answers(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        Get the ground truth answer from the example.
        Override this method for different datasets.
        """
        answer_maps = {0: "A", 1: "B"}
        return [answer_maps[example[self.output_key]] for example in batch_examples]
