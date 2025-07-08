from typing import Dict, Any, List
from ..eval import Evaluator

class MuSR(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "TAUR-Lab/MuSR", split: str = "murder_mysteries", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, **kwargs)
        self.choices = ["A", "B", "C", "D"]
        self.input_keys = ["narrative", "question"]
        self.output_key = "answer_index"
        
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        "narrative": "In an adrenaline inducing bungee jumping site, Mack's thrill-seeking adventure came to a gruesome end by a nunchaku; 
                      now, it's up to Detective Winston to unravel the deadly secrets between Mackenzie and Ana.",
        "question": "Who is the most likely murderer?",
        "choices": "['Mackenzie', 'Ana']",
        "answer_index": 0,
        "answer_choice": "Mackenzie",
        """
        prompts = []
        for example in batch_examples:
            # Filter to only include input keys
            prompt = "\n".join([example["narrative"], example["question"]])
            prompt += "Choose the correct option from the following options (for final answer, please only output the letter choice): \n"
            for i, choice in enumerate(eval(example["choices"])):
                prompt += f"{self.choices[i]}. {choice}\n"
            prompts.append(prompt)
            
        return prompts

    def get_ground_truth_answers(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        Get the ground truth answer from the example.
        Override this method for different datasets.
        """
        answer_maps = {0: "A", 1: "B", 2: "C", 3: "D"}
        return [answer_maps[example[self.output_key]] for example in batch_examples]
