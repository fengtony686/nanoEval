from ..eval import Evaluator
from typing import Dict, Any, List

class Winogrande(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "allenai/winogrande", split: str = "validation", subset: str = "winogrande_xl", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, subset=subset, **kwargs)
        self.input_keys = ["sentence", "option1", "option2"]
        self.output_key = "answer"
        self.choices = ["A", "B"]
        
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        prompts = []
        for example in batch_examples:
            prompt = example["sentence"] + "\n"
            for i, input_key in enumerate(self.input_keys[1:]):
                prompt += f"{self.choices[i]}. {example[input_key]}\n"
            prompt += "Choose the correct option from the following options (for final answer, please only output the letter choice): \n"
            prompts.append(prompt)
        return prompts
    
    def get_ground_truth_answers(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        answer_maps = {'1': 'A', '2': 'B'}
        return [answer_maps[example["answer"]] for example in batch_examples]