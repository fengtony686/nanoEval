from typing import Dict, Any, List
from ..eval import Evaluator

class CommonsenseQA(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "tau/commonsense_qa", split: str = "validation", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, **kwargs)
        self.choices = ["A", "B", "C", "D", "E"]
        self.input_keys = ["question"]
        self.output_key = "choices"
         
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        "question": The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? 
        "choices": { "label": [ "A", "B", "C", "D", "E" ], "text": [ "ignore", "enforce", "authoritarian", "yell at", "avoid" ] }
        """
        # Format prompts
        prompts = []
        for example in batch_examples:
            # Filter to only include input keys
            prompt = example["question"]
            prompt += "\n"
            prompt += "Choose the correct option from the following options (for final answer, please only output the letter choice): \n"
            for i, choice in enumerate(example["choices"]["label"]):
                prompt += f"{choice}. {example['choices']['text'][i]}\n"
            prompts.append(prompt)
            
        return prompts
