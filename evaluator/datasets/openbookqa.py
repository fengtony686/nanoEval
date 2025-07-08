from typing import Dict, Any, List
from ..eval import Evaluator

class OpenbookQA(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "allenai/openbookqa", split: str = "test", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, subset="main", **kwargs)
        self.choices = ["A", "B", "C", "D"]
        self.output_key = "answerKey"

        
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        "question_stem": "The sun is responsible for" 
        "choices": { "label": ['A', 'B', 'C', 'D'], "text": [ 'puppies learning new tricks','children growing up and getting old','flowers wilting in a vase','plants sprouting, blooming and wilting' ] }
        """
        # Format prompts
        prompts = []
        for example in batch_examples:
            # Filter to only include input keys
            prompt = ""

            prompt += example["question_stem"]
            prompt += "\n"
            prompt += "Choose the correct option from the following options (for final answer, please only output the letter choice): \n"
            for i, choice in enumerate(example["choices"]["label"]):
                prompt += f"{choice}. {example['choices']['text'][i]}\n"
            prompts.append(prompt)
            
        return prompts

