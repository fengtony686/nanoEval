from typing import Dict, Any, List
from ..eval import Evaluator

class Quality(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "emozilla/quality", split: str = "validation", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, **kwargs)
        self.choices = [0, 1, 2, 3] #TODO string?  int?
        self.output_key = "answer"
        
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        "article": ... long text
        "question": "Why is Si retirement so significant to the Space Exploration Team?"
        "options": [ "There aren’t enough working people in the world. They won’t be able to find a replacement.", "As one of two remaining spacemen, it would likely mean the defunding and shut down of the Space Exploration Team.", "Training new spacemen is costly and time consuming. They won’t have anyone else ready after him.", "His retirement may inspire others to stop working as well, which would be hugely detrimental as most people don't feel the drive to work as is." ]
        "answer": 2
        """
        # Format prompts
        prompts = []
        for example in batch_examples:
            # Filter to only include input keys
            prompt = ""

            prompt += "Context" + example["article"]
            prompt += "\n"
            prompt += "Question: " + example["question"]
            prompt += "\n"
            prompt += "Based on the givencontext, choose the correct option for the question from the following options (for final answer, please only output the letter choice): \n"
            for i, choice in enumerate(example["options"]): 
                #TODO Not sure if the pure number i. is good enough for extraction
                prompt += f"{i}.{choice}\n"
            prompts.append(prompt)
            
        return prompts
