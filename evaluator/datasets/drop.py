from typing import Dict, Any, List
from ..eval import Evaluator

class DROP(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "ucinlp/drop", split: str = "validation", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, **kwargs)
        self.choices = None
        
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        passage: "\" Hoping to rebound from their loss to the Patriots, ..."
        question: "Who scored the first touchdown of the game?"
        answers_spans:  {
            "spans": ["Chaz Schilens"]
        },
        """
        # Format prompts
        prompts = []
        for example in batch_examples:
            # Filter to only include input keys
            prompt = ""

            prompt += "Context" + example["passage"]
            prompt += "\n"
            prompt += "Question: " + example["question"]
            prompts.append(prompt)
        return prompts
    
    def get_ground_truth_answers(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        Get the ground truth answer from the example.
        """
        return [example["answers_spans"]['spans'][0] for example in batch_examples]
