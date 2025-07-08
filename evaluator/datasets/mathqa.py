from typing import Dict, Any, List
from ..eval import Evaluator

class MathQA(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "lucasmccabe/mathqa", split: str = "test", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, **kwargs)
        self.choices = ["A", "B"]
        self.input_keys = ["Problem"]
        self.output_key = "correct"
        
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        "Problem": "a multiple choice test consists of 4 questions , and each question has 5 answer choices . in how many r ways can the test be completed if every question is unanswered ?",
        "Rationale": "\"5 choices for each of the 4 questions , thus total r of 5 * 5 * 5 * 5 = 5 ^ 4 = 625 ways to answer all of them . answer : c .\"",
        "annotated_formula": "power(5, 4)",
        "category": "general",
        "correct": "c",
        "linear_formula": "power(n1,n0)|",
        "options": "a ) 24 , b ) 120 , c ) 625 , d ) 720 , e ) 1024"
        """
        prompts = []
        for example in batch_examples:
            # Filter to only include input keys
            prompt = example["Problem"]
            prompt += "\n"
            prompt += "Choose the correct option from the following options (for final answer, please only output the letter choice): \n"
            prompt += example["options"].replace("a )", "A. ").replace("b )", "B. ").replace("c )", "C. ").replace("d )", "D. ").replace("e )", "E. ")+"\n"
            prompts.append(prompt)
            
        return prompts

    def get_ground_truth_answers(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        Get the ground truth answer from the example.
        Override this method for different datasets.
        """
        return [example[self.output_key].capitalize() for example in batch_examples]
