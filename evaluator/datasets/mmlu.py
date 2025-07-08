# datasets/mmlu.py
from typing import Dict, Any, List
from ..eval import Evaluator
# from ..hf_eval import HFEvaluator as Evaluator

class MMLU(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "cais/mmlu", split: str = "test", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, **kwargs)
        self.choices = ["A", "B", "C", "D"]
        self.input_keys = ["question", "choices"]
        self.output_key = "answer"
    
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        prompts = []
        for example in batch_examples:
            prompt = example["question"] + "\n\n"
            for i, choice_text in enumerate(example["choices"]):
                prompt += f"{self.choices[i]}. {choice_text}\n"
            prompt += "\nChoose the correct option from the following options (for final answer, please only output the letter choice): "
            prompts.append(prompt)
        return prompts
    
    def extract_answer(self, response: str) -> str:
        """Extract the answer letter from the model's response."""
        # First try to get the answer after "the answer is" pattern
        answer = super().extract_answer(response)
        
        # If that doesn't work, look for a standalone letter
        if len(answer) > 5:  # If answer is too long, try to find just the letter
            for choice in self.choices:
                if choice in response:
                    return choice
            
            # Look for letter at beginning of response or at end
            first_word = response.strip().split()[0] if response.strip() else ""
            if first_word in self.choices:
                return first_word
                
            # Try a more lenient approach - any standalone A, B, C, D
            import re
            matches = re.findall(r'\b([A-D])\b', response)
            if matches:
                return matches[0]
        
        return answer
    def get_ground_truth_answers(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """Converts integer answer index to letter string."""
        # self.choices is ["A", "B", "C", "D"] defined in __init__
        # self.output_key is "answer" defined in __init__
        return [self.choices[example[self.output_key]] for example in batch_examples]