# datasets/arc.py
from typing import Dict, Any, List
# from ..hf_eval import HFEvaluator as Evaluator
from ..eval import Evaluator


class ARC(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "allenai/ai2_arc", split: str = "test", subset: str = "ARC-Easy", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, subset=subset, **kwargs)
        self.choices = ["A", "B", "C", "D", "E"]  # Some ARC questions have 5 options
        self.input_keys = ["question", "choices"]
        self.output_key = "answerKey"
    
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        prompts = []
        for example in batch_examples:
            prompt = f"{example['question']}\n\n"
            for i, (label, text) in enumerate(zip(example["choices"]["label"], example["choices"]["text"])):
                prompt += f"{label}. {text}\n"
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
                
            # Try a more lenient approach - any standalone A-E
            import re
            matches = re.findall(r'\b([A-E])\b', response)
            if matches:
                return matches[0]
        
        return answer