# datasets/hellaswag.py
from typing import Dict, Any, List
# from ..hf_eval import HFEvaluator as Evaluator
from ..eval import Evaluator


class HellaSWAG(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "Rowan/hellaswag", split: str = "validation", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, **kwargs)
        self.choices = ["A", "B", "C", "D"]
        self.input_keys = ["ctx", "endings"]
        self.output_key = "label"
    
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        prompts = []
        for example in batch_examples:
            prompt = f"{example['ctx']}\n\nChoose the most appropriate ending for the above text:\n"
            for i, ending in enumerate(example["endings"]):
                prompt += f"{self.choices[i]}. {ending}\n"
            prompt += "\nChoose the correct option from the following options (for final answer, please only output the letter choice): "
            prompts.append(prompt)
        return prompts
    
    def get_ground_truth_answers(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """Convert numeric labels to letter choices, ensuring integer type."""
        answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        ground_truths = []
        for example in batch_examples:
            try:
                # Explicitly convert the label to an integer
                label_int = int(example["label"])
                if label_int in answer_map:
                    ground_truths.append(answer_map[label_int])
                else:
                    # Handle case where integer label is out of expected range (0-3)
                    print(f"Warning: Label index out of bounds '{label_int}' in example: {example.get('ind', 'N/A')}")
                    ground_truths.append("ERROR_LABEL_OUT_OF_BOUNDS")
            except KeyError:
                # Handle cases where the key 'label' might be missing
                print(f"Warning: Missing 'label' key in example: {example.get('ind', 'N/A')}")
                ground_truths.append("ERROR_MISSING_LABEL")
            except (ValueError, TypeError):
                # Handle cases where the label cannot be converted to int
                print(f"Warning: Invalid label format '{example.get('label', 'N/A')}' in example: {example.get('ind', 'N/A')}")
                ground_truths.append("ERROR_INVALID_LABEL")
        return ground_truths
    
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