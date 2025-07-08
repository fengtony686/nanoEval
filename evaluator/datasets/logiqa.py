from typing import Dict, Any, List
from ..eval import Evaluator

class LogiQA(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "ybisk/piqa", split: str = "test", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, **kwargs)
        self.choices = ["A", "B", "C", "D"]
        self.input_keys = ["context", "query"]
        self.output_key = "correct_option"
        
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        'context': 'Continuous exposure to indoor fluorescent lights is beneficial to the health of hamsters with heart disease. 
                    One group of hamsters exposed to continuous exposure to fluorescent lights has an average lifespan that 
                    is 2.5% longer than another one of the same species but living in a black wall.',
        'query': 'Which of the following questions was the initial motivation for conducting the above experiment?',
        'options': ['Can hospital light therapy be proved to promote patient recovery?',
                    'Which one lives longer, the hamster living under the light or the hamster living in the dark?',
                    'What kind of illness does the hamster have?',
                    'Do some hamsters need a period of darkness?'],
        'correct_option': 0
        """
        prompts = []
        for example in batch_examples:
            # Filter to only include input keys
            prompt = "\n".join([example["context"], example["query"]])
            prompt += "Choose the correct option from the following options (for final answer, please only output the letter choice): \n"
            for i, option in enumerate(example["options"]):
                prompt += f"{self.choices[i]}. {option}\n"
            prompts.append(prompt)
            
        return prompts

    def get_ground_truth_answers(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        Get the ground truth answer from the example.
        Override this method for different datasets.
        """
        answer_maps = {0: "A", 1: "B", 2: "C", 3: "D"}
        return [answer_maps[example[self.output_key]] for example in batch_examples]
