import random
from typing import Any, Dict, List
from datasets import Dataset
from ..eval import Evaluator


class SciQ(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "allenai/sciq", split: str = "test", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, **kwargs)

        random.seed(42)
        # We randomly shuffle the options for each example to avoid bias
        # Then change the correct answer position accordingly
        processed_dataset = []
        for example in self.dataset:
            all_options = [
                example["correct_answer"],
                example["distractor1"],
                example["distractor2"],
                example["distractor3"],
            ]
            shuffled_options = all_options.copy()
            random.shuffle(shuffled_options)
            correct_pos = shuffled_options.index(example["correct_answer"])

            tmp = {
                "question": example["question"],
                "options": shuffled_options,
                "correct_answer": f"({correct_pos})",
                "subset": example["subset"],
            }
            processed_dataset.append(tmp)
        
        self.dataset = Dataset.from_list(processed_dataset)
        self.choices = [0, 1, 2, 3]
        self.output_key = "correct_answer"

    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """(Following is the processed dataset example)

        "question": "What phenomenon makes global winds blow northeast to southwest
        or the reverse in the northern hemisphere and northwest to southeast or the
        reverse in the southern hemisphere?",
        "options": [ "coriolis effect", "muon effect", "centrifugal effect",
        "tropical effect" ]
        "correct_answer": 0
        """
        # Format prompts
        prompts = []
        for example in batch_examples:
            # Filter to only include input keys
            prompt = ""
            prompt += "Question: " + example["question"]
            prompt += "\n"
            prompt += (
                "Based on the givencontext, choose the correct option for the"
                " question from the following options (for final answer, please only"
                " output the letter choice): \n"
            )

            for i, choice in enumerate(example["options"]):
                # TODO better to have some extraction format instead of pure number
                prompt += f"({i}).{choice}\n"
            prompts.append(prompt)
        return prompts

