import random
from typing import Any, Dict, List
from datasets import Dataset
from ..eval import Evaluator


class GPQA(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "Idavidrein/gpqa", split: str = "train", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, subset="gpqa_main", **kwargs)

        random.seed(42)
        # We randomly shuffle the options for each example to avoid bias
        # Then change the correct answer position accordingly
        processed_dataset = []
        for example in self.dataset:
            all_options = [
                example["Correct Answer"],
                example["Incorrect Answer 1"],
                example["Incorrect Answer 2"],
                example["Incorrect Answer 3"],
            ]
            shuffled_options = all_options.copy()
            random.shuffle(shuffled_options)
            correct_pos = shuffled_options.index(example["Correct Answer"])

            tmp = {
                "question": example["Question"],
                "options": shuffled_options,
                "correct_answer": f"({correct_pos})",
                "subset": example["subset"]
            }
            processed_dataset.append(tmp)
        
        self.dataset = Dataset.from_list(processed_dataset)
        self.choices = [0, 1, 2, 3]
        self.output_key = "correct_answer"

    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """(Following is the processed dataset example)

        "question": "methyl 2-oxocyclohexane-1-carboxylate is heated in the presence of aqueous NaOH. Then the reaction mixture is acidified with aqueous HCl, after which heating is continued. How many oxygen atoms are there in the main product of this reaction?",
        "options": [ "1", "2", "3",
        "4" ]
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

