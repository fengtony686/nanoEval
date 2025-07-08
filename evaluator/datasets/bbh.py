from typing import Dict, Any, List
from ..eval import Evaluator
from datasets import load_dataset

class BBH(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "lukaemon/bbh", split: str = "test", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, **kwargs)
        self.input_keys = ["input"]
        self.output_key = "target"
        self.choices = None
    
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        prompts = []
        for example in batch_examples:
            prompt = example["input"]
            prompts.append(prompt)
        return prompts


class BBH_synthetic(Evaluator):
    def __init__(self, model_name_or_path: str, dataset_name_or_path: str = "lukaemon/bbh", split: str = "test", **kwargs):
        super().__init__(model_name_or_path, dataset_name_or_path, split, subset="dummy",**kwargs)
        self.input_keys = ["problem"]
        self.output_key = "answer"
        self.choices = None
    
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        prompts = []
        for example in batch_examples:
            prompt = example["problem"]
            prompts.append(prompt)
        return prompts
    
    def _load_and_concatenate_datasets(self, dataset_name_or_path: str, split: str) -> Any:
        """Load jsonl file and convert to dataset."""
        assert dataset_name_or_path.endswith(".json"), (
            "Currently only json files are supported in BBH_synthetic"
        )
        dataset = load_dataset("json", data_files=dataset_name_or_path)["train"]
        return dataset
