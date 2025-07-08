import os
import re
import json
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from .utils import get_choices

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Stores the result of a single evaluation."""
    prompt: str
    model_response: str
    id: str = None
    extracted_answer: Optional[str] = None
    ground_truth_answer: Optional[str] = None
    ground_truth_reasoning: Optional[str] = None
    is_correct: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class Evaluator:
    """Evaluates language models on various benchmarks."""
    
    def __init__(
        self,
        model_name_or_path: str,
        dataset_name_or_path: str = "lukaemon/bbh",
        split: str = "test",
        output_dir: str = "results",
        max_new_tokens: int = 1024,
        batch_size: int = 8,
        temperature: float = 0.7,
        top_p: float = 0.95,
        tensor_parallel_size: int = 4,
        seed: int = 42,
        max_model_len: int = 4096,
        input_keys: List[str] = ["input"],
        output_key: str = "target",
        save_logprobs: bool = False,
        subset: Optional[str] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_name_or_path: Path to the model or model name
            dataset_name_or_path: Path to the dataset or dataset name
            split: Dataset split to use
            output_dir: Directory to save results
            max_new_tokens: Maximum number of tokens to generate
            batch_size: Batch size for inference
            temperature: Temperature for sampling
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            seed: Random seed
        """
        self.model_name_or_path = model_name_or_path
        self.dataset_name_or_path = dataset_name_or_path
        self.split = split
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p
        self.tensor_parallel_size = tensor_parallel_size
        self.seed = seed
        self.save_logprobs = save_logprobs
        try:
            self.input_keys = eval(input_keys)
        except:
            self.input_keys = input_keys
        self.output_key = output_key
        if subset:
            self.subsets = [subset]
        else:
            self.subsets = get_dataset_config_names(dataset_name_or_path, trust_remote_code=True)
        self.is_chat_model = "chat" in model_name_or_path.lower() or "instruct" in model_name_or_path.lower()
        
        self.instruction = "Let's think step by step and output the final answer after \"The answer is:\"."
        self.system_prompt = "You are a helpful assistant that can answer questions and help with tasks."
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        logger.info(f"Loading model {model_name_or_path}...")
        self.model = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            max_model_len=max_model_len,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Load dataset
        logger.info(f"Loading dataset {dataset_name_or_path}, split: {split}...")
        self.dataset = self._load_and_concatenate_datasets(dataset_name_or_path, split)
        logger.info(f"Loaded {len(self.dataset)} examples in total")
        
        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=None,  # Will be set per task
        )
        if self.save_logprobs:
            self.sampling_params.logprobs = 20 # note that vllm can only return top-20 logprobs instead of self.tokenizer.vocab_size
    
    def _load_and_concatenate_datasets(self, dataset_name_or_path: str, split: str) -> Any:
        """Load and concatenate datasets from specified subsets, maintaining subset information."""
        dataset = None
        for subset in self.subsets:
            subset_dataset = load_dataset(dataset_name_or_path, subset, split=split)
            
            # Add a new column to store the subset name
            subset_dataset = subset_dataset.map(lambda example: {"subset": subset})
            
            if dataset is None:
                dataset = subset_dataset
            else:
                dataset = concatenate_datasets([dataset, subset_dataset])
            
            logger.info(f"Loaded {len(subset_dataset)} examples for subset {subset}")
        
        return dataset
    
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        Format the prompt for the model based on the dataset.
        Override this method for different datasets.
        
        Args:
            example: A single example from the dataset
            
        Returns:
            Formatted prompt string
        """
        prompt = "\n ".join([self.instruction, example])
        if self.is_chat_model:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt
    
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        Get prompts for the model based on the dataset.
        Override this method for different datasets.
        """
        # Format prompts
        prompts = []
        for example in batch_examples:
            # Filter to only include input keys
            prompt = ""
            for i, input_key in enumerate(self.input_keys):
                if input_key in example:
                    prompt += example[input_key]
                    if i < len(self.input_keys) - 1:
                        prompt += "\n"
                    if i == 0 and len(self.input_keys) > 1:
                        prompt += "Choose the correct option from the following options: \n"
                else:
                    raise ValueError(f"Input key {input_key} not found in example")
            prompts.append(prompt)
            
        return prompts

    def get_ground_truth_answers(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """
        Get the ground truth answer from the example.
        Override this method for different datasets.
        """
        return [example[self.output_key] for example in batch_examples]
    
    def extract_answer(self, response: str) -> str:
        """
        Extract the answer from the model's response.
        Override this method for different datasets.
        
        Args:
            response: Model's response
            
        Returns:
            Extracted answer
        """
        # Default implementation tries to extract after "Answer:" if present
        answer_keys = ["answer is:", "answer:"]
        response_lower = response.lower()
        
        # Create a regex pattern to match any of the answer keys
        pattern = r'(' + '|'.join(re.escape(key) for key in answer_keys) + r')'
        
        # Find all matches and extract the last one
        matches = list(re.finditer(pattern, response_lower))
        if matches:
            last_match = matches[-1]
            return response_lower[last_match.end():].strip("\n .")
        
        return response_lower.strip("\n .")
    
    def evaluate_correctness(self, extracted_answer: str, ground_truth: str) -> bool:
        """
        Evaluate if the extracted answer is correct.
        Override this method for different datasets.
        
        Args:
            extracted_answer: Extracted answer from model response
            ground_truth: Ground truth answer
            
        Returns:
            Boolean indicating correctness
        """
        # Simple exact match for now
        if isinstance(ground_truth, int):
            ground_truth = str(ground_truth)
        return extracted_answer.lower().strip(". ") == ground_truth.lower().strip(". ")

    def process_batch(self, batch_examples: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """
        Process a batch of examples.
        
        Args:
            batch_examples: List of examples to process
            
        Returns:
            List of evaluation results
        """

        prompts = self.get_prompts(batch_examples)
        prompts = [self.format_prompt(prompt) for prompt in prompts]

        # Get ground truth answers from each example
        ground_truth_answers = self.get_ground_truth_answers(batch_examples)
        
        # Generate responses
        outputs = self.model.generate(
            prompts, 
            self.sampling_params,
        )
        
        results = []
        for i, (prompt, ground_truth_answer, output) in enumerate(zip(prompts, ground_truth_answers, outputs)):
            response = output.outputs[0].text
            extracted_answer = self.extract_answer(response)
                        
            # Evaluate correctness if ground truth is available
            is_correct = None
            if ground_truth_answer is not None:
                is_correct = self.evaluate_correctness(extracted_answer, ground_truth_answer)
            
            # Get subset information from the example or metadata of the example
            if "subset" in batch_examples[i]:
                subset_info = batch_examples[i]["subset"]
            elif "metadata" in batch_examples[i] and "subset" in batch_examples[i]["metadata"]:
                subset_info = batch_examples[i]["metadata"]["subset"]
            else:
                subset_info = "unknown"
            
            result = EvaluationResult(
                # id=example.get("key", str(example.get("id", ""))),
                prompt=prompt,
                model_response=response,
                extracted_answer=extracted_answer,
                ground_truth_answer=ground_truth_answer,
                is_correct=is_correct,
                metadata={
                    "dataset": self.dataset_name_or_path,
                    "model": self.model_name_or_path,
                    "subset": subset_info,
                }
            )
            if self.save_logprobs:
                try:
                    logprobs = output.outputs[0].logprobs[0].values() # the logprobs of the first token
                except:
                    # If output is "".
                    logprobs = []
                if self.choices is None:
                    choices = get_choices(ground_truth_answer)
                else:
                    choices = self.choices
                logprob_dict = {entry.decoded_token: entry.logprob for entry in logprobs if entry.decoded_token in choices}
                result.logprobs = logprob_dict
            results.append(result)
        
        return results
    
    def evaluate(self, save_frequency: int = 10) -> Dict[str, Any]:
        """
        Evaluate the model on the dataset.
        
        Args:
            save_frequency: How often to save intermediate results
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_results = []
        batch_results = []
        
        # Create output filenames
        dataset_name = self.dataset_name_or_path.split("/")[-1]
        dataset_name = dataset_name.split(".")[0]
        model_name = self.model_name_or_path.split("/")[-1]
        output_jsonl = f"{self.output_dir}/{model_name}/{dataset_name}_{self.split}.jsonl"
        output_json = f"{self.output_dir}/{model_name}/{dataset_name}_{self.split}_all.json"
        os.makedirs(f"{self.output_dir}/{model_name}", exist_ok=True)
        
        # Process dataset in batches
        num_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        
        start_time = time.time()
        for i in tqdm(range(num_batches), desc="Processing batches"):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, len(self.dataset))
            batch_examples = self.dataset.to_list()[batch_start:batch_end]
            
            # Process batch
            results = self.process_batch(batch_examples)
            all_results.extend(results)
            batch_results.extend(results)
            
            # Save intermediate results
            if (i + 1) % save_frequency == 0:
                self._save_jsonl(output_jsonl, batch_results)
                logger.info(f"Saved batch results to '{output_jsonl}' ({len(batch_results)} samples)")
                batch_results = []
            

            # NOTE: for debug
            # if i == 5:
            #     break
        
        # Save any remaining results
        if batch_results:
            self._save_jsonl(output_jsonl, batch_results)
            logger.info(f"Saved final batch to '{output_jsonl}' ({len(batch_results)} samples)")
        
        # Save all results
        self._save_json(output_json, all_results)
        logger.info(f"Saved all results to '{output_json}' ({len(all_results)} samples)")
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_results)
        metrics["total_time"] = time.time() - start_time
        metrics["samples_per_second"] = len(all_results) / metrics["total_time"]
        
        # Save metrics
        metrics_file = f"{self.output_dir}/{model_name}/{dataset_name}_{self.split}_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Evaluation complete! Metrics saved to '{metrics_file}'")
        logger.info(f"Processed {len(all_results)} samples in {metrics['total_time']:.2f} seconds")
        
        return metrics
    
    def _save_jsonl(self, filename: str, results: List[EvaluationResult]) -> None:
        """Save results to a JSONL file."""
        with open(filename, "a", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
    
    def _save_json(self, filename: str, results: List[EvaluationResult]) -> None:
        """Save results to a JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([result.__dict__ for result in results], f, indent=2, ensure_ascii=False)
    
    def _calculate_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        metrics = {
            "total_samples": len(results),
        }
        
        # Calculate accuracy if we have ground truth
        correct_samples = [r for r in results if r.is_correct is True]
        if correct_samples:
            metrics["accuracy"] = len(correct_samples) / len(results)
        else:
            metrics["accuracy"] = 0
        
        return metrics