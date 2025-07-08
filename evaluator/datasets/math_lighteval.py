# datasets/math_lighteval.py

import re
import logging
from typing import Dict, Any, List, Optional

# Adjust the import path based on your project structure
# Assumes eval.py is in the parent directory of this file's directory
try:
    from ..eval import Evaluator
except (ImportError, ValueError):
    try:
        from eval import Evaluator
    except ImportError:
        raise ImportError("Could not import the base 'Evaluator' class. Ensure eval.py is accessible.")


# --- Math Verify Integration ---
try:
    from math_verify import is_equiv
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
# --- End Math Verify Integration ---

logger = logging.getLogger(__name__)

# --- Helper functions mimicking verl logic for extraction ---
def _find_last_boxed(text: Optional[str]) -> Optional[str]:
    """Finds the last occurrence of \\boxed{...} in a string."""
    if text is None: return None
    matches = re.findall(r'\\boxed\{.+?\}', str(text))
    return matches[-1] if matches else None

def _remove_box_wrapper(boxed_text: Optional[str]) -> Optional[str]:
    """Extracts the content from a \\boxed{...} string."""
    if boxed_text is None: return None
    match = re.search(r'\\boxed\{(.+?)\}', boxed_text)
    return match.group(1).strip(' $') if match else None
# --- End Helper Functions ---


class MATHLightEval(Evaluator):
    """
    Evaluator subclass for MATH dataset, compatible with the user-provided eval.py.

    Uses `math-verify` if available. Forces logprobs saving.
    Overrides prompt formatting and answer extraction/evaluation logic.
    Extraction prioritizes the *last* boxed answer.
    """
    def __init__(
        self,
        model_name_or_path: str,
        dataset_name_or_path: str = "lighteval/MATH", # Or DigitalLearningGmbH/MATH-lighteval
        split: str = "test",
        subset: str = "default",
        temperature: float = 0.0, # Override default temp for MATH
        top_p: float = 1.0,       # Override default top_p for MATH
        **kwargs # Accept other base class args from command line/script
    ):
        """
        Initializes the MATHLightEval evaluator.
        """
        logger.info(f"Initializing MATHLightEval for model: {model_name_or_path}")
        logger.info(f"Dataset: {dataset_name_or_path}, Subset: {subset}, Split: {split}")

        if not MATH_VERIFY_AVAILABLE:
             logger.warning("Package 'math-verify' not found. Correctness evaluation will fallback to basic string comparison. "
                            "Install with: pip install math-verify")

        # Call the base class constructor, passing only arguments it accepts
        # Get defaults from kwargs or use base class defaults if not provided
        super().__init__(
             model_name_or_path=model_name_or_path,
             dataset_name_or_path=dataset_name_or_path,
             split=split,
             subset=subset, # Pass subset correctly
             output_dir=kwargs.get('output_dir', 'results'),
             max_new_tokens=kwargs.get('max_new_tokens', 1024),
             batch_size=kwargs.get('batch_size', 8), # As defined in user's eval.py
             temperature=temperature, # Use MATH specific default
             top_p=top_p,           # Use MATH specific default
             tensor_parallel_size=kwargs.get('tensor_parallel_size', 4), # Match user's default
             seed=kwargs.get('seed', 42),
             max_model_len=kwargs.get('max_model_len', 4096), # Match user's default
             # --- Force/Set Settings Specific to this Evaluator ---
             save_logprobs=True,      # Force logprobs saving
             input_keys=["problem"],  # Specific input key for MATH dataset
             output_key="solution"   # Specific output key for MATH dataset
             # *** Arguments NOT accepted by user's eval.py are omitted: ***
             # trust_remote_code, gpu_memory_utilization, dtype
        )

        # Note: Base class sets self.tokenizer and self.model (using trust_remote_code=True internally)
        # Base class also sets self.sampling_params (without stop tokens by default)

        self.choices = None # MATH is typically not multiple choice

        # Custom instruction for MATH dataset evaluation
        # NOTE: The base format_prompt might prepend its own instruction if not overridden!
        self.instruction = (
            "Solve the following math problem step by step. "
            "Think step-by-step to reach the solution. "
            "Present the final numerical or symbolic answer enclosed in \\boxed{}. "
            "The final answer should be the only thing within the \\boxed{}."
        )
        logger.info(f"Using MATH instruction prompt: {self.instruction}")
        logger.info(f"Using deterministic sampling for MATH: temp={self.temperature}, top_p={self.top_p}")
        if self.save_logprobs:
            logger.warning("save_logprobs=True is set, but the base eval.py uses get_choices for saving. "
                           "Logprobs field in output may be empty for MATH.")


    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """Formats the core prompt content for the MATH dataset."""
        prompts = []
        problem_key = self.input_keys[0] # Should be "problem"
        for example in batch_examples:
            if problem_key not in example or not example[problem_key]:
                logger.error(f"Input key '{problem_key}' not found or empty in example: {example.keys()}")
                problem_text = "[Problem text not found or empty]"
            else:
                problem_text = example[problem_key]
            # Combine the problem text with the specific instruction FOR THIS EVALUATOR
            prompt_content = f"{problem_text}\n\n{self.instruction}"
            prompts.append(prompt_content)
        return prompts


    # --- Override format_prompt to prevent base class duplication ---
    def format_prompt(self, prompt_content: str) -> str:
        """
        Formats the prompt string for the specific model type.
        Prevents base class from adding its instruction again.
        Handles chat templating if necessary.

        Args:
            prompt_content: The prompt string generated by this class's get_prompts.

        Returns:
            The final formatted prompt string ready for the model.
        """
        # The base __init__ sets self.is_chat_model and self.tokenizer
        if self.is_chat_model:
            # Apply the chat template using the tokenizer loaded in the base class
            # Base class defines self.system_prompt
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_content}
            ]
            try:
                 # Base class has self.tokenizer
                 final_prompt = self.tokenizer.apply_chat_template(
                     messages,
                     tokenize=False,
                     add_generation_prompt=True
                 )
                 logger.debug(f"Applied chat template. Final prompt starts with: {final_prompt[:250]}...")
                 return final_prompt
            except Exception as e:
                 logger.error(f"Failed to apply chat template: {e}. Using raw prompt content.", exc_info=True)
                 return prompt_content # Fallback
        else:
            # For base models, the prompt from get_prompts is the final prompt
            logger.debug(f"Using raw prompt (not chat). Final prompt starts with: {prompt_content[:250]}...")
            return prompt_content
    # --- End Override format_prompt ---


    def _extract_final_answer_from_string(self, text: Optional[str]) -> Optional[str]:
        """
        Helper function to extract the likely final answer from a string.
        Prioritizes the *last* \\boxed{}, then common phrases, then last line.
        """
        # (Implementation remains the same as previous version)
        if text is None: return None
        text_str = str(text).strip()
        if not text_str: return None

        last_box_str = _find_last_boxed(text_str)
        if last_box_str:
            content = _remove_box_wrapper(last_box_str)
            if content is not None:
                logger.debug(f"Extracted answer from last box: '{content}'")
                return content
            else:
                logger.warning(f"Found last box '{last_box_str}' but failed to extract content.")
                return None

        logger.debug(f"No \\boxed{{...}} found in: '{text_str[:100]}...'. Trying fallbacks.")
        answer_keys = ["final answer is:", "the answer is:", "answer is:", "answer:"]
        pattern = r'(?:' + '|'.join(re.escape(key) for key in answer_keys) + r')\s*([\s\S]*)$'
        matches = list(re.finditer(pattern, text_str, re.IGNORECASE))
        if matches:
            last_match_content = text_str[matches[-1].end():].strip()
            if last_match_content:
                first_line = last_match_content.split('\n')[0].strip(' .$')
                if first_line:
                    logger.debug(f"Extracted answer via keyword fallback: '{first_line}'")
                    return first_line

        lines = text_str.split('\n')
        for line in reversed(lines):
            stripped_line = line.strip(' .$')
            if stripped_line:
                if len(stripped_line) < 80 and not re.fullmatch(r'[a-zA-Z\s]+', stripped_line):
                    if re.search(r'[\d\=\$\(\)\^]', stripped_line) or len(stripped_line) < 10:
                        logger.debug(f"Extracted answer via last line fallback: '{stripped_line}'")
                        return stripped_line
                break

        logger.debug(f"Could not extract specific answer pattern using fallbacks either for: '{text_str[:100]}...'")
        return None


    def extract_answer(self, response: str) -> str:
        """Extract the final answer from the model's response string."""
        extracted = self._extract_final_answer_from_string(response)
        return extracted if extracted is not None else ""


    def evaluate_correctness(self, extracted_answer: str, ground_truth: Optional[str]) -> Optional[bool]:
        """
        Check if the extracted math answer is equivalent to the ground truth answer.
        Uses `math-verify.is_equiv` if available, otherwise falls back.
        """
        # (Implementation remains the same as previous version)
        if ground_truth is None:
             logger.debug("Ground truth is None, cannot evaluate correctness.")
             return None

        if not extracted_answer:
             logger.debug("Evaluating correctness: Model extracted answer is empty.")
             return False

        gt_final_answer = self._extract_final_answer_from_string(ground_truth)

        if gt_final_answer is None:
            logger.warning(f"Could not extract final answer from ground truth: '{ground_truth[:100]}...'. Treating as incorrect.")
            return False

        if MATH_VERIFY_AVAILABLE:
            try:
                is_equivalent = is_equiv(str(extracted_answer), str(gt_final_answer))
                logger.debug(f"Math-Verify comparison: Extracted='{extracted_answer}', GT_Final='{gt_final_answer}', Result={is_equivalent}")
                return is_equivalent
            except Exception as e:
                logger.error(f"Math-Verify comparison error for Extracted='{extracted_answer}' and GT_Final='{gt_final_answer}': {e}", exc_info=False)
                return False
        else:
            logger.debug("Using basic string comparison fallback (math-verify not available).")
            norm_extracted = re.sub(r'\s+', '', extracted_answer.lower()).strip(' .,$')
            norm_gt = re.sub(r'\s+', '', gt_final_answer.lower()).strip(' .,$')
            is_equivalent = norm_extracted == norm_gt
            logger.debug(f"Basic comparison: NormExtracted='{norm_extracted}', NormGT='{norm_gt}', Result={is_equivalent}")
            return is_equivalent