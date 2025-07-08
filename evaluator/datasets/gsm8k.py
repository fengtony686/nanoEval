# datasets/gsm8k.py

import re
import logging
from typing import Dict, Any, List, Optional
from ..eval import Evaluator

# --- Math Verify Integration ---
try:
    from math_verify import is_equiv, extract_answer as mv_extract
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
# --- End Math Verify Integration ---

logger = logging.getLogger(__name__)

class GSM8K(Evaluator):
    """
    Evaluator for GSM8K dataset with enhanced answer extraction and verification.
    Utilizes math-verify package if available for better answer extraction and equivalence checking.
    """
    def __init__(
        self, 
        model_name_or_path: str, 
        dataset_name_or_path: str = "openai/gsm8k", 
        split: str = "test", 
        subset: str = "main", 
        temperature: float = 0.0,  # Override default for deterministic answers
        top_p: float = 1.0,        # Override default for deterministic answers
        **kwargs
    ):
        """Initialize the GSM8K evaluator with appropriate settings."""
        logger.info(f"Initializing GSM8K evaluator for model: {model_name_or_path}")
        logger.info(f"Dataset: {dataset_name_or_path}, Subset: {subset}, Split: {split}")
        
        if not MATH_VERIFY_AVAILABLE:
            logger.warning("Package 'math-verify' not found. Correctness evaluation will fallback to basic comparison. "
                          "Install with: pip install math-verify")
            
        # Call the base class constructor
        super().__init__(
            model_name_or_path=model_name_or_path, 
            dataset_name_or_path=dataset_name_or_path, 
            split=split, 
            subset=subset,
            temperature=temperature,  # Use deterministic sampling
            top_p=top_p,              # Use deterministic sampling
            save_logprobs=True,       # Force logprobs saving
            input_keys=["question"],  # Specific input key for GSM8K
            output_key="answer",      # Specific output key for GSM8K
            **kwargs
        )
        
        self.choices = None  # GSM8K is not multiple choice
        
        # Custom instruction for GSM8K
        self.instruction = (
            "Solve the following math word problem step by step. "
            "Think carefully and break down the problem. "
            "Make sure to include the final numerical answer after 'The answer is:'"
        )
        
        logger.info(f"Using GSM8K instruction prompt: {self.instruction}")
        logger.info(f"Using deterministic sampling for GSM8K: temp={self.temperature}, top_p={self.top_p}")
        
    def get_prompts(self, batch_examples: List[Dict[str, Any]]) -> List[str]:
        """Format prompts for GSM8K dataset."""
        prompts = []
        question_key = self.input_keys[0]  # Should be "question"
        
        for example in batch_examples:
            if question_key not in example or not example[question_key]:
                logger.error(f"Input key '{question_key}' not found or empty in example: {example.keys()}")
                question_text = "[Question text not found or empty]"
            else:
                question_text = example[question_key]
                
            # Combine the question text with the specific instruction
            prompt_content = f"{question_text}\n\n{self.instruction}"
            prompts.append(prompt_content)
            
        return prompts
        
    def format_prompt(self, prompt_content: str) -> str:
        """
        Format the prompt for the specific model type.
        Handle chat templating if necessary.
        """
        if self.is_chat_model:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_content}
            ]
            try:
                final_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                logger.debug(f"Applied chat template. Final prompt starts with: {final_prompt[:250]}...")
                return final_prompt
            except Exception as e:
                logger.error(f"Failed to apply chat template: {e}. Using raw prompt content.", exc_info=True)
                return prompt_content  # Fallback
        else:
            logger.debug(f"Using raw prompt (not chat). Final prompt starts with: {prompt_content[:250]}...")
            return prompt_content
            
    def extract_answer(self, response: str) -> str:
        """
        Extract the numerical answer from the response.
        Uses math-verify's extract_answer if available.
        """
        if not response:
            logger.debug("Response is empty, cannot extract answer")
            return ""
            
        # Try math-verify extraction first if available
        if MATH_VERIFY_AVAILABLE:
            try:
                extracted = mv_extract(response)
                if extracted is not None:
                    logger.debug(f"Extracted answer using math-verify: '{extracted}'")
                    return str(extracted)
            except Exception as e:
                logger.warning(f"math-verify extraction failed: {e}. Falling back to regex methods.")
                
        # First look for "the answer is: X" pattern
        answer_patterns = [
            r'(?:the\s+)?answer\s+(?:is|=)\s*[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?',
            r'(?:the\s+)?final\s+(?:answer|result)\s+(?:is|=)\s*[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?',
            r'(?:Therefore|Thus|So|Hence),\s+(?:the\s+)?(?:answer|result)\s+(?:is|=)\s*[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                # Extract just the number from the matched text
                num_match = re.search(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', match.group(0))
                if num_match:
                    logger.debug(f"Extracted answer via answer pattern: '{num_match.group(0)}'")
                    return num_match.group(0).replace(',', '')
        
        # Try to find the last number preceded by an equals sign
        equals_match = re.findall(r'=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', response)
        if equals_match:
            logger.debug(f"Extracted answer via equals pattern: '{equals_match[-1]}'")
            return equals_match[-1].replace(',', '')
        
        # Look for the last standalone numeric value (possibly with commas and decimals)
        numbers = re.findall(r'(?:^|\s)([-+]?(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?))(?=$|\s)', response)
        if numbers:
            logger.debug(f"Extracted last standalone number: '{numbers[-1]}'")
            return numbers[-1].replace(',', '')
            
        # If all else fails, look for dollar amounts
        dollar_match = re.search(r'\$\s*([-+]?[0-9]*\.?[0-9]+)', response)
        if dollar_match:
            logger.debug(f"Extracted dollar amount: '{dollar_match.group(1)}'")
            return dollar_match.group(1).replace(',', '')
            
        logger.warning(f"Could not extract an answer from response: '{response[:100]}...'")
        return ""
        
    def evaluate_correctness(self, extracted_answer: str, ground_truth: str) -> bool:
        """
        Check if the numerical answer is correct.
        Uses math-verify's is_equiv if available.
        """
        if not extracted_answer:
            logger.debug("Extracted answer is empty, marking as incorrect")
            return False
            
        if ground_truth is None:
            logger.debug("Ground truth is None, cannot evaluate correctness")
            return None
            
        # Clean up the ground truth (GSM8K often has explanatory text)
        gt_number_match = re.search(r'[-+]?(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)', ground_truth)
        if not gt_number_match:
            logger.warning(f"Could not extract a numeric ground truth from: '{ground_truth}'")
            return False
            
        gt_number = gt_number_match.group(0).replace(',', '')
        
        # Try math-verify equivalence checking first if available
        if MATH_VERIFY_AVAILABLE:
            try:
                is_equivalent = is_equiv(extracted_answer, gt_number)
                logger.debug(f"Math-verify comparison: Extracted='{extracted_answer}', GT='{gt_number}', Result={is_equivalent}")
                return is_equivalent
            except Exception as e:
                logger.warning(f"math-verify equivalence check failed: {e}. Falling back to numeric comparison.")
        
        # Fallback to direct numeric comparison
        try:
            # Convert both to floats for comparison
            extracted_num = float(extracted_answer)
            ground_truth_num = float(gt_number)
            
            # Check if they're equal (allowing for small floating-point errors)
            is_equal = abs(extracted_num - ground_truth_num) < 1e-6
            logger.debug(f"Numeric comparison: Extracted={extracted_num}, GT={ground_truth_num}, Result={is_equal}")
            return is_equal
        except ValueError as e:
            logger.error(f"Error converting to float for comparison: {e}")
            return False