
from typing import List

def get_choices(ground_truth_answer: str) -> List[str]:
    """
    Determine the list of choices based on the ground truth answer.
    
    Args:
        ground_truth_answer: The ground truth answer from the dataset.
    
    Returns:
        A list of choices.
    """
    if ground_truth_answer in ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)", "(K)", "(L)", "(M)", "(N)", "(O)", "(P)", "(Q)"]:
        return ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)", "(K)", "(L)", "(M)", "(N)", "(O)", "(P)", "(Q)"] # BBH
    elif ground_truth_answer in ["A", "B", "C", "D", "E", "F"]:
        return ["A", "B", "C", "D", "E", "F"]
    elif ground_truth_answer in ["True", "False"]:
        return ["True", "False"]
    elif ground_truth_answer in ["Yes", "No"]:
        return ["Yes", "No"]
    elif ground_truth_answer in ["yes", "no"]:
        return ["yes", "no"]
    else:
        return [ground_truth_answer]