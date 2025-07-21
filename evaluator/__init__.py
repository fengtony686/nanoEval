from .eval import Evaluator
from .datasets.commonsense_qa import CommonsenseQA
from .datasets.neqa import NeQA
from .datasets.winogrande import Winogrande
from .datasets.bbh import BBH, BBH_synthetic
from .datasets.drop import DROP
from .datasets.openbookqa import OpenbookQA
from .datasets.gpqa import GPQA
from .datasets.quality import Quality
from .datasets.sciq import SciQ
from .datasets.piqa import PIQA
from .datasets.logiqa import LogiQA
from .datasets.musr import MuSR
from .datasets.mathqa import MathQA
# from .selection import BenchmarkSelector
from .datasets.mmlu import MMLU
from .datasets.hellaswag import HellaSWAG
from .datasets.arc import ARC
from .datasets.gsm8k import GSM8K
from .datasets.math_lighteval import MATHLightEval


def get_evaluator(dataset_name: str) -> Evaluator:
    if dataset_name == "tau/commonsense_qa":
        return CommonsenseQA
    elif dataset_name == "inverse-scaling/NeQA":
        return NeQA
    elif dataset_name == "allenai/winogrande":
        return Winogrande
    elif dataset_name == "lukaemon/bbh":
        return BBH
    elif dataset_name == "cais/mmlu":
        return MMLU
    elif dataset_name == "Rowan/hellaswag":
        return HellaSWAG
    elif dataset_name == "allenai/ai2_arc":
        return ARC
    elif dataset_name == "openai/gsm8k":
        return GSM8K
    elif dataset_name == "DigitalLearningGmbH/MATH-lighteval":
        return MATHLightEval
    elif dataset_name.endswith("bbh_syn_all.json"):
        return BBH_synthetic
    elif dataset_name == "ucinlp/drop":
        return DROP
    elif dataset_name == "allenai/openbookqa":
        return OpenbookQA
    elif dataset_name == "Idavidrein/gpqa":
        return GPQA
    elif dataset_name == "emozilla/quality":
        return Quality
    elif dataset_name == "allenai/sciq":
        return SciQ
    elif dataset_name == "lucasmccabe/logiqa":
        return LogiQA
    elif dataset_name == "TAUR-Lab/MuSR":
        return MuSR
    elif dataset_name == "ybisk/piqa":
        return PIQA
    elif dataset_name == "allenai/math_qa":
        return MathQA
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

__all__ = ['BenchmarkSelector']
