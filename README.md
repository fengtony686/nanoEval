# nanoEval

We introduce a minimal codebase for evaluating language models on benchmarks.


## Setup

```bash
uv venv $SCRATCH/envs/eval --python 3.10 && source $SCRATCH/envs/eval/bin/activate  && uv pip install pip
uv pip install setuptools torch vllm datasets transformers
source $SCRATCH/envs/eval/bin/activate
```

## Usage

```bash
python main.py --model Qwen/Qwen2.5-3B --dataset lukaemon/bbh --output_dir results
```

## Results

```bash
python aggregate_results.py --results_dir results --output_dir results
```

## Citation
To cite this repository:


```bibtex
@software{nanoeval,
  author = {Hanlin Zhang},
  title = {nanoEval: A Minimal Codebase for Evaluating Language Models on Benchmarks},
  url = {https://github.com/hanlinzhang/nanoEval},
  version = {0.1.0},
  year = {2025},
}
```