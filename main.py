import argparse
from evaluator import get_evaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate language models on benchmarks")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B", help="Model name or path")
    parser.add_argument("--dataset", type=str, default="lukaemon/bbh", help="Dataset name or path")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--max_model_len", type=int, default=4096, help="Max model_len allowed.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_frequency", type=int, default=10, help="How often to save results (in batches)")
    parser.add_argument("--input_keys", type=str, default="['prompt']", help="Input keys")
    parser.add_argument("--output_key", type=str, default="target", help="Output key")
    parser.add_argument("--save_logprobs", type=bool, default=True, help="Save logprobs")
    args = parser.parse_args()
    
    evaluator = get_evaluator(args.dataset)(
        model_name_or_path=args.model,
        dataset_name_or_path=args.dataset,
        input_keys=args.input_keys,
        output_key=args.output_key,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len, # For some model like Phythia, only 2048 length is allowed
        batch_size=args.batch_size,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        save_logprobs=args.save_logprobs,
    )
    
    # Run evaluation
    metrics = evaluator.evaluate(save_frequency=args.save_frequency)
    
    # Print summary metrics
    print("\nEvaluation Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
