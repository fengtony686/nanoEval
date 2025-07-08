import os
import json
import csv
import glob
from collections import defaultdict

def aggregate_metrics(results_dir="results", output_dir=None):
    """
    Recursively search through the results directory, find all metrics.json files,
    and aggregate their contents into a CSV file.
    
    Args:
        results_dir: Path to the results directory
        output_dir: Directory to save the output CSV file (default: current directory)
    """
    # Dictionary to store all metrics data
    all_metrics = []
    
    # Walk through the results directory
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        
        # Skip if not a directory
        if not os.path.isdir(model_path):
            continue
        
        # Find all metrics.json files for this model
        metrics_files = glob.glob(os.path.join(model_path, "*_metrics.json"))
        
        for metrics_file in metrics_files:
            # Extract dataset name from the filename
            # Format is: dataset_name_split_metrics.json
            base_filename = os.path.basename(metrics_file)
            parts = base_filename.split('_')
            dataset_name = parts[0]
            if len(parts) > 2:
                # If filename has multiple underscores, combine the parts except the last two
                dataset_name = '_'.join(parts[:-2])
                split = parts[-2]
            else:
                # If only one underscore, use the first part as dataset
                split = parts[0]
            
            # Load metrics from the file
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                    
                # Add model and dataset info to the metrics
                metrics['model'] = model_dir
                metrics['dataset'] = dataset_name
                metrics['split'] = split
                
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error processing {metrics_file}: {e}")
    
    # Determine all possible metric columns
    metric_columns = set()
    for metrics in all_metrics:
        metric_columns.update(metrics.keys())
    
    # Remove model, dataset, split which we'll handle separately
    metric_columns.discard('model')
    metric_columns.discard('dataset')
    metric_columns.discard('split')
    
    # Create ordered columns list with model first, accuracy second, and the rest alphabetically
    columns = ['model', 'accuracy']
    # Add remaining columns alphabetically
    remaining_columns = [col for col in sorted(list(metric_columns)) if col != 'accuracy']
    columns.extend(remaining_columns)
    # Add dataset and split after all metric columns
    columns.append('dataset')
    columns.append('split')
    
    # Set output file paths
    if output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        all_metrics_file = os.path.join(output_dir, "aggregated_metrics.csv")
        acc_results_file = os.path.join(output_dir, "acc_results.csv")
    else:
        all_metrics_file = "aggregated_metrics.csv"
        acc_results_matrix_file = "acc_results.csv"
    
    # Write to CSV (all metrics)
    with open(all_metrics_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for metrics in all_metrics:
            # Make sure all columns have values (even if None)
            row = {col: metrics.get(col, None) for col in columns}
            writer.writerow(row)
    
    print(f"Successfully aggregated metrics from {len(all_metrics)} files to {all_metrics_file}")
    
    # Create model-dataset matrix with accuracy values
    # Get unique models and datasets
    models = sorted(list(set(m['model'] for m in all_metrics)))
    datasets = sorted(list(set(m['dataset'] for m in all_metrics)))

    # Create dictionary to store accuracy values
    acc_results = defaultdict(dict)

    # Populate the matrix with accuracy values
    for metric in all_metrics:
        model = metric['model']
        dataset = metric['dataset']
        split = metric['split']
        # Use accuracy if available, otherwise None
        accuracy = metric.get('accuracy', 0)
        
        # Use dataset_split as the key to handle multiple splits
        dataset_key = f"{dataset}"
        acc_results[model][dataset_key] = accuracy

    # Get all unique dataset_split combinations
    dataset_keys = sorted(list(set(f"{m['dataset']}" for m in all_metrics)))

    # Write the model-dataset matrix to CSV
    with open(acc_results_file, 'w', newline='', encoding='utf-8') as f:
        # First column is model, then one column per dataset
        fieldnames = ['model'] + dataset_keys
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write each model's row
        for model in models:
            row = {'model': model}
            # Add accuracy for each dataset (or None if not available)
            for dataset_key in dataset_keys:
                row[dataset_key] = acc_results[model].get(dataset_key, None)
            writer.writerow(row)

    print(f"Successfully created model-dataset acc matrix at {acc_results_file}")
    
    return all_metrics_file, acc_results_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate metrics from JSON files to CSV")
    parser.add_argument("--results_dir", type=str, default="results", 
                        help="Directory containing the results")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Directory to save the output CSV files")
    
    args = parser.parse_args()
    
    aggregate_metrics(args.results_dir, args.output_dir)
