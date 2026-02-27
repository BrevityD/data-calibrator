import ast
import json
import os

def read_log(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return data
        
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Some lines might be plain text or errors, skip them if they don't look like dicts
            if not line.startswith('{'):
                continue
            try:
                # The log format seems to use single quotes, so ast.literal_eval is safer
                entry = ast.literal_eval(line)
                data.append(entry)
            except Exception as e:
                # Skip invalid lines
                continue
    return data

def process_data(data):
    # Group by epoch
    epochs = {}
    for entry in data:
        if 'epoch' not in entry:
            continue
        epoch = entry['epoch']
        if epoch not in epochs:
            epochs[epoch] = {}
        epochs[epoch].update(entry)
    
    # Sort by epoch to ensure chronological order
    sorted_epochs = sorted(epochs.keys())
    
    processed_results = {}
    
    # Use integer incrementing keys as requested
    for i, epoch in enumerate(sorted_epochs):
        merged_entry = epochs[epoch]
        
        # Filter keys: remove 'runtime' and 'second' related items
        clean_entry = {'epoch': epoch}
        for k, v in merged_entry.items():
            if 'runtime' in k or 'second' in k:
                continue
            clean_entry[k] = v
            
        processed_results[i] = clean_entry
        
    return processed_results

def main():
    from pathlib import Path
    from loguru import logger

    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parents[4]
    
    wandb_base_rel_path = 'samples/experiment0/heatmap/wandb'
    wandb_base = project_root / wandb_base_rel_path
    output_rel_path = 'samples/experiment0/heatmap/log_data/training_results.json'
    output_path = project_root / output_rel_path

    if not wandb_base.exists():
        logger.error(f"Wandb directory not found: {wandb_base}")
        return

    # Find run directories
    run_dirs = []
    # Patterns: run-20260202* and run-20260203*
    for pattern in ['run-20260202*', 'run-20260203*']:
        run_dirs.extend(wandb_base.glob(pattern))
    
    # Sort by name (timestamp is in the name)
    run_dirs.sort(key=lambda x: x.name)
    
    master_results = {}
    
    for idx, run_dir in enumerate(run_dirs):
        log_path = run_dir / 'files/output.log'
        if not log_path.exists():
            logger.warning(f"Log file not found in {run_dir}")
            continue
            
        logger.info(f"Processing run {idx}: {run_dir.name}")
        data = read_log(log_path)
        
        if not data:
            logger.warning(f"No valid data found in log file: {log_path}")
            continue

        results = process_data(data)
        master_results[idx] = results

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(master_results, f, indent=4)

    logger.info(f"Processed {len(master_results)} runs. Results saved to {output_path}")

if __name__ == '__main__':
    main()
