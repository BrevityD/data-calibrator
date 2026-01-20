import os
import json
import re
import matplotlib.pyplot as plt
from pathlib import Path

def parse_checkpoint_step(name):
    match = re.search(r"checkpoint-(\d+)", name)
    if match:
        return int(match.group(1))
    return -1

def main():
    base_model_dir = "/public/home/dzj/data-calibrator/samples/experiment1/code_domain/outputs-5e7-bs16-ep3-sgd"
    reports_dir = "outputs/reports"
    output_plot = "outputs/reports/bigcodebench_out_domain.png"
    
    # Get all potential checkpoints
    all_ckpts = [d for d in os.listdir(base_model_dir) if d.startswith("checkpoint-")]
    all_ckpts.sort(key=parse_checkpoint_step)
    
    data = []
    
    for ckpt in all_ckpts:
        step = parse_checkpoint_step(ckpt)
        if step == -1:
            continue
            
        report_file = Path(reports_dir) / ckpt / "bigcodebench.json"
        score = 0.0
        
        if report_file.exists():
            try:
                with open(report_file, 'r') as f:
                    content = json.load(f)
                    # Try to find out_domain score
                    found_subset = False
                    if "metrics" in content:
                        for metric in content["metrics"]:
                            if metric["name"] == "mean_acc":
                                for cat in metric.get("categories", []):
                                    for subset in cat.get("subsets", []):
                                        if subset["name"] == "out_domain":
                                            score = subset["score"]
                                            found_subset = True
                                            break
                                    if found_subset: break
                            if found_subset: break
                    
                    if not found_subset:
                        # Fallback to top-level score if subsets not found as expected
                        score = content.get("score", 0.0)
            except Exception as e:
                print(f"Error reading {report_file}: {e}")
        
        data.append((step, score))
    
    # Sort by step
    data.sort(key=lambda x: x[0])
    steps = [x[0] for x in data]
    scores = [x[1] for x in data]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, scores, marker='.', linestyle='-', color='b', markersize=4)
    
    # Highlight non-zero points
    non_zero_steps = [s for s, v in data if v > 0]
    non_zero_scores = [v for s, v in data if v > 0]
    plt.scatter(non_zero_steps, non_zero_scores, color='red', s=20, label='Evaluated')

    plt.title('BigCodeBench (out_domain) Accuracy vs Checkpoint Step')
    plt.xlabel('Step')
    plt.ylabel('Score')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.ylim(-0.01, max(scores + [0.1]) * 1.2)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    
    # Also print some stats
    evaluated_count = len(non_zero_steps)
    total_count = len(steps)
    print(f"Total checkpoints: {total_count}")
    print(f"Evaluated checkpoints: {evaluated_count}")

if __name__ == "__main__":
    main()
