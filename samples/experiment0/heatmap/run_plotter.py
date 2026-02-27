import sys
from pathlib import Path

# Ensure the project root is in the python path to import modules
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from datacalibrator.visualization.comet_plotter import CometPlotter

def main():
    base_dir = Path(__file__).parent
    # The JSON file is expected to be in log_data/training_results.json relative to this script
    json_path = base_dir / 'log_data/training_results.json'
    output_plot = base_dir / 'comet_plot.png'
    
    if not json_path.exists():
        print(f"Error: {json_path} does not exist.")
        # Fallback to check if it's in the current working directory or check absolute path
        # But per instruction, we use the extracted file.
        return

    print(f"Generating plot from {json_path}...")
    try:
        plotter = CometPlotter(json_path)
        plotter.plot(
            x_metric='eval_code_loss',
            y_metric='eval_math_loss',
            output_path=output_plot
        )
        print(f"Successfully generated {output_plot}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
