import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
from loguru import logger

class CometPlotter:
    def __init__(self, train_results_file: Union[str, Path]):
        """
        Initialize the CometPlotter.
        
        Args:
            train_results_file: Path to the JSON file containing training results.
        """
        if isinstance(train_results_file, str):
            self.result_file = Path(train_results_file)
        elif isinstance(train_results_file, Path):
            self.result_file = train_results_file
        else:
            logger.error(f"Invalid type for train_results_file: {type(train_results_file)}")
            raise TypeError("train_results_file must be a string or Path")
            
        if not self.result_file.exists():
            logger.error(f"File not found: {self.result_file}")
            raise FileNotFoundError(f"File not found: {self.result_file}")
            
        logger.info(f"Initialized CometPlotter for file: {self.result_file.name}")
        self.data = self.load_data()

    def load_data(self) -> Dict:
        """Load data from the JSON file."""
        try:
            with open(self.result_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load data from {self.result_file}: {e}")
            raise

    def get_series(self, run_data: Dict, x_metric: str, y_metric: str) -> Tuple[List[float], List[float]]:
        """Extract X and Y series from a single run data."""
        # Sort steps numerically
        try:
            steps = sorted([int(k) for k in run_data.keys()])
        except ValueError:
             # Fallback if keys are not integers
            steps = sorted(run_data.keys())
            
        xs = []
        ys = []
        for s in steps:
            step_key = str(s)
            step_data = run_data[step_key]
            if x_metric in step_data and y_metric in step_data:
                xs.append(step_data[x_metric])
                ys.append(step_data[y_metric])
        return xs, ys

    def plot(self, 
             x_metric: str = 'eval_code_loss', 
             y_metric: str = 'eval_math_loss', 
             output_path: Optional[Union[str, Path]] = None,
             show_pareto: bool = True):
        """
        Generate the Comet Plot.

        Args:
            x_metric: Metric name for the X-axis (L1).
            y_metric: Metric name for the Y-axis (L2).
            output_path: Path to save the plot. If None, shows the plot.
            show_pareto: Whether to draw the Pareto Frontier connecting branch endpoints.
        """
        
        plt.figure(figsize=(12, 10))
        
        # Sort run IDs numerically to ensure order
        try:
            run_ids = sorted([int(k) for k in self.data.keys()])
            run_ids = [str(k) for k in run_ids]
        except ValueError:
            run_ids = sorted(self.data.keys())

        # 1. Plot Backbone (0th step of each run)
        backbone_x = []
        backbone_y = []
        
        # 2. Plot Branches
        branch_endpoints = []
        
        logger.info("Processing data for Backbone and Branches...")

        for run_id in run_ids:
            run_data = self.data[run_id]
            
            # Get full branch trajectory
            rx, ry = self.get_series(run_data, x_metric, y_metric)
            
            if not rx:
                continue
            
            # Add the first point of this run to the backbone
            backbone_x.append(rx[0])
            backbone_y.append(ry[0])
            
            # Plot branch
            plt.plot(rx, ry, color='red', alpha=0.3, linewidth=1, marker='.', markersize=1, zorder=1)
            
            # Collect endpoint for Pareto Frontier
            branch_endpoints.append((rx[-1], ry[-1]))

        # Plot Backbone
        if backbone_x:
            logger.info(f"Plotting backbone with {len(backbone_x)} points")
            plt.plot(backbone_x, backbone_y, label='Backbone (L1 Training)', color='black', linewidth=2, marker='o', markersize=4, alpha=0.8, zorder=10)
            
            # Mark start and end of backbone
            plt.scatter(backbone_x[0], backbone_y[0], color='green', s=100, label='Start', zorder=11, edgecolors='black')
            plt.scatter(backbone_x[-1], backbone_y[-1], color='blue', s=100, label='End', zorder=11, edgecolors='black')
        else:
            logger.warning("No backbone data found.")

        # 3. Plot Pareto Frontier
        if show_pareto and branch_endpoints:
            # Sort endpoints by x (L1) to draw a line connecting them
            branch_endpoints.sort(key=lambda p: p[0])
            px, py = zip(*branch_endpoints)
            plt.plot(px, py, color='purple', linestyle='--', label='Pareto Frontier', linewidth=2, alpha=0.7, zorder=5)
            plt.scatter(px, py, color='purple', s=30, zorder=5)

        plt.xlabel(x_metric)
        plt.ylabel(y_metric)
        plt.title(f"Comet Plot: {x_metric} vs {y_metric}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if output_path:
            save_path = Path(output_path)
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
            logger.info(f"Plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        plotter = CometPlotter(file_path)
        plotter.plot(output_path="comet_plot.png")
    else:
        logger.warning("No input file provided. Usage: python comet_plotter.py <path_to_training_results.json>")
