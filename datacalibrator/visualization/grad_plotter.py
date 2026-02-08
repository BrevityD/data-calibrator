import wandb
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List, Dict
from loguru import logger

class GradientPlotter:
    def __init__(self, entity: str, project: str, run_ids: Dict[str, str]):
        """
        Initialize the GradientPlotter.
        
        Args:
            entity: WandB entity (username or team name).
            project: WandB project name.
            run_ids: Dictionary mapping run name (or label) to WandB run ID.
        """
        self.api = wandb.Api()
        self.entity = entity
        self.project = project
        self.run_ids = run_ids
        self.histories = {}
        logger.info(f"Initialized GradientPlotter for runs: {self.run_ids}")

    def fetch_data(self):
        """Fetch history data from WandB for all runs."""
        for label, run_id in self.run_ids.items():
            run_path = f"{self.entity}/{self.project}/{run_id}"
            try:
                logger.info(f"Fetching run history for {label} ({run_path})...")
                run = self.api.run(run_path)
                # Scan history allows retrieving all logged keys
                history = pd.DataFrame(run.scan_history())
                self.histories[label] = history
                logger.info(f"Fetched {len(history)} steps of data for {label}.")
            except Exception as e:
                logger.error(f"Failed to fetch data for run {run_path}: {e}")

    def plot_3d_gradients(
        self, 
        metrics: List[str], 
        x_dataset: str = "code", 
        y_dataset: str = "math", 
        output_dir: str = "plots",
        show_plot: bool = False
    ):
        """
        Plot 3D graphs for gradient metrics with multiple runs.
        
        Args:
            metrics: List of metric suffixes to plot (e.g., ['grad_norm', 'grad_mean']).
            x_dataset: Name of the dataset for X-axis.
            y_dataset: Name of the dataset for Y-axis.
            output_dir: Directory to save plots.
            show_plot: Whether to display the plot interactively.
        """
        if not self.histories:
            self.fetch_data()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        splits = ['train', 'test']
        
        # Colors for different runs
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for metric in metrics:
            for split in splits:
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                has_data = False
                
                for i, (label, history) in enumerate(self.histories.items()):
                    x_col = f"{split}-{x_dataset}/{metric}"
                    y_col = f"{split}-{y_dataset}/{metric}"
                    step_col = "step"

                    if x_col not in history.columns or y_col not in history.columns:
                        logger.warning(f"Columns for {metric} in {split} split not found in run {label}. Skipping.")
                        continue

                    # Drop NaNs for these columns
                    df = history[[step_col, x_col, y_col]].dropna()
                    
                    if df.empty:
                        logger.warning(f"No valid data for {metric} in {split} split for run {label} after dropping NaNs.")
                        continue
                    
                    has_data = True

                    xs = df[x_col]
                    ys = df[y_col]
                    zs = df[step_col]
                    
                    color = colors[i % len(colors)]

                    # Plot scatter points
                    ax.scatter(xs, ys, zs, c=color, marker='o', s=50, label=f'{label}')
                    
                    # Draw lines connecting the points to visualize trajectory
                    ax.plot(xs, ys, zs, color=color, alpha=0.3)
                
                if not has_data:
                    plt.close(fig)
                    continue
                
                # Set origin to (0, 0, 0) if possible, but step usually starts >0.
                # However, for metric values, 0 is a meaningful baseline.
                # We enforce lower bounds for axes to include 0 for metrics.
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)
                ax.set_zlim(bottom=0)

                # Ensure axes direction (standard cartesian usually points out)
                # In matplotlib 3d, this is standard, but we can fix view if needed.
                # For "00 point fixed" usually means we want to see (0,0,0) clearly.

                # Labels
                ax.set_xlabel(f"{x_dataset.capitalize()} {metric}")
                ax.set_ylabel(f"{y_dataset.capitalize()} {metric}")
                ax.set_zlabel("Step")
                
                ax.set_title(f"3D Gradient Evolution: {split.capitalize()} {metric}")
                ax.legend()
                
                # Save
                filename = f"grad_3d_{split}_{metric}.png"
                save_path = output_path / filename
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved plot to {save_path}")
                
                if show_plot:
                    plt.show()
                else:
                    plt.close(fig)

    # def plot_comparative_evolution(
    #     self,
    #     metrics: List[str],
    #     datasets: List[str] = ["code", "math"],
    #     output_dir: str = "plots",
    #     show_plot: bool = False
    # ):
    #     """
    #     Plot 2D evolution of metrics over steps for multiple datasets.
    #     """
    #     if self.history is None:
    #         self.fetch_data()
            
    #     output_path = Path(output_dir)
    #     output_path.mkdir(parents=True, exist_ok=True)
        
    #     splits = ['train', 'test']
        
    #     for metric in metrics:
    #         for split in splits:
    #             plt.figure(figsize=(10, 6))
                
    #             has_data = False
    #             for ds in datasets:
    #                 col = f"{split}-{ds}/{metric}"
    #                 if col in self.history.columns:
    #                     df = self.history[["step", col]].dropna()
    #                     if not df.empty:
    #                         plt.plot(df["step"], df[col], marker='o', label=ds.capitalize())
    #                         has_data = True
                
    #             if has_data:
    #                 plt.xlabel("Step")
    #                 plt.ylabel(metric)
    #                 plt.title(f"{split.capitalize()} {metric} Evolution")
    #                 plt.legend()
    #                 plt.grid(True, alpha=0.3)
                    
    #                 filename = f"evolution_{split}_{metric}.png"
    #                 save_path = output_path / filename
    #                 plt.savefig(save_path, dpi=300)
    #                 logger.info(f"Saved plot to {save_path}")
                    
    #                 if show_plot:
    #                     plt.show()
    #                 else:
    #                     plt.close()
