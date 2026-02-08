from dataclasses import dataclass, field
from typing import List, Dict
import sys
from pathlib import Path

# Add project root to path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[3]
sys.path.append(str(project_root))

from datacalibrator.visualization.grad_plotter import GradientPlotter

@dataclass
class VizConfig:
    entity: str
    project: str
    # Dictionary of label -> run_id
    run_ids: Dict[str, str]
    output_dir: str
    metrics: List[str]
    show_plot: bool

def main():
    # Configuration
    config = VizConfig(
        entity="brevity",
        project="data-calibrator-gradient-distance",
        run_ids={
            "Code Model": "idccss11",
            "Math Model": "idcmss11"
        },
        output_dir="plots",
        metrics=["grad_norm", "grad_mean", "grad_var", "grad_l1_norm"],
        show_plot=False
    )

    plotter = GradientPlotter(
        entity=config.entity, 
        project=config.project, 
        run_ids=config.run_ids
    )
    
    # Plot 3D gradients (Code vs Math over Steps)
    plotter.plot_3d_gradients(
        metrics=config.metrics,
        x_dataset="code",
        y_dataset="math",
        output_dir=config.output_dir,
        show_plot=config.show_plot
    )
    
    # plot_comparative_evolution is commented out in grad_plotter.py as requested
    # plotter.plot_comparative_evolution(...)

if __name__ == "__main__":
    main()
