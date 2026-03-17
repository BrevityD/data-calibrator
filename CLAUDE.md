# Project Context 

The current directory is a Git worktree for a branch.

# Setup 

Use `uv` to manage the environment.

```bash
source /public/home/jza/data_calibrate/data_mixture/.venv/bin/activate
```

# Standards

- Create a commit after every change. Any result file or model weights file should not be added.(e.g. *json,.pth,.log)
- Only push after confirmation from the user.

## Output File Organization

All output files (`.png`, `.json`, `.log`) from `fit_metric/*.py` scripts must be saved to `fit_metric/result/<script_name>/`:

- `draw_geo.py` → `fit_metric/result/draw_geo/`
- `draw_geo_orthogonal.py` → `fit_metric/result/draw_geo_orthogonal/`
- `variational_geodesic.py` → `fit_metric/result/variational_geodesic/`
- `train_with_geo.py` → `fit_metric/result/train_with_geo/`

The entire `fit_metric/result/` directory is ignored by git and should not be committed.