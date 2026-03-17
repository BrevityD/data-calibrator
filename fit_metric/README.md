# fit_metric

## Overview

The `fit_metric` module implements a Riemannian metric fitting pipeline over a 2D space defined by (math_loss, code_loss). The goal is to:

1. Process training data and fit a Riemannian metric tensor G(x, y) that captures the geometry of the loss landscape
2. Compute geodesic paths in this metric space
3. Use geodesics to guide model training and interpolation

## Workflow

```
data_process.py
    ↓
(train v1.pth / v2.pth externally)
    ↓
draw_geo.py
    ↓
draw_geo_orthogonal.py / variational_geodesic.py
    ↓
train_with_geo.py (TBD)
```

## File Index

| Script | Description |
|--------|-------------|
| `data_process.py` | [Data processing and preparation](description/data_process.md) |
| `draw_geo.py` | [Metric visualization and geodesic computation](description/draw_geo.md) |
| `draw_geo_orthogonal.py` | [Orthogonal geodesic analysis](description/draw_geo_orthogonal.md) |
| `variational_geodesic.py` | [Variational geodesic computation](description/variational_geodesic.md) |
| `train_with_geo.py` | [Training with geodesic guidance](description/train_with_geo.md) |

## Output Convention

All output files (`.png`, `.json`, `.log`) from scripts in this directory are saved to `result/<script_name>/` and are ignored by git:

- `data_process.py` → `result/data_process/`
- `draw_geo.py` → `result/draw_geo/`
- `draw_geo_orthogonal.py` → `result/draw_geo_orthogonal/`
- `variational_geodesic.py` → `result/variational_geodesic/`
- `train_with_geo.py` → `result/train_with_geo/`

The entire `result/` directory is ignored by git and should not be committed.
