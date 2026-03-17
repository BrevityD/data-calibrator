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
- Save results in the corresponding directory with an ordered naming scheme.