# Paddle Continuous Evaluation Baselines

## Howtos


### Add New Evaluation Task

Reference [mnist task](https://github.com/PaddlePaddle/paddle-ce-latest-kpis/tree/develop/models/mnist), 
the following files are required by CE framework:

- `run.xsh` , a script to start this evaluation execution
  - this script can be any bash script, just place `#!/bin/bash` or 
  `#/bin/xonsh` to the head if it is written in the `bash` or `xonsh` language
- `continuous_evaluation.py` to include all the `KPI`s this task tracks
- `latest_kpis` directory, include all the baseline files

### PR and Add to Service
- PR to `fast` branch, and run `ce-kpi-fast-test` test on teamcity,
- if passed, PR from `fast` to `master` branch.

### Add new KPI to track
Reference the interface [kpi.py](https://github.com/PaddlePaddle/continuous_evaluation/blob/develop/continuous_evaluation_py23/kpi.py), there are two basic KPIs:

- LessWorseKpi
- GreaterWorseKpi
