# MixTailor

Features below are included in this framework:

* Heterogenous Data Set
* Gradient Sampling
* Robust Aggregation Mechanisms

## Generating Experiments

```
python3 -m dbqf.wg.grid_run --grid mnist --run_name mnist --cluster slurm --cluster_args 2,5,p100,t4 --prefix mnh
```
