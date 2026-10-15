# OCFW

Implementation of **Object-centric forecast residuals for manipulation failure warning.**

## Install

Use Python 3.11 in an isolated environment, then run from the repository root:

```bash
python -m pip install -e .
```

## Data

Export [RLBench](https://github.com/stepjam/RLBench) episodes as NPZ with `rgb: uint8 (T,H,W,3)`, `command: (T,U)` and annotation `masks: (T,H,W)`. Background is 0; object identities are 1–6. The first slots are gripper, manipulated object and target. A CSV manifest contains `episode,split,task,failure,terminal,cause`; terminal is a frame index and cause is a zero-based object ID. Paths are relative to the manifest. Predictor training requires successful episodes only. Final-test masks/cause labels are read by evaluation, not the RGB monitor.

## Workflow

```bash
python -m ocfr prepare --input data/episodes.csv
python -m ocfr train
python -m ocfr calibrate
python -m ocfr run
python -m ocfr evaluate
python -m ocfr plot
```

Main commands accept `--config config.json` and `--output runs/default`. Keep each seed/condition in its own output directory. The resolved configuration is written to `settings.json`.

## Configuration

`nodes,size` set object capacity and perception/render resolution. `horizon` sets the future archive; `render_horizon` chooses its auxiliary pixel target. `identity_penalty` augments geometric/appearance/IoU Hungarian matching. `relation_weight,progress_bins,ewma` control residual scoring. `target_horizon,normal_penalty` define validation risk targets, while `persistence,fps` define warning timing. Training epochs, batch size, learning rate and seed use the defaults shown in `config.json` where applicable.

## Outputs

`perception.pt` and `dynamics.pt` contain segmentation and graph forecasting. `calibration.npz` stores phase statistics and the validation risk mapping. `predictions.csv` contains causal risk, warnings, attribution and host costs; `pixels_*.npz` supplies the background-dominance diagnostic. `metrics.csv` contains event AUROC/F1, median lead, FA/min, attribution, ECE, soft-target Brier and runtime.

## Implementation and controls

A compact U-Net learns masks on training episodes. Test masks are predicted from RGB. Thirty-nine-dimensional tracks encode position, box, velocity, appearance and confidence. Graph/GRU forecasts use the current executed command only; state/relation scores compare observations with archived 1/4/8/12-frame forecasts. Rendering is an auxiliary training signal. A validation-only logistic calibrator and persistent threshold produce warnings.

Controls: `full`, `no-action`, `no-relation`, `no-phase`, `no-variance`, `single-horizon`. Choose `--variant` and use the same preparation and fitting choices in each separate output directory.

```bash
python -m ocfr.compare --inputs "runs/*/metrics.csv" --reference full --metrics event_f1 median_lead_seconds object_attribution_accuracy --unit seed
```

The comparison command averages repeated rows within the specified independent unit, bootstraps units and applies paired Wilcoxon/Holm comparisons per metric.

