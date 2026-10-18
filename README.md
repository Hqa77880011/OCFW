# OCFW

Implementation of **Object-centric forecast residuals for manipulation failure warning.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Export [RLBench](https://github.com/stepjam/RLBench) episodes as NPZ with `rgb: uint8 (T,H,W,3)`, `command: (T,U)` and annotation `masks: (T,H,W)`. Background is 0; object identities are 1–6. The first slots are gripper, manipulated object and target. A CSV manifest contains `episode,split,task,failure,terminal,cause`; terminal is a frame index and cause is a zero-based object ID. Paths are relative to the manifest. Use `train`, `val` and `test` roles. Predictor training requires successful episodes only; validation must include successful episodes for phase calibration. Final-test masks/cause labels are read by evaluation, not the RGB monitor.

```bash
python -m ocfr prepare --input data/episodes.csv --config config.json --output runs/full
```

## Training and inference

```bash
python -m ocfr train --config config.json --output runs/full
python -m ocfr calibrate --config config.json --output runs/full
python -m ocfr run --config config.json --output runs/full
```

`config.json` records the method settings and seed `13`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m ocfr evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `full`, `no-action`, `no-relation`, `no-phase`, `no-variance`, `single-horizon`.

Repeat the same data and configuration in a separate directory, for example:

```bash
python -m ocfr prepare --input data/episodes.csv --config config.json --output runs/no-action --variant no-action
python -m ocfr train --config config.json --output runs/no-action --variant no-action
python -m ocfr calibrate --config config.json --output runs/no-action --variant no-action
python -m ocfr run --config config.json --output runs/no-action --variant no-action
python -m ocfr evaluate --config config.json --output runs/no-action --variant no-action
```

## Outputs

All main outputs are written below `runs/full/`.

`perception.pt` and `dynamics.pt` contain segmentation and graph forecasting. `calibration.npz` stores phase statistics and the validation risk mapping. `predictions.csv` contains causal risk, warnings, attribution and host costs; `pixels_*.npz` supplies the background-dominance diagnostic. `metrics.csv` contains event AUROC/F1, median lead, FA/min, attribution, ECE, soft-target Brier and runtime.

```bash
python -m ocfr plot --config config.json --output runs/full
```
