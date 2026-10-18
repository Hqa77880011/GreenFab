# GreenFab

Implementation of **GreenFab: Calibrated Surrogate Learning and Preference-Guided Robust Optimization for Semiconductor Recipe Allocation**.

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

`x[N,D]` contains pre-decision recipe/context descriptors and `y[N,4]` yield/energy/water/time outcomes in declared normalized units. For robust allocation execute, supply scenario objectives `[candidate,scenario,objective]`, scenario probabilities and transport distances, Wasserstein radius, candidate feasibility, and optional pairwise preferences. Convert yield to a minimization objective before optimization.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent source units; related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

`demo` generates synthetic inputs for this tensor interface; application-specific results require acquired data.

```bash
greenfab demo --output data/example.npz --count 96 --config config.json --seed 13
greenfab prepare --input data/example.npz --output data/prepared
```

## Training and inference

```bash
greenfab train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16 --seed 13
```

For finite allocation without model fitting, use `greenfab execute --input example.json --output runs/allocation --variant full`. Learned allocation uses `calibrate`, then `export` from an actual candidate/scenario NPZ with `x[A,S,8], probabilities[S], distance[S,S], radius, feasible[A]` and optional preference pairs; the exported JSON feeds `execute`.

## Evaluation

```bash
greenfab evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
greenfab plot --input runs --output runs/figures
```

## Ablations

Supported controls: `baseline`, `no_robustness`, `no_preferences`. Keep the input groups, seed and budget fixed.

```bash
greenfab execute --input example.json --output runs/no_robustness --variant no_robustness
```

## Outputs

`data/prepared/` contains train/validation/calibration/test NPZs. Training writes `model.pt`, `history.json` and `training.json` with run arguments. Evaluation writes `predictions.npz` and `metrics.json`; plotting reads saved metric files.

Keep `config.json` with the run; it contains model/data settings and the seed where the CLI does not expose one. Use separate output directories for different seeds/conditions.

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
