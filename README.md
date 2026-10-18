# GreenFab

Implementation of **GreenFab: Calibrated Surrogate Learning and Preference-Guided Robust Optimization for Semiconductor Recipe Allocation**.

A multi-output surrogate predicts mean and ordered 5/95 percent outcome bounds. Finite-scenario Wasserstein transport LPs produce worst-case objectives. Preference comparisons cut a simplex polytope, and minimax regret enumerates competitor candidates with a linear program over that polytope. Execute exports the chosen feasible candidate, all robust objectives, regrets, and the complete preference constraints.

## Install

Use Python 3.10 or later. Create and activate a virtual environment, then install:

```bash
python -m pip install -e .
```

## Prepare data

`x[N,D]` contains pre-decision recipe/context descriptors and `y[N,4]` yield/energy/water/time outcomes in declared normalized units. For robust allocation execute, supply scenario objectives `[candidate,scenario,objective]`, scenario probabilities and transport distances, Wasserstein radius, candidate feasibility, and optional pairwise preferences. Convert yield to a minimization objective before optimization.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent subjects, devices, scenes, lots, or source images. Related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

```bash
greenfab demo --output data/example.npz --count 96 --config config.json
greenfab prepare --input data/example.npz --output data/prepared
```

The demo defines a small synthetic problem with the documented tensor interface. Use application data in that schema for the paper's evaluation settings.

## Train and execute

```bash
greenfab train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16
greenfab evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
greenfab plot --input runs --output runs/figures
```

Training uses AdamW and selects the minimum validation objective. `model.pt` stores parameters and configuration; `history.json` records training/validation objectives. Evaluation exports predictions and mechanism tensors to `predictions.npz` and computed scores to `metrics.json`. Plotting exports comparison charts from those actual evaluation files.

`--device cpu` is the default; `--device cuda` selects CUDA. `--lr`, `--epochs`, `--batch-size`, and `--seed` control optimization. Method parameters are in `config.json`; `--set key=value` overrides one value using JSON syntax. Use identical splits and seeds across comparisons.

## Calibration

For interval or classification confidence calibration, use independent groups excluded from fitting and checkpoint selection:

```bash
greenfab calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
```

Evaluate `calibrated.pt` with the same evaluation command. Classification uses temperature scaling; numeric intervals use a 90% group-maximum split-calibration expansion. Insufficient independent units retain infinite intervals. Pretraining checkpoints can initialize supervised training with `train --init path/to/model.pt`.

## Baselines and ablations

Available variants: `full`, `baseline`, `no_robustness`, `no_preferences`. `baseline` removes the paper-specific mechanism while retaining the input and prediction task. Other variants disable their named component.

```bash
greenfab train --input data/prepared --output runs/baseline --config config.json --variant baseline
greenfab evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Metrics and implementation conventions

Classification reports accuracy, balanced accuracy, macro-F1, NLL, confidence ECE, and worst-group F1; binary tasks add AUROC and AUPRC. Unknown labels use -1 and enable unknown AUROC and OSCR. Segmentation reports Dice, IoU, precision, recall, MAE, Brier/ECE, two-pixel boundary F1, symmetric contour distance, and HD95. Reconstruction reports RMSE, MAE, signed bias, unit-range PSNR, and Gaussian-window SSIM. Each method adds its mechanism metrics where applicable.

Parameter counts and forward timing describe the executing process. Use batch size one and a declared device configuration for device latency comparisons. Physical energy and scores from external task evaluators require measured inputs; they are not inferred from parameter counts. All prediction arrays remain available for task-specific evaluation.

The implementation uses the explicit core operations described above with configurable compact networks. Numeric conventions and application-specific information are stated with the method and data schema. The parameter file contains the current defaults.

## Execute the mechanism

```bash
greenfab execute --input example.json --output runs/execution --variant full
```

The example is an explicit analytical/simulated input specification, not an experiment result. Replace it with the schema described above for application data. The execute path writes the actual selected actions, constraints, and mechanism metrics.

## Selected calibrated allocation setting

Use eight input features, four dimensionless minimized outcomes, the included
width-32 surrogate, 90% independent fab-day group calibration, finite scenario
transport and the preference simplex. Normalize outcome units using training
statistics before preparing NPZ files; reverse signs of maximized objectives
before fitting. The chosen robust objective is the calibrated surrogate upper
endpoint, so interval uncertainty actually enters allocation.

```bash
greenfab train --input data/prepared --config config.json --output runs/surrogate --variant full --epochs 100 --seed 13
greenfab calibrate --input data/prepared/calibration.npz --model runs/surrogate/model.pt --output runs/surrogate/calibrated.pt
greenfab export --input data/candidate_scenarios.npz --model runs/surrogate/calibrated.pt --output runs/allocation_problem.json
greenfab execute --input runs/allocation_problem.json --output runs/allocation --variant full
```

Candidate/scenario NPZ contains x[A,S,8], probabilities[S] summing to one,
symmetric nonnegative distance[S,S] with zero diagonal, scalar radius>=0,
physical feasible[A], and optional integer preferences[L,2], candidate_ids and
scenario_ids. The export path executes the frozen calibrated model, preserves
candidate/scenario order and writes the allocation JSON consumed by execute.
Optional group IDs are checked against stored fitting/validation/calibration
roles. The feasible mask is an input physical constraint, never inferred from a
surrogate prediction. Infinite calibration expansion cannot become a finite
allocation objective. At least nine independent calibration groups permit a
finite 90% bound. The generated feature/outcome setting and explicit example.json
remain self-contained analytic controls; real factory inputs/constraints are
needed only for factory-specific claims.
