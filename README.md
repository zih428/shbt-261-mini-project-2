# Pascal VOC 2007 Semantic Segmentation Stack

This repository implements a configuration-driven experiment pipeline for the Pascal VOC 2007 semantic segmentation mini-project. It is structured to support:

- internal train/dev splitting with multilabel stratification
- rigorous metric computation, including `mIoU`, `Dice`, `Pixel Accuracy`, and `HD95`
- multiple model families:
  - `U-Net`
  - `DeepLabV3+`
  - `SegFormer`
  - `SAM2` semantic adaptation
- ablation studies, qualitative analysis, and aggregated reporting assets

## Install

```bash
uv sync --group dev
uv sync --group dev --extra models
uv sync --group dev --extra models --extra sam2
```

`SAM2` support is optional. If you are not running the SAM2 experiment track, you can skip the `--extra sam2` install.

## Expected Dataset Layout

The scripts assume the Pascal VOC 2007 directory layout expected by `torchvision.datasets.VOCSegmentation`, for example:

```text
VOCtrainval_06-Nov-2007/
  VOCdevkit/
    VOC2007/
      JPEGImages/
      SegmentationClass/
      ImageSets/
```

## Workflow

1. Prepare metadata and the internal split:

```bash
uv run python scripts/prepare_voc.py \
  --data-root /path/to/VOCtrainval_06-Nov-2007 \
  --output-dir ./artifacts/dataset
```

2. Train a model on the internal train/dev split:

```bash
uv run python scripts/train.py \
  --config configs/experiments/unet_resnet34.yaml \
  --data-root /path/to/VOCtrainval_06-Nov-2007 \
  --artifact-root ./artifacts
```

Training writes `progress.json` in the run directory and resumes automatically from `checkpoint_last.pth` unless `--no-resume` is passed.

To run multiple models with overall progress tracking and suite-level resume:

```bash
uv run python scripts/run_suite.py \
  --suite-name headline_models \
  --configs \
    configs/experiments/unet_resnet34.yaml \
    configs/experiments/deeplabv3plus_resnet50.yaml \
    configs/experiments/segformer_b2.yaml \
    configs/experiments/sam2_hiera_s_frozen.yaml \
  --data-root /path/to/VOCtrainval_06-Nov-2007 \
  --artifact-root ./artifacts
```

To inspect saved progress after an interruption:

```bash
uv run python scripts/show_progress.py --path ./artifacts/suites/headline_models
uv run python scripts/show_progress.py --path ./artifacts/runs/unet_resnet34
```

3. Evaluate a checkpoint on the locked official VOC validation split:

```bash
uv run python scripts/evaluate.py \
  --config configs/experiments/unet_resnet34.yaml \
  --checkpoint ./artifacts/runs/unet_resnet34/checkpoint_best.pth \
  --data-root /path/to/VOCtrainval_06-Nov-2007 \
  --artifact-root ./artifacts \
  --split official_val
```

4. Aggregate completed runs into comparison tables and figures:

```bash
uv run python scripts/aggregate_results.py \
  --run-dirs \
    ./artifacts/evals/unet_resnet34_official_val \
    ./artifacts/evals/deeplabv3plus_resnet50_official_val \
    ./artifacts/evals/segformer_b2_official_val \
    ./artifacts/evals/sam2_hiera_s_frozen_official_val \
  --output-dir ./artifacts/aggregate
```

## Outputs

### Training Run Outputs

Each training run writes artifacts under:

```text
artifacts/runs/<experiment_name>/
```

Training produces:

- `resolved_config.yaml`
- `checkpoint_last.pth`
- `checkpoint_best.pth`
- `train_log.csv`
- `metrics.json`
- `progress.json`

`metrics.json` contains run-level summary fields such as:

- `training_time_seconds`
- `peak_gpu_memory_gb`
- `total_parameters`
- `trainable_parameters`
- `best_dev_metrics`

### Evaluation Outputs

Each evaluation run writes artifacts under:

```text
artifacts/evals/<experiment_name>_<split>/
```

Evaluation produces:

- `metrics.json`
- `per_class.csv`
- `per_image.csv`
- `subset_metrics.json`
- `resolved_config.yaml`
- qualitative figures
- prediction masks

This directory also includes:

- `per_class_iou.png`
- `best_worst_panel.png`
- `person_panel.png`
- `qualitative/inputs/*.png`
- `qualitative/ground_truth/*.png`
- `qualitative/predictions/*.png`
- `qualitative/triptychs/*.png`

### Suite Progress Outputs

If you use `scripts/run_suite.py`, suite-level progress is written under:

```text
artifacts/suites/<suite_name>/progress.json
```

## Notes

- `SegFormer` requires `timm`.
- `SAM2` requires valid checkpoint/config paths in the experiment config.
- The code preserves the Pascal VOC `255` void label as `ignore_index=255`.
