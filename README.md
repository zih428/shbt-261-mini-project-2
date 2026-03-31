# Pascal VOC 2007 Semantic Segmentation Stack

This repository implements a configuration-driven experiment pipeline for the Pascal VOC 2007 semantic segmentation mini-project. It is structured to support:

- internal train/dev splitting with multilabel stratification
- rigorous metric computation, including `mIoU`, `Dice`, `Pixel Accuracy`, and `HD95`
- multiple model families:
  - `U-Net`
  - `DeepLabV3+`
  - `SegFormer`
  - `Mask2Former`
  - `SAM2` semantic adaptation
- ablation studies, qualitative analysis, and aggregated reporting assets

## Install

```bash
uv sync --group dev
uv sync --group dev --extra models
uv sync --group dev --extra models --extra sam2
```

`SAM2` support is optional. If you are not running the SAM2 experiment track, you can skip the `--extra sam2` install.

## Model Specs

The repo currently supports five headline model tracks.

### U-Net

- Config: `configs/experiments/unet_resnet34.yaml`
- Local MPS config: `configs/local_runs/unet_resnet34_mps.yaml`
- Family: `U-Net`
- Encoder: `ResNet-34`
- Pretraining: `ImageNet-1K V1`
- Decoder: custom 4-stage upsampling decoder with skip connections
- Output classes: `21` (`20` VOC classes + `background`)
- Optimizer default: `AdamW`, `lr=3e-4`

### DeepLabV3+

- Config: `configs/experiments/deeplabv3plus_resnet50.yaml`
- Local MPS config: `configs/local_runs/deeplabv3plus_resnet50_mps.yaml`
- Family: `DeepLabV3+`
- Encoder: `ResNet-50`
- Pretraining: `ImageNet-1K V2`
- Decoder: ASPP + low-level feature fusion head
- Output classes: `21`
- Optimizer default: `AdamW`, `lr=2e-4`

### SegFormer-B2

- Config: `configs/experiments/segformer_b2.yaml`
- Local MPS config: `configs/local_runs/segformer_b2_mps.yaml`
- Family: `SegFormer`
- Backbone/model id: `nvidia/segformer-b2-finetuned-ade-512-512`
- Pretraining: ADE20K semantic-segmentation checkpoint with the classifier reinitialized to `21` VOC classes
- Decoder: native `SegformerForSemanticSegmentation` decode head
- Output classes: `21`
- Optimizer default: `AdamW`, `lr=2e-4`

### Mask2Former Swin-Tiny

- Config: `configs/experiments/mask2former_swin_tiny.yaml`
- Local MPS config: `configs/local_runs/mask2former_swin_tiny_mps.yaml`
- Family: `Mask2Former`
- Backbone/model id: `facebook/mask2former-swin-tiny-ade-semantic`
- Pretraining: ADE20K semantic-segmentation checkpoint with the classifier reinitialized to `21` VOC classes
- Decoder: Mask2Former pixel decoder + transformer query decoder, aggregated into semantic logits for the common training/evaluation pipeline
- Output classes: `21`
- Optimizer default: `AdamW`, `lr=6e-5`

### SAM2 Hiera-S Semantic Adapter

- Config: `configs/experiments/sam2_hiera_s_frozen.yaml`
- Local MPS config: `configs/local_runs/sam2_hiera_s_frozen_mps.yaml`
- Family: `SAM2` semantic adaptation
- Backbone/model id: `facebook/sam2-hiera-small`
- Pretraining: official SAM2 Hiera-S checkpoint from Hugging Face
- Semantic head: 2-layer convolutional decoder + `1x1` classifier
- Main setting: frozen backbone, train decoder only
- Optional setting: LoRA adapters on matched backbone modules via `configs/experiments/sam2_hiera_s_lora.yaml`
- Output classes: `21`
- Optimizer default: `AdamW`, `lr=1e-4`

## Pretrained Assets

Before running pretrained experiments on this machine, cache the required weights:

```bash
uv run python scripts/fetch_pretrained_assets.py
```

This caches:

- torchvision `ResNet-18`, `ResNet-34`, and `ResNet-50` checkpoints under `~/.cache/torch/hub/checkpoints`
- `SegFormer-B2` weights from Hugging Face
- `Mask2Former Swin-Tiny` weights from Hugging Face
- `SAM2 Hiera-S` weights from Hugging Face

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

To run the local pretrained MPS headline suite with overall progress tracking and suite-level resume:

```bash
uv run python scripts/run_suite.py \
  --suite-name headline_models_mps \
  --configs \
    configs/local_runs/unet_resnet34_mps.yaml \
    configs/local_runs/deeplabv3plus_resnet50_mps.yaml \
    configs/local_runs/segformer_b2_mps.yaml \
    configs/local_runs/mask2former_swin_tiny_mps.yaml \
    configs/local_runs/sam2_hiera_s_frozen_mps.yaml \
  --data-root /path/to/VOCtrainval_06-Nov-2007 \
  --artifact-root ./artifacts
```

The local MPS configs keep pretrained initialization enabled while using MPS-safe batch sizes and `amp: false`.

To inspect saved progress after an interruption:

```bash
uv run python scripts/show_progress.py --path ./artifacts/suites/headline_models_mps
uv run python scripts/show_progress.py --path ./artifacts/runs/unet_resnet34_pretrained_mps
```

3. Evaluate a checkpoint on the locked official VOC validation split:

```bash
uv run python scripts/evaluate.py \
  --config configs/local_runs/unet_resnet34_mps.yaml \
  --checkpoint ./artifacts/runs/unet_resnet34_pretrained_mps/checkpoint_best.pth \
  --data-root /path/to/VOCtrainval_06-Nov-2007 \
  --artifact-root ./artifacts \
  --split official_val
```

4. Aggregate completed runs into comparison tables and figures:

```bash
uv run python scripts/aggregate_results.py \
  --run-dirs \
    ./artifacts/evals/unet_resnet34_pretrained_mps_official_val \
    ./artifacts/evals/deeplabv3plus_resnet50_pretrained_mps_official_val \
    ./artifacts/evals/segformer_b2_pretrained_mps_official_val \
    ./artifacts/evals/mask2former_swin_tiny_pretrained_mps_official_val \
    ./artifacts/evals/sam2_hiera_s_frozen_pretrained_mps_official_val \
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

- `SegFormer` requires `transformers`.
- `Mask2Former` requires `transformers`.
- `SAM2` can load from either an explicit local checkpoint/config pair or a Hugging Face model id.
- The code preserves the Pascal VOC `255` void label as `ignore_index=255`.
