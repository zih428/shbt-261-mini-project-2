# Semantic Segmentation with Pascal VOC 2007

## Abstract
- State the problem, models compared, and top-line findings in 4-6 sentences.

## 1. Problem Framing
- Semantic segmentation objective
- Why Pascal VOC 2007 is a useful benchmark
- Why the study goes beyond the assignment minimum

## 2. Dataset and Experimental Protocol
- Pascal VOC 2007 splits
- Internal train/dev construction from official train
- Locked use of official VOC val as the final test set
- Preprocessing, augmentations, ignore-label handling, and reproducibility controls

## 3. Model Architectures
- U-Net with ResNet encoder
- DeepLabV3+
- SegFormer-B2
- SAM2 semantic adapter

## 4. Training Setup
- Optimizer, schedule, batch sizes, mixed precision
- Loss definitions
- Hardware and runtime details

## 5. Main Results
- Main comparison table
- Runtime-vs-accuracy tradeoff
- Per-class IoU figure

## 6. Ablation Studies
- Backbone size
- Loss design
- Augmentation policy
- SAM2 tuning strategy

## 7. Generalization and Failure Analysis
- Person-present subset
- Small-object subset
- Crowded-scene subset
- High-boundary-complexity subset
- Best/worst qualitative panels

## 8. Lessons Learned and Limitations
- What actually worked
- What underperformed and why
- Remaining risks / limitations

## 9. Conclusion
- One paragraph with the main takeaways
