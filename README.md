# Robust CIFAR-10 Classification with 500 Training Images

Train a small Inception-style CNN **from scratch** on only 500 CIFAR-10 images (50 per class) and test it on a **shifted** version of the CIFAR-10 test set (rotation, brightness/contrast change, occlusion and a hidden corruption).

## Results (mean ± std over 3 seeds)

| Configuration | Val Acc (%) | Clean Test (%) | Shifted Test (%) |
|---|---|---|---|
| No augmentation | 39.50 ± 0.41 | 48.25 ± 1.06 | 14.52 ± 0.72 |
| Geometric only | 38.67 ± 1.89 | 43.60 ± 1.85 | 16.66 ± 1.14 |
| Photometric only | 42.00 ± 2.68 | 43.25 ± 1.86 | 19.78 ± 1.40 |
| Geometric + photometric | 41.17 ± 0.24 | 44.26 ± 0.90 | 15.28 ± 0.73 |
| + Cutout / Random Erasing | 47.67 ± 0.85 | 43.25 ± 0.41 | 26.94 ± 0.63 |
| + Mixup/CutMix + label smoothing | 46.17 ± 0.24 | 41.86 ± 1.21 | 23.85 ± 0.65 |
| **Final model (= + Cutout, best validation)** | **47.67 ± 0.85** | **43.25 ± 0.41** | **26.94 ± 0.63** |

- **Most helpful change:** Cutout / Random Erasing (+11.7 points on shifted test). It matches the random occlusion in the shifted test set.
- **Mixup/CutMix did not help.** The small model and 60 epochs lead to underfitting with too much regularization.
- **Val Acc** is the average of the accuracy on the 100 clean validation images and on a corrupted copy of the same 100 images.

## Method

| Item | Setting |
|---|---|
| Model | Inception-style CNN: 2 conv layers, 2 Inception modules (1×1, 3×3, 5×5 and pooling branches with 1×1 reductions), global average pooling, dropout 0.3 |
| Parameters | 251,050 trainable (limit: 1,000,000) |
| Pretrained weights | None (trained from scratch) |
| Optimizer | AdamW, lr 2e-3, weight decay 5e-4 |
| Schedule | Cosine annealing, 60 epochs, batch size 16 |
| Model selection | Best validation epoch (early stopping) |
| Augmentation | On the fly, training data only: crop with padding, horizontal flip, rotation ±25°, color jitter, random grayscale, channel shuffle, blur, noise, Cutout, Mixup/CutMix (ablation only) |

## Data and seeds

- **Dataset:** CIFAR-10 from `torchvision` (downloaded automatically, no manual download needed).
- **train_small:** 500 images (50 per class), split into **400 train / 100 validation** (40 + 10 per class).
- **clean_test:** original 10,000 CIFAR-10 test images (reporting only).
- **shifted_test:** 10,000 test images with rotation (±25°), brightness and contrast change, a 16×16 occlusion (25% of the image) and a hidden corruption.
- **Seeds:** data split = `42`, shifted test = `123`, training = `0, 1, 2`.
- The test sets are used only for the final reporting. All choices are made on the validation split.

## How to reproduce

1. Open the notebook `DL_ASS01_XXF_YYYY.ipynb` in [Google Colab](https://colab.research.google.com) (File → Upload notebook).
2. Select **Runtime → Change runtime type → T4 GPU → Save**.
3. Select **Runtime → Run all**. The full run (6 configurations × 3 seeds) takes about 30 to 60 minutes.
4. The notebook prints the baseline results, the ablation table, the confusion matrix, misclassified images, the hidden-corruption test and the TTA result.

**Requirements** (already installed on Colab): `torch`, `torchvision`, `scikit-learn`, `pandas`, `matplotlib`, `numpy`.

## Notebook structure

| Cell | Content |
|---|---|
| 1 | Imports, seeds, constants |
| 2 | Data loading and the 400/100 split |
| 3 | Shifted test set, hidden-corruption test set, corrupted validation copy |
| 4 | Inception-style model and parameter count check |
| 5 | Augmentations, training and evaluation functions |
| 6 | Ablation runs (6 configurations × 3 seeds) |
| 7 | Part A: baseline results and learning curves |
| 8 | Part D: ablation table and final model selection |
| 9 | Part E: per-class accuracy, confusion matrix, misclassified images, hidden-corruption test |
| 10 | Part F: test-time augmentation (bonus) |

## Notes

- The hidden corruption in the notebook is implemented as random grayscale or channel swap (50/50), because the official implementation was not part of the provided data.
- GPU operations can differ slightly between hardware, so numbers may vary by a small amount on a different GPU even with the same seeds.

**Author:** Zainab Noor
