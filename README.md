#  Transformer Captioning & Self-Supervised Learning



## What This Project Does

This project covers two major deep learning systems built and validated end-to-end:

1. **Transformer-based Image Captioning** — implement a full Transformer decoder that attends over visual features to generate natural-language captions from images.
2. **Self-Supervised Representation Learning (SimCLR)** — implement contrastive pretraining from scratch, then evaluate learned representations on downstream image classification.

Both pipelines are trained and evaluated with **CUDA-accelerated GPU execution**, which is required for the large-batch SimCLR pretraining and the hundred-epoch Transformer captioning training loops.

---

## Results

| Task | Metric | Value |
|---|---|---|
| Transformer Captioning | Final training loss (100 epochs) | **0.0224** (↓ from ~5.0) |
| ViT Sanity Check | Top-1 accuracy (single batch, 100 epochs) | **100%** |
| SimCLR Pretraining | Model scale | 24.62M params, 1.31G FLOPs |
| SimCLR Linear Eval | kNN Acc@1 (CIFAR-10 test) | **83.31%** |
| SimCLR Linear Eval | kNN Acc@5 (CIFAR-10 test) | **99.36%** |
| Contrastive Loss | Numerical error (naive & vectorized) | < 5.66e-08 |

---

## Project Structure

```

├── Transformer_Captioning.ipynb   # Transformer decoder + ViT training
├── Self_Supervised_Learning.ipynb # SimCLR pretraining + linear eval
├── transformer_layers.py      # Multi-head attention, positional encoding
├── classifiers/transformer.py # Full Transformer captioning model
├── contrastive_loss.py    # NT-Xent loss (naive + vectorized)
├── model.py               # SimCLR encoder + projection head
└── utils.py               # Training and kNN evaluation loops
```

---

## Part 1 — Transformer Image Captioning

**What was implemented:**
- Multi-head self-attention and cross-attention layers
- Positional encoding for sequence-aware caption generation
- Full autoregressive Transformer decoder that attends over CNN/VGG image features
- Captioning solver with ADAM optimization over 100 epochs on COCO captions

**Key result:**  
Training loss dropped from ~5.0 → **0.0224** in 100 epochs, well below the required threshold of 0.05. Caption samples at test time generate coherent natural-language descriptions.

---

## Part 2 — Vision Transformer (ViT) Sanity Check

**What was implemented:**
- Patch-based image tokenization and linear projection
- ViT classification head with class token
- CIFAR-10 single-batch overfitting as a correctness check

**Key result:**  
Top-1 accuracy climbed from 12.5% (random) → **100%** by epoch 80, confirming correct forward/backward pass implementation.

---

## Part 3 — Self-Supervised Learning with SimCLR

**What was implemented:**
- **NT-Xent contrastive loss** in both naive (loop-based) and vectorized (matrix) forms — validated to numerical precision with maximum relative error < 5.66e-08
- SimCLR data augmentation pipeline (random crop, color jitter, grayscale, Gaussian blur)
- ResNet-based encoder (`f`) + 2-layer MLP projection head (`g`)
- Full CUDA-accelerated pretraining loop (24.62M params, 1.31G FLOPs/forward)
- kNN-based linear evaluation on CIFAR-10 test set

**Key result:**  
After 1 epoch of pretraining on CIFAR-10 using a pretrained SimCLR backbone, kNN evaluation achieves **83.31% Acc@1** and **99.36% Acc@5** — demonstrating high-quality visual representations learned without any labels.

---


> **GPU is required** for the SimCLR pretraining cell. The captioning notebook can run on CPU but will be significantly slower.

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` + `torchvision` | Model training (CUDA-enabled) |
| `numpy` | Numerical ops |
| `matplotlib` | Visualization |
| `thop` | FLOPs/param profiling |
| `tqdm` | Training progress bars |

Full dependency list: [`requirements.cuda.txt`](requirements.cuda.txt) (CUDA) or [`requirements.txt`](requirements.txt) (CPU/MPS).

---

## One-Line Summary (Resume Format)

> **Transformer Image Captioning & Self-Supervised Learning (SimCLR, CUDA)**  
> Built and validated a CUDA-accelerated PyTorch pipeline implementing Transformer captioning (final loss 0.0224 over 100 epochs) and SimCLR contrastive pretraining (24.62M params, 1.31G FLOPs), achieving 83.31% top-1 and 99.36% top-5 accuracy on CIFAR-10 kNN linear evaluation.
