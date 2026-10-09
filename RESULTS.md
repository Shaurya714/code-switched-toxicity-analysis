# Final Experimental Results

## Task

Binary toxicity classification for code-switched Hinglish and English social-media-style text.

## Final Dataset

- **Total examples:** 2,375
- **Unique examples:** 2,375
- **Non-toxic:** 1,110
- **Toxic:** 1,265

## Data Split

- **Training:** 1,662
- **Validation:** 356
- **Test:** 357
- **Split method:** Stratified
- **Random state:** 42

## Model

- **Base model:** `shae2977/xlm-roberta-hinglish-sentiment-analysis`
- **Architecture:** XLM-RoBERTa
- **Fine-tuning:** LoRA
- **Classification:** Binary

## LoRA Configuration

- **Rank:** 8
- **Alpha:** 16
- **Dropout:** 0.1
- **Bias:** none
- **Target modules:** `query`, `value`
- **Modules saved:** `classifier`

## Training Configuration

- **Epochs:** 5
- **Training batch size:** 2
- **Evaluation batch size:** 2
- **Gradient accumulation:** 4
- **Effective batch size:** 8
- **Learning rate:** 2e-4
- **Weight decay:** 0.01
- **Maximum sequence length:** 128
- **FP16:** Enabled with CUDA
- **Best model metric:** Weighted F1

## Hardware

- **GPU:** NVIDIA RTX 3050
- **VRAM:** 6 GB
- **CUDA:** Enabled

## Final Test Results

| Metric | Score |
|---|---:|
| Accuracy | 96.92% |
| Weighted Precision | 96.96% |
| Weighted Recall | 96.92% |
| Weighted F1 | 96.92% |
| Macro Precision | 96.86% |
| Macro Recall | 97.00% |
| Macro F1 | 96.91% |

## Per-Class Results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Non-toxic | 95.35% | 98.20% | 96.76% | 167 |
| Toxic | 98.38% | 95.79% | 97.07% | 190 |

## Confusion Matrix

| Actual / Predicted | Non-toxic | Toxic |
|---|---:|---:|
| Non-toxic | 164 | 3 |
| Toxic | 8 | 182 |

- **Total correct predictions:** **346 / 357**
- **Total incorrect predictions:** **11 / 357**

## Final Status

The final prototype successfully performs binary toxicity classification on Hinglish-focused text using a LoRA-adapted XLM-RoBERTa model.

The dataset and final model checkpoint should be treated as frozen for the reported experiment.

## Important Limitation

The dataset is curated/synthetic and is not a large naturally collected social-media benchmark. Therefore, the reported performance demonstrates prototype effectiveness on the constructed evaluation set and should not be interpreted as equivalent to performance on an independent real-world benchmark.
