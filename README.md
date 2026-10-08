# Code-Switched Sentiment & Toxicity Analysis in Regional Social Media using Fine-Tuned Small LLMs (Phi-3 / Llama-3-8B) and LoRA

## Overview

This project explores toxicity and sentiment analysis for code-switched regional social-media text, with a primary focus on **Hinglish** — the natural combination of Hindi and English commonly used in Roman script on digital platforms.

Code-switched social-media text presents challenges for conventional NLP systems because users frequently combine languages, informal spellings, slang, abbreviations, and conversational expressions within the same sentence.

The project investigates **parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA)** for adapting pretrained language models to regional code-switched text.

The current implemented prototype focuses on **binary Hinglish toxicity detection**:

- **Non-toxic**
- **Toxic**

The implemented prototype uses a **Hinglish-adapted XLM-RoBERTa checkpoint with LoRA**. The broader project direction includes experimentation with small language models such as Phi-3 and Llama-3-8B.

---

## Objectives

The main objectives are:

1. Detect toxic content in code-switched Hinglish text.
2. Handle Roman-script Hindi-English expressions.
3. Investigate parameter-efficient fine-tuning using LoRA.
4. Improve recognition of regional vocabulary, informal expressions, and social-media language.
5. Evaluate the model using standard classification metrics.
6. Provide a lightweight terminal-based inference system.

---

## Current Implementation

### Task

Binary toxicity classification:

| Label | Meaning |
|---|---|
| 0 | Non-toxic |
| 1 | Toxic |

### Base Model

The implemented model is:

**`shae2977/xlm-roberta-hinglish-sentiment-analysis`**

This checkpoint is based on **XLM-RoBERTa** and has been adapted for Hinglish text.

The original checkpoint uses a three-class sentiment classification head. For this project, the classification head was replaced with a two-class toxicity classification head.

### Fine-Tuning Method

The model is adapted using **LoRA (Low-Rank Adaptation)**.

Instead of updating the complete pretrained model, LoRA introduces a small number of trainable parameters into selected transformer layers.

#### LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (`r`) | 8 |
| Alpha | 16 |
| Dropout | 0.1 |
| Bias | None |
| Task Type | Sequence Classification |
| Target Modules | `query`, `value` |
| Saved Module | `classifier` |

---

## Dataset

The current prototype uses a curated Hinglish toxicity dataset containing:

- **2,375 unique text samples**
- **1,110 non-toxic samples**
- **1,265 toxic samples**

The dataset is divided into:

| Split | Samples |
|---|---:|
| Training | 1,662 |
| Validation | 356 |
| Test | 357 |

The dataset is used specifically for the binary toxicity classification task.

> **Research limitation:** The current dataset is curated/synthetic rather than a large naturally collected social-media benchmark. Therefore, the reported results should be interpreted as prototype results rather than evidence of general real-world performance.

---

## Training

Training uses Hugging Face Transformers, PEFT, and PyTorch.

### Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 5 |
| Training Batch Size | 2 |
| Evaluation Batch Size | 2 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 8 |
| Learning Rate | `2e-4` |
| Weight Decay | `0.01` |
| Maximum Sequence Length | 128 |
| Evaluation Strategy | Every Epoch |
| Best Model Metric | Weighted F1 |
| Mixed Precision | FP16 when CUDA is available |

The model was trained using an NVIDIA RTX 3050 GPU.

---

## Results

Final held-out test performance:

| Metric | Score |
|---|---:|
| Accuracy | **96.92%** |
| Weighted Precision | **96.96%** |
| Weighted Recall | **96.92%** |
| Weighted F1 | **96.92%** |
| Macro Precision | **96.86%** |
| Macro Recall | **97.00%** |
| Macro F1 | **96.91%** |

### Test Set

- Total test samples: **357**
- Correct predictions: **346**
- Incorrect predictions: **11**

### Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Non-toxic | 95.35% | 98.20% | 96.76% |
| Toxic | 98.38% | 95.79% | 97.07% |

Detailed evaluation results are available in [`RESULTS.md`](RESULTS.md).

---

## Project Structure


code-switched-toxicity-analysis/
│
├── DATASET/
│   ├── toxicity_binary_dataset.csv
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── toxicity_lora_output/
│   └── final/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_args.bin
│
├── analyzer.py
├── evaluate_model.py
├── manual_test.py
├── prepare_data.py
├── train_model.py
├── requirements.txt
├── RESULTS.md
├── README.md
└── .gitignore


## Installation
Clone the repository and enter the project directory:

git clone https://github.com/Shaurya714/code-switched-toxicity-analysis.git
cd code-switched-toxicity-analysis
Create and activate a virtual environment:
Windows
python -m venv .venv
.venv\Scripts\activate

Install the required packages:
pip install -r requirements.txt

Running the Analyzer
The saved LoRA model can be used directly for terminal-based inference.
python analyzer.py
example -
<img width="467" height="186" alt="image" src="https://github.com/user-attachments/assets/34db1818-81f7-49ea-a828-5adcf5fa4c53" />

Type exit to stop the analyzer.
Running Evaluation
To evaluate the saved model on the held-out test set:
python evaluate_model.py

This reports:
- Accuracy
- Precision
- Recall
- F1-score
- Per-class metrics
- Confusion matrix
Manual Testing
For interactive testing of individual inputs:
python manual_test.py

The script displays the predicted class and probability distribution.
Dataset Preparation
The dataset preparation script performs:
- Dataset loading
- Duplicate checking
- Label conversion
- Stratified train/validation/test splitting
- CSV generation
Run:
python prepare_data.py

Training the Model
To reproduce the LoRA fine-tuning process:
python train_model.py

The trained adapter is saved under:
toxicity_lora_output/final/

Why LoRA?
Full fine-tuning requires updating the parameters of the entire pretrained model.
LoRA instead introduces trainable low-rank matrices into selected transformer layers while keeping most of the pretrained model frozen.
This provides:
- Lower memory requirements
- Faster fine-tuning
- Smaller trainable parameter count
- Efficient experimentation on consumer GPUs
- Easy storage and deployment of the trained adapter
Limitations
The current prototype has several limitations:
1. The dataset is curated/synthetic and relatively small.
2. The model has primarily been evaluated on Hinglish text.
3. Real-world social-media language is substantially more diverse.
4. Sarcasm, implicit toxicity, context-dependent toxicity, and multilingual code-switching require further investigation.
5. High test performance on a curated dataset does not guarantee equivalent performance on unseen real-world data.
6. The current implementation uses XLM-RoBERTa rather than Phi-3 or Llama-3-8B.
Future Work
Planned extensions include:
- Evaluation on larger naturally collected datasets.
- Expansion to additional Indian regional languages.
- Comparison with Phi-3 and Llama-3-8B based approaches.
- More extensive robustness and out-of-distribution testing.
- Context-aware toxicity detection.
- Investigation of sarcasm and implicit toxicity.
- Deployment as an API or lightweight web application.
Research Direction
The broader research direction is to investigate how parameter-efficient fine-tuning of small language models can improve NLP systems for regional and code-switched social-media language.
The current Hinglish toxicity classifier serves as the implemented prototype for this direction.
Status
Current status: Working prototype
The repository contains the dataset, preprocessing pipeline, LoRA training pipeline, trained adapter, evaluation pipeline, inference scripts, and documentation required to reproduce and demonstrate the current prototype.
