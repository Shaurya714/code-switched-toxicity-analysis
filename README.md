# Code-Switched Sentiment \& Toxicity Analysis in Regional Social Media using Fine-Tuned Small LLMs (Phi-3 / Llama-3-8B) and LoRA

## Overview

This project explores toxicity and sentiment analysis for code-switched regional social-media text, with a primary focus on \*\*Hinglish\*\* — the natural combination of Hindi and English commonly used in Roman script on digital platforms.



Code-switched social-media text presents challenges for conventional NLP systems because users frequently combine languages, informal spellings, slang, abbreviations, and conversational expressions within the same sentence.



The project investigates parameter-efficient fine-tuning using \*\*Low-Rank Adaptation (LoRA)\*\* for adapting pretrained language models to regional code-switched text.



The current implemented prototype focuses on \*\*binary Hinglish toxicity detection\*\*:

- non-toxic
- toxic
The implemented prototype uses a \*\*Hinglish-adapted XLM-RoBERTa checkpoint with LoRA\*\*. The broader project direction includes experimentation with small language models such as Phi-3 and Llama-3-8B.


## Objectives

The main objectives are:

1. Detect toxic content in code-switched Hinglish text.

2. Handle Roman-script Hindi-English expressions.

3. Investigate parameter-efficient fine-tuning using LoRA.

4. Improve recognition of regional vocabulary, informal expressions, and social-media language.

5. Evaluate the model using standard classification metrics.

6. Provide a lightweight terminal-based inference system.

## Current Implementation

### Task

Binary toxicity classification:

| Label | Meaning |

|---|---|

|   0   | Non-toxic |

|   1   | Toxic |



### Base Model

The implemented model is:
"shae2977/xlm-roberta-hinglish-sentiment-analysis
This checkpoint is based on \*\*XLM-RoBERTa\*\* and has been adapted for Hinglish text.
The original checkpoint uses a three-class sentiment classification head. For this project, the classification head was replaced with a two-class toxicity classification head.


### Fine-Tuning Method
The model is adapted using \*\*LoRA (Low-Rank Adaptation)\*\*.
Instead of updating the complete pretrained model, LoRA introduces a small number of trainable parameters into selected transformer layers.



Configuration:
text

LoRA rank (r):             8

LoRA alpha:                16

LoRA dropout:              0.1

Bias:                      none

Task type:                 sequence classification

Target modules:            query, value

Saved module:              classifier

