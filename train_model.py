import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODEL_NAME = (
    "shae2977/"
    "xlm-roberta-hinglish-sentiment-analysis"
)

TRAIN_FILE = "DATASET/train.csv"
VALIDATION_FILE = "DATASET/validation.csv"

OUTPUT_DIR = "toxicity_lora_output"

NUM_LABELS = 2

LABELS = [
    "non_toxic",
    "toxic"
]


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print(
    f"Training examples:   {len(train_df)}"
)

print(
    f"Validation examples: {len(validation_df)}"
)


# ------------------------------------------------------------
# Convert to Hugging Face datasets
# ------------------------------------------------------------

train_dataset = Dataset.from_pandas(
    train_df[
        [
            "tweet",
            "label"
        ]
    ],
    preserve_index=False
)

validation_dataset = Dataset.from_pandas(
    validation_df[
        [
            "tweet",
            "label"
        ]
    ],
    preserve_index=False
)


# ------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_function(examples):

    return tokenizer(
        examples["tweet"],
        truncation=True,
        max_length=128
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True
)


# ------------------------------------------------------------
# Load base model
# ------------------------------------------------------------

print("\nLoading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True
)


# ------------------------------------------------------------
# Binary labels
# ------------------------------------------------------------

model.config.id2label = {
    0: "non_toxic",
    1: "toxic"
}

model.config.label2id = {
    "non_toxic": 0,
    "toxic": 1
}


# ------------------------------------------------------------
# LoRA configuration
# ------------------------------------------------------------

print("\nApplying LoRA...")

lora_config = LoraConfig(

    r=8,

    lora_alpha=16,

    lora_dropout=0.1,

    bias="none",

    task_type=TaskType.SEQ_CLS,

    target_modules=[
        "query",
        "value"
    ],

    modules_to_save=[
        "classifier"
    ]
)


model = get_peft_model(
    model,
    lora_config
)


model.print_trainable_parameters()


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    return {

        "accuracy": accuracy,

        "weighted_precision": precision,

        "weighted_recall": recall,

        "weighted_f1": f1

    }


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=5,

    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    weight_decay=0.01,

    fp16=torch.cuda.is_available(),

    logging_steps=20,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="weighted_f1",

    greater_is_better=True,

    save_total_limit=2,

    report_to="none"
)


# ------------------------------------------------------------
# Data collator
# ------------------------------------------------------------

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STARTING BINARY TOXICITY TRAINING")
print("=" * 60)

trainer.train()


# ------------------------------------------------------------
# Save final model
# ------------------------------------------------------------

print("\nSaving model...")

final_dir = os.path.join(
    OUTPUT_DIR,
    "final"
)

trainer.save_model(
    final_dir
)

tokenizer.save_pretrained(
    final_dir
)


print("\n" + "=" * 60)
print("BINARY TRAINING COMPLETE")
print("=" * 60)

print(
    f"Model saved to: {final_dir}"
)