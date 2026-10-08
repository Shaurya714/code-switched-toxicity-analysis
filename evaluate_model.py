import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from peft import PeftModel

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODEL_PATH = "toxicity_lora_output/final"

TEST_FILE = "DATASET/test.csv"

BASE_MODEL = (
    "shae2977/"
    "xlm-roberta-hinglish-sentiment-analysis"
)

LABEL_NAMES = [
    "non_toxic",
    "toxic"
]


# ------------------------------------------------------------
# Load test data
# ------------------------------------------------------------

print("=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

df = pd.read_csv(
    TEST_FILE
)

print(
    f"Test examples: {len(df)}"
)


# ------------------------------------------------------------
# Load tokenizer
# ------------------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


# ------------------------------------------------------------
# Load base model
# ------------------------------------------------------------

print("Loading model...")

base_model = AutoModelForSequenceClassification.from_pretrained(

    BASE_MODEL,

    num_labels=2,

    ignore_mismatched_sizes=True
)


# ------------------------------------------------------------
# Load LoRA adapter
# ------------------------------------------------------------

model = PeftModel.from_pretrained(
    base_model,
    MODEL_PATH
)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model.to(device)

model.eval()

print(
    f"Device: {device}"
)


# ------------------------------------------------------------
# Prepare labels
# ------------------------------------------------------------

label_map = {

    "non_toxic": 0,

    "toxic": 1
}


df["label"] = df[
    "category"
].map(label_map)


if df["label"].isna().any():

    print(
        "\nERROR: Unknown category found:"
    )

    print(
        df[
            df["label"].isna()
        ]["category"].value_counts()
    )

    raise ValueError(
        "Test dataset contains an unknown category."
    )


texts = (
    df["tweet"]
    .astype(str)
    .tolist()
)

true_labels = (
    df["label"]
    .astype(int)
    .tolist()
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

predictions = []

print("\nRunning predictions...")


for i in range(
    0,
    len(texts),
    16
):

    batch_texts = texts[
        i:i + 16
    ]

    inputs = tokenizer(

        batch_texts,

        padding=True,

        truncation=True,

        max_length=128,

        return_tensors="pt"
    )

    inputs = {

        key: value.to(device)

        for key, value in inputs.items()

    }


    with torch.no_grad():

        outputs = model(
            **inputs
        )


    batch_predictions = (
        torch.argmax(
            outputs.logits,
            dim=-1
        )
        .cpu()
        .numpy()
    )


    predictions.extend(
        batch_predictions.tolist()
    )


# ------------------------------------------------------------
# Overall metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    true_labels,
    predictions
)


precision, recall, f1, _ = (
    precision_recall_fscore_support(

        true_labels,

        predictions,

        average="weighted",

        zero_division=0
    )
)


macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(

        true_labels,

        predictions,

        average="macro",

        zero_division=0
    )
)


print("\n" + "=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

print(
    f"Accuracy          : {accuracy:.4f}"
)

print(
    f"Weighted Precision: {precision:.4f}"
)

print(
    f"Weighted Recall   : {recall:.4f}"
)

print(
    f"Weighted F1       : {f1:.4f}"
)

print(
    f"Macro Precision   : {macro_precision:.4f}"
)

print(
    f"Macro Recall      : {macro_recall:.4f}"
)

print(
    f"Macro F1          : {macro_f1:.4f}"
)


# ------------------------------------------------------------
# Per-class report
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PER-CLASS RESULTS")
print("=" * 60)

print(

    classification_report(

        true_labels,

        predictions,

        labels=[
            0,
            1
        ],

        target_names=LABEL_NAMES,

        digits=4,

        zero_division=0
    )
)


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(

    true_labels,

    predictions,

    labels=[
        0,
        1
    ]
)


print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(
    "Rows = Actual"
)

print(
    "Columns = Predicted\n"
)


print(
    "                    "
    + "  ".join(

        f"{name[:18]:>18}"

        for name in LABEL_NAMES

    )
)


for i, row in enumerate(cm):

    print(

        f"{LABEL_NAMES[i]:<20}"

        + "  ".join(

            f"{value:>18}"

            for value in row

        )

    )


print(
    "\n" + "=" * 60
)

print(
    "BINARY EVALUATION COMPLETE"
)

print(
    "=" * 60
)