import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from peft import PeftModel

import torch.nn.functional as F


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODEL_PATH = "toxicity_lora_output/final"

BASE_MODEL = (
    "shae2977/"
    "xlm-roberta-hinglish-sentiment-analysis"
)


LABELS = {

    0: "non_toxic",

    1: "toxic"

}


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("=" * 60)
print("LOADING BINARY TOXICITY MODEL")
print("=" * 60)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


base_model = AutoModelForSequenceClassification.from_pretrained(

    BASE_MODEL,

    num_labels=2,

    ignore_mismatched_sizes=True
)


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

print("\nModel ready.")

print(
    "Type a sentence to test it."
)

print(
    "Type 'exit' to stop."
)


# ------------------------------------------------------------
# Interactive testing
# ------------------------------------------------------------

while True:

    print(
        "\n" + "-" * 60
    )

    text = input(
        "Enter sentence: "
    ).strip()


    if text.lower() == "exit":

        break


    if not text:

        continue


    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        max_length=128
    )


    inputs = {

        key: value.to(device)

        for key, value in inputs.items()

    }


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs
        )


    probabilities = F.softmax(

        outputs.logits,

        dim=-1

    )[0]


    predicted_id = torch.argmax(
        probabilities
    ).item()


    predicted_label = LABELS[
        predicted_id
    ]


    confidence = (
        probabilities[
            predicted_id
        ].item()
        * 100
    )


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\nPrediction:")

    print(
        f"  {predicted_label}"
    )

    print(
        f"  Confidence: {confidence:.2f}%"
    )


    print(
        "\nAll probabilities:"
    )


    for class_id, probability in enumerate(
        probabilities
    ):

        print(

            f"  {LABELS[class_id]:<12}: "

            f"{probability.item() * 100:6.2f}%"

        )