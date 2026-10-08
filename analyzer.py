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
# Toxicity Analyzer
# ------------------------------------------------------------

class ToxicityAnalyzer:

    def __init__(self):

        print("Loading toxicity analyzer...")

        self.device = torch.device(

            "cuda"
            if torch.cuda.is_available()
            else "cpu"

        )


        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_PATH
            )
        )


        base_model = (
            AutoModelForSequenceClassification
            .from_pretrained(

                BASE_MODEL,

                num_labels=2,

                ignore_mismatched_sizes=True

            )
        )


        self.model = (
            PeftModel.from_pretrained(

                base_model,

                MODEL_PATH

            )
        )


        self.model.to(
            self.device
        )

        self.model.eval()


        print(
            f"Model loaded on: {self.device}"
        )


    # --------------------------------------------------------
    # Analyze text
    # --------------------------------------------------------

    def analyze(self, text):

        if not isinstance(
            text,
            str
        ):

            raise TypeError(
                "Input must be a string."
            )


        text = text.strip()


        if not text:

            raise ValueError(
                "Input text cannot be empty."
            )


        inputs = self.tokenizer(

            text,

            return_tensors="pt",

            truncation=True,

            max_length=128

        )


        inputs = {

            key: value.to(
                self.device
            )

            for key, value in inputs.items()

        }


        with torch.no_grad():

            outputs = self.model(
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


        return {

            "text": text,

            "prediction": predicted_label,

            "is_toxic": predicted_id == 1,

            "confidence": round(
                confidence,
                2
            ),

            "non_toxic_probability": round(

                probabilities[0].item() * 100,

                2

            ),

            "toxic_probability": round(

                probabilities[1].item() * 100,

                2

            )

        }


# ------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------

if __name__ == "__main__":

    analyzer = ToxicityAnalyzer()


    print(
        "\n" + "=" * 60
    )

    print(
        "TOXICITY ANALYZER"
    )

    print(
        "=" * 60
    )

    print(
        "Enter text to analyze."
    )

    print(
        "Type 'exit' to stop."
    )


    while True:

        text = input(
            "\nText: "
        ).strip()


        if text.lower() == "exit":

            break


        if not text:

            continue


        result = analyzer.analyze(
            text
        )


        print(
            "\nPrediction :",
            result["prediction"]
        )

        print(
            "Toxic      :",
            result["is_toxic"]
        )

        print(
            "Confidence :",
            f'{result["confidence"]:.2f}%'
        )

        print(
            "Non-toxic  :",
            f'{result["non_toxic_probability"]:.2f}%'
        )

        print(
            "Toxic      :",
            f'{result["toxic_probability"]:.2f}%'
        )