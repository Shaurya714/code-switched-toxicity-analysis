import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = "DATASET/toxicity_binary_dataset.csv"
OUTPUT_DIR = "DATASET"


print("=" * 60)
print("DATASET CHECK")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Total rows: {len(df)}")
print(f"Unique tweets: {df['tweet'].nunique()}")


# ------------------------------------------------------------
# Label mapping
# ------------------------------------------------------------

label_map = {
    "non_toxic": 0,
    "toxic": 1
}

df["label"] = df["category"].map(label_map)

if df["label"].isna().any():

    print("\nERROR: Unknown category found:")
    print(
        df[df["label"].isna()]["category"].value_counts()
    )

    raise ValueError(
        "Dataset contains an unknown category."
    )


# Keep only required columns
df = df[
    [
        "tweet",
        "category",
        "label"
    ]
]


# ------------------------------------------------------------
# First split
# 70% train
# 30% temporary
# ------------------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)


# ------------------------------------------------------------
# Second split
# 15% validation
# 15% test
# ------------------------------------------------------------

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)


# ------------------------------------------------------------
# Save datasets
# ------------------------------------------------------------

train_df.to_csv(
    f"{OUTPUT_DIR}/train.csv",
    index=False
)

validation_df.to_csv(
    f"{OUTPUT_DIR}/validation.csv",
    index=False
)

test_df.to_csv(
    f"{OUTPUT_DIR}/test.csv",
    index=False
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL SPLIT")
print("=" * 60)

print(f"Train:      {len(train_df)}")
print(f"Validation: {len(validation_df)}")
print(f"Test:       {len(test_df)}")


print("\nTRAIN DISTRIBUTION:")
print(
    train_df["category"].value_counts()
)


print("\nVALIDATION DISTRIBUTION:")
print(
    validation_df["category"].value_counts()
)


print("\nTEST DISTRIBUTION:")
print(
    test_df["category"].value_counts()
)


print("\nFiles created:")

print("DATASET/train.csv")
print("DATASET/validation.csv")
print("DATASET/test.csv")


print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)