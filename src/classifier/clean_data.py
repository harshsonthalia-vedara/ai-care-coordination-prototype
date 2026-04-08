from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_FILE = DATA_DIR / "Train Dataset.xlsx"
TEST_FILE = DATA_DIR / "Test Data.xlsx"
TEST_LABELS_FILE = DATA_DIR / "test_labels.xlsx"


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\.\,\!\?\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "text" in df.columns:
        df["text"] = df["text"].apply(normalize_text)

    for col in ["channel", "category", "resolution_status", "agent_id"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()

    for col in ["sentiment_label", "intent_label"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(x).strip().lower() if pd.notna(x) else pd.NA
            )

    if "date_submitted" in df.columns:
        df["date_submitted"] = pd.to_datetime(df["date_submitted"], errors="coerce")

    for col in ["word_count", "response_time_hours", "csat_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "word_count" in df.columns and "text" in df.columns:
        missing_wc = df["word_count"].isna()
        df.loc[missing_wc, "word_count"] = (
            df.loc[missing_wc, "text"].fillna("").str.split().str.len()
        )

    if "ticket_id" in df.columns:
        df = df.drop_duplicates(subset=["ticket_id"]).reset_index(drop=True)

    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_excel(TRAIN_FILE, sheet_name="train")
    test_df = pd.read_excel(TEST_FILE, sheet_name="test")
    test_labels_df = pd.read_excel(TEST_LABELS_FILE, sheet_name="test_labels")

    train_df = clean_dataframe(train_df)
    test_df = clean_dataframe(test_df)
    test_labels_df = clean_dataframe(test_labels_df)

    train_out = PROCESSED_DIR / "train_clean.csv"
    test_out = PROCESSED_DIR / "test_clean.csv"
    labels_out = PROCESSED_DIR / "test_labels_clean.csv"

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    test_labels_df.to_csv(labels_out, index=False)

    print("Saved cleaned datasets:")
    print(f"  - {train_out}")
    print(f"  - {test_out}")
    print(f"  - {labels_out}")

    print("\nShapes:")
    print("  train:", train_df.shape)
    print("  test:", test_df.shape)
    print("  test_labels:", test_labels_df.shape)

    print("\nTrain sentiment distribution:")
    print(train_df["sentiment_label"].value_counts(dropna=False))

    print("\nTrain intent distribution:")
    print(train_df["intent_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()