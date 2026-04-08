from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

TEXT_COL = "text"
FEATURE_COLS = [TEXT_COL]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=5000), TEXT_COL),
        ],
        remainder="drop",
    )


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )


def metric_summary(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = y_true.astype(str)
    y_pred = pd.Series(y_pred).astype(str)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = ""
    return df


def normalize_label_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: str(x).strip().lower() if pd.notna(x) else pd.NA)
    return df


def grouped_train_val_split(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data so that identical text values do not appear in both train and validation.
    """
    df = df.dropna(subset=[label_col]).copy()
    df["group_text"] = df[TEXT_COL].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(df, groups=df["group_text"]))

    train_split = df.iloc[train_idx].copy()
    val_split = df.iloc[val_idx].copy()

    return train_split, val_split


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
    test_df = pd.read_csv(DATA_DIR / "test_clean.csv")

    train_df = normalize_label_column(train_df, "sentiment_label")
    train_df = normalize_label_column(train_df, "intent_label")

    train_df = ensure_columns(train_df)
    test_df = ensure_columns(test_df)

    # ---------------------------
    # Sanity check for overlap
    # ---------------------------
    train_unique_texts = set(train_df[TEXT_COL].astype(str).unique())
    test_unique_texts = set(test_df[TEXT_COL].astype(str).unique())
    overlap_count = len(train_unique_texts.intersection(test_unique_texts))

    # ---------------------------
    # Sentiment: grouped validation
    # ---------------------------
    sentiment_all = train_df.dropna(subset=["sentiment_label"]).copy()
    sentiment_train, sentiment_val = grouped_train_val_split(sentiment_all, "sentiment_label")

    X_train_sent = sentiment_train[FEATURE_COLS]
    y_train_sent = sentiment_train["sentiment_label"].astype(str)

    X_val_sent = sentiment_val[FEATURE_COLS]
    y_val_sent = sentiment_val["sentiment_label"].astype(str)

    sentiment_model = build_pipeline()
    sentiment_model.fit(X_train_sent, y_train_sent)

    sentiment_val_preds = sentiment_model.predict(X_val_sent)
    sentiment_metrics = metric_summary(y_val_sent, sentiment_val_preds)

    # Final sentiment model trained on all labeled train rows
    sentiment_model_final = build_pipeline()
    sentiment_model_final.fit(
        sentiment_all[FEATURE_COLS],
        sentiment_all["sentiment_label"].astype(str),
    )
    test_sentiment_preds = sentiment_model_final.predict(test_df[FEATURE_COLS])

    # ---------------------------
    # Intent: grouped validation
    # ---------------------------
    intent_all = train_df.dropna(subset=["intent_label"]).copy()
    intent_train, intent_val = grouped_train_val_split(intent_all, "intent_label")

    X_train_intent = intent_train[FEATURE_COLS]
    y_train_intent = intent_train["intent_label"].astype(str)

    X_val_intent = intent_val[FEATURE_COLS]
    y_val_intent = intent_val["intent_label"].astype(str)

    intent_model = build_pipeline()
    intent_model.fit(X_train_intent, y_train_intent)

    intent_val_preds = intent_model.predict(X_val_intent)
    intent_metrics = metric_summary(y_val_intent, intent_val_preds)

    # Final intent model trained on all labeled train rows
    intent_model_final = build_pipeline()
    intent_model_final.fit(
        intent_all[FEATURE_COLS],
        intent_all["intent_label"].astype(str),
    )
    test_intent_preds = intent_model_final.predict(test_df[FEATURE_COLS])

    # ---------------------------
    # Save predictions for provided test set
    # ---------------------------
    predictions_df = pd.DataFrame(
        {
            "ticket_id": test_df["ticket_id"],
            "predicted_sentiment": pd.Series(test_sentiment_preds).astype(str),
            "predicted_intent": pd.Series(test_intent_preds).astype(str),
        }
    )

    predictions_path = OUTPUT_DIR / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    # ---------------------------
    # Save realistic metrics from grouped validation
    # ---------------------------
    metrics = {
        "dataset_note": (
            "Provided train/test split contains complete overlap of unique test texts with train texts. "
            "Reported metrics below come from grouped validation on train data, not from the provided test set."
        ),
        "overlap_summary": {
            "train_unique_texts": int(len(train_unique_texts)),
            "test_unique_texts": int(len(test_unique_texts)),
            "overlapping_unique_texts": int(overlap_count),
        },
        "sentiment": sentiment_metrics,
        "intent": intent_metrics,
        "train_rows_total": int(len(train_df)),
        "test_rows_total": int(len(test_df)),
        "sentiment_train_rows_used": int(len(sentiment_all)),
        "intent_train_rows_used": int(len(intent_all)),
        "sentiment_grouped_train_rows": int(len(sentiment_train)),
        "sentiment_grouped_val_rows": int(len(sentiment_val)),
        "intent_grouped_train_rows": int(len(intent_train)),
        "intent_grouped_val_rows": int(len(intent_val)),
    }

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved outputs:")
    print(f"  - {predictions_path}")
    print(f"  - {metrics_path}")

    print("\nOverlap summary:")
    print(json.dumps(metrics["overlap_summary"], indent=2))

    print("\nSentiment grouped-validation metrics:")
    print(json.dumps(
        {
            "accuracy": sentiment_metrics["accuracy"],
            "macro_f1": sentiment_metrics["macro_f1"],
            "weighted_f1": sentiment_metrics["weighted_f1"],
        },
        indent=2,
    ))

    print("\nIntent grouped-validation metrics:")
    print(json.dumps(
        {
            "accuracy": intent_metrics["accuracy"],
            "macro_f1": intent_metrics["macro_f1"],
            "weighted_f1": intent_metrics["weighted_f1"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()