import pandas as pd
import unicodedata
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# Input CSV
INPUT_CSV = "local_costs_on_ground_truth_with_cleaned_results.csv"

# Output CSV
OUTPUT_CSV = "metrics.csv"

# Separator used when one document has multiple valid ground truth categories
GROUND_TRUTH_SEPARATOR = " | "

# Value used when the model did not reply / reply is missing
MISSING_REPLY_LABEL = "__missing_reply__"


def normalize_label(value):
    """
    Normalize labels so comparisons are more consistent.

    Example:
    ' Expedientes de Trámites Laborales '
    becomes:
    'expedientes de tramites laborales'

    This makes comparisons:
    - case-insensitive
    - accent-insensitive
    - whitespace-insensitive
    """
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower()

    # Remove accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))

    # Normalize repeated spaces
    text = " ".join(text.split())

    if text == "":
        return pd.NA

    return text


def get_ground_truth_options(value):
    """
    Convert a GROUND_TRUTH value into a list of valid labels.

    Example:
    'expedientes de trámites laborales | informes'

    becomes:
    [
        'expedientes de tramites laborales',
        'informes'
    ]
    """
    if pd.isna(value):
        return []

    parts = str(value).split(GROUND_TRUTH_SEPARATOR)

    options = []
    for part in parts:
        normalized = normalize_label(part)

        if pd.notna(normalized):
            options.append(normalized)

    # Remove duplicates while preserving order
    options = list(dict.fromkeys(options))

    return options


def get_reply_label(value):
    """
    Convert model reply into a clean comparable label.
    Missing replies are replaced with a special label.
    """
    normalized = normalize_label(value)

    if pd.isna(normalized):
        return MISSING_REPLY_LABEL

    return normalized


# Read CSV
df = pd.read_csv(INPUT_CSV, encoding="utf-8")

# Make sure TIME and COST are numeric
df["TIME"] = pd.to_numeric(df["TIME"], errors="coerce")
df["COST"] = pd.to_numeric(df["COST"], errors="coerce")

# Remove rows without ground truth, because they cannot be evaluated
df = df.dropna(subset=["GROUND_TRUTH"])

results = []

# Group by both model and type so the TYPE column is preserved
for (model, type_value), subset in df.groupby(["LLM_NAME", "TYPE"], dropna=False):

    subset = subset.copy()

    y_true_for_metrics = []
    y_pred_for_metrics = []

    correct_count = 0
    multi_ground_truth_count = 0

    for _, row in subset.iterrows():

        ground_truth_options = get_ground_truth_options(row["GROUND_TRUTH"])
        reply = get_reply_label(row["REPLY"])

        if len(ground_truth_options) > 1:
            multi_ground_truth_count += 1

        is_correct = reply in ground_truth_options

        if is_correct:
            correct_count += 1

            # For sklearn metrics, if the reply is one of the valid answers,
            # we treat that reply as the true label.
            y_true_for_metrics.append(reply)
            y_pred_for_metrics.append(reply)

        else:
            # If wrong, use the first ground truth option as the expected label.
            # This lets precision/recall/f1 still work with sklearn.
            if ground_truth_options:
                y_true_for_metrics.append(ground_truth_options[0])
            else:
                y_true_for_metrics.append("__missing_ground_truth__")

            y_pred_for_metrics.append(reply)

    accuracy = correct_count / len(subset) if len(subset) > 0 else 0

    results.append({
        "model": model,
        "type": type_value,
        "accuracy": accuracy,
        "precision": precision_score(
            y_true_for_metrics,
            y_pred_for_metrics,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            y_true_for_metrics,
            y_pred_for_metrics,
            average="weighted",
            zero_division=0
        ),
        "f1_score": f1_score(
            y_true_for_metrics,
            y_pred_for_metrics,
            average="weighted",
            zero_division=0
        ),
        "documents": len(subset),
        "correct": correct_count,
        "incorrect": len(subset) - correct_count,
        "missing_replies": subset["REPLY"].isna().sum(),
        "multi_ground_truth_rows": multi_ground_truth_count,
        "avg_time_seconds": subset["TIME"].mean(),
        "total_time_seconds": subset["TIME"].sum(),
        "avg_cost": subset["COST"].mean(),
        "total_cost": subset["COST"].sum()
    })

metrics_df = pd.DataFrame(results)

# CER = Cost / Effectiveness(F1)
metrics_df["CER"] = np.where(
    metrics_df["f1_score"] > 0,
    metrics_df["total_cost"] / metrics_df["f1_score"],
    np.nan
)

print(metrics_df)

metrics_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8"
)