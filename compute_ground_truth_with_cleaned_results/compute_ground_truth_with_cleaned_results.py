import pandas as pd
import re
from pathlib import PurePath

# Input files
CLEANED_RESULTS_CSV = "cleaned_results.csv"
GROUND_TRUTH_CSV = "ground_truth.csv"

# Output file
OUTPUT_CSV = "ground_truth_with_cleaned_results.csv"

# Separator used when one file has multiple ground truth categories
GROUND_TRUTH_SEPARATOR = " | "

# If True:
# folder_a/document.md and folder_b/document.md are treated as document.md
USE_ONLY_FILENAME_FOR_MATCH = True


def clean_text_column(series):
    """
    Convert values to pandas string type and strip spaces.
    Keeps missing values as missing instead of turning them into 'nan'.
    """
    return series.astype("string").str.strip()


def normalize_document_name(value):
    """
    Normalizes document names for matching.

    Examples:
        document.md     -> document.md
        document_1.md   -> document.md
        document_2.md   -> document.md

    Also works with paths:
        folder_a/document_1.md -> document.md
    """
    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    if not value:
        return pd.NA

    # Make Windows paths behave like normal paths
    value = value.replace("\\", "/")

    if USE_ONLY_FILENAME_FOR_MATCH:
        value = PurePath(value).name

    path = PurePath(value)
    stem = path.stem
    suffix = path.suffix

    # Remove trailing _number before extension
    # Example: document_1.md -> document.md
    normalized_stem = re.sub(r"_\d+$", "", stem)

    return normalized_stem + suffix


def combine_categories(categories):
    """
    Combines multiple ground truth categories into one string.

    Example:
        ['informes', 'actas'] -> 'informes | actas'
    """
    unique_categories = (
        categories
        .dropna()
        .astype(str)
        .str.strip()
    )

    unique_categories = unique_categories[unique_categories != ""]

    # Remove duplicates while preserving order
    unique_categories = list(dict.fromkeys(unique_categories))

    if not unique_categories:
        return pd.NA

    return GROUND_TRUTH_SEPARATOR.join(unique_categories)


def main():
    # Read CSV files
    cleaned_results_df = pd.read_csv(CLEANED_RESULTS_CSV, encoding="utf-8")
    ground_truth_df = pd.read_csv(GROUND_TRUTH_CSV, encoding="utf-8")

    # Required columns validation
    required_cleaned_columns = {
        "DATASET", "DOCUMENT", "LLM_NAME", "TYPE", "REPLY", "TIME", "COST"
    }

    required_ground_truth_columns = {
        "DATASET", "CATEGORY", "FILE"
    }

    missing_cleaned_columns = required_cleaned_columns - set(cleaned_results_df.columns)
    missing_ground_truth_columns = required_ground_truth_columns - set(ground_truth_df.columns)

    if missing_cleaned_columns:
        raise ValueError(
            f"{CLEANED_RESULTS_CSV} is missing columns: {missing_cleaned_columns}"
        )

    if missing_ground_truth_columns:
        raise ValueError(
            f"{GROUND_TRUTH_CSV} is missing columns: {missing_ground_truth_columns}"
        )

    # Clean text columns
    cleaned_results_df["DATASET"] = clean_text_column(cleaned_results_df["DATASET"])
    cleaned_results_df["DOCUMENT"] = clean_text_column(cleaned_results_df["DOCUMENT"])

    ground_truth_df["DATASET"] = clean_text_column(ground_truth_df["DATASET"])
    ground_truth_df["FILE"] = clean_text_column(ground_truth_df["FILE"])
    ground_truth_df["CATEGORY"] = clean_text_column(ground_truth_df["CATEGORY"])

    # Create normalized matching keys
    cleaned_results_df["DOCUMENT_MATCH_KEY"] = cleaned_results_df["DOCUMENT"].apply(
        normalize_document_name
    )

    ground_truth_df["FILE_MATCH_KEY"] = ground_truth_df["FILE"].apply(
        normalize_document_name
    )

    # Group ground truth rows by DATASET + normalized FILE key
    # This allows:
    #   1. One file to have multiple valid categories
    #   2. document.md and document_1.md to match the same ground truth
    ground_truth_lookup = (
        ground_truth_df
        .groupby(["DATASET", "FILE_MATCH_KEY"], dropna=False)["CATEGORY"]
        .apply(combine_categories)
        .reset_index()
        .rename(columns={"CATEGORY": "GROUND_TRUTH"})
    )

    # Merge cleaned_results with grouped ground_truth
    results_df = cleaned_results_df.merge(
        ground_truth_lookup,
        how="left",
        left_on=["DATASET", "DOCUMENT_MATCH_KEY"],
        right_on=["DATASET", "FILE_MATCH_KEY"]
    )

    # Remove helper columns
    results_df = results_df.drop(
        columns=["DOCUMENT_MATCH_KEY", "FILE_MATCH_KEY"]
    )

    # Keep original cleaned_results columns plus GROUND_TRUTH at the end
    original_columns = [
        "DATASET", "DOCUMENT", "LLM_NAME", "TYPE", "REPLY", "TIME", "COST"
    ]

    final_columns = original_columns + ["GROUND_TRUTH"]
    results_df = results_df[final_columns]

    # Save output CSV
    results_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    # Print summary
    total_rows = len(results_df)
    missing_ground_truth = results_df["GROUND_TRUTH"].isna().sum()

    files_with_multiple_ground_truths = ground_truth_lookup[
        ground_truth_lookup["GROUND_TRUTH"].astype("string").str.contains(
            r"\|",
            regex=True,
            na=False
        )
    ]

    print(f"Generated: {OUTPUT_CSV}")
    print(f"Total rows: {total_rows}")
    print(f"Rows without ground truth: {missing_ground_truth}")
    print(f"Files with multiple ground truth categories: {len(files_with_multiple_ground_truths)}")


if __name__ == "__main__":
    main()