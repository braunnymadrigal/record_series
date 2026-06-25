import pandas as pd

# Input files
GROUND_TRUTH_CSV = "ground_truth.csv"
RECORDS_SERIES_CSV = "records_series.csv"

# Output file
OUTPUT_CSV = "records_series_times_used.csv"

# Read CSV files
ground_truth_df = pd.read_csv(GROUND_TRUTH_CSV, encoding="utf-8")
records_series_df = pd.read_csv(RECORDS_SERIES_CSV, encoding="utf-8")

# Count how many times each DATASET + CATEGORY appears in ground_truth.csv
usage_counts_df = (
    ground_truth_df
    .groupby(["DATASET", "CATEGORY"])
    .size()
    .reset_index(name="TIMES_USED")
)

# Merge with records_series.csv so categories with 0 files are also included
result_df = records_series_df.merge(
    usage_counts_df,
    on=["DATASET", "CATEGORY"],
    how="left"
)

# Replace NaN with 0 for categories that were never used
result_df["TIMES_USED"] = result_df["TIMES_USED"].fillna(0).astype(int)

# Save result
result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"{OUTPUT_CSV} generated with {len(result_df)} rows.")