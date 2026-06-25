import pandas as pd
import numpy as np

# Input / Output files
INPUT_CSV = "ground_truth_with_cleaned_results.csv"
OUTPUT_CSV = "local_costs_on_ground_truth_with_cleaned_results.csv"


# Cost per hour (USD/hour)
LOCAL_MODEL_COSTS = {
    "google/gemma-4-E2B-it": 0.112,
    "google/gemma-2b-it": 0.0392,
    "Qwen/Qwen3.5-2B": 0.0392,
    "Qwen/Qwen1.5-0.5B": 0.03072,
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": 0.03072,
    "HuggingFaceTB/SmolLM-135M-Instruct": 0.0392,
}

# Load CSV
df = pd.read_csv(INPUT_CSV)

# Convert cost column to numeric if possible
df["COST"] = pd.to_numeric(df["COST"], errors="coerce")

# Fill costs only for local models and only when cost is missing
for model, hourly_cost in LOCAL_MODEL_COSTS.items():

    mask = (
        (df["LLM_NAME"] == model)
        & (df["COST"].isna())
    )

    df.loc[mask, "COST"] = (
        df.loc[mask, "TIME"] / 3600
    ) * hourly_cost

# Optional: round to 8 decimals
df["COST"] = df["COST"].round(6)

# Save
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved: {OUTPUT_CSV}")