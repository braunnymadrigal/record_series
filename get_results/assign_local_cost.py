import pandas as pd
import numpy as np

# Input / Output files
INPUT_CSV = "clean_results\\results_cleaned.csv"
OUTPUT_CSV = "clean_results\\results_with_costs.csv"


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
df["cost"] = pd.to_numeric(df["cost"], errors="coerce")

# Fill costs only for local models and only when cost is missing
for model, hourly_cost in LOCAL_MODEL_COSTS.items():

    mask = (
        (df["llm"] == model)
        & (df["cost"].isna())
    )

    df.loc[mask, "cost"] = (
        df.loc[mask, "time"] / 3600
    ) * hourly_cost

# Optional: round to 8 decimals
df["cost"] = df["cost"].round(6)

# Save
df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved: {OUTPUT_CSV}")