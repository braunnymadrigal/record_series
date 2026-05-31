import pandas as pd
import numpy as np

INPUT_CSV = "clean_results\\metrics_by_model.csv"
OUTPUT_CSV = "clean_results\\metrics_with_cer.csv"

df = pd.read_csv(INPUT_CSV)

# CER = Cost / Effectiveness(F1)
df["CER"] = np.where(
    df["f1_score"] > 0,
    df["total_cost"] / df["f1_score"],
    np.nan
)

df.to_csv(OUTPUT_CSV, index=False)

print(df[[
    "model",
    "f1_score",
    "avg_cost",
    "CER"
]])

print(f"\nSaved: {OUTPUT_CSV}")