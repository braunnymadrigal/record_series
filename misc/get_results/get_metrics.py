import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv("clean_results\\results_with_costs.csv")

results = []

for model in df["llm"].unique():

    subset = df[df["llm"] == model]

    y_true = subset["expected"].astype(str)
    y_pred = subset["reply"].astype(str)

    results.append({
        "model": model,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "documents": len(subset),
        "avg_time_seconds": subset["time"].mean(),
        "total_time_seconds": subset["time"].sum(),
        "avg_cost": subset["cost"].mean(),
        "total_cost": subset["cost"].sum()
    })

metrics_df = pd.DataFrame(results)

print(metrics_df)

metrics_df.to_csv(
    "clean_results\\metrics_by_model.csv",
    index=False
)