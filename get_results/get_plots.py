import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("clean_results//metrics_with_cer.csv")

# Quitar modelos con F1=0 para la gráfica CER
df_cer = df[df["CER"].notna()].copy()

# Nombres más cortos
df["short_model"] = [
    m.replace("google/", "")
     .replace("Qwen/", "")
     .replace("HuggingFaceTB/", "")
    for m in df["model"]
]

df_cer["short_model"] = [
    m.replace("google/", "")
     .replace("Qwen/", "")
     .replace("HuggingFaceTB/", "")
    for m in df_cer["model"]
]

# ==========================
# GRAPH 1
# ACCURACY VS F1
# ==========================

df_plot = df.sort_values(
    by="f1_score",
    ascending=False
)

x = np.arange(len(df_plot))
width = 0.38

plt.figure(figsize=(18, 8))

plt.bar(
    x - width/2,
    df_plot["accuracy"],
    width,
    label="Accuracy",
    color="#002751"
)

plt.bar(
    x + width/2,
    df_plot["f1_score"],
    width,
    label="F1 Score",
    color="#9B140A"
)

plt.xticks(
    x,
    df_plot["short_model"],
    rotation=45,
    ha="right"
)

plt.ylabel("Score")
plt.xlabel("Model")
plt.ylim(0, 1)

plt.title(
    "Accuracy and F1 Score by Model (Sorted by F1 Score)"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "plots//accuracy_vs_f1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================
# GRAPH 2
# F1 + AVG TIME
# ==========================

df_plot = df.sort_values(
    by="f1_score",
    ascending=False
)

x = np.arange(len(df_plot))

fig, ax1 = plt.subplots(figsize=(18, 8))

# --------------------------
# F1 SCORE (BARS)
# --------------------------

ax1.bar(
    x,
    df_plot["f1_score"],
    color="#9B140A",
    alpha=0.85,
    label="F1 Score"
)

ax1.set_ylabel("F1 Score")
ax1.set_xlabel("Model")
ax1.set_ylim(0, 1)

ax1.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

# --------------------------
# TIME (LINE)
# --------------------------

ax2 = ax1.twinx()

ax2.plot(
    x,
    df_plot["avg_time_seconds"],
    color="#002751",
    marker="o",
    linewidth=2.5,
    markersize=6,
    label="Average Time (s)"
)

ax2.set_ylabel("Average Time (seconds)")

# --------------------------
# X AXIS
# --------------------------
ax1.set_xticks(x)

ax1.set_xticklabels(
    df_plot["short_model"],
    rotation=45,
    ha="right"
)
# --------------------------
# TITLE
# --------------------------

plt.title(
    "F1 Score and Average Classification Time (Sorted by F1 Score)"
)

# --------------------------
# LEGEND
# --------------------------

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper right"
)

plt.tight_layout()

plt.savefig(
    "plots//f1_vs_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================
# GRAPH 3
# F1 VS CER
# ==========================

df_plot = (
    df[df["CER"].notna()]
    .sort_values(
        by="f1_score",
        ascending=False
    )
)

x = np.arange(len(df_plot))

fig, ax1 = plt.subplots(figsize=(18, 8))

# --------------------------
# F1 SCORE (BARS)
# --------------------------

ax1.bar(
    x,
    df_plot["f1_score"],
    color="#9B140A",
    alpha=0.85,
    label="F1 Score"
)

ax1.set_ylabel("F1 Score")
ax1.set_ylim(0, 1)

ax1.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

# --------------------------
# CER (LINE)
# --------------------------

ax2 = ax1.twinx()

ax2.plot(
    x,
    df_plot["CER"],
    color="#002751",
    marker="o",
    linewidth=2.5,
    markersize=6,
    label="CER"
)

ax2.set_ylabel("CER")

# --------------------------
# X AXIS
# --------------------------

ax1.set_xticks(x)

ax1.set_xticklabels(
    df_plot["short_model"],
    rotation=45,
    ha="right"
)

# --------------------------
# TITLE
# --------------------------

plt.title(
    "F1 Score and CER by Model (Sorted by F1 Score)"
)

# --------------------------
# LEGEND
# --------------------------

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "plots//f1_vs_cer.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================
# GRAPH 4
# F1 VS AVERAGE COST
# ==========================

df_plot = df.sort_values(
    by="f1_score",
    ascending=False
).copy()

# Convertir a centavos de dólar
df_plot["avg_cost_cents"] = df_plot["avg_cost"] * 100

x = np.arange(len(df_plot))

fig, ax1 = plt.subplots(figsize=(18, 8))

# ==========================
# GRAPH 4
# F1 VS AVERAGE COST
# ==========================

df_plot = df.sort_values(
    by="f1_score",
    ascending=False
)

x = np.arange(len(df_plot))

fig, ax1 = plt.subplots(figsize=(18, 8))

# --------------------------
# F1 SCORE (BARS)
# --------------------------

ax1.bar(
    x,
    df_plot["f1_score"],
    color="#9B140A",
    alpha=0.85,
    label="F1 Score"
)

ax1.set_ylabel("F1 Score")
ax1.set_ylim(0, 1)

ax1.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

# --------------------------
# AVERAGE COST (LINE)
# --------------------------

ax2 = ax1.twinx()

ax2.plot(
    x,
    df_plot["avg_cost"],
    color="#002751",
    marker="o",
    linewidth=2.5,
    markersize=6,
    label="Average Cost ($)"
)

ax2.set_ylabel("Average Cost per Classification (USD)")

# --------------------------
# X AXIS
# --------------------------

ax1.set_xticks(x)

ax1.set_xticklabels(
    df_plot["short_model"],
    rotation=45,
    ha="right"
)

# --------------------------
# TITLE
# --------------------------

plt.title(
    "F1 Score and Average Cost per Classification by Model (Sorted by F1 Score)"
)

# --------------------------
# LEGEND
# --------------------------

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "plots//f1_vs_avg_cost.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Graphs generated successfully.")