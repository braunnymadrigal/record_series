import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Input CSV
INPUT_CSV = "records_series_times_used.csv"

# Output image
OUTPUT_IMAGE = "records_series_times_used.png"

# Display names for datasets
DATASET_LABELS = {
    "museo": "Museo Nacional",
    "archivo": "Archivo Nacional",
    "jurídico": "Archivo Asesoría Jurídica UCR"
}

# Specific category replacements
CATEGORY_REPLACEMENTS = {
    "informes de participación en comisión interinstitucional de encargados de archivos públicos": "Informes de participación"
}

# Font sizes
HEADER_FONT_SIZE = 20
CATEGORY_FONT_SIZE = 16
AXIS_TICK_FONT_SIZE = 14
VALUE_FONT_SIZE = 15

# Visual spacing
BAR_HEIGHT = 0.60
GAP_BETWEEN_DATASETS = 2

def uppercase_first(text):
    text = str(text).strip()
    if not text:
        return text
    return text[0].upper() + text[1:]

# Read CSV
df = pd.read_csv(INPUT_CSV, encoding="utf-8")

# Make sure TIMES_USED is integer
df["TIMES_USED"] = df["TIMES_USED"].fillna(0).astype(int)

# Replace category names
df["CATEGORY"] = df["CATEGORY"].replace(CATEGORY_REPLACEMENTS)

# Make all category names start with uppercase
df["CATEGORY"] = df["CATEGORY"].apply(uppercase_first)

# Dataset order
datasets = ["museo", "archivo", "jurídico"]

# Build one single graph with gaps between datasets
plot_rows = []
y_positions = []
y_labels = []
dataset_headers = []

current_y = 0

for dataset in datasets:
    dataset_df = df[df["DATASET"] == dataset].copy()

    # Sort categories inside each dataset
    dataset_df = dataset_df.sort_values("TIMES_USED", ascending=True)

    total_used = dataset_df["TIMES_USED"].sum()
    total_series = len(dataset_df)

    dataset_headers.append({
        "dataset": DATASET_LABELS.get(dataset, dataset),
        "documentos": total_used,
        "series": total_series,
        "y": current_y + len(dataset_df)
    })

    for _, row in dataset_df.iterrows():
        plot_rows.append(row)
        y_positions.append(current_y)
        y_labels.append(row["CATEGORY"])
        current_y += 1

    current_y += GAP_BETWEEN_DATASETS

plot_df = pd.DataFrame(plot_rows)

# Global x-axis scale
global_max = int(df["TIMES_USED"].max())
x_limit = max(1, math.ceil(global_max * 1.20))

# Figure size:
# wide rectangular chart, but still enough height for readability
fig_height = max(8, len(plot_df) * 0.42)
fig, ax = plt.subplots(figsize=(18, fig_height))

bars = ax.barh(
    y_positions,
    plot_df["TIMES_USED"],
    height=BAR_HEIGHT
)

# Category labels
ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=CATEGORY_FONT_SIZE)

# Same scale for everything
ax.set_xlim(0, x_limit)

# Force integer x-axis values
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.tick_params(axis="x", labelsize=AXIS_TICK_FONT_SIZE)
ax.tick_params(axis="y", labelsize=CATEGORY_FONT_SIZE, length=0)

# Add actual values next to bars, but hide zeros
for bar in bars:
    width = int(bar.get_width())

    if width > 0:
        y_position = bar.get_y() + bar.get_height() / 2

        ax.text(
            width + (x_limit * 0.01),
            y_position,
            f"{width}",
            va="center",
            ha="left",
            fontsize=VALUE_FONT_SIZE,
            fontweight="bold"
        )

# Dataset headers
for header in dataset_headers:
    ax.text(
        0,
        header["y"],
        f'{header["dataset"]} | Documentos: {header["documentos"]} | Series: {header["series"]}',
        fontsize=HEADER_FONT_SIZE,
        fontweight="bold",
        va="bottom"
    )

# Remove title and axis labels
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")

# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# Grid only on x-axis
ax.grid(axis="x", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

# Better margins for long labels and slide readability
plt.subplots_adjust(
    left=0.34,
    right=0.97,
    top=0.98,
    bottom=0.08
)

plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight")

print(f"Graph saved as {OUTPUT_IMAGE}")