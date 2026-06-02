import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# COLORS
RED_COLOR = "#de0606"
BLUE_COLOR = "#007fd7"

# Legend
extra_legend = [
    Line2D([0], [0], color="none", label="■Local   ☁Cloud"),
    Line2D([0], [0], color="none", label="¢Barato   $Caro"),
]

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(f"metrics_for_plotting.csv")

# Poner F1 y Accuracy como porcentajes enteros
def reformat(s):
    s = float(s)
    s = s * 100.0
    s = int(round(s))
    return s
df['accuracy'] = df['accuracy'].apply(reformat)
df['f1_score'] = df['f1_score'].apply(reformat)

# Nombres más bonitos
df["model"] = df["model"].str.replace(r"\\n", "\n", regex=True)

# Agregar 'emojis' según el tipo al nombre
type_icon = {
    "local": "■",
    "cloud": "☁"
}

cheap_icon = {
    "yes": "¢",
    "no": "$"
}

df["model"] = (
    df["model"]
    + "\n"
    + df["type"].map(type_icon)
    + df["is_cheap"].map(cheap_icon)
)

# ==========================
# GRAPH 1
# F1 + ACCURACY
# ==========================

df_plot = df.sort_values(
    by="f1_score",
    ascending=False
)

y = np.arange(len(df_plot))
height = 0.38

plt.figure(figsize=(16, 16))

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,
})

plt.barh(
    y - height/2,
    df_plot["f1_score"],
    height,
    label="F1 promedio (porcentaje)",
    color=RED_COLOR
)

plt.barh(
    y + height/2,
    df_plot["accuracy"],
    height,
    label="Accuracy promedio (porcentaje)",
    color=BLUE_COLOR
)

plt.yticks(
    y,
    df_plot["model"]
)

plt.xlim(0, 100)
ax = plt.gca()
ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
# Highest F1 on top
ax.invert_yaxis()

plt.grid(
    axis="x",
    linestyle="--",
    alpha=0.4
)

ax.set_xlabel(
    "Porcentaje",
    labelpad=20
)

handles, labels = plt.gca().get_legend_handles_labels()

plt.legend(
    handles + extra_legend,
    labels + [x.get_label() for x in extra_legend],
    loc="best"
)

plt.tight_layout()

plt.savefig(
    "../plots/accuracy_vs_f1_horizontal.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()


# ==========================
# GRAPH 2
# F1 + AVG TIME (HORIZONTAL)
# ==========================

df_plot = df.sort_values(
    by="f1_score",
    ascending=False
)

y = np.arange(len(df_plot))

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,
})

fig, ax1 = plt.subplots(figsize=(16, 16))

# --------------------------
# F1 SCORE (BARS)
# --------------------------

bars = ax1.barh(
    y,
    df_plot["f1_score"],
    height=0.6,
    color=RED_COLOR,
    alpha=0.85,
    label="F1 promedio (porcentaje)"
)

ax1.set_xlim(0, 100)

ax1.xaxis.set_major_formatter(
    PercentFormatter(xmax=100)
)

ax1.grid(
    axis="x",
    linestyle="--",
    alpha=0.4
)

# --------------------------
# MODELS
# --------------------------

ax1.set_yticks(y)

ax1.set_yticklabels(
    df_plot["model"]
)

# Highest F1 on top
ax1.invert_yaxis()

# --------------------------
# TIME (DOTS)
# --------------------------

ax2 = ax1.twiny()

ax2.scatter(
    df_plot["avg_time_seconds"],
    y,
    color=BLUE_COLOR,
    s=150,  # dot size
    label="Tiempo promedio (segundos)",
    zorder=3
)

ax2.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"{x:.0f} s")
)

ax1.set_xlabel(
    "F1 (porcentaje)",
    color=RED_COLOR,
    labelpad=20
)

ax2.set_xlabel(
    "Tiempo promedio (segundos)",
    color=BLUE_COLOR,
    labelpad=20
)

ax1.xaxis.label.set_color(RED_COLOR)
ax1.tick_params(axis="x", colors=RED_COLOR)

ax2.xaxis.label.set_color(BLUE_COLOR)
ax2.tick_params(axis="x", colors=BLUE_COLOR)

# --------------------------
# LEGEND
# --------------------------

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    handles1 + handles2 + extra_legend,
    labels1 + labels2 + [x.get_label() for x in extra_legend],
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    "../plots/f1_vs_time_horizontal.png",
    dpi=600,
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

y = np.arange(len(df_plot))

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,
})

fig, ax1 = plt.subplots(figsize=(16, 16))

# --------------------------
# F1 SCORE (BARS)
# --------------------------

bars = ax1.barh(
    y,
    df_plot["f1_score"],
    height=0.6,
    color=RED_COLOR,
    alpha=0.85,
    label="F1 promedio (porcentaje)"
)

ax1.set_xlim(0, 100)

ax1.xaxis.set_major_formatter(
    PercentFormatter(xmax=100)
)

ax1.grid(
    axis="x",
    linestyle="--",
    alpha=0.4
)

# --------------------------
# MODELS
# --------------------------

ax1.set_yticks(y)

ax1.set_yticklabels(
    df_plot["model"]
)

# Highest F1 on top
ax1.invert_yaxis()

# --------------------------
# CER (DOTS)
# --------------------------

ax2 = ax1.twiny()

ax2.scatter(
    df_plot["CER"],
    y,
    color=BLUE_COLOR,
    s=150,  # dot size
    label="CER promedio (USD por unidad de F1)",
    zorder=3
)

ax2.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"${x:.2f}")
)

ax1.set_xlabel(
    "F1 (porcentaje)",
    color=RED_COLOR,
    labelpad=20
)

ax2.set_xlabel(
    "CER promedio (USD por unidad de F1)",
    color=BLUE_COLOR,
    labelpad=20
)

ax1.xaxis.label.set_color(RED_COLOR)
ax1.tick_params(axis="x", colors=RED_COLOR)

ax2.xaxis.label.set_color(BLUE_COLOR)
ax2.tick_params(axis="x", colors=BLUE_COLOR)

# --------------------------
# LEGEND
# --------------------------

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    handles1 + handles2 + extra_legend,
    labels1 + labels2 + [x.get_label() for x in extra_legend],
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    "../plots/f1_vs_cer_horizontal.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ==========================
# GRAPH 3
# F1 VS avg_cost
# ==========================

df_plot = df.sort_values(
    by="f1_score",
    ascending=False
)

y = np.arange(len(df_plot))

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 16,
    "legend.fontsize": 20,
})

fig, ax1 = plt.subplots(figsize=(16, 16))

# --------------------------
# F1 SCORE (BARS)
# --------------------------

bars = ax1.barh(
    y,
    df_plot["f1_score"],
    height=0.6,
    color=RED_COLOR,
    alpha=0.85,
    label="F1 promedio (porcentaje)"
)

ax1.set_xlim(0, 100)

ax1.xaxis.set_major_formatter(
    PercentFormatter(xmax=100)
)

ax1.grid(
    axis="x",
    linestyle="--",
    alpha=0.4
)

# --------------------------
# MODELS
# --------------------------

ax1.set_yticks(y)

ax1.set_yticklabels(
    df_plot["model"]
)

# Highest F1 on top
ax1.invert_yaxis()

# --------------------------
# CER (DOTS)
# --------------------------

ax2 = ax1.twiny()

ax2.scatter(
    df_plot["avg_cost"],
    y,
    color=BLUE_COLOR,
    s=150,  # dot size
    label="Costo promedio (USD)",
    zorder=3
)

ax2.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: f"${x:.3f}")
)

ax1.set_xlabel(
    "F1 promedio (porcentaje)",
    color=RED_COLOR,
    labelpad=20
)

ax2.set_xlabel(
    "Costo promedio (USD)",
    color=BLUE_COLOR,
    labelpad=20
)

ax1.xaxis.label.set_color(RED_COLOR)
ax1.tick_params(axis="x", colors=RED_COLOR)

ax2.xaxis.label.set_color(BLUE_COLOR)
ax2.tick_params(axis="x", colors=BLUE_COLOR)

# --------------------------
# LEGEND
# --------------------------

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    handles1 + handles2 + extra_legend,
    labels1 + labels2 + [x.get_label() for x in extra_legend],
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    "../plots/f1_vs_cost_horizontal.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("Graphs generated successfully.")