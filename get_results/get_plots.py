import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CONFIG
# ==========================================

INPUT_CSV = "metrics_by_model.csv"

# ==========================================
# LEER CSV
# ==========================================

df = pd.read_csv(INPUT_CSV)

# ==========================================
# ORDENAR POR F1
# ==========================================

df = df.sort_values(
    by="f1_score",
    ascending=False
)

# ==========================================
# GRAFICO 1
# METRICAS POR MODELO
# ==========================================

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1_score"
]

x = range(len(df))

width = 0.2

plt.figure(figsize=(18, 8))

for i, metric in enumerate(metrics):

    plt.bar(
        [p + i * width for p in x],
        df[metric],
        width=width,
        label=metric
    )

plt.xticks(
    [p + width * 1.5 for p in x],
    df["model_name"],
    rotation=45,
    ha="right"
)

plt.ylabel("Score")
plt.xlabel("Modelo")
plt.title("Metricas de Clasificacion por Modelo")
plt.legend()
plt.tight_layout()

plt.savefig(
    "classification_metrics_barplot.png",
    dpi=300
)

plt.close()

# ==========================================
# GRAFICO
# F1 SCORE VS TIEMPO PROMEDIO
# CON DOBLE EJE
# ==========================================

fig, ax1 = plt.subplots(figsize=(16, 8))

# ------------------------------------------
# EJE IZQUIERDO -> F1 SCORE
# ------------------------------------------

ax1.plot(
    df["model_name"],
    df["f1_score"],
    marker="o",
    linewidth=2,
    label="F1 Score"
)

ax1.set_xlabel("Modelo")

ax1.set_ylabel("F1 Score")

ax1.tick_params(axis="y")

plt.xticks(
    rotation=45,
    ha="right"
)

# ------------------------------------------
# EJE DERECHO -> TIEMPO
# ------------------------------------------

ax2 = ax1.twinx()

ax2.plot(
    df["model_name"],
    df["average_time_seconds"],
    marker="s",
    linewidth=2,
    linestyle="--",
    label="Average Time (s)"
)

ax2.set_ylabel("Tiempo Promedio (s)")

ax2.tick_params(axis="y")

# ------------------------------------------
# TITULO
# ------------------------------------------

plt.title(
    "Relacion entre F1 Score y Tiempo Promedio"
)

# ------------------------------------------
# LEYENDA
# ------------------------------------------

lines1, labels1 = ax1.get_legend_handles_labels()

lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "f1_vs_time_dual_axis.png",
    dpi=300
)

plt.close()

print(
    "Grafico generado: f1_vs_time_dual_axis.png"
)