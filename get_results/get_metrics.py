import pandas as pd
import unicodedata
import re
import warnings

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")


INPUT_CSV = "results_evaluated.csv"
OUTPUT_CSV = "metrics_by_model.csv"


VALID_CATEGORIES = [

    # JURIDICO
    "consultoria",
    "gestion judicial",
    "gestion notarial y registral",

    # MUSEO
    "correspondencia",
    "circulares",
    "memorandos",
    "informes de labores",
    "informes tecnicos producidos",
    "informes de inspeccion",
    "informes del indice de desarrollo archivistico",
    "minutas de reuniones",
    "controles de prestamo de documentos",
    "solicitudes administrativas",
    "expedientes de proyectos y programas institucionales",
    "expedientes de tramites laborales",
    "expedientes de transferencias de documentos",
    "instrumentos tecnicos archivisticos",
    "expedientes sobre voluntariado",
    "expedientes de tablas de plazos"
]


def normalize(text):

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = unicodedata.normalize('NFKD', text)

    text = ''.join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = re.sub(r'\s+', ' ', text)

    return text

def sanitize_prediction(pred):

    pred = normalize(pred)

    # MATCH EXACTO
    for cat in VALID_CATEGORIES:
        if pred == cat:
            return cat

    # MATCH PARCIAL
    for cat in VALID_CATEGORIES:

        pattern = r'\b' + re.escape(cat) + r'\b'

        if re.search(pattern, pred):
            return cat

    return "unknown"


df = pd.read_csv(INPUT_CSV)


df["expected_clean"] = (
    df["expected_label"]
    .apply(normalize)
)

df["prediction_clean"] = (
    df["response_normalized"]
    .fillna(df["response"])
    .apply(sanitize_prediction)
)


numeric_columns = [
    "elapsed_time",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "exact_match"
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)


results = []

grouped = df.groupby("model_name")

for model_name, group in grouped:

    y_true = group["expected_clean"]
    y_pred = group["prediction_clean"]


    # Accuracy basado en exact_match
    accuracy = group["exact_match"].mean()

    # Precision / Recall / F1 multicategoria
    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    total_time = group["elapsed_time"].sum()

    avg_time = group["elapsed_time"].mean()


    avg_input_tokens = group["input_tokens"].mean()

    avg_output_tokens = group["output_tokens"].mean()

    avg_total_tokens = group["total_tokens"].mean()

    total_input_tokens = group["input_tokens"].sum()

    total_output_tokens = group["output_tokens"].sum()

    total_tokens = group["total_tokens"].sum()


    efficiency = 0

    if total_time > 0:
        efficiency = f1 / total_time

  

    results.append({

        # MODELO
        "model_name": model_name,

        "provider": group["provider"].iloc[0],

        "model_group": (
            group["model_group"].iloc[0]
            if "model_group" in group.columns
            else ""
        ),

        "model_access_type": group["model_access_type"].iloc[0],

        # DATASET
        "dataset_size": len(group),


        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),


        "total_time_seconds": round(total_time, 4),
        "average_time_seconds": round(avg_time, 4),


        "average_input_tokens": round(avg_input_tokens, 2),
        "average_output_tokens": round(avg_output_tokens, 2),
        "average_total_tokens": round(avg_total_tokens, 2),


        "total_input_tokens": int(total_input_tokens),
        "total_output_tokens": int(total_output_tokens),
        "total_tokens": int(total_tokens),


        "input_token_price": "",
        "output_token_price": "",


        "gpu_price": "",
        "ram_price": "",
        "hardware_total_price": "",


        "average_cost": "",
        "total_cost": "",


        "efficiency_f1_over_time": round(efficiency, 8),


        "cer": "",
        "cer_correctness_oriented": "",
        "icer": ""
    })


metrics_df = pd.DataFrame(results)


metrics_df = metrics_df.sort_values(
    by="f1_score",
    ascending=False
)


metrics_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("\n===================================")
print("Metricas calculadas correctamente")
print(f"Modelos evaluados: {len(metrics_df)}")
print(f"CSV generado: {OUTPUT_CSV}")
print("===================================")
