import os
import csv
import unicodedata
from pathlib import Path


ROOT_DIR = r"C:\\Users\\PC\\Downloads\\A"

OUTPUT_CSV = "ground_truth_full_path.csv"


def normalize(text):
    """
    Convierte a minúsculas, elimina tildes y simplifica texto.
    """

    text = text.lower()

    # quitar tildes
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))

    replacements = {
        "exp.": "expediente",
        "exp ": "expediente ",
        "expedientes": "expediente",
        "tramites": "tramites",
        "minuta": "minutas",
        "reunion": "reuniones",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text



RULES = [

    # MUSEO
    {
        "dataset": "museo",
        "categoria": "circulares",
        "keywords": ["circular"]
    },

    {
        "dataset": "museo",
        "categoria": "memorandos",
        "keywords": ["memorando", "memo"]
    },

    {
        "dataset": "museo",
        "categoria": "minutas de reuniones",
        "keywords": ["minutas", "reuniones"]
    },

    {
        "dataset": "museo",
        "categoria": "expedientes de trámites laborales",
        "keywords": ["expediente", "tramites laborales"]
    },

    {
        "dataset": "museo",
        "categoria": "expedientes de transferencias de documentos",
        "keywords": ["transferencias de documentos"]
    },

    {
        "dataset": "museo",
        "categoria": "expedientes sobre voluntariado",
        "keywords": ["voluntariado"]
    },

    {
        "dataset": "museo",
        "categoria": "expedientes de proyectos y programas institucionales",
        "keywords": ["proyecto"]
    },

    {
        "dataset": "museo",
        "categoria": "informes técnicos producidos",
        "keywords": ["informes"]
    },

    {
        "dataset": "museo",
        "categoria": "informes del índice de desarrollo archivístico",
        "keywords": ["desarrollo archivistico"]
    },

    {
        "dataset": "museo",
        "categoria": "correspondencia",
        "keywords": ["correspondencia"]
    },

    {
        "dataset": "museo",
        "categoria": "solicitudes administrativas",
        "keywords": ["solicitud"]
    },

    {
        "dataset": "museo",
        "categoria": "instrumentos técnicos archivísticos",
        "keywords": ["instrumento"]
    },

    # JURIDICO
    {
        "dataset": "jurídico",
        "categoria": "consultoría",
        "keywords": ["consultoria"]
    },

    {
        "dataset": "jurídico",
        "categoria": "gestión judicial",
        "keywords": ["judicial"]
    },

    {
        "dataset": "jurídico",
        "categoria": "gestión notarial y registral",
        "keywords": ["notarial", "registral"]
    },

]


def detect_category(path_text):

    normalized = normalize(path_text)

    for rule in RULES:

        matched = True

        for keyword in rule["keywords"]:

            if normalize(keyword) not in normalized:
                matched = False
                break

        if matched:
            return rule["dataset"], rule["categoria"]

    return None, None


rows = []
not_found = []

for root, dirs, files in os.walk(ROOT_DIR):

    for file in files:

        if file.lower().endswith(".pdf"):

            full_path = os.path.join(root, file)

            dataset, categoria = detect_category(full_path)

            if dataset:

                rows.append({
                    "pdf": full_path,
                    "dataset": dataset,
                    "categoria": categoria
                })

            else:
                not_found.append(full_path)
                print(f"[WARNING] No se encontró categoría para: {full_path}")


with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["pdf", "dataset", "categoria"]
    )

    writer.writeheader()
    writer.writerows(rows)


print("\n==============================")
print(f"PDFs clasificados: {len(rows)}")
print(f"PDFs sin clasificar: {len(not_found)}")
print(f"CSV generado: {OUTPUT_CSV}")
print("==============================")