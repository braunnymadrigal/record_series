import pandas as pd
import unicodedata
import re
from pathlib import Path

RESULTS_CSV = "results_all.csv"
GROUND_TRUTH_CSV = "ground_truth.csv"

OUTPUT_CSV = "results_evaluated.csv"

def normalize(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # quitar tildes
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    # espacios múltiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_pdf_name(path_text):

    if pd.isna(path_text):
        return ""

    filename = Path(str(path_text)).name

    # quitar .pdf
    filename = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)

    # quitar duplicados tipo _1 _2
    filename = re.sub(r"_\d+$", "", filename)

    return normalize(filename)


results_df = pd.read_csv(RESULTS_CSV)

ground_truth_df = pd.read_csv(GROUND_TRUTH_CSV)


ground_truth_df["pdf_normalized"] = (
    ground_truth_df["pdf"]
    .apply(extract_pdf_name)
)

ground_truth_df["categoria_normalized"] = (
    ground_truth_df["categoria"]
    .apply(normalize)
)



ground_truth_map = dict(
    zip(
        ground_truth_df["pdf_normalized"],
        ground_truth_df["categoria_normalized"]
    )
)


FLEXIBLE_TERMS = {
    "minutas de reuniones": [
        "minuta",
        "minutas",
        "reunion",
        "reuniones"
    ],

    "instrumentos tecnicos archivisticos": [
        "instrumento",
        "archivistico",
        "archivisticos"
    ],

    "expedientes de tramites laborales": [
        "tramites laborales",
        "laborales"
    ]
}


expected_labels = []
exact_match = []
regex_match = []

for _, row in results_df.iterrows():


    pdf_name = extract_pdf_name(row["document"])

    expected = ground_truth_map.get(pdf_name, "")

    expected_labels.append(expected)

    response = normalize(row.get("response", ""))

    exact = 1 if response == expected else 0

    exact_match.append(exact)

    partial = 0

    if expected:

        # match exacto dentro del texto
        pattern = r"\b" + re.escape(expected) + r"\b"

        if re.search(pattern, response):
            partial = 1

        else:

            # buscar términos flexibles
            synonyms = FLEXIBLE_TERMS.get(expected, [])

            for term in synonyms:

                if re.search(r"\b" + re.escape(term) + r"\b", response):
                    partial = 1
                    break

    regex_match.append(partial)


results_df["expected_label"] = expected_labels

results_df["response_normalized"] = (
    results_df["response"]
    .apply(normalize)
)

results_df["exact_match"] = exact_match

results_df["regex_match"] = regex_match

results_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print("\n===================================")
print(f"Archivo generado: {OUTPUT_CSV}")
print("===================================")

print("\nResumen:")

print(f"Exact matches: {sum(exact_match)}")
print(f"Regex/Flexible matches: {sum(regex_match)}")
print(f"Total: {len(results_df)}")