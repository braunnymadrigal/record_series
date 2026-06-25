import pandas as pd
from pathlib import Path

# Base directory
BASE_DIR = Path("classified_datasets_md")

# Expected dataset folders
DATASETS = ["museo", "archivo", "jurídico"]

rows = []

for dataset in DATASETS:
    dataset_dir = BASE_DIR / dataset

    if not dataset_dir.exists():
        print(f"Warning: directory not found: {dataset_dir}")
        continue

    # CATEGORY = each folder inside museo / archivo / jurídico
    for category_dir in dataset_dir.iterdir():
        if not category_dir.is_dir():
            continue

        category = category_dir.name

        # Files inside the category folder
        for file_path in category_dir.rglob("*"):
            if file_path.is_file():
                rows.append({
                    "DATASET": dataset,
                    "CATEGORY": category,
                    "FILE": file_path.name
                })

# Create DataFrame
ground_truth_df = pd.DataFrame(rows, columns=["DATASET", "CATEGORY", "FILE"])

# Save CSV
ground_truth_df.to_csv("ground_truth.csv", index=False, encoding="utf-8")

print(f"ground_truth.csv generated with {len(ground_truth_df)} rows.")