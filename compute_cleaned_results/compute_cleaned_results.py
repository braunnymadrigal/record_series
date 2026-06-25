import pandas as pd

results_df = pd.read_csv("all_results.csv", encoding="utf-8")

# Drop useless columns
results_df = results_df.drop(
    columns=[
        "expected_label",
        "provider",
        "model_group",
        "model_api_id",
        "is_valid_category",
        "normalized_response",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "error_message",
        "timestamp"
    ]
)

# Rename columns
results_df = results_df.rename(
    columns={
        "dataset": "DATASET",
        "document": "DOCUMENT",
        "model_name": "LLM_NAME",
        "model_access_type": "TYPE",
        "response": "REPLY",
        "elapsed_time" : "TIME",
        "estimated_cost" : "COST"
    }
)

# Clean document name
results_df['DOCUMENT'] = results_df['DOCUMENT'].str.extract(r'.*\\.*\\(.*)')

# Column reply: To lower case
results_df['REPLY'] = results_df['REPLY'].str.lower()

# Column reply: Remove everything that is not a normal character
results_df['REPLY'] = results_df['REPLY'].str.replace("[^a-záéíóúñ]", "")

# Column reply: New categories
results_df['REPLY'] = results_df['REPLY'].str.replace("informesdelabores", "informes")
results_df['REPLY'] = results_df['REPLY'].str.replace("informestécnicosproducidos", "informes")
results_df['REPLY'] = results_df['REPLY'].str.replace("informesdeinspección", "informes")
results_df['REPLY'] = results_df['REPLY'].str.replace("informesdelíndicededesarrolloarchivístico", "informes")
results_df['REPLY'] = results_df['REPLY'].str.replace("controlesdepréstamodedocumentos", "controles")

# Create csv file
results_df.to_csv("cleaned_results.csv", index=False, encoding="utf-8")