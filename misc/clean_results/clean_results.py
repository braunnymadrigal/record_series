import pandas as pd
#import re

results_df = pd.read_csv("results.csv", encoding="utf-8")
classifications_df = pd.read_csv("classifications.csv", encoding="utf-8")

# Drop useless columns
results_df = results_df.drop(columns=["expected_label", "provider", "model_group", "model_access_type", "model_api_id", "normalized_response", "is_valid_category", "input_tokens", "output_tokens", "total_tokens", "error_message", "timestamp"])

# Rename columns
results_df = results_df.rename(columns={"model_name": "llm", "response": "reply", "elapsed_time" : "time", "estimated_cost" : "cost"})

# Clean document name
results_df['document'] = results_df['document'].str.extract(r'.*\\.*\\(.*)')

# Column reply: Remove spaces
results_df['reply'] = results_df['reply'].str.replace(" ", "")
results_df['reply'] = results_df['reply'].str.replace("\n", "")
results_df['reply'] = results_df['reply'].str.replace("\t", "")
results_df['reply'] = results_df['reply'].str.replace("\r", "")

# Column reply: To lower case
results_df['reply'] = results_df['reply'].str.lower()

# Column reply: New categories
results_df['reply'] = results_df['reply'].str.replace("informesdelabores", "informes")
results_df['reply'] = results_df['reply'].str.replace("informestécnicosproducidos", "informes")
results_df['reply'] = results_df['reply'].str.replace("informesdeinspección", "informes")
results_df['reply'] = results_df['reply'].str.replace("informesdelíndicededesarrolloarchivístico", "informes")
results_df['reply'] = results_df['reply'].str.replace("controlesdepréstamodedocumentos", "controles")

# Column document: Removes rows containing a document not in classifications.csv
valid_documents_list = classifications_df['document'].values.tolist()
results_df = results_df[results_df.document.isin(valid_documents_list)]

# Column expected: Create new row expected with expected category for each document
expected_list = list()
documents_list = results_df['document'].values.tolist()
valid_categories_list = classifications_df['category'].values.tolist()
valid_document_category_dict = dict(zip(valid_documents_list, valid_categories_list))
for i in range(len(documents_list)):
    expected_list.append(valid_document_category_dict[documents_list[i]])
results_df['expected'] = expected_list

# Create csv file
results_df.to_csv("results_cleaned.csv", index=False)