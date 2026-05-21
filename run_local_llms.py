import time
import pymupdf4llm
import torch
import pandas as pd
import gc
from pathlib import Path
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText

# Constants
DEVICE = "cuda"
MAX_NEW_TOKENS = 64

# DATAFRAMES
LLMS_DF = pd.read_csv("llms.csv", encoding="utf-8")
RECORDS_SERIES_DF = pd.read_csv("records_series.csv", encoding="utf-8")
results_df = pd.read_csv("results.csv", encoding="utf-8")

# LLMs engine and model
engine = None
model = None

# LLMs prompt
prompt = ""

# Methods
def load_engine(llm_name, llm_engine):
    global engine
    if llm_engine == "processor":
        engine = AutoProcessor.from_pretrained(llm_name)
    elif llm_engine == "tokenizer":
        engine = AutoTokenizer.from_pretrained(llm_name)

def load_model(llm_name, llm_type):
    global model
    if llm_type == "AutoModelForCausalLM":
        model = AutoModelForCausalLM.from_pretrained(llm_name, dtype=torch.float16).to(DEVICE)
    elif llm_type == "AutoModelForImageTextToText":
        model = AutoModelForImageTextToText.from_pretrained(llm_name, dtype=torch.float16).to(DEVICE)

def run_spec_llm_with_spec_prompt(llm_name, current_prompt):
    global prompt
    prompt = current_prompt
    
    start_time = time.perf_counter()

    inputs = engine.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE)
    


    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )

    response = engine.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time)

    del inputs
    del outputs
    gc.collect()
    torch.cuda.empty_cache()
    
    return response, elapsed_time

def create_prompt(categories, file):
    current_prompt = [
        {
            "role": "user",
            "content": f"""
                        Por favor, realice la siguiente tarea actuando como un archivista experto.\n
                        Categorice el documento en una categoría perteneciente a las series documentales. 
                        Dé como respuesta una única serie documental dentro de la lista de series documentales disponibles. 
                        No produzca comentarios adicionales.\n
                       La lista de series documentales disponibles es: {categories}.\n
                        El documento es:\n\n\n {file}"""
        },
    ]
    return current_prompt

def main():
    global model
    global engine

    llms_names = LLMS_DF["LLM_NAME"].tolist()
    llms_types = LLMS_DF["LLM_TYPE"].tolist()
    llms_engines = LLMS_DF["LLM_ENGINE"].tolist()
    
    for i in range(len(llms_names)):
        llm_name = llms_names[i]
        llm_type = llms_types[i]
        llm_engine = llms_engines[i]
        
        load_engine(llm_name, llm_engine)
        load_model(llm_name, llm_type)
        
        datasets = RECORDS_SERIES_DF["DATASET"].tolist()
        datasets = list(dict.fromkeys(datasets))
        
        for dataset in datasets:
            categories = RECORDS_SERIES_DF.loc[RECORDS_SERIES_DF["DATASET"] == dataset]
            categories = categories["CATEGORY"].tolist()
            
            files = [
                f for f in Path(f"datasets/{dataset}").iterdir()
                if f.is_file()
            ]
            
            for file_path in files:
                document = pymupdf4llm.to_markdown(str(file_path))
                current_prompt = create_prompt(categories, document)
                response, elapsed_time = run_spec_llm_with_spec_prompt(llm_name, current_prompt)
                response = response.lower()
                
                if response not in categories:
                    response = ""
                
                results_df.loc[len(results_df)] = [dataset, file_path, llm_name, response, elapsed_time]
                
                print()
                print([dataset, file_path, llm_name, response, elapsed_time])
                print()
        
        if model is not None:
            model.cpu()

        del model
        del engine

        model = None
        engine = None

        gc.collect()

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    results_df.to_csv("results.csv", index=False)

# Invoke
if __name__ == "__main__":
    main()
