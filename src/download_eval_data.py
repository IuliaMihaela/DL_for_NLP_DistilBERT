import evaluate
from datasets import load_dataset

def download_eval_data():
    print("=== Starting Download of Evaluation Datasets ===")
    
    # 1. GLUE Tasks (Table 1)
    glue_tasks = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    print(f"\n>> Downloading GLUE tasks: {glue_tasks}")
    for task in glue_tasks:
        print(f"   - Fetching glue/{task}...")
        load_dataset("glue", task)
        evaluate.load("glue", task) # caches the metric script

    # 2. IMDb (Table 2)
    print("\n>> Downloading IMDb (Table 2)...")
    load_dataset("imdb")
    evaluate.load("accuracy")

    # 3. SQuAD (Table 2)
    print("\n>> Downloading SQuAD v1.1 (Table 2)...")
    load_dataset("squad")
    evaluate.load("squad")

    # 4. WikiText-103 (For Paper Training)
    # This fixes the "ValueError: Couldn't find cache" error
    print("\n>> Downloading WikiText-103 (For Paper Training)...")
    # We download the 'raw' version which matches what we use in paper_corpus.py
    #load_dataset("wikitext", "wikitext-103-raw-v1", revision="main")
    load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")


    print("\n=== All Evaluation Data Downloaded and Cached Successfully! ===")

if __name__ == "__main__":
    download_eval_data()