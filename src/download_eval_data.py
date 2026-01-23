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

    print("\n=== All Evaluation Data Downloaded and Cached Successfully! ===")

if __name__ == "__main__":
    download_eval_data()