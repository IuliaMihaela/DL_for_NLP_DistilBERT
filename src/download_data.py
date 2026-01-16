import os
from datasets import load_dataset

def setup_data_folder(data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    print("Downloading WikiText-2 subset...")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    for split in ['train', 'validation', 'test']:
        file_path = os.path.join(data_dir, f"{split}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            for item in dataset[split]:
                if len(item['text'].strip()) > 0:
                    f.write(item['text'] + "\n")
    
    print(f"Data folder ready! Files saved to: {os.path.abspath(data_dir)}")

if __name__ == "__main__":
    setup_data_folder()