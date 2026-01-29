import torch
from model import DistilBertStudent
from dataset import DistillationDataset

def sanity_check():
    print("--- Testing Model Architecture ---")
    # Initialize student
    try:
        student_wrapper = DistilBertStudent()
        # checking if manual mapping code works
        student_wrapper.initialize_from_teacher()
        print("Student initialized and weights loaded from Teacher!")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return

    print("\n--- Testing Data Pipeline ---")
    try:
        # Load just a subset of data
        data_handler = DistillationDataset(subset_size=50) 
        
        # Get dataloader
        loader = data_handler.get_data_loader(batch_size=2)
        print(f"Data loader created. Dataset size: {len(data_handler.dataset)}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    print("\n--- Testing Forward Pass ---")
    try:
        # take one batch
        batch = next(iter(loader))
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        print(f"Input shape: {input_ids.shape}") 

        # Pass through the model
        logits, hidden_states = student_wrapper(input_ids, attention_mask)

        print(f"Forward pass successful!")
        print(f"Logits shape: {logits.shape}")  
        
        # Checking the right number of hidden states 
        print(f"Hidden states collected: {len(hidden_states)}") 
        if len(hidden_states) == 7:
            print("Hidden states count matches paper (Embeddings + 6 Layers)")
        else:
            print(f"Warning: Expected 7 hidden states, got {len(hidden_states)}")

        print("\nTesting done successfully!")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    sanity_check()