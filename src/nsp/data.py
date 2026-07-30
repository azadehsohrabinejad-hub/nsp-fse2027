import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

class NSPSequenceDataset(Dataset):
    def __init__(self, seq_dir):
        self.seq_dir = seq_dir
        self.file_list = []
        
        # Find all JSON sequence files
        for root, dirs, files in os.walk(seq_dir):
            for file in files:
                if file.endswith(".json"):
                    self.file_list.append(os.path.join(root, file))
                    
        print(f"Found {len(self.file_list)} sequences.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract the sequence Y and semantic label
        Y = data.get("Y", [])
        semantic_label = data.get("semantic_label", None)
        
        # Convert to PyTorch tensor
        # Shape: (T, 21) where T is sequence length
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        
        # For now, we return the sequence and the file name for tracking
        return Y_tensor, data.get("run_id", "unknown")

# Function to test the loader
if __name__ == "__main__":
    seq_dir = r"D:\raz\razieh\data\nsp_sequences"
    dataset = NSPSequenceDataset(seq_dir)
    
    # Get the first item
    seq, run_id = dataset[0]
    print(f"\nSample Run ID: {run_id}")
    print(f"Sequence Shape: {seq.shape}")
    print(f"First step vector (y_1): {seq[0]}")