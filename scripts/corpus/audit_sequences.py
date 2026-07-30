import os
import json
import numpy as np

# 1. Define path
seq_dir = r"D:\raz\razieh\data\nsp_sequences"

print("Auditing generated NSP sequences...")
lengths = []

# 2. Read all sequence files
for root, dirs, files in os.walk(seq_dir):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract sequence length (T)
            seq_len = data.get("sequence_length", 0)
            lengths.append(seq_len)

# 3. Analyze the lengths
lengths = np.array(lengths)
print("\n--- Sequence Length Audit ---")
print(f"Total sequences checked: {len(lengths)}")
print(f"Min sequence length (T_min): {lengths.min()}")
print(f"Max sequence length (T_max): {lengths.max()}")
print(f"Average sequence length (T_avg): {lengths.mean():.2f}")

print("\nDistribution of sequence lengths:")
unique, counts = np.unique(lengths, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  T = {u}: {c} sequences")

# 4. Conclusion
if lengths.max() <= 1:
    print("\nWARNING: All sequences have T=1. Data is single-step and cannot be used for Transition/Drift modeling.")
else:
    print("\nSUCCESS: Multi-step sequences exist. Data is theoretically suitable for NSP.")