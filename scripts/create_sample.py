"""Create a balanced training sample from the full CICIDS2017 dataset."""
import pandas as pd
import numpy as np
import os

data_path = "data/cicids2017/cicids2017.csv"

# Detect label column
df_head = pd.read_csv(data_path, nrows=5)
label_col = [c for c in df_head.columns if "label" in c.lower()][0]
print(f"Label col: '{label_col}'")

# Sample ~1% from each chunk for memory efficiency
rng = np.random.RandomState(42)
chunks = []
total = 0
for chunk in pd.read_csv(data_path, chunksize=100_000):
    total += len(chunk)
    n = max(1, int(len(chunk) * 0.01))
    chunks.append(chunk.sample(n=n, random_state=42))

df = pd.concat(chunks, ignore_index=True)
print(f"Sampled {len(df)} rows from {total} total")
print(df[label_col].value_counts())
binary = (df[label_col] != "BENIGN").astype(int)
print(f"\nBinary: Benign={int((binary==0).sum())}, Attack={int((binary==1).sum())}")

out_path = "data/cicids2017/cicids2017_train_sample.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved {len(df)} rows to {out_path} ({os.path.getsize(out_path)/(1024**2):.1f} MB)")
