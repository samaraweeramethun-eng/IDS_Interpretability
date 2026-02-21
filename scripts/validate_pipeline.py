"""Quick end-to-end pipeline validation.

Runs on the 5000-row CICIDS2017 sample with tiny model sizes.
Validates: data loading → training → Integrated Gradients → Grad-CAM → SHAP.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"   # skip heavy sympy / dynamo imports

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time

# Force imports up-front so they don't hit any timeouts
print("Importing libraries...", flush=True)
t0 = time.time()
import numpy as np
import pandas as pd
import torch
print(f"  torch {torch.__version__} (CUDA: {torch.cuda.is_available()})", flush=True)
import sklearn
print(f"  sklearn {sklearn.__version__}", flush=True)
print(f"  Imports done in {time.time()-t0:.1f}s", flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from methun_research.config import CNNTransformerConfig, EnhancedConfig
from methun_research.training.cnn_trainer import train_cnn_transformer
from methun_research.training.enhanced_trainer import train_enhanced
from methun_research.interpretability.shap_runner import run_shap

os.makedirs("artifacts/shap", exist_ok=True)
DATA = "data/cicids2017/cicids2017_sample.csv"

# ── 1. CNN-Transformer ──────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("  STEP 1/3: CNN-Transformer Training", flush=True)
print("=" * 60, flush=True)

cnn_cfg = CNNTransformerConfig(
    input_path=DATA, output_dir="artifacts",
    epochs=3, batch_size=32, val_batch_size=64,
    lr=1e-3, num_workers=0,
    d_model=32, conv_channels=16, num_layers=1, num_heads=4,
    d_ff=128, dropout=0.1, undersampling_ratio=0.5,
    ig_steps=4, ig_samples=64,
)
t0 = time.time()
cnn_path = train_cnn_transformer(cnn_cfg)
print(f"\n  [OK] CNN-Transformer: {time.time()-t0:.1f}s  ->  {cnn_path}", flush=True)

# ── 2. Enhanced Transformer ─────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("  STEP 2/3: Enhanced Transformer Training", flush=True)
print("=" * 60, flush=True)

enh_cfg = EnhancedConfig(
    input_path=DATA, output_dir="artifacts",
    epochs=3, batch_size=32, val_batch_size=64,
    lr=1e-3, num_workers=0,
    d_model=32, num_layers=1, heads=4, d_ff=128,
    dropout=0.1, group_size=8,
    use_swa=False, use_mixup=False,
    val_size=0.1, test_size=0.2, undersampling_ratio=0.5,
)
t0 = time.time()
enh_path = train_enhanced(enh_cfg)
print(f"\n  [OK] Enhanced Transformer: {time.time()-t0:.1f}s  ->  {enh_path}", flush=True)

# ── 3. SHAP on CNN-Transformer ──────────────────────────────────────
if cnn_path:
    print("\n" + "=" * 60, flush=True)
    print("  STEP 3/3: SHAP Analysis (CNN-Transformer)", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()
    shap_csv = run_shap(
        checkpoint_path=cnn_path, data_path=DATA,
        output_dir="artifacts/shap",
        background_size=100, eval_size=100, eval_pool=400, chunk_size=64,
    )
    print(f"\n  [OK] SHAP: {time.time()-t0:.1f}s  ->  {shap_csv}", flush=True)

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("  [OK] FULL PIPELINE VALIDATION COMPLETE", flush=True)
print("=" * 60, flush=True)
print("\nArtifacts:", flush=True)
for root, dirs, files in os.walk("artifacts"):
    for f in sorted(files):
        fp = os.path.join(root, f)
        sz = os.path.getsize(fp) / 1024
        print(f"  {fp}  ({sz:.1f} KB)", flush=True)
