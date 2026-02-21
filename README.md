# Methun Research — XAI for Intrusion Detection with CNN-Transformer

> Explainable AI (XAI) pipeline for network intrusion detection using CNN-Transformer hybrids on the CICIDS2017 dataset.

This project trains two Transformer-based IDS models and explains their predictions with **four XAI methods**: Integrated Gradients, Grad-CAM, SHAP, and learned feature-importance gates.

---

## VM Terminal Quick-Start (Recommended)

Run the entire pipeline from an SSH terminal on a **Google Cloud Compute Engine VM** with T4 GPU.  
Total cost: **~$0.12/hr** (Spot pricing). Full pipeline finishes in **~65 min**.

### Step 1 — Create the VM

```bash
gcloud compute instances create ids-training \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --boot-disk-size=100GB \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud
```

### Step 2 — SSH into the VM

```bash
gcloud compute ssh ids-training --zone=us-central1-a
```

### Step 3 — Run the setup script (installs everything from scratch)

This single command installs NVIDIA drivers, CUDA 12.1, Python 3, creates a venv, clones the repo, and installs all dependencies:

```bash
curl -sL https://raw.githubusercontent.com/samaraweeramethun-eng/IDS_Interpretability/main/vm_setup.sh | bash
```

After it finishes, activate the environment:

```bash
cd ~/IDS_Interpretability && source .venv/bin/activate
```

### Step 4 — Upload the dataset

Choose **any one** of these methods:

**Option A — SCP from your local machine** (run this on your **local** terminal, not the VM):

```bash
gcloud compute scp /path/to/cicids2017.csv \
    ids-training:~/IDS_Interpretability/data/cicids2017/cicids2017.csv \
    --zone=us-central1-a
```

**Option B — Google Cloud Storage bucket:**

```bash
# On the VM:
gsutil cp gs://YOUR_BUCKET/cicids2017.csv data/cicids2017/cicids2017.csv
```

**Option C — Google Drive via `gdown`:**

```bash
# On the VM:
pip install gdown
gdown --id YOUR_GDRIVE_FILE_ID -O data/cicids2017/cicids2017.csv
```

**Option D — Direct URL (if you have one):**

```bash
# On the VM:
wget -O data/cicids2017/cicids2017.csv "https://your-download-link.com/cicids2017.csv"
```

Verify the dataset is in place:

```bash
ls -lh data/cicids2017/cicids2017.csv
# Should show ~919 MB
```

### Step 5 — Run the full pipeline

```bash
python scripts/run_full_pipeline.py --data data/cicids2017/cicids2017.csv
```

This runs **everything** end-to-end:
1. **CNN-Transformer training** (25 epochs) + Integrated Gradients + Grad-CAM
2. **Enhanced Transformer training** (35 epochs) + test evaluation
3. **SHAP analysis** on both models
4. **Results summary** — prints test metrics + top features

**Optional flags:**

```bash
# Override epochs
python scripts/run_full_pipeline.py --data data/cicids2017/cicids2017.csv --cnn-epochs 30 --enh-epochs 40

# Skip one model
python scripts/run_full_pipeline.py --data data/cicids2017/cicids2017.csv --skip-enhanced

# Change batch size or training sample cap
python scripts/run_full_pipeline.py --data data/cicids2017/cicids2017.csv --batch-size 2048 --max-train-samples 1000000

# Smoke test with the 5 000-row sample (~2 min)
python scripts/run_full_pipeline.py --data data/cicids2017/cicids2017_sample.csv
```

**Run in background** (keeps running even if SSH disconnects):

```bash
nohup python scripts/run_full_pipeline.py --data data/cicids2017/cicids2017.csv > pipeline.log 2>&1 &
tail -f pipeline.log   # watch progress
```

### Step 6 — Download results

From your **local** terminal:

```bash
gcloud compute scp --recurse \
    ids-training:~/IDS_Interpretability/artifacts/ \
    ./artifacts/ \
    --zone=us-central1-a
```

### Step 7 — Delete the VM (stop charges)

```bash
gcloud compute instances delete ids-training --zone=us-central1-a
```

---

## What the pipeline produces

After a full run, the `artifacts/` directory contains:

| File | Description |
|------|-------------|
| `cnn_transformer_ids.pth` | CNN-Transformer checkpoint (weights + preprocessor + test metrics) |
| `enhanced_binary_rtids_model.pth` | Enhanced Transformer checkpoint (weights + preprocessor + test metrics) |
| `cnn_transformer_integrated_gradients.csv` | IG feature importance ranking |
| `cnn_transformer_grad_cam.csv` | Grad-CAM feature importance ranking |
| `grad_cam_importance.png` | Grad-CAM bar chart |
| `shap/shap_global_importance_attack.csv` | SHAP importance (CNN-Transformer) |
| `shap/shap_summary_attack.png` | SHAP beeswarm plot |
| `shap/shap_waterfall_attack.png` | SHAP waterfall (highest-confidence attack) |
| `shap_enhanced/shap_global_importance_attack.csv` | SHAP importance (Enhanced Transformer) |

---

## Data Split

Both models use a proper **70/10/20** three-way split:

| Split | % | Purpose |
|-------|---|---------|
| **Train** | 70% | Balanced by intelligent undersampling, capped at 500K |
| **Validation** | 10% | Early stopping / model selection |
| **Test** | 20% | Final evaluation on completely unseen data |

Test metrics (ROC-AUC, F1, Precision, Recall, Accuracy) are printed at the end of training and saved inside each checkpoint.

---

## Models

| Model | Architecture | XAI Methods |
|-------|-------------|-------------|
| **CNN-Transformer** | 1D-CNN tokenizer → Transformer encoder → CLS classifier | Integrated Gradients, Grad-CAM, SHAP |
| **Enhanced Transformer** | Feature-gated group tokenizer → Transformer encoder → Attention pooling | SHAP, learned feature gates |

## XAI Methods

| Method | Type | What it explains |
|--------|------|-----------------|
| **Integrated Gradients** | Post-hoc, gradient-based | Per-feature attribution from a zero baseline |
| **Grad-CAM** | Post-hoc, activation-based | Which CNN feature positions activate for attacks |
| **SHAP (GradientExplainer)** | Post-hoc, Shapley-value | Global & local feature importance with theory guarantees |
| **Feature Gates** | Intrinsic (built-in) | Learned sigmoid attention over raw features |

---

## Google Colab (Alternative)

If you don't want to set up a VM, use the Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samaraweeramethun-eng/IDS_Interpretability/blob/main/colab_pipeline.ipynb)

1. Click the badge → set runtime to **GPU → T4**
2. Run all cells — the notebook auto-detects Colab vs VM

---

## Project Structure

```
vm_setup.sh                   # <<< One-shot VM bootstrap (drivers + CUDA + deps)
scripts/
  run_full_pipeline.py        # <<< Headless full pipeline (both models + SHAP)
  run_pipeline.py             # CLI: single model train + SHAP
  run_shap.py                 # CLI: standalone SHAP on existing checkpoint
colab_pipeline.ipynb          # Colab/VM interactive notebook
src/methun_research/
  config.py                   # Dataclass configs for both models
  data.py                     # Preprocessing, balancing, dataloaders
  models/
    cnn_transformer.py        # CNN + Transformer hybrid
    transformer.py            # Enhanced Transformer with group tokenizer
  training/
    cnn_trainer.py             # Full training loop (CNN model)
    enhanced_trainer.py        # Full training loop (Enhanced model)
  interpretability/
    integrated_gradients.py   # IG attributions
    grad_cam.py               # Grad-CAM for CNN layers
    shap_runner.py            # SHAP GradientExplainer pipeline
  utils/
    device.py                 # GPU/CPU auto-detection
data/cicids2017/              # Place dataset CSVs here
artifacts/                    # Checkpoints + XAI outputs (auto-created)
```

## Local Setup (CPU-only)

For development or testing on a machine without a GPU:

```bash
git clone https://github.com/samaraweeramethun-eng/IDS_Interpretability.git
cd IDS_Interpretability
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Smoke test (~2 min on CPU)
python scripts/run_full_pipeline.py --data data/cicids2017/cicids2017_sample.csv
```

## Citation

```bibtex
@misc{methun2026xai_ids,
  title={Explainable CNN-Transformer Intrusion Detection on CICIDS2017},
  author={Methun Research},
  year={2026},
  url={https://github.com/samaraweeramethun-eng/IDS_Interpretability}
}
```

## License

MIT
