# Methun Research — XAI for Intrusion Detection with CNN-Transformer

> Explainable AI (XAI) pipeline for network intrusion detection using CNN-Transformer hybrids on the CICIDS2017 dataset.

This project trains two Transformer-based IDS models and explains their predictions with **four XAI methods**: Integrated Gradients, Grad-CAM, SHAP, and learned feature-importance gates.

## Quick Start — Google Colab (Recommended)

The fastest way to run everything is with the **ready-made Colab notebook** — no local setup needed.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samaraweeramethun-eng/IDS_Interpretability/blob/main/colab_pipeline.ipynb)

1. Click the badge above (or upload `colab_pipeline.ipynb` manually to Colab)
2. Set runtime to **GPU → T4**: *Runtime → Change runtime type → T4 GPU*
3. Run all cells — the notebook walks through setup, training, and XAI automatically

> **Estimated time on T4 GPU:** ~10 min (sample) / ~45 min (full 2.8 M row dataset)

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

## Project Structure

```
colab_pipeline.ipynb          # <<< One-click Colab notebook (start here)
scripts/
  run_pipeline.py             # CLI: train + SHAP in one command
  run_shap.py                 # CLI: standalone SHAP on existing checkpoint
  train_cnn_transformer.py    # CLI: train CNN-Transformer only
  train_enhanced.py           # CLI: train Enhanced Transformer only
  validate_pipeline.py        # Quick local smoke test (5 K rows)
  train_cpu.py                # CPU-optimised training (small model)
src/methun_research/
  config.py                   # Dataclass configs for both models
  data.py                     # Preprocessing, balancing, dataloaders
  models/
    cnn_transformer.py        # CNN + Transformer hybrid
    transformer.py            # Enhanced Transformer with group tokenizer
  training/
    cnn_trainer.py            # Full training loop (CNN model)
    enhanced_trainer.py       # Full training loop (Enhanced model)
  interpretability/
    integrated_gradients.py   # IG attributions
    grad_cam.py               # Grad-CAM for CNN layers
    shap_runner.py            # SHAP GradientExplainer pipeline
  utils/
    device.py                 # GPU/CPU auto-detection
data/cicids2017/              # Place dataset CSVs here
artifacts/                    # Checkpoints + XAI outputs (auto-created)
```

## Local Setup

### Prerequisites

- Python 3.10+
- 16 GB+ RAM (32 GB recommended for the full dataset)
- (Optional) CUDA GPU — without it, use the sample dataset or Colab

### Install

```bash
git clone https://github.com/samaraweeramethun-eng/IDS_Interpretability.git
cd IDS_Interpretability

python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Data

1. Download CICIDS2017 from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Place the consolidated CSV at `data/cicids2017/cicids2017.csv`
3. A 5 000-row sample is included at `data/cicids2017/cicids2017_sample.csv` for smoke tests

### Train + XAI (one command)

```bash
# Full pipeline: train CNN-Transformer then run SHAP
python scripts/run_pipeline.py \
  --model cnn \
  --data data/cicids2017/cicids2017.csv \
  --epochs 25 \
  --batch-size 256 \
  --background 2000 --eval 2000 --pool 150000
```

### Train models individually

```bash
# CNN-Transformer (generates IG + Grad-CAM automatically)
python scripts/run_pipeline.py --model cnn --epochs 25

# Enhanced Transformer
python scripts/run_pipeline.py --model enhanced --epochs 35
```

### Run SHAP on an existing checkpoint

```bash
python scripts/run_shap.py \
  --checkpoint artifacts/cnn_transformer_ids.pth \
  --data data/cicids2017/cicids2017.csv \
  --output artifacts/shap
```

### Quick smoke test (CPU, ~1 min)

```bash
python scripts/validate_pipeline.py
```

This runs all 3 steps (CNN training → Enhanced training → SHAP) on the 5 000-row sample with tiny model sizes to verify everything works.

## Google Colab — Step-by-Step

If you prefer running cells manually instead of using the notebook:

### 1. Create a GPU runtime

*Runtime → Change runtime type → T4 GPU*

### 2. Clone & install

```python
!git clone https://github.com/samaraweeramethun-eng/IDS_Interpretability.git
%cd IDS_Interpretability
!pip install -q torch --index-url https://download.pytorch.org/whl/cu121
!pip install -q -r requirements.txt
```

### 3. Upload or mount data

**Option A — Google Drive (recommended for full dataset):**

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p data/cicids2017
!ln -sf /content/drive/MyDrive/cicids2017/cicids2017.csv data/cicids2017/cicids2017.csv
```

**Option B — Direct upload:**

```python
from google.colab import files
uploaded = files.upload()
!mkdir -p data/cicids2017
!mv "{list(uploaded.keys())[0]}" data/cicids2017/cicids2017.csv
```

**Option C — Use included sample (no upload needed):**

The 5 000-row sample at `data/cicids2017/cicids2017_sample.csv` is already in the repo.

### 4. Train CNN-Transformer

```python
import os, warnings
os.environ['TORCHDYNAMO_DISABLE'] = '1'
warnings.filterwarnings('ignore')

from methun_research.config import CNNTransformerConfig
from methun_research.training.cnn_trainer import train_cnn_transformer

cfg = CNNTransformerConfig(
    input_path='data/cicids2017/cicids2017.csv',
    epochs=25, batch_size=256, num_workers=2,
)
cnn_path = train_cnn_transformer(cfg)
```

### 5. Train Enhanced Transformer

```python
from methun_research.config import EnhancedConfig
from methun_research.training.enhanced_trainer import train_enhanced

cfg = EnhancedConfig(
    input_path='data/cicids2017/cicids2017.csv',
    epochs=35, batch_size=512, num_workers=2,
)
enh_path = train_enhanced(cfg)
```

### 6. Run SHAP

```python
from methun_research.interpretability.shap_runner import run_shap

shap_csv = run_shap(
    checkpoint_path=cnn_path,
    data_path='data/cicids2017/cicids2017.csv',
    output_dir='artifacts/shap',
    background_size=2000, eval_size=2000, eval_pool=150000,
)
```

### 7. View results

```python
import pandas as pd
from IPython.display import display, Image

# Feature rankings
for f in ['artifacts/cnn_transformer_integrated_gradients.csv',
          'artifacts/cnn_transformer_grad_cam.csv',
          'artifacts/shap/shap_global_importance_attack.csv']:
    if os.path.exists(f):
        print(f'\n--- {os.path.basename(f)} ---')
        display(pd.read_csv(f).head(15))

# Plots
for f in ['artifacts/grad_cam_importance.png',
          'artifacts/shap/shap_summary_attack.png',
          'artifacts/shap/shap_waterfall_attack.png']:
    if os.path.exists(f):
        display(Image(filename=f, width=700))
```

### 8. Save to Google Drive

```python
import shutil
shutil.copytree('artifacts', '/content/drive/MyDrive/methun_artifacts', dirs_exist_ok=True)
```

## Outputs

After a full run, the `artifacts/` directory contains:

| File | Description |
|------|-------------|
| `cnn_transformer_ids.pth` | CNN-Transformer checkpoint (weights + preprocessor + config) |
| `enhanced_binary_rtids_model.pth` | Enhanced Transformer checkpoint |
| `cnn_transformer_integrated_gradients.csv` | IG feature importance ranking |
| `cnn_transformer_grad_cam.csv` | Grad-CAM feature importance ranking |
| `grad_cam_importance.png` | Grad-CAM bar chart |
| `shap/shap_global_importance_attack.csv` | SHAP feature importance ranking |
| `shap/shap_summary_attack.png` | SHAP beeswarm summary plot |
| `shap/shap_waterfall_attack.png` | SHAP waterfall for highest-confidence attack |

## Citation

If you use this code in your research, please cite:

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
