# Methun Research IDS Experiments

This repository contains experiments on the CICIDS2017 intrusion detection dataset using two model families:

- A CNN + Transformer hybrid tuned for packet-feature sequences
- An enhanced Transformer with focal loss, SWA, mixup, and additional regularization

The project also includes SHAP-based interpretability utilities to audit trained checkpoints.

## 1. Prerequisites

- Python 3.10+ (3.11 recommended)
- Git
- At least 20 GB of free disk space for data + checkpoints
- (Optional) CUDA-capable GPU for faster training

## 2. Clone and Environment Setup

```bash
# 1) Clone
git clone https://github.com/<your-org>/methun-research.git
cd methun-research

# 2) Create a virtual environment (Windows PowerShell example)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
# (Optional) install the package in editable mode if you plan to import it elsewhere
# pip install -e .
```

The `requirements.txt` file installs the project (through `-e .`) together with every dependency declared in `pyproject.toml`, so sharing that single file is enough to recreate the environment on a fresh machine.

## 3. Data Preparation

1. Download the CICIDS2017 dataset from the Canadian Institute for Cybersecurity portal.
2. Consolidate the CSV files into `data/cicids2017/cicids2017.csv` (matching the default config path).
3. Large raw splits are already listed under `data/network intrusion dataset/` for reproducibility—verify integrity before training.

> Tip: Keep the processed CSV compressed elsewhere; the working copy should remain under `data/` so scripts can resolve relative paths.

## 4. Training Entry Points

Both scripts live in `scripts/` and simply instantiate the configs in `src/methun_research/config.py`.

### 4.1 CNN + Transformer Hybrid

```bash
python scripts/train_cnn_transformer.py \
  --epochs 25 \
  --batch-size 256
```

Arguments are defined in `CNNTransformerConfig`. Override any attribute by editing the dataclass or by wiring up argparse (coming soon). Artifacts (checkpoints, logs, metrics) are saved to `artifacts/` by default.

### 4.2 Enhanced Transformer Baseline

```bash
python scripts/train_enhanced.py
```

Key differences vs. the hybrid:

- Focal loss and optional class weighting
- Mixup augmentation and gradient accumulation
- Stochastic Weight Averaging (SWA)

Adjust hyperparameters inside `EnhancedConfig` before launching jobs.

### 4.3 One-Command Pipeline (Train ➜ Test ➜ SHAP)

Prefer to train, evaluate on a held-out test split, and immediately run SHAP without juggling multiple commands? Use the pipeline helper:

```bash
python scripts/run_pipeline.py \
  --epochs 5 \
  --background 400 --eval 400 --pool 5000 \
  --chunk 128 --topk 15
```

What happens when you launch this command:

- **Train/Val/Test split** – `EnhancedConfig` now exposes `val_size` (default 0.1) and `test_size` (default 0.2). The pipeline first stratifies CICIDS2017 into train/val/test, balances only the training fold, then reports validation metrics each epoch and prints a final `Test Loss … | Test AUC …` line for the untouched test set.
- **Training overrides** – Pass any of the familiar knobs (`--epochs`, `--batch-size`, `--val-batch-size`, `--lr`, `--undersample`, `--num-workers`, `--seed`) to customize the Enhanced Transformer run without editing code.
- **SHAP controls** – Flags (`--background`, `--eval`, `--pool`, `--chunk`, `--topk`) mirror `scripts/run_shap.py`. The SHAP runner streams the dataset, so modest values (e.g., 400/400/5000) work on CPU-only machines; scale them up on beefier hardware.
- **Artifacts** – Checkpoints land in `--output` (default `artifacts/`); SHAP exports (`shap_global_importance_attack.csv`, `shap_summary_attack.png`, `shap_waterfall_attack.png`, plus logs) are written to `<output>/shap/` unless `--shap-dir` is provided.
- **Checkpoint reuse** – To skip retraining, add `--skip-training --checkpoint artifacts/enhanced_binary_rtids_model.pth`; the script will jump straight to SHAP using your existing model.

## 5. Interpretability with SHAP

After training, run SHAP to inspect feature importance:

```bash
python scripts/run_shap.py \
  --checkpoint artifacts/best_model.pt \
  --data data/cicids2017/cicids2017.csv \
  --output artifacts/shap \
  --chunk 256 --background 2000 --eval 2000 --pool 150000
```

Outputs include summary plots and serialized SHAP values for reproducibility.

## 6. Project Layout

```
artifacts/                # Checkpoints, tensorboard runs, SHAP outputs
scripts/                  # Thin CLI wrappers for training + SHAP
src/methun_research/      # Core package (configs, models, trainers, utils)
data/                     # CICIDS2017 csv + raw splits
pyproject.toml            # Python dependency spec
methun-research.ipynb     # Scratchpad for dataset exploration
```

## 7. Running on Google Colab

Need GPU acceleration or a clean slate? Follow these steps to reproduce the full training + SHAP pipeline inside [Google Colab](https://colab.research.google.com/):

1. **Create a GPU runtime** - `Runtime > Change runtime type > GPU`. (High-RAM runtimes reduce pandas/SHAP memory pressure.)
2. **Clone the repo & install dependencies** in the first cell:

  ```bash
  !git clone https://github.com/<your-org>/methun-research.git
  %cd methun-research
  !pip install -U pip
  !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  !pip install -r requirements.txt
  ```

  > The extra `pip install torch ... --index-url ...` grabs the latest CUDA wheel that matches Colab's GPU drivers; the subsequent `requirements.txt` install reuses that wheel.

3. **Make the CICIDS2017 CSV available**. Two common options:

  - *Mount Google Drive* (recommended for the full dataset):

    ```python
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    !mkdir -p data/cicids2017
    !ln -s /content/drive/MyDrive/cicids2017/cicids2017.csv data/cicids2017/cicids2017.csv
    ```

  - *Upload a quick sample* (good for smoke tests):

    ```python
    from google.colab import files
    uploaded = files.upload()
    uploaded_name = next(iter(uploaded))
    !mkdir -p data/cicids2017
    !mv "$uploaded_name" data/cicids2017/cicids2017.csv
    ```

4. **Run the pipeline** (artifacts are saved locally unless you point `--output` at Drive):

  ```bash
  !python scripts/run_pipeline.py \
     --model cnn \
     --epochs 10 \
     --batch-size 512 --val-batch-size 1024 \
     --chunk 128 --background 800 --eval 800 --pool 6000 \
     --output /content/drive/MyDrive/methun_artifacts
  ```

  - Lower the SHAP knobs (`--chunk`, `--background`, `--eval`, `--pool`) if you hit RAM limits.
  - `build_dataloaders()` now auto-disables `pin_memory` when CUDA isn't available, so CPU-only Colab runtimes also work out of the box.

## 8. Next Steps

- Add argparse overrides to the training scripts for ad-hoc sweeps
- Integrate Weights & Biases or MLflow tracking
- Extend the interpretability suite with Integrated Gradients comparisons

Feel free to open issues or PRs for enhancements, data-processing scripts, or bug fixes.
