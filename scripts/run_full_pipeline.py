#!/usr/bin/env python3
"""
run_full_pipeline.py — Run the entire IDS training + XAI pipeline end-to-end.

Designed for headless execution on a GCE VM with T4 GPU.
Runs both models, test evaluation, SHAP, IG, Grad-CAM — then prints a summary.

Usage:
    python scripts/run_full_pipeline.py --data data/cicids2017/cicids2017.csv
    python scripts/run_full_pipeline.py --data data/cicids2017/cicids2017_sample.csv  # smoke test
"""

import argparse
import gc
import os
import sys
import time

# Suppress torch dynamo warnings
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="Full IDS pipeline: train + XAI")
    p.add_argument("--data", required=True, help="Path to CICIDS2017 CSV")
    p.add_argument("--output", default="artifacts", help="Output directory")
    p.add_argument("--cnn-epochs", type=int, default=None, help="CNN-Transformer epochs (auto-detected if omitted)")
    p.add_argument("--enh-epochs", type=int, default=None, help="Enhanced Transformer epochs (auto-detected if omitted)")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--max-train-samples", type=int, default=None, help="Cap training samples (0=unlimited)")
    p.add_argument("--skip-cnn", action="store_true", help="Skip CNN-Transformer training")
    p.add_argument("--skip-enhanced", action="store_true", help="Skip Enhanced Transformer training")
    p.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def detect_mode(data_path: str) -> bool:
    """Return True if using a small sample file."""
    return "sample" in os.path.basename(data_path).lower()


def print_header(title: str):
    w = 64
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}\n")


def print_gpu_status():
    import torch
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        peak = torch.cuda.max_memory_allocated(0)
        print(f"  GPU VRAM: {(total-free)/1024**3:.1f}/{total/1024**3:.1f} GB used, peak {peak/1024**3:.2f} GB")
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"  System RAM: {ram.used/1024**3:.1f}/{ram.total/1024**3:.1f} GB ({ram.percent}%)")
    except ImportError:
        pass


def main():
    args = parse_args()
    is_sample = detect_mode(args.data)
    os.makedirs(args.output, exist_ok=True)

    # Validate dataset exists
    if not os.path.exists(args.data):
        print(f"ERROR: Dataset not found at {args.data}")
        print("Upload your dataset first. See vm_setup.sh for instructions.")
        sys.exit(1)

    data_size_mb = os.path.getsize(args.data) / 1024**2
    print_header("IDS Interpretability — Full Pipeline")
    print(f"  Dataset:    {args.data} ({data_size_mb:.0f} MB)")
    print(f"  Mode:       {'SAMPLE (smoke test)' if is_sample else 'FULL DATASET (production)'}")
    print(f"  Output:     {args.output}")

    import torch
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  CUDA:       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"  VRAM:       {total/1024**3:.1f} GB total, {free/1024**3:.1f} GB free")
    print()

    # ── Import after environment check ───────────────────────────────
    from methun_research.config import CNNTransformerConfig, EnhancedConfig
    from methun_research.training.cnn_trainer import train_cnn_transformer
    from methun_research.training.enhanced_trainer import train_enhanced
    from methun_research.interpretability.shap_runner import run_shap

    # ── Build configs ────────────────────────────────────────────────
    batch = args.batch_size or (64 if is_sample else 1024)
    val_batch = batch * 2
    max_train = args.max_train_samples if args.max_train_samples is not None else (0 if is_sample else 500_000)

    cnn_cfg = CNNTransformerConfig(
        input_path=args.data,
        output_dir=args.output,
        epochs=args.cnn_epochs or (5 if is_sample else 25),
        batch_size=batch,
        val_batch_size=val_batch,
        lr=1.5e-3 if is_sample else 2e-3,
        num_workers=2,
        d_model=64 if is_sample else 192,
        conv_channels=32 if is_sample else 96,
        num_layers=1 if is_sample else 3,
        num_heads=4 if is_sample else 8,
        d_ff=256 if is_sample else 768,
        dropout=0.2,
        val_size=0.1,
        ig_steps=8 if is_sample else 32,
        ig_samples=128 if is_sample else 512,
        max_train_samples=max_train,
        random_state=args.seed,
    )

    enh_cfg = EnhancedConfig(
        input_path=args.data,
        output_dir=args.output,
        epochs=args.enh_epochs or (5 if is_sample else 35),
        batch_size=batch,
        val_batch_size=val_batch,
        lr=2e-3,
        num_workers=2,
        d_model=64 if is_sample else 160,
        num_layers=2 if is_sample else 4,
        heads=4 if is_sample else 10,
        d_ff=256 if is_sample else 640,
        dropout=0.15,
        group_size=8,
        use_swa=not is_sample,
        use_mixup=not is_sample,
        max_train_samples=max_train,
        random_state=args.seed,
    )

    print(f"  CNN config:      {cnn_cfg.epochs} epochs, d={cnn_cfg.d_model}, batch={cnn_cfg.batch_size}")
    print(f"  Enhanced config: {enh_cfg.epochs} epochs, d={enh_cfg.d_model}, batch={enh_cfg.batch_size}")
    print(f"  Max train samples: {max_train or 'unlimited'}")

    pipeline_start = time.time()
    cnn_path = None
    enh_path = None

    # ── Step 1: CNN-Transformer ──────────────────────────────────────
    if not args.skip_cnn:
        print_header("Step 1/4 — CNN-Transformer Training")
        print_gpu_status()
        t0 = time.time()
        cnn_path = train_cnn_transformer(cnn_cfg)
        elapsed = (time.time() - t0) / 60
        print(f"\n  ✓ CNN-Transformer done in {elapsed:.1f} min → {cnn_path}")
        print_gpu_status()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        # Try to find existing checkpoint
        cnn_ckpt = os.path.join(args.output, "cnn_transformer_ids.pth")
        if os.path.exists(cnn_ckpt):
            cnn_path = cnn_ckpt
            print(f"\n  [skip-cnn] Using existing checkpoint: {cnn_path}")
        else:
            print("\n  [skip-cnn] No existing checkpoint found.")

    # ── Step 2: Enhanced Transformer ─────────────────────────────────
    if not args.skip_enhanced:
        print_header("Step 2/4 — Enhanced Transformer Training")
        print_gpu_status()
        t0 = time.time()
        enh_path = train_enhanced(enh_cfg)
        elapsed = (time.time() - t0) / 60
        print(f"\n  ✓ Enhanced Transformer done in {elapsed:.1f} min → {enh_path}")
        print_gpu_status()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        enh_ckpt = os.path.join(args.output, "enhanced_binary_rtids_model.pth")
        if os.path.exists(enh_ckpt):
            enh_path = enh_ckpt
            print(f"\n  [skip-enhanced] Using existing checkpoint: {enh_path}")
        else:
            print("\n  [skip-enhanced] No existing checkpoint found.")

    # ── Step 3: SHAP ─────────────────────────────────────────────────
    if not args.skip_shap:
        shap_bg = 200 if is_sample else 2000
        shap_eval = 200 if is_sample else 2000
        shap_pool = 500 if is_sample else 150_000

        if cnn_path:
            print_header("Step 3/4 — SHAP Analysis (CNN-Transformer)")
            shap_dir = os.path.join(args.output, "shap")
            os.makedirs(shap_dir, exist_ok=True)
            t0 = time.time()
            run_shap(
                checkpoint_path=cnn_path,
                data_path=args.data,
                output_dir=shap_dir,
                background_size=shap_bg,
                eval_size=shap_eval,
                eval_pool=shap_pool,
                chunk_size=256,
            )
            print(f"\n  ✓ SHAP (CNN) done in {(time.time()-t0)/60:.1f} min")
            gc.collect()

        if enh_path:
            print_header("Step 3b/4 — SHAP Analysis (Enhanced Transformer)")
            shap_enh_dir = os.path.join(args.output, "shap_enhanced")
            os.makedirs(shap_enh_dir, exist_ok=True)
            t0 = time.time()
            run_shap(
                checkpoint_path=enh_path,
                data_path=args.data,
                output_dir=shap_enh_dir,
                background_size=shap_bg,
                eval_size=shap_eval,
                eval_pool=shap_pool,
                chunk_size=256,
            )
            print(f"\n  ✓ SHAP (Enhanced) done in {(time.time()-t0)/60:.1f} min")
            gc.collect()

    # ── Step 4: Summary ──────────────────────────────────────────────
    print_header("Step 4/4 — Results Summary")
    total_min = (time.time() - pipeline_start) / 60

    # Load and display test metrics from checkpoints
    import pandas as pd

    def load_test_metrics(ckpt_path, name):
        if not ckpt_path or not os.path.exists(ckpt_path):
            return None
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = ckpt.get("test_metrics")
        if not m:
            return None
        return {
            "Model": name,
            "ROC-AUC": m["auc_roc"],
            "PR-AUC": m["auc_pr"],
            "F1": m["f1_score"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "Accuracy": m["accuracy"],
        }

    rows = []
    r = load_test_metrics(cnn_path, "CNN-Transformer")
    if r:
        rows.append(r)
    r = load_test_metrics(enh_path, "Enhanced Transformer")
    if r:
        rows.append(r)

    if rows:
        df = pd.DataFrame(rows).set_index("Model")
        print("  Held-out Test Set Results (20% of data, never seen during training):\n")
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    else:
        print("  No test metrics available.")

    # List top XAI features
    for label, path, col in [
        ("Integrated Gradients", f"{args.output}/cnn_transformer_integrated_gradients.csv", "avg_abs_integrated_grad"),
        ("Grad-CAM", f"{args.output}/cnn_transformer_grad_cam.csv", "grad_cam_importance"),
        ("SHAP (CNN)", f"{args.output}/shap/shap_global_importance_attack.csv", "mean_abs_shap"),
        ("SHAP (Enhanced)", f"{args.output}/shap_enhanced/shap_global_importance_attack.csv", "mean_abs_shap"),
    ]:
        if os.path.exists(path):
            top = pd.read_csv(path).sort_values(col, ascending=False).head(10)
            features = ", ".join(top.iloc[:, 0].tolist())
            print(f"\n  Top-10 {label}:")
            print(f"    {features}")

    # List artifacts
    print(f"\n\n  Generated artifacts:")
    for root, dirs, files in os.walk(args.output):
        for f in sorted(files):
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            if sz > 1024 * 1024:
                print(f"    {fp}  ({sz/1024**2:.1f} MB)")
            else:
                print(f"    {fp}  ({sz/1024:.1f} KB)")

    print(f"\n  Total pipeline time: {total_min:.1f} min")
    print(f"\n{'='*64}")
    print("  Pipeline complete! Download artifacts with:")
    print(f"    scp -r USER@VM_IP:~/IDS_Interpretability/{args.output}/ .")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
