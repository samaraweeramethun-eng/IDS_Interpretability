"""Train both models on the sample dataset (CPU-optimized settings)."""
import sys
import os
import time

# ── CNN-Transformer ─────────────────────────────────────────────────
def train_cnn():
    from methun_research.config import CNNTransformerConfig
    from methun_research.training.cnn_trainer import train_cnn_transformer

    cfg = CNNTransformerConfig(
        input_path="data/cicids2017/cicids2017_train_sample.csv",
        output_dir="artifacts",
        epochs=5,
        batch_size=64,
        val_batch_size=128,
        lr=1.5e-3,
        num_workers=0,          # 0 workers = main-process loading (avoids Windows multiprocessing overhead)
        d_model=64,             # small for CPU speed
        conv_channels=32,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        dropout=0.2,
        ig_steps=8,
        ig_samples=128,
    )
    print("=" * 60)
    print("  CNN-Transformer Training (CPU-optimised)")
    print("=" * 60)
    t0 = time.time()
    path = train_cnn_transformer(cfg)
    elapsed = time.time() - t0
    print(f"\nCNN-Transformer done in {elapsed/60:.1f} min  ->  {path}")
    return path


# ── Enhanced Transformer ────────────────────────────────────────────
def train_enhanced():
    from methun_research.config import EnhancedConfig
    from methun_research.training.enhanced_trainer import train_enhanced as _train

    cfg = EnhancedConfig(
        input_path="data/cicids2017/cicids2017_train_sample.csv",
        output_dir="artifacts",
        epochs=5,
        batch_size=64,
        val_batch_size=128,
        lr=2e-3,
        num_workers=0,
        d_model=64,
        num_layers=2,
        heads=4,
        d_ff=256,
        dropout=0.15,
        group_size=8,
        use_swa=False,          # disable SWA to save memory on CPU
        use_mixup=True,
        val_size=0.1,
        test_size=0.2,
    )
    print("\n" + "=" * 60)
    print("  Enhanced Transformer Training (CPU-optimised)")
    print("=" * 60)
    t0 = time.time()
    path = _train(cfg)
    elapsed = time.time() - t0
    print(f"\nEnhanced Transformer done in {elapsed/60:.1f} min  ->  {path}")
    return path


if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)

    which = sys.argv[1] if len(sys.argv) > 1 else "cnn"

    if which in ("cnn", "both"):
        cnn_path = train_cnn()

    if which in ("enhanced", "both"):
        enh_path = train_enhanced()

    print("\n✓ Training complete!")
