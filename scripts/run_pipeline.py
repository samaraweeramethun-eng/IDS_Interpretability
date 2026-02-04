import argparse
import os
from pathlib import Path

from methun_research.config import CNNTransformerConfig, EnhancedConfig
from methun_research.training import train_cnn_transformer, train_enhanced
from methun_research.interpretability import run_shap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the enhanced transformer and immediately run SHAP interpretability."
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "enhanced"],
        default="cnn",
        help="Model to train before running SHAP (default: cnn).",
    )
    parser.add_argument("--data", default="data/cicids2017/cicids2017.csv", help="Path to CICIDS2017 CSV")
    parser.add_argument("--output", default="artifacts", help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=1024, help="Validation batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate if provided")
    parser.add_argument("--undersample", type=float, default=None, help="Override undersampling ratio")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader worker count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training and SHAP sampling")
    parser.add_argument("--checkpoint", help="Optional existing checkpoint to skip training")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the training phase and use --checkpoint for SHAP",
    )
    parser.add_argument("--shap-dir", help="Directory to store SHAP outputs (default: <output>/shap)")
    parser.add_argument("--chunk", type=int, default=128, help="Batch size (rows) for SHAP evaluation chunks")
    parser.add_argument("--background", type=int, default=500, help="Number of background samples for SHAP")
    parser.add_argument("--eval", type=int, default=500, help="Number of evaluation samples for SHAP")
    parser.add_argument("--pool", type=int, default=10_000, help="Candidate pool size sampled before SHAP")
    parser.add_argument("--topk", type=int, default=20, help="Waterfall plot feature count")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EnhancedConfig | CNNTransformerConfig:
    if args.model == "cnn":
        config: EnhancedConfig | CNNTransformerConfig = CNNTransformerConfig()
    else:
        config = EnhancedConfig()
    config.input_path = args.data
    config.output_dir = args.output
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.val_batch_size = args.val_batch_size
    config.random_state = args.seed
    if args.lr is not None:
        config.lr = args.lr
    if args.undersample is not None:
        config.undersampling_ratio = args.undersample
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    return config


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    config = build_config(args)
    trainer = train_cnn_transformer if args.model == "cnn" else train_enhanced

    checkpoint_path = args.checkpoint
    if args.skip_training:
        if not checkpoint_path:
            raise ValueError("--skip-training requires --checkpoint to be provided")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Skipping training; using checkpoint: {checkpoint_path}")
    else:
        print(f"Starting {args.model.upper()} training phase...")
        checkpoint_path = trainer(config)
        if checkpoint_path is None:
            raise RuntimeError("Training did not produce a checkpoint; aborting SHAP phase")
        print(f"Training complete -> {checkpoint_path}")

    shap_dir = args.shap_dir or os.path.join(args.output, "shap")
    os.makedirs(shap_dir, exist_ok=True)
    print(f"Running SHAP interpretability (outputs -> {shap_dir})")
    run_shap(
        checkpoint_path=checkpoint_path,
        data_path=args.data,
        output_dir=shap_dir,
        chunk_size=args.chunk,
        background_size=args.background,
        eval_size=args.eval,
        eval_pool=args.pool,
        random_seed=args.seed,
        plot_topk=args.topk,
    )
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
