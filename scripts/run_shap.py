import argparse

from methun_research.interpretability import run_shap


def parse_args():
    parser = argparse.ArgumentParser(description="Run SHAP on IDS checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", required=True, help="Path to CICIDS2017 CSV")
    parser.add_argument("--output", required=True, help="Directory to store SHAP artifacts")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--background", type=int, default=2000)
    parser.add_argument("--eval", type=int, default=2000)
    parser.add_argument("--pool", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    run_shap(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_dir=args.output,
        chunk_size=args.chunk,
        background_size=args.background,
        eval_size=args.eval,
        eval_pool=args.pool,
        random_seed=args.seed,
        plot_topk=args.topk,
    )


if __name__ == "__main__":
    main()
