import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from methun_research.models.transformer import EnhancedBinaryTransformerClassifier
from methun_research.utils.device import setup_device
from methun_research.data import detect_label_column


def load_checkpoint(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def build_model_from_ckpt(ckpt, device):
    state = ckpt["model_state_dict"]
    cfg = ckpt.get("config", {})
    input_dim = state["feature_embedder.projection.weight"].shape[1]
    model = EnhancedBinaryTransformerClassifier(
        input_dim=input_dim,
        d_model=cfg.get("d_model", 160),
        num_layers=cfg.get("num_layers", 4),
        num_heads=cfg.get("heads", 10),
        d_ff=cfg.get("d_ff", 640),
        dropout=cfg.get("dropout", 0.15),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def prepare_eval_matrix(data_path: str, label_col: str, feature_cols: list, preprocessor=None):
    usecols = list(dict.fromkeys(feature_cols + [label_col]))
    df = pd.read_csv(data_path, usecols=usecols)
    df["binary_label"] = (df[label_col] != "BENIGN").astype(int)
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    if preprocessor is not None:
        return preprocessor.transform(X), df["binary_label"].values
    from sklearn.preprocessing import QuantileTransformer

    qt = QuantileTransformer(output_distribution="uniform", random_state=42)
    return qt.fit_transform(X), df["binary_label"].values


def run_shap(
    checkpoint_path: str,
    data_path: str,
    output_dir: str,
    chunk_size: int = 256,
    background_size: int = 2000,
    eval_size: int = 2000,
    eval_pool: int = 150_000,
    random_seed: int = 42,
    plot_topk: int = 20,
):
    device, _ = setup_device()
    ckpt = load_checkpoint(checkpoint_path)
    model = build_model_from_ckpt(ckpt, device)
    preprocessor = ckpt.get("preprocessor")
    df_head = pd.read_csv(data_path, nrows=5)
    label_col = detect_label_column(df_head)
    exclude_cols = [label_col, "binary_label", "Flow ID", "Source IP", "Destination IP", "Timestamp"]
    feature_cols = [col for col in df_head.columns if col not in exclude_cols]
    X_all, y_all = prepare_eval_matrix(data_path, label_col, feature_cols, preprocessor)
    pool_n = min(eval_pool, len(y_all))
    rng = np.random.RandomState(random_seed)
    pool_idx = rng.choice(len(y_all), size=pool_n, replace=False)
    X_pool = X_all[pool_idx]
    y_pool = y_all[pool_idx]
    bg_size = min(background_size, X_pool.shape[0])
    eval_n = min(eval_size, X_pool.shape[0])
    background = torch.from_numpy(X_pool[:bg_size]).float().to(device)
    eval_tensor = torch.from_numpy(X_pool[:eval_n]).float().to(device)
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap must be installed to run SHAP analysis") from exc
    class LogitsOnly(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            out = self.base(x)
            return out[0] if isinstance(out, (tuple, list)) else out
    logits_model = LogitsOnly(model).to(device)
    explainer = shap.DeepExplainer(logits_model, background)
    shap_values = []
    start = 0
    while start < eval_tensor.shape[0]:
        end = min(start + chunk_size, eval_tensor.shape[0])
        shap_values.append(
            explainer.shap_values(eval_tensor[start:end], check_additivity=False)[1]
        )
        start = end
    sv = np.concatenate(shap_values, axis=0)
    mean_abs = np.abs(sv).mean(axis=0)
    importance = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
    importance = importance.sort_values("mean_abs_shap", ascending=False)
    csv_path = os.path.join(output_dir, "shap_global_importance_attack.csv")
    importance.to_csv(csv_path, index=False)
    print(f"Saved SHAP CSV -> {csv_path}")
    try:
        import matplotlib.pyplot as plt
        shap.summary_plot(
            sv,
            features=eval_tensor.detach().cpu().numpy(),
            feature_names=feature_cols,
            show=False,
        )
        plt.tight_layout()
        summary_path = os.path.join(output_dir, "shap_summary_attack.png")
        plt.savefig(summary_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary plot -> {summary_path}")
    except Exception as exc:
        print(f"Skipping SHAP plot: {exc}")
    probs = torch.softmax(logits_model(eval_tensor), dim=1)[:, 1].detach().cpu().numpy()
    attack_idx = np.where(y_pool[:eval_n] == 1)[0]
    if len(attack_idx) > 0:
        target_idx = attack_idx[np.argmax(probs[attack_idx])]
    else:
        target_idx = int(np.argmax(probs))
    try:
        explanation = shap.Explanation(
            values=sv[target_idx],
            base_values=float(np.array(explainer.expected_value[1]).mean()),
            data=eval_tensor[target_idx].detach().cpu().numpy(),
            feature_names=feature_cols,
        )
        shap.plots.waterfall(explanation, max_display=plot_topk, show=False)
        import matplotlib.pyplot as plt
        waterfall_path = os.path.join(output_dir, "shap_waterfall_attack.png")
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP waterfall plot -> {waterfall_path}")
    except Exception as exc:
        print(f"Skipping SHAP waterfall: {exc}")
    return csv_path
