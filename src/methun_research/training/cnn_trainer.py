import gc
import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from methun_research.config import CNNTransformerConfig
from methun_research.data import (
    IntelligentDataBalancer,
    detect_label_column,
    prepare_features,
    build_dataloaders,
    calculate_comprehensive_metrics,
)
from methun_research.models.cnn_transformer import CNNTransformerIDS
from methun_research.utils.device import setup_device
from methun_research.interpretability.integrated_gradients import generate_ig_report
from methun_research.interpretability.grad_cam import generate_gradcam_report


def _set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _prepare_scaled_data(X: pd.DataFrame, y: np.ndarray, config: CNNTransformerConfig):
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )
    train_medians = X_train_raw.median()
    X_train_raw = (
        X_train_raw.replace([np.inf, -np.inf], np.nan)
        .fillna(train_medians)
    )
    X_val_raw = (
        X_val_raw.replace([np.inf, -np.inf], np.nan)
        .fillna(train_medians)
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    del X_train_raw, X_val_raw
    gc.collect()
    return X_train, X_val, y_train, y_val, scaler, train_medians


def _train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


def _eval_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(data)
            loss = criterion(logits, target)
            losses.append(loss.item())
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    metrics = calculate_comprehensive_metrics(
        np.array(all_targets), np.array(all_preds), np.array(all_probs)
    )
    return np.mean(losses) if losses else 0.0, metrics, np.array(all_probs), np.array(all_targets)


def train_cnn_transformer(config: CNNTransformerConfig | None = None):
    config = config or CNNTransformerConfig()
    _set_seeds(config.random_state)
    device, multi_gpu = setup_device()
    if multi_gpu:
        config.batch_size = 512
        config.val_batch_size = 1024
    print("Loading dataset for CNN-Transformer training...")
    df = pd.read_csv(config.input_path, engine="python")
    label_col = detect_label_column(df)
    X, y, feature_cols = prepare_features(df, label_col)
    del df; gc.collect()  # free ~1.7 GB
    X_train, X_val, y_train, y_val, scaler, medians = _prepare_scaled_data(X, y, config)
    del X; gc.collect()  # free unsplit feature DataFrame
    balancer = IntelligentDataBalancer(config.undersampling_ratio, config.random_state)
    X_train_bal, y_train_bal = balancer.balance_classes(X_train, y_train)
    input_dim = X_train.shape[1]
    del X_train, y_train; gc.collect()  # free pre-balance arrays
    train_loader, val_loader, _ = build_dataloaders(
        X_train_bal,
        y_train_bal,
        X_val,
        y_val,
        batch_size=config.batch_size,
        val_batch_size=config.val_batch_size,
        num_workers=config.num_workers,
    )
    del X_train_bal, y_train_bal; gc.collect()  # now in TensorDataset
    model = CNNTransformerIDS(
        input_dim=input_dim,
        d_model=config.d_model,
        conv_channels=config.conv_channels,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
    ).to(device)
    if multi_gpu:
        model = DataParallel(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = (
        optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr,
            epochs=config.epochs,
            steps_per_epoch=max(len(train_loader), 1),
        )
        if len(train_loader) > 0
        else None
    )
    best_auc = 0.0
    best_state = None
    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, metrics, _, _ = _eval_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"ROC-AUC {metrics['auc_roc']:.4f} | F1 {metrics['f1_score']:.4f}"
        )
        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            state_dict = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
            preprocess_state = {
                "type": "standard_scaler",
                "medians": medians.to_dict(),
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            }
            best_state = {
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": config.__dict__,
                "feature_columns": feature_cols,
                "preprocessor": preprocess_state,
                "model_type": "cnn_transformer",
            }
    if best_state is None:
        print("Training failed to improve beyond initialization.")
        return None
    model_path = os.path.join(config.output_dir, "cnn_transformer_ids.pth")
    torch.save(best_state, model_path)
    print(f"Saved CNN-Transformer checkpoint -> {model_path}")
    preprocess_artifacts = {
        "feature_columns": feature_cols,
        "medians": medians.to_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    preprocess_path = os.path.join(config.output_dir, "cnn_transformer_preprocess.pkl")
    joblib.dump(preprocess_artifacts, preprocess_path)
    print(f"Saved preprocessing artifacts -> {preprocess_path}")
    final_model = model.module if isinstance(model, DataParallel) else model
    generate_ig_report(
        final_model,
        X_val,
        feature_cols,
        config.output_dir,
        steps=config.ig_steps,
        sample_size=config.ig_samples,
        seed=config.random_state,
    )
    generate_gradcam_report(
        final_model,
        X_val,
        feature_cols,
        config.output_dir,
        sample_size=config.ig_samples,
        seed=config.random_state,
    )
    return model_path
