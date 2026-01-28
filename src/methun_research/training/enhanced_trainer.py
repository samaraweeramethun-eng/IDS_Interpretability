import os
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

from methun_research.config import EnhancedConfig
from methun_research.data import (
    IntelligentDataBalancer,
    RobustPreprocessor,
    detect_label_column,
    prepare_features,
    stratified_split,
    build_dataloaders,
    calculate_comprehensive_metrics,
)
from methun_research.models.transformer import (
    EnhancedBinaryTransformerClassifier,
    AdaptiveFocalLoss,
    TabularMixup,
    SWAOptimizer,
)
from methun_research.utils.device import setup_device


def _set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _create_model(config: EnhancedConfig, input_dim: int, device: torch.device, multi_gpu: bool):
    model = EnhancedBinaryTransformerClassifier(
        input_dim=input_dim,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
    ).to(device)
    if multi_gpu:
        model = DataParallel(model)
    return model


def _train_epoch(model, loader, optimizer, scheduler, criterion, mixup, device):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if mixup is not None and mixup.get("enabled") and mixup.get("epoch") > 5:
            mixed_x, y_a, y_b, lam = mixup["fn"](data, target)
            output = model(mixed_x)
            loss = mixup["fn"].mixup_criterion(output, y_a, y_b, lam, criterion)
        else:
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
        if batch_idx % 100 == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            gamma_val = criterion.gamma.item() if hasattr(criterion, "gamma") else 0.0
            print(
                f"Batch {batch_idx:04d} | Loss {loss.item():.4f} | LR {current_lr:.2e} | gamma {gamma_val:.2f}"
            )
    return float(np.mean(losses)) if losses else 0.0


def _eval_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target)
            losses.append(loss.item())
            probs = F.softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    metrics = calculate_comprehensive_metrics(
        np.array(all_targets), np.array(all_preds), np.array(all_probs)
    )
    return float(np.mean(losses)) if losses else 0.0, metrics


def train_enhanced(config: EnhancedConfig | None = None):
    config = config or EnhancedConfig()
    _set_seeds(config.random_state)
    device, multi_gpu = setup_device()
    if multi_gpu:
        config.batch_size = 1024
        config.val_batch_size = 2048
    print("Loading dataset...")
    df = pd.read_csv(config.input_path)
    label_col = detect_label_column(df)
    X_raw, y, feature_cols = prepare_features(df, label_col)
    X_train_raw, X_val_raw, y_train, y_val = stratified_split(
        X_raw.values,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    X_train_raw = pd.DataFrame(X_train_raw, columns=feature_cols)
    X_val_raw = pd.DataFrame(X_val_raw, columns=feature_cols)
    preprocessor = RobustPreprocessor(
        scaling="quantile" if config.use_robust_scaling else "standard",
        handle_outliers=True,
        n_features=120,
    )
    X_train = preprocessor.fit_transform(X_train_raw, y_train)
    X_val = preprocessor.transform(X_val_raw)
    balancer = IntelligentDataBalancer(config.undersampling_ratio, config.random_state)
    X_train_bal, y_train_bal = balancer.balance_classes(X_train, y_train)
    train_loader, val_loader, _ = build_dataloaders(
        X_train_bal,
        y_train_bal,
        X_val,
        y_val,
        batch_size=config.batch_size,
        val_batch_size=config.val_batch_size,
        num_workers=config.num_workers,
    )
    class_counts = np.bincount(y_train_bal)
    total = len(y_train_bal)
    class_weights = torch.FloatTensor([
        total / (2 * class_counts[0]),
        total / (2 * class_counts[1]),
    ]).to(device)
    model = _create_model(config, X_train.shape[1], device, multi_gpu)
    criterion = AdaptiveFocalLoss(
        alpha=config.focal_alpha,
        class_weights=class_weights if config.use_class_weights else None,
        label_smoothing=config.label_smoothing,
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.lr,
        epochs=config.epochs,
        steps_per_epoch=max(len(train_loader), 1),
        pct_start=0.1,
        anneal_strategy="cos",
    )
    mixup = {"fn": TabularMixup(alpha=config.mixup_alpha), "enabled": config.use_mixup, "epoch": 0}
    swa_optimizer = SWAOptimizer(optimizer, config.swa_start, config.swa_freq) if config.use_swa else None
    best_auc = 0.0
    best_state = None
    for epoch in range(1, config.epochs + 1):
        mixup["epoch"] = epoch
        train_loss = _train_epoch(model, train_loader, optimizer, scheduler, criterion, mixup, device)
        val_loss, metrics = _eval_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d} | Train {train_loss:.4f} | Val {val_loss:.4f} | AUC {metrics['auc_roc']:.4f} | "
            f"F1 {metrics['f1_score']:.4f}"
        )
        if swa_optimizer is not None:
            updated = swa_optimizer.update_swa(model, epoch)
            if updated:
                print(f"SWA updated after epoch {epoch}")
        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            state_dict = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
            best_state = {
                "model_state_dict": copy.deepcopy(state_dict),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": config.__dict__,
                "feature_columns": feature_cols,
            }
    if best_state is None:
        print("Training did not improve over initialization.")
        return None
    model_path = os.path.join(config.output_dir, "enhanced_binary_rtids_model.pth")
    torch.save(best_state, model_path)
    print(f"Saved best enhanced model -> {model_path}")
    if swa_optimizer is not None and swa_optimizer.get_swa_model() is not None:
        torch.save(swa_optimizer.get_swa_model().state_dict(), os.path.join(config.output_dir, "enhanced_swa.pth"))
    return model_path
