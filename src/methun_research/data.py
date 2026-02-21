from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch


class IntelligentDataBalancer:
    def __init__(self, undersampling_ratio: float = 0.12, random_state: int = 42):
        self.undersampling_ratio = undersampling_ratio
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)

    def balance_classes(self, X: np.ndarray, y: np.ndarray):
        majority_idx = np.where(y == 0)[0]
        minority_idx = np.where(y == 1)[0]
        if len(majority_idx) == 0 or len(minority_idx) == 0:
            return X, y
        target_majority = max(
            len(minority_idx) * 3,
            int(len(majority_idx) * self.undersampling_ratio)
        )
        if len(majority_idx) > target_majority:
            minority_center = X[minority_idx].mean(axis=0)
            distances = np.linalg.norm(X[majority_idx] - minority_center, axis=1)
            weights = 1 / (distances + 1e-8)
            weights /= weights.sum()
            selected_majority = self._rng.choice(
                majority_idx,
                size=target_majority,
                replace=False,
                p=weights
            )
        else:
            selected_majority = majority_idx
        combined_idx = np.concatenate([selected_majority, minority_idx])
        return X[combined_idx], y[combined_idx]


class RobustPreprocessor:
    def __init__(self, scaling: str = "quantile", handle_outliers: bool = True, n_features: int = 120):
        self.scaling = scaling
        self.handle_outliers = handle_outliers
        self.n_features = n_features
        self.scaler = None
        self.feature_selector = None
        self.cap_bounds = None

    def _cap_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.handle_outliers or self.cap_bounds is None:
            return frame
        capped = frame.copy()
        for idx, col in enumerate(capped.columns):
            low, high = self.cap_bounds[idx]
            capped[col] = capped[col].clip(low, high)
        return capped

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        X_proc = X.replace([np.inf, -np.inf], np.nan)
        for col in X_proc.columns:
            if X_proc[col].isna().any():
                X_proc[col] = X_proc[col].fillna(X_proc[col].median())
        if self.handle_outliers:
            bounds = []
            for col in X_proc.columns:
                q_low = X_proc[col].quantile(0.005)
                q_high = X_proc[col].quantile(0.995)
                bounds.append((q_low, q_high))
            self.cap_bounds = bounds
            X_proc = self._cap_frame(X_proc)
        if y is not None and self.n_features < X_proc.shape[1]:
            selector = SelectKBest(mutual_info_classif, k=self.n_features)
            X_selected = selector.fit_transform(X_proc, y)
            self.feature_selector = selector
            X_proc = pd.DataFrame(X_selected, index=X_proc.index)
        if self.scaling == "quantile":
            self.scaler = QuantileTransformer(output_distribution="uniform", random_state=42)
        elif self.scaling == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        return self.scaler.fit_transform(X_proc).astype(np.float32)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_proc = X.replace([np.inf, -np.inf], np.nan)
        for col in X_proc.columns:
            if X_proc[col].isna().any():
                X_proc[col] = X_proc[col].fillna(X_proc[col].median())
        X_proc = self._cap_frame(X_proc)
        if self.feature_selector is not None:
            X_proc = self.feature_selector.transform(X_proc)
            X_proc = pd.DataFrame(X_proc)
        if self.scaler is not None:
            return self.scaler.transform(X_proc).astype(np.float32)
        return X_proc.values.astype(np.float32)


def detect_label_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "label" in col.lower():
            return col
    raise ValueError("Label column not found in dataset")


def prepare_features(df: pd.DataFrame, label_col: str):
    binary_label = (df[label_col] != "BENIGN").astype(int).values
    blacklist = {label_col, "Flow ID", "Source IP", "Destination IP", "Timestamp"}
    feature_cols = [col for col in df.columns if col not in blacklist]
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    # Downcast float64 -> float32 to halve memory (saves ~850 MB on full dataset)
    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = X[col].astype(np.float32)
    return X, binary_label, feature_cols


def stratified_split(X: np.ndarray, y: np.ndarray, test_size: float, random_state: int):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    pin_memory: bool | None = None,
):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    class_counts = np.bincount(y_train)
    weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = weights[y_train]
    sampler = WeightedRandomSampler(
        torch.DoubleTensor(sample_weights),
        len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, val_dataset


def calculate_comprehensive_metrics(y_true, y_pred, y_prob):
    if len(y_true) == 0:
        return {key: 0.0 for key in ["accuracy", "auc_roc", "auc_pr", "f1_score", "precision", "recall"]}
    accuracy = float(np.mean(y_true == y_pred))
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return {
            "accuracy": accuracy,
            "auc_roc": 0.5,
            "auc_pr": float(np.mean(y_true)),
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
    from sklearn.metrics import (
        roc_auc_score,
        precision_recall_curve,
        auc,
        f1_score,
        precision_score,
        recall_score,
    )

    auc_roc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(recall, precision)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return {
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
    }
