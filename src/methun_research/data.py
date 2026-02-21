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
            # Compute distances in chunks to avoid a massive temp array
            CHUNK = 100_000
            distances = np.empty(len(majority_idx), dtype=np.float32)
            for start in range(0, len(majority_idx), CHUNK):
                end = min(start + CHUNK, len(majority_idx))
                diff = X[majority_idx[start:end]] - minority_center
                distances[start:end] = np.linalg.norm(diff, axis=1).astype(np.float32)
                del diff
            weights = 1.0 / (distances.astype(np.float64) + 1e-8)
            weights /= weights.sum()  # float64 ensures sum == 1.0
            selected_majority = self._rng.choice(
                majority_idx,
                size=target_majority,
                replace=False,
                p=weights,
            )
            del distances, weights
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

    def fit_transform(self, X, y: np.ndarray) -> np.ndarray:
        # Accept either DataFrame or numpy array
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)
        # In-place clean
        np.nan_to_num(X_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if self.handle_outliers:
            bounds = []
            for col_idx in range(X_np.shape[1]):
                q_low = float(np.percentile(X_np[:, col_idx], 0.5))
                q_high = float(np.percentile(X_np[:, col_idx], 99.5))
                bounds.append((q_low, q_high))
                np.clip(X_np[:, col_idx], q_low, q_high, out=X_np[:, col_idx])
            self.cap_bounds = bounds
        if y is not None and self.n_features < X_np.shape[1]:
            selector = SelectKBest(mutual_info_classif, k=self.n_features)
            X_np = selector.fit_transform(X_np, y).astype(np.float32)
            self.feature_selector = selector
        if self.scaling == "quantile":
            self.scaler = QuantileTransformer(output_distribution="uniform", random_state=42)
        elif self.scaling == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        return self.scaler.fit_transform(X_np).astype(np.float32)

    def transform(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)
        np.nan_to_num(X_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if self.handle_outliers and self.cap_bounds is not None:
            for col_idx in range(X_np.shape[1]):
                low, high = self.cap_bounds[col_idx]
                np.clip(X_np[:, col_idx], low, high, out=X_np[:, col_idx])
        if self.feature_selector is not None:
            X_np = self.feature_selector.transform(X_np).astype(np.float32)
        if self.scaler is not None:
            return self.scaler.transform(X_np).astype(np.float32)
        return X_np


def detect_label_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "label" in col.lower():
            return col
    raise ValueError("Label column not found in dataset")


def prepare_features(df: pd.DataFrame, label_col: str):
    """Extract features as a float32 numpy array to minimise memory."""
    binary_label = (df[label_col] != "BENIGN").astype(np.int8).values
    blacklist = {label_col, "Flow ID", "Source IP", "Destination IP", "Timestamp"}
    feature_cols = [col for col in df.columns if col not in blacklist]
    X = df[feature_cols]
    # Convert any non-numeric columns
    obj_cols = X.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        X = X.copy()  # only copy if we need to mutate
        for col in obj_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    # Convert to float32 numpy immediately — avoids keeping a fat DataFrame alive
    X_np = X.values.astype(np.float32)
    # Replace inf / NaN in-place
    np.nan_to_num(X_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X_np, binary_label, feature_cols


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
    max_train_samples: int = 0,
):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    # Cap training set for speed (data is already balanced by IntelligentDataBalancer)
    if max_train_samples > 0 and len(X_train) > max_train_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), size=max_train_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    use_persistent = num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=4 if use_persistent else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=4 if use_persistent else None,
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
