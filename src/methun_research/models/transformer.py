import math
import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureImportanceLayer(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.Tanh(),
            nn.Linear(d_model, input_dim),
            nn.Sigmoid(),
        )
        self.projection = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        importance = self.feature_attention(x)
        attended = x * importance
        embedded = self.projection(attended)
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)
        return embedded.unsqueeze(1), importance


class EnhancedMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv_projection = nn.Linear(d_model, d_model * 3, bias=False)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_projection(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.d_model)
        return self.output_projection(output)


class EnhancedTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attention = EnhancedMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attention(self.norm1(x))
        x = x + self.dropout(attended) * self.residual_scale
        fed_forward = self.feed_forward(self.norm2(x))
        x = x + self.dropout(fed_forward) * self.residual_scale
        return x


class AttentionPoolingClassifier(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.15):
        super().__init__()
        self.attention_pool = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x_with_cls = torch.cat([cls_token, x], dim=1)
        pooled, _ = self.attention_pool(cls_token, x_with_cls, x_with_cls)
        return self.classifier(pooled.squeeze(1))


class EnhancedBinaryTransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.feature_embedder = FeatureImportanceLayer(input_dim, d_model)
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = AttentionPoolingClassifier(d_model, dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x, importance = self.feature_embedder(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.classifier(x)
        return logits, importance


class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma_min=1.0, gamma_max=3.0, class_weights=None, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.gamma = nn.Parameter(torch.tensor(1.8))

    def forward(self, inputs, targets):
        gamma = torch.clamp(self.gamma, self.gamma_min, self.gamma_max)
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_weight * (1 - pt) ** gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()


class TabularMixup:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    @staticmethod
    def mixup_criterion(pred, y_a, y_b, lam, criterion):
        if isinstance(pred, tuple):
            pred, _ = pred
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class SWAOptimizer:
    def __init__(self, base_optimizer, swa_start=20, swa_freq=3, swa_lr=0.001):
        self.base_optimizer = base_optimizer
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        self.swa_model = None
        self.n_averaged = 0

    def update_swa(self, model, epoch):
        if epoch >= self.swa_start and (epoch - self.swa_start) % self.swa_freq == 0:
            if self.swa_model is None:
                self.swa_model = copy.deepcopy(model)
            else:
                for swa_param, param in zip(self.swa_model.parameters(), model.parameters()):
                    swa_param.data = (swa_param.data * self.n_averaged + param.data) / (self.n_averaged + 1)
            self.n_averaged += 1
            return True
        return False

    def get_swa_model(self):
        return self.swa_model
