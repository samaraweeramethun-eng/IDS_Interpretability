"""Grad-CAM for the CNN-Transformer IDS model.

Generates class-discriminative activation maps by weighting the CNN
tokenizer's feature maps with gradients flowing from the target class.
Each feature position receives an importance score that reveals *where*
the convolutional tokenizer focuses when predicting attacks.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


class GradCAM:
    """Gradient-weighted Class Activation Mapping for 1-D CNN layers.

    Hooks into a target convolutional layer and produces per-feature-position
    importance scores that show *where* the CNN focuses when making a
    prediction.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list = []
        self._register_hooks()

    def _register_hooks(self):
        def _fwd(module, inp, out):
            self.activations = out.detach()

        def _bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(_fwd))
        self._handles.append(
            self.target_layer.register_full_backward_hook(_bwd)
        )

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> torch.Tensor:
        """Return a Grad-CAM heatmap of shape ``(B, L)`` over feature positions.

        Parameters
        ----------
        input_tensor : (B, F) float tensor on the model's device.
        target_class : class index to explain (1 = attack for binary IDS).

        Returns
        -------
        cam : (B, L) tensor with values in [0, 1].
        """
        self.model.eval()
        with torch.enable_grad():
            inp = input_tensor.detach().clone().requires_grad_(True)
            output = self.model(inp)
            if isinstance(output, (tuple, list)):
                output = output[0]
            self.model.zero_grad()
            target = output[:, target_class].sum()
            target.backward()

        # activations / gradients: (B, C, L) for Conv1d layers
        weights = self.gradients.mean(dim=-1, keepdim=True)  # (B, C, 1)
        cam = (weights * self.activations).sum(dim=1)  # (B, L)
        cam = F.relu(cam)

        # Per-sample min-max normalisation
        cam_min = cam.min(dim=-1, keepdim=True)[0]
        cam_max = cam.max(dim=-1, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam


def _resolve_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last Conv1d inside the CNN tokenizer."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError(
            "Model has no 'tokenizer' attribute – Grad-CAM requires a CNNTokenizer."
        )
    conv_seq = getattr(tokenizer, "conv", None)
    if conv_seq is None:
        raise ValueError("Tokenizer has no 'conv' Sequential.")
    for layer in reversed(list(conv_seq.children())):
        if isinstance(layer, torch.nn.Conv1d):
            return layer
    raise ValueError("No Conv1d layer found in tokenizer.conv")


def generate_gradcam_report(
    model: torch.nn.Module,
    X_val: np.ndarray,
    feature_names: list[str],
    output_dir: str,
    sample_size: int = 512,
    target_class: int = 1,
    seed: int = 42,
) -> str:
    """Compute Grad-CAM attributions and save a per-feature importance CSV.

    Parameters
    ----------
    model : A *CNNTransformerIDS* model (must have a ``tokenizer.conv``).
    X_val : Preprocessed validation features, shape ``(N, F)``.
    feature_names : Human-readable feature names matching columns of *X_val*.
    output_dir : Directory where the CSV (and optional plot) are written.
    sample_size : Number of samples to average Grad-CAM over.
    target_class : Target class for Grad-CAM (1 = attack).
    seed : Random seed for reproducible sub-sampling.

    Returns
    -------
    csv_path : Absolute path to the saved CSV.
    """
    if X_val.shape[0] == 0:
        return ""

    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    target_layer = _resolve_target_layer(model)
    cam_extractor = GradCAM(model, target_layer)

    sample_count = min(sample_size, X_val.shape[0])
    rng = np.random.RandomState(seed)
    idx = rng.choice(X_val.shape[0], sample_count, replace=False)
    data = torch.FloatTensor(X_val[idx]).to(device)

    # Process in chunks to limit memory
    all_cams: list[torch.Tensor] = []
    for chunk in torch.split(data, 128):
        cam = cam_extractor.generate(chunk, target_class=target_class)
        all_cams.append(cam.detach().cpu())

    cam_extractor.remove_hooks()

    cam_tensor = torch.cat(all_cams, dim=0)  # (N_sample, L)
    # L == input_dim because Conv1d preserves length with padding='same'
    mean_cam = cam_tensor.mean(dim=0).numpy()  # (L,)

    # Truncate or pad to match feature_names length
    num_features = len(feature_names)
    if mean_cam.shape[0] >= num_features:
        mean_cam = mean_cam[:num_features]
    else:
        mean_cam = np.pad(mean_cam, (0, num_features - mean_cam.shape[0]))

    importance_df = pd.DataFrame(
        {"feature": feature_names, "grad_cam_importance": mean_cam}
    ).sort_values("grad_cam_importance", ascending=False)

    csv_path = os.path.join(output_dir, "cnn_transformer_grad_cam.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved Grad-CAM feature ranking -> {csv_path}")

    # Optional: save a bar plot
    try:
        import matplotlib.pyplot as plt

        top_k = min(20, len(importance_df))
        top = importance_df.head(top_k)
        plt.figure(figsize=(10, 6))
        plt.barh(top["feature"][::-1], top["grad_cam_importance"][::-1])
        plt.xlabel("Mean Grad-CAM Activation")
        plt.title("Top Feature Positions by Grad-CAM (Attack Class)")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "grad_cam_importance.png")
        plt.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Saved Grad-CAM plot -> {plot_path}")
    except Exception as exc:
        print(f"Skipping Grad-CAM plot: {exc}")

    return csv_path
