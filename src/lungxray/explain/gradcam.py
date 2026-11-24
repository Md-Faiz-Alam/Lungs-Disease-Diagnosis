"""Grad-CAM (Selvaraju et al., 2017), implemented from scratch with hooks.

Shows which regions of the radiograph drove a prediction. Used for (1) clinician-facing
explanations in the app and (2) debugging shortcut learning: a heatmap on text markers,
corners or outside the lungs means the model learned the dataset, not the disease.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inp, out):
        self.activations = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "gradients", g))

    def remove(self) -> None:
        self._handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, torch.Tensor]:
        """x: (1, 3, H, W). Returns (heatmap HxW in [0,1], logits)."""
        self.model.eval()
        x = x.clone().requires_grad_(True)  # ensures a graph even if the backbone is frozen
        with torch.enable_grad():
            logits = self.model(x)
            idx = int(logits.argmax(1)) if class_idx is None else class_idx
            self.model.zero_grad(set_to_none=True)
            logits[0, idx].backward()
        acts, grads = self.activations.detach(), self.gradients.detach()
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.cpu().numpy(), logits.detach()


def overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    base = image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    colored = (colormaps["jet"](heatmap)[..., :3] * 255).astype(np.uint8)
    blended = (1 - alpha) * np.asarray(base, dtype=np.float32) + alpha * colored
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))
