from pathlib import Path

import numpy as np
import pytest
import torch

from lungxray.explain.gradcam import GradCAM
from lungxray.models.bundle import import_legacy_state_dict, load_bundle
from lungxray.models.factory import build_model, gradcam_target_layer, set_backbone_trainable

ARCHS = ["custom_cnn", "resnet18", "densenet121", "efficientnet_b0", "convnext_tiny"]


@pytest.mark.parametrize("arch", ARCHS)
def test_forward_and_gradcam(arch):
    model = build_model(arch, 3, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert model(x).shape == (1, 3)
    cam = GradCAM(model, gradcam_target_layer(model, arch))
    heat, logits = cam(x)
    cam.remove()
    assert heat.shape == (224, 224) and heat.min() >= 0 and heat.max() <= 1
    assert logits.shape == (1, 3)


def test_freeze_backbone_leaves_head_trainable():
    model = build_model("resnet18", 3, pretrained=False)
    set_backbone_trainable(model, "resnet18", trainable=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("fc.") for n in trainable)


LEGACY = Path("models/legacy/custom_cnn_v0.pth")


@pytest.mark.skipif(not LEGACY.exists(), reason="legacy weights not present")
def test_legacy_weights_load(tmp_path):
    out = tmp_path / "legacy.pt"
    import_legacy_state_dict(LEGACY, out, ["COVID19", "NORMAL", "PNEUMONIA"])
    model, meta = load_bundle(out)
    assert meta["preprocessing"]["resize"] == "center_crop"
    with torch.no_grad():
        p = model(torch.randn(2, 3, 224, 224)).softmax(1).numpy()
    assert np.allclose(p.sum(1), 1, atol=1e-5)
