import pytest
import torch


transformers = pytest.importorskip("transformers")

from vocseg.models.mask2former import Mask2Former


def test_mask2former_forward_shape():
    model = Mask2Former(num_classes=21, backbone_name="mask2former_swin_tiny", pretrained=False)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    assert y.shape == (1, 21, 128, 128)
