import pytest
import torch


transformers = pytest.importorskip("transformers")

from vocseg.models.segformer import SegFormer


def test_segformer_forward_shape():
    model = SegFormer(num_classes=21, backbone_name="segformer_b2", pretrained=False, embedding_dim=256)
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 21, 128, 128)
