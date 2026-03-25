import torch

from vocseg.models.deeplabv3plus import DeepLabV3Plus
from vocseg.models.unet import UNetResNet


def test_unet_forward_shape():
    model = UNetResNet(num_classes=21, backbone="resnet18", pretrained=False)
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 21, 128, 128)


def test_deeplab_forward_shape():
    model = DeepLabV3Plus(num_classes=21, backbone="resnet18", pretrained=False)
    x = torch.randn(2, 3, 128, 128)
    y = model(x)
    assert y.shape == (2, 21, 128, 128)
