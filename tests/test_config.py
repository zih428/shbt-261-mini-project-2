from pathlib import Path

from vocseg.config import load_config


def test_config_inheritance():
    config_path = Path("configs/ablations/unet_resnet18.yaml")
    config = load_config(config_path)
    assert config["experiment_name"] == "unet_resnet18"
    assert config["model"]["family"] == "unet"
    assert config["model"]["backbone"] == "resnet18"
    assert config["scheduler"]["max_epochs"] == 60
