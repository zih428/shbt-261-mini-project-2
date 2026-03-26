from pathlib import Path

from vocseg.config import load_config


def test_config_inheritance():
    config_path = Path("configs/ablations/unet_resnet18.yaml")
    config = load_config(config_path)
    assert config["experiment_name"] == "unet_resnet18"
    assert config["model"]["family"] == "unet"
    assert config["model"]["backbone"] == "resnet18"
    assert config["scheduler"]["name"] == "plateau"
    assert config["scheduler"]["max_epochs"] == 150
    assert config["training"]["policy_version"] == 2
    assert config["training"]["monitor"] == "mIoU"
    assert config["training"]["min_epochs"] == 20
    assert config["training"]["early_stopping_patience"] == 15
    assert config["training"]["early_stopping_min_delta"] == 0.002
