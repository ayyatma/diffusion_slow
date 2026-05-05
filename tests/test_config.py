import yaml


def test_mobilevit_config_has_phase1_thresholds():
    with open("configs/mobilevit_s.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["model"] == "mobilevit_s"
    assert config["num_classes"] == 10
    assert config["input_size"] == 256

    thresholds = config["thresholds"]
    assert [thresholds[f"exit{i}"] for i in range(1, 6)] == [0.70, 0.80, 0.85, 0.88, 0.90]
    assert thresholds["calibrate_from_val"] is True
