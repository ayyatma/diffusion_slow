import yaml


def test_mobilevit_config_has_phase1_thresholds():
    with open("configs/mobilevit_s.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["model"] == "mobilevit_s"
    assert config["num_classes"] == 10
    assert config["input_size"] == 256

    assert config["dataset"]["name"] == "CIFAR10"
    assert config["dataset"]["val_size"] == 5000

    thresholds = config["thresholds"]
    assert [thresholds[f"exit{i}"] for i in range(1, 6)] == [0.70, 0.80, 0.85, 0.88, 0.90]
    assert thresholds["calibrate_from_val"] is True
    assert thresholds["target_exit_rate"] == 0.20

    exit_heads = config["exit_heads"]
    assert exit_heads["lph_hidden_mult"] == 2
    assert exit_heads["gah_hidden_dim"] == 256

    training = config["training"]
    assert training["batch_size"] == 128
    assert len(training["stage1"]["exit_loss_weights"]) == 6
    assert len(training["stage2"]["exit_loss_weights"]) == 5
    assert training["stage1"]["exit_loss_weights"] == [0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
    assert training["stage2"]["exit_loss_weights"] == [1.0, 1.0, 0.8, 0.6, 0.4]
    assert training["stage2"]["kd_alpha"] == 0.5
    assert training["stage1"]["checkpoint"].endswith("_stage1_exp1.pt")
    assert training["stage2"]["input_checkpoint"].endswith("_stage1_exp1.pt")
    assert training["stage2"]["checkpoint"].endswith("_stage2_exp1.pt")
    assert config["eval"]["checkpoint"].endswith("_stage2_exp1.pt")
