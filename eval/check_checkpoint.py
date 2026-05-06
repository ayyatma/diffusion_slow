import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_clean import evaluate_accuracy
from models.mobilevit_s_exits import MobileViTSWithExits
from training.common import (
    add_common_args,
    build_cifar10_loaders,
    get_device,
    load_config,
    load_state_dict,
    resolve_project_path,
    write_json,
)


def check_checkpoint(
    checkpoint,
    config_path="configs/mobilevit_s.yaml",
    device_name=None,
    max_batches=10,
    output_path=None,
):
    config = load_config(config_path)
    device = get_device(device_name)
    checkpoint_path = resolve_project_path(checkpoint)

    model = MobileViTSWithExits(
        num_classes=config["num_classes"],
        pretrained=False,
        exit_head_config=config.get("exit_heads"),
    ).to(device)
    state = load_state_dict(checkpoint_path, device)
    load_result = model.load_state_dict(state, strict=True)

    _, _, test_loader = build_cifar10_loaders(config)
    accuracies = evaluate_accuracy(model, test_loader, device, max_batches=max_batches)

    result = {
        "config": str(resolve_project_path(config_path)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "device": str(device),
        "max_batches": max_batches,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "strict_load_ok": not load_result.missing_keys and not load_result.unexpected_keys,
        "accuracies": {
            **{f"exit{i+1}": accuracies[i] for i in range(5)},
            "final": accuracies[5],
        },
        "accuracies_percent": {
            **{f"exit{i+1}": accuracies[i] * 100 for i in range(5)},
            "final": accuracies[5] * 100,
        },
    }

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Strict load OK: {result['strict_load_ok']}")
    for name, value in result["accuracies_percent"].items():
        print(f"{name}: {value:.2f}%")

    if output_path:
        saved_path = write_json(output_path, result)
        print(f"Checkpoint report saved to {saved_path}.")

    return result


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    check_checkpoint(
        args.checkpoint,
        args.config,
        args.device,
        args.max_batches,
        args.output,
    )
