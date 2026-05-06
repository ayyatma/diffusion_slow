import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mobilevit_s_exits import MobileViTSWithExits
from attacks.utils.entropy import exit_decision, max_confidence
from training.common import (
    add_common_args,
    build_cifar10_loaders,
    get_device,
    limited_batches,
    load_config,
    load_state_dict,
    progress_batches,
    resolve_project_path,
)


def calibrate_thresholds(model, val_loader, device, target_exit_rate=0.20, max_batches=None):
    model.eval()
    remaining_confidences = [[] for _ in range(5)]
    thresholds = []

    with torch.no_grad():
        all_exit_logits = [[] for _ in range(5)]
        batches = progress_batches(val_loader, max_batches, desc="Calibrating thresholds")
        for X, _ in batches:
            X = X.to(device)
            logits = model(X)
            for i in range(5):
                all_exit_logits[i].append(logits[i].cpu())

    if not all_exit_logits[0]:
        raise ValueError("Cannot calibrate thresholds from an empty validation loader")

    all_exit_logits = [torch.cat(exit_logits, dim=0) for exit_logits in all_exit_logits]
    active_mask = torch.ones(all_exit_logits[0].shape[0], dtype=torch.bool)

    for i in range(5):
        active_logits = all_exit_logits[i][active_mask]
        if active_logits.numel() == 0:
            thresholds.append(0.99)
            continue

        probs = max_confidence(active_logits)
        remaining_confidences[i].extend(probs.tolist())
        t = torch.quantile(probs, 1.0 - target_exit_rate).item()
        t = float(min(max(t, 0.5), 0.99))
        thresholds.append(t)

        should_exit = exit_decision(all_exit_logits[i], t)
        active_mask = active_mask & ~should_exit

    return thresholds


def evaluate_accuracy(model, data_loader, device, max_batches=None):
    model.eval()
    corrects = [0] * 6
    total = 0

    with torch.no_grad():
        batches = progress_batches(data_loader, max_batches, desc="Evaluating clean accuracy")
        for X, y in batches:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total += y.size(0)

            for i in range(6):
                preds = logits[i].argmax(dim=1)
                corrects[i] += (preds == y).sum().item()
            batches.set_postfix(final_acc=f"{corrects[-1] / total * 100:.2f}%")

    if total == 0:
        raise ValueError("Cannot evaluate accuracy from an empty data loader")

    return [correct / total for correct in corrects]


def eval_clean(config_path="configs/mobilevit_s.yaml", device_name=None, max_batches=None):
    config = load_config(config_path)
    device = get_device(device_name)
    _, val_loader, test_loader = build_cifar10_loaders(config)

    model = MobileViTSWithExits(num_classes=config["num_classes"], pretrained=False).to(device)
    weights_path = resolve_project_path(config["eval"]["checkpoint"])
    if weights_path.exists():
        model.load_state_dict(load_state_dict(weights_path, device))
    else:
        print(f"Warning: weights not found at {weights_path}; using initialized weights.")

    accuracies = evaluate_accuracy(model, test_loader, device, max_batches=max_batches)
    for i in range(5):
        print(f"Exit {i+1} Accuracy: {accuracies[i] * 100:.2f}%")
    print(f"Final Exit Accuracy: {accuracies[5] * 100:.2f}%")

    if config["thresholds"]["calibrate_from_val"]:
        thresholds = calibrate_thresholds(
            model,
            val_loader,
            device,
            target_exit_rate=config["thresholds"].get("target_exit_rate", 0.20),
            max_batches=max_batches,
        )
        print("Calibrated thresholds:", thresholds)
    else:
        print("Using thresholds from config")

if __name__ == "__main__":
    import argparse

    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    eval_clean(args.config, args.device, args.max_batches)
