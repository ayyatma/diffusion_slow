import torch
import torch.nn as nn
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mobilevit_s_exits import MobileViTSWithExits
from training.common import (
    add_common_args,
    build_cifar10_loaders,
    get_device,
    load_config,
    load_matching_state_dict,
    progress_batches,
    resolve_project_path,
    save_checkpoint_with_metadata,
)


def train_stage1(config_path="configs/mobilevit_s.yaml", device_name=None, max_batches=None, epochs=None):
    config = load_config(config_path)
    stage_cfg = config["training"]["stage1"]
    device = get_device(device_name)
    train_loader, _, _ = build_cifar10_loaders(config)

    model = MobileViTSWithExits(
        num_classes=config["num_classes"],
        pretrained=False,
        exit_head_config=config.get("exit_heads"),
    ).to(device)
    input_checkpoint = resolve_project_path(stage_cfg["input_checkpoint"])
    load_report = load_matching_state_dict(model, input_checkpoint, device, exclude_prefixes=("exits.",))
    print(
        f"Loaded {len(load_report['loaded_keys'])} matching tensors from {input_checkpoint}; "
        f"skipped {len(load_report['skipped_keys'])} exit/incompatible tensors.",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stage_cfg["lr"],
        weight_decay=stage_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_cfg["t_max"])
    criterion = nn.CrossEntropyLoss()
    exit_weights = stage_cfg["exit_loss_weights"]

    num_epochs = epochs or stage_cfg["epochs"]
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        steps = 0
        batches = progress_batches(train_loader, max_batches, desc=f"Stage 1 epoch {epoch+1}/{num_epochs}")
        for X, y in batches:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)

            loss = 0
            for w, out in zip(exit_weights, logits):
                loss += w * criterion(out, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            batches.set_postfix(loss=f"{total_loss/steps:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/max(steps, 1):.4f}", flush=True)

    checkpoint, metadata_path = save_checkpoint_with_metadata(
        model,
        stage_cfg["checkpoint"],
        {
            "stage": "stage1",
            "config": str(resolve_project_path(config_path)),
            "epochs": num_epochs,
            "max_batches": max_batches,
            "input_checkpoint": str(input_checkpoint),
            "input_checkpoint_loaded_keys": len(load_report["loaded_keys"]),
            "input_checkpoint_skipped_keys": len(load_report["skipped_keys"]),
            "final_train_loss": total_loss / max(steps, 1),
            "optimizer": stage_cfg["optimizer"],
            "lr": stage_cfg["lr"],
            "weight_decay": stage_cfg["weight_decay"],
            "scheduler": stage_cfg["scheduler"],
            "t_max": stage_cfg["t_max"],
            "exit_loss_weights": stage_cfg["exit_loss_weights"],
            "exit_heads": config.get("exit_heads"),
        },
    )
    print(f"Stage 1 complete. Weights saved to {checkpoint}.")
    print(f"Stage 1 metadata saved to {metadata_path}.")
    return model

if __name__ == "__main__":
    import argparse

    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    train_stage1(args.config, args.device, args.max_batches, args.epochs)
