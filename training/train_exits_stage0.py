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
    progress_batches,
    resolve_project_path,
)


def train_stage0(
    config_path="configs/mobilevit_s.yaml",
    device_name=None,
    max_batches=None,
    epochs=None,
    pretrained=True,
):
    config = load_config(config_path)
    stage_cfg = config["training"]["stage0"]
    device = get_device(device_name)
    train_loader, _, _ = build_cifar10_loaders(config)

    model = MobileViTSWithExits(num_classes=config["num_classes"], pretrained=pretrained).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stage_cfg["lr"],
        weight_decay=stage_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_cfg["t_max"])
    criterion = nn.CrossEntropyLoss()

    num_epochs = epochs or stage_cfg["epochs"]
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        steps = 0
        batches = progress_batches(train_loader, max_batches, desc=f"Stage 0 epoch {epoch+1}/{num_epochs}")
        for X, y in batches:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            final_out = logits[-1]
            loss = criterion(final_out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            batches.set_postfix(loss=f"{total_loss/steps:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/max(steps, 1):.4f}", flush=True)

    checkpoint = resolve_project_path(stage_cfg["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    print("Stage 0 complete. Weights saved.")
    return model

if __name__ == "__main__":
    import argparse

    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    train_stage0(
        args.config,
        args.device,
        args.max_batches,
        args.epochs,
        pretrained=not args.no_pretrained,
    )
