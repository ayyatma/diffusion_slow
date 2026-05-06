import torch
import torch.nn as nn

from models.mobilevit_s_exits import MobileViTSWithExits
from training.common import (
    add_common_args,
    build_cifar10_loaders,
    get_device,
    limited_batches,
    load_config,
    resolve_project_path,
)


def train_stage1(config_path="configs/mobilevit_s.yaml", device_name=None, max_batches=None):
    config = load_config(config_path)
    stage_cfg = config["training"]["stage1"]
    device = get_device(device_name)
    train_loader, _, _ = build_cifar10_loaders(config)

    model = MobileViTSWithExits(num_classes=config["num_classes"], pretrained=False).to(device)
    input_checkpoint = resolve_project_path(stage_cfg["input_checkpoint"])
    model.load_state_dict(torch.load(input_checkpoint, map_location=device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=stage_cfg["lr"],
        weight_decay=stage_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_cfg["t_max"])
    criterion = nn.CrossEntropyLoss()
    exit_weights = stage_cfg["exit_loss_weights"]

    for epoch in range(stage_cfg["epochs"]):
        model.train()
        total_loss = 0
        steps = 0
        for X, y in limited_batches(train_loader, max_batches):
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

        scheduler.step()
        print(f"Epoch {epoch+1}/{stage_cfg['epochs']}, Loss: {total_loss/max(steps, 1):.4f}")

    checkpoint = resolve_project_path(stage_cfg["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    print("Stage 1 complete. Weights saved.")
    return model

if __name__ == "__main__":
    import argparse

    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    train_stage1(args.config, args.device, args.max_batches)
