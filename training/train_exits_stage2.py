import torch
import torch.nn as nn
import torch.nn.functional as F
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
    limited_batches,
    load_config,
    load_state_dict,
    resolve_project_path,
)

def kd_loss(student_logits, teacher_logits, temperature):
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)

def train_stage2(config_path="configs/mobilevit_s.yaml", device_name=None, max_batches=None, epochs=None):
    config = load_config(config_path)
    stage_cfg = config["training"]["stage2"]
    device = get_device(device_name)
    train_loader, _, _ = build_cifar10_loaders(config)

    model = MobileViTSWithExits(num_classes=config["num_classes"], pretrained=False).to(device)
    input_checkpoint = resolve_project_path(stage_cfg["input_checkpoint"])
    model.load_state_dict(load_state_dict(input_checkpoint, device))

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.exits.parameters(),
        lr=stage_cfg["lr"],
        weight_decay=stage_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_cfg["t_max"])
    criterion_ce = nn.CrossEntropyLoss()
    kd_temp = stage_cfg["kd_temperature"]
    kd_alpha = stage_cfg["kd_alpha"]
    exit_weights = stage_cfg["exit_loss_weights"]

    num_epochs = epochs or stage_cfg["epochs"]
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        steps = 0
        for X, y in limited_batches(train_loader, max_batches):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)

            teacher_logits = logits[-1].detach()

            loss = 0
            for i in range(5):
                out = logits[i]
                loss_ce = criterion_ce(out, y)
                loss_kd = kd_loss(out, teacher_logits, kd_temp)
                combined = kd_alpha * loss_kd + (1 - kd_alpha) * loss_ce
                loss += exit_weights[i] * combined

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/max(steps, 1):.4f}", flush=True)

    checkpoint = resolve_project_path(stage_cfg["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    print("Stage 2 complete. Weights saved.")
    return model

if __name__ == "__main__":
    import argparse

    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    train_stage2(args.config, args.device, args.max_batches, args.epochs)
