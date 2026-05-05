import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.mobilevit_s_exits import MobileViTSWithExits

def kd_loss(student_logits, teacher_logits, temperature):
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)

def train_stage2():
    # Exit distillation with frozen backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256, padding=16),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    model = MobileViTSWithExits(num_classes=10, pretrained=False).to(device)
    model.load_state_dict(torch.load("models/mobilevit_s_cifar10_stage1.pt"))
    
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    if hasattr(model, 'feature_extractor'):
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
            
    optimizer = torch.optim.AdamW(model.exits.parameters(), lr=1e-3, weight_decay=1e-4) # Train exits only
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion_ce = nn.CrossEntropyLoss()
    
    kd_temp = 4.0
    kd_alpha = 0.7
    exit_weights = [0.1, 0.2, 0.4, 0.6, 0.8]  # Weights for 5 exits (final exit is teacher)

    for epoch in range(30):
        model.train()
        total_loss = 0
        for X, y in train_loader:
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
            
        scheduler.step()
        print(f"Epoch {epoch+1}/30, Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "models/mobilevit_s_cifar10_stage2.pt")
    print("Stage 2 complete. Weights saved.")

if __name__ == "__main__":
    train_stage2()
