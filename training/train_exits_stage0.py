import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.mobilevit_s_exits import MobileViTSWithExits
import yaml

def train_stage0():
    # Backbone adaptation to CIFAR
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
    
    model = MobileViTSWithExits(num_classes=10, pretrained=True).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            # Logits are a list: [exit1, exit2, exit3, exit4, exit5, final_head]
            final_out = logits[-1]
            loss = criterion(final_out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "models/mobilevit_s_cifar10_stage0.pt")
    print("Stage 0 complete. Weights saved.")

if __name__ == "__main__":
    train_stage0()
