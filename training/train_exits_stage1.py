import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.mobilevit_s_exits import MobileViTSWithExits

def train_stage1():
    # Joint exit supervision
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
    model.load_state_dict(torch.load("models/mobilevit_s_cifar10_stage0.pt"))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    criterion = nn.CrossEntropyLoss()
    
    # 6 outputs: 5 early exits + 1 final exit
    exit_weights = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    for epoch in range(40):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            
            loss = 0
            for w, out in zip(exit_weights, logits):
                loss += w * criterion(out, y)
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch+1}/40, Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "models/mobilevit_s_cifar10_stage1.pt")
    print("Stage 1 complete. Weights saved.")

if __name__ == "__main__":
    train_stage1()
