import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.mobilevit_s_exits import MobileViTSWithExits
from attacks.utils.entropy import max_confidence
import yaml
import os

def calibrate_thresholds(model, val_loader, device, target_exit_rate=0.20):
    model.eval()
    all_max_probs = [[] for _ in range(5)]
    
    with torch.no_grad():
        for X, _ in val_loader:
            X = X.to(device)
            logits = model(X)
            for i in range(5):
                probs = max_confidence(logits[i])
                all_max_probs[i].extend(probs.cpu().tolist())

    thresholds = []
    # Simplified percentiles for demonstration
    for i in range(5):
        probs = torch.tensor(all_max_probs[i])
        # Find threshold where approx target_exit_rate % bounds the exiting.
        # This is a simplified version; ideally we do this sequentially removing early exited.
        t = torch.quantile(probs, 1.0 - target_exit_rate).item()
        thresholds.append(t)
        
    return thresholds

def eval_clean():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    model = MobileViTSWithExits(num_classes=10, pretrained=False).to(device)
    weights_path = "models/mobilevit_s_cifar10_stage2.pt"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Warning: Weights not found, using initialized weights!")
        
    model.eval()
    corrects = [0]*6
    total = 0
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total += y.size(0)
            
            for i in range(6):
                preds = logits[i].argmax(dim=1)
                corrects[i] += (preds == y).sum().item()
                
    for i in range(5):
        print(f"Exit {i+1} Accuracy: {corrects[i]/total * 100:.2f}%")
    print(f"Final Exit Accuracy: {corrects[5]/total * 100:.2f}%")
    
    # Calibration
    with open("configs/mobilevit_s.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    if config['thresholds']['calibrate_from_val']:
        thresh = calibrate_thresholds(model, val_loader, device)
        print("Calibrated thresholds:", thresh)
    else:
        print("Using thresholds from config")

if __name__ == "__main__":
    eval_clean()
