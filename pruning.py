import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
class CFG:
    seed = 42
    batch_size = 128
    epochs = 5 
    lr = 1e-2  # High learning rate to force gate movement
    lambdas = [0.1, 1.0, 5.0] # High lambdas to dominate CE loss
    gate_threshold = 0.01 # Specified in JD 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Part 1: Prunable Linear Layer [cite: 68-75] ---
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        # Start gates VERY low so they cross the 0.01 threshold easily
        # Sigmoid(-5) is ~0.006, Sigmoid(-4) is ~0.018
        nn.init.normal_(self.gate_scores, mean=-4.5, std=0.1)

    def forward(self, x):
        # Apply Sigmoid to turn scores into gates [cite: 77]
        gates = torch.sigmoid(self.gate_scores)
        # Element-wise multiplication [cite: 79]
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

# --- Part 2: Neural Network Definition [cite: 111] ---
class SelfPruningMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_sparsity_loss(self):
        # L1 Norm (Sum of absolute values) of all gates [cite: 89, 90, 91]
        layers = [self.fc1, self.fc2, self.fc3]
        total_sum = sum(torch.sigmoid(l.gate_scores).sum() for l in layers)
        total_params = sum(l.gate_scores.numel() for l in layers)
        return total_sum / total_params

# --- Part 3: Training & Evaluation [cite: 98, 99] ---
def train_and_report(lam):
    torch.manual_seed(CFG.seed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # 5k samples (10% subset) to ensure CPU speed for the deadline
    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    indices = np.random.choice(len(train_ds), 5000, replace=False)
    train_loader = DataLoader(Subset(train_ds, indices), batch_size=CFG.batch_size, shuffle=True)
    
    test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False)

    model = SelfPruningMLP().to(CFG.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    
    print(f"\n>> Training Lambda: {lam}")
    for epoch in range(1, CFG.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(CFG.device), labels.to(CFG.device)
            optimizer.zero_grad()
            
            # Total Loss = Classification + Lambda * Sparsity [cite: 87]
            ce_loss = F.cross_entropy(model(images), labels)
            s_loss = model.get_sparsity_loss()
            total_loss = ce_loss + lam * s_loss
            
            total_loss.backward()
            optimizer.step()
        
        # Report Sparsity Level [cite: 101, 102]
        with torch.no_grad():
            layers = [model.fc1, model.fc2, model.fc3]
            total = sum(l.gate_scores.numel() for l in layers)
            pruned = sum((torch.sigmoid(l.gate_scores) < CFG.gate_threshold).sum().item() for l in layers)
            sp_pct = (pruned / total) * 100
            print(f"Epoch {epoch} | Sparsity: {sp_pct:.2f}%")

    # Final Evaluation [cite: 103]
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(CFG.device), labels.to(CFG.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return model, (100 * correct / total), sp_pct

if __name__ == "__main__":
    final_results = []
    best_model, best_acc, best_lam = None, 0, 0

    for l in CFG.lambdas:
        m, acc, sp = train_and_report(l)
        final_results.append((l, acc, sp))
        if acc > best_acc:
            best_acc, best_model, best_lam = acc, m, l

    # Summary Table [cite: 116]
    print("\n" + "="*45)
    print(f"{'Lambda':<10} | {'Test Acc (%)':<15} | {'Sparsity (%)':<10}")
    print("-" * 45)
    for l, acc, sp in final_results:
        print(f"{l:<10} | {acc:<15.2f} | {sp:<10.2f}")