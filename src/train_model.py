import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_valid = np.load("data/X_valid.npy")
y_valid = np.load("data/y_valid.npy")

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_valid_t = torch.FloatTensor(X_valid)
y_valid_t = torch.FloatTensor(y_valid)

train_ds = TensorDataset(X_train_t, y_train_t)
valid_ds = TensorDataset(X_valid_t, y_valid_t)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=32)

class SignClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 28)
        )
    def forward(self, x):
        return self.network(x)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

model = SignClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss+= loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            predicted = output.argmax(dim=1)
            correct+= (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    accuracy = correct / total * 100
    avg_loss = train_loss / len(train_loader)

    print(f"Epoch {epoch+1:02d}/{EPOCHS}) | Loss: {avg_loss:.4f} | Val Accuracy:{accuracy:.1f}%")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/sign_classifier.path")
    print("\n Model saved to models/sign_classifier.pth")