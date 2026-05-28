import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


SELECTED_CLASSES = {
    'street_music': 0,
    'children_playing': 1,
    'siren': 2,
    'car_horn': 3,
    'dog_bark': 4
}

def extract_features(file_path, sr=22050, n_mels=64):
    try:
        audio, sr = librosa.load(file_path, sr=sr, duration=4.0)
        if len(audio) < sr * 4:
            audio = np.pad(audio, (0, sr * 4 - len(audio)))
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        return mel_db
    except:
        return None

def prepare_audio_dataset():
    
    csv_path = "data/audio_samples/UrbanSound8K.csv"
    df = pd.read_csv(csv_path)
    
    
    df = df[df['class'].isin(SELECTED_CLASSES.keys())]
    
    X, y = [], []
    skipped = 0
    
    print(f"📂 Processing {len(df)} audio files...")
    
    for i, row in df.iterrows():
        fold = row['fold']
        filename = row['slice_file_name']
        class_name = row['class']
        
        file_path = f"data/audio_samples/fold{fold}/{filename}"
        
        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(SELECTED_CLASSES[class_name])
        else:
            skipped += 1
        
        if len(X) % 100 == 0 and len(X) > 0:
            print(f"   Processed: {len(X)}, Skipped: {skipped}")
    
    X = np.array(X)[:, np.newaxis, :, :]  
    y = np.array(y)
    
    print(f"\n Done! X: {X.shape}, y: {y.shape}")
    return X, y


class AudioCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.conv_layers(x))

if __name__ == "__main__":
    
    X, y = prepare_audio_dataset()
    
    
    split = int(len(X) * 0.8)
    idx = np.random.permutation(len(X))
    X_train, X_valid = X[idx[:split]], X[idx[split:]]
    y_train, y_valid = y[idx[:split]], y[idx[split:]]
    
    
    device = torch.device("cpu")
    print(f"\n🖥️ Device: {device}")
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    valid_ds = TensorDataset(torch.FloatTensor(X_valid), torch.LongTensor(y_valid))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=32)
    
    
    model = AudioCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    EPOCHS = 20
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predicted = model(X_batch).argmax(dim=1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
        
        accuracy = correct / total * 100
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val Accuracy: {accuracy:.1f}%")
    
    
    torch.save(model.state_dict(), "models/audio_cnn.pth")
    print("\n Audio model saved!")