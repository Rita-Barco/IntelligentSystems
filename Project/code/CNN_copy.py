import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score,classification_report
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas

#load mnist dataset
mnist = datasets.load_digits()
X = mnist.data
y = mnist.target

#train test spliting
test_size=0.2
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)

# Standardize features
scaler=MinMaxScaler()
Xtr= scaler.fit_transform(Xtr)
Xte= scaler.transform(Xte)

Xtr=Xtr.reshape(-1,1, 8, 8)
Xte=Xte.reshape(-1,1, 8, 8)
Xtr.shape

#image in black and white
plt.imshow(np.transpose(Xtr[0], (1, 2, 0)), cmap='gray')
plt.show()

class CNN(nn.Module):
    def __init__(self, in_channels=1, dropout_prob=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)   # (1,8,8) → (8,8,8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, padding=1)  # (8,4,4) → (64,4,4)
        self.pool = nn.MaxPool2d(2, 2)                            # halves spatial size
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes (digits 0–9)

    def forward(self, x):      # input x shape: (batch_size, 1, 8, 8)
        x = self.pool(F.relu(self.conv1(x)))  # conv1 + relu + pool → (8,4,4)
        x = self.pool(F.relu(self.conv2(x)))  # conv2 + relu + pool → (64,2,2)
        x = torch.flatten(x, 1)               # flatten except batch dim
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

num_epochs=100
lr=0.0005
dropout=0.1
batch_size=64

Xtr = torch.tensor(Xtr, dtype=torch.float32)
Xte = torch.tensor(Xte, dtype=torch.float32)
ytr = torch.tensor(ytr, dtype=torch.long)
yte = torch.tensor(yte, dtype=torch.long)


# Create DataLoaders
train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN( dropout_prob=dropout).to(device)
criterion = nn.CrossEntropyLoss()  # for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


# test the model
model.eval()
all_preds = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        
accuracy = accuracy_score(yte.numpy(), all_preds)
print(f"Test Accuracy: {accuracy:.4f}")


print(classification_report(yte.numpy(), all_preds))
