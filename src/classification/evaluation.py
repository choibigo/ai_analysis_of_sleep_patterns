import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('classification_trainset.csv')
X = df.drop('Label', axis=1).values
y = df['Label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_p=0.1):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        
    def forward(self, x):
        residual = x
        x = self.relu(self.layer1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = x + residual
        x = self.relu(x)
        return x
    
class ResidualNN(nn.Module):
    def __init__(self):
        super(ResidualNN, self).__init__()
        self.layer1 = nn.Linear(10, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.residual_block1 = ResidualBlock(16, 16)
        self.residual_block2 = ResidualBlock(16, 16)
        self.residual_block3 = ResidualBlock(16, 16)
        self.output = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.bn1(self.layer1(x))
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.sigmoid(self.output(x))
        return x
    
model = ResidualNN()

model.load_state_dict(torch.load('best_model_0.1629_899_95.56.pth'))
model.eval()

X_test_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.BCELoss()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_test_loss = test_loss / len(test_loader)
test_accuracy = 100 * correct / total

print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")