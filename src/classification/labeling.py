import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

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
        self.layer1 = nn.Linear(9, 16)
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

model.load_state_dict(torch.load('best_model/best_model_0.1501_994_96.30.pth'))
model.eval()

current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, '../../data/classification_testset.csv')
test_df = pd.read_csv(data_file_path)
X_test = test_df.drop(['Label', 'Feature8'], axis=1).values

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs > 0.5).float()

test_df['Label'] = predicted.numpy().astype(int)
test_df.to_csv('classification_testset_with_labels.csv', index=False)