import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np


def dataset_geration(csv_path):
    data = pd.read_csv(csv_path)  # 'data.csv' 파일 경로를 입력하세요.

    # Feature와 Target을 분리합니다.
    # selected_columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature6', 'Feature7', 'Feature8', 'Feature10'] 
    # x = data[selected_columns].values

    x = data.iloc[:, 1:].values  # Feature: 1열부터 끝까지
    y = data.iloc[:, 0].values   # Target: 0열 (Age)

    # 데이터를 float32 타입의 numpy array로 변환합니다.
    x = x.astype(np.float32)
    y = y.astype(np.float32).reshape(-1, 1)  # y를 2D 배열로 변환

    # 데이터를 Tensor로 변환합니다.
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)

    return x_tensor, y_tensor

class ResidualBlock(nn.Module):
    def __init__(self, size, dropout_prob=0.5):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(size, size)
        self.bn = nn.BatchNorm1d(size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        identity = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out += identity
        out = self.relu(out)
        return out

class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.residual_block1 = ResidualBlock(hidden_size, dropout_prob)
        self.residual_block2 = ResidualBlock(hidden_size, dropout_prob)
        
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        
        out = self.fc2(out)
        return out

if __name__ =="__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_csv_path = '.\\src\\regression\\origin_traindata.csv'
    valid_csv_path = '.\\src\\regression\\origin_validdata.csv'
    save_folder = '.\\src\\regression\\artifacts'
    os.makedirs(save_folder, exist_ok=True)

    train_x, train_y = dataset_geration(train_csv_path)
    valid_x, valid_y = dataset_geration(valid_csv_path)

    batch_size = 64
    train_dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    input_size = 10
    hidden_size = 32
    output_size = 1
    num_epochs = 10000

    model = RegressionModel(input_size, hidden_size, output_size).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(1, num_epochs):
        for inputs, targets in train_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch) % 100 == 0:
            model.eval()
            with torch.no_grad():
                valid_x = valid_x.to(device)
                valid_y = valid_y.to(device)

                predictions = model(valid_x)
                valid_loss = criterion(predictions, valid_y)

                predictions = predictions.cpu().numpy().astype(int)
                true_values = valid_y.cpu().numpy()

                predictions_diff = np.abs(true_values.flatten() - predictions.flatten())  # 절대값으로 변환
                ave_diff = np.sum(predictions_diff) / len(predictions_diff)
                df = pd.DataFrame({'True Values': true_values.flatten(), 'Predictions': predictions.flatten(), 'Difference': predictions_diff})
                df.to_csv(os.path.join(save_folder, f'predictions_{epoch}_{ave_diff:.3f}.csv'), index=False)

            torch.jit.save(torch.jit.script(model), os.path.join(save_folder, f'model_{epoch}.pt'))

            model.train()

            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}, Valid Loss: {valid_loss.item():.4f}, Avg diff: {ave_diff:.4f}')