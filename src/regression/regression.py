import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np


def dataset_geeration(csv_path):
    data = pd.read_csv(csv_path)  # 'data.csv' 파일 경로를 입력하세요.

    # Feature와 Target을 분리합니다.
    # selected_columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10'] # 2200 - 5.108
    # selected_columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature8', 'Feature9', 'Feature10'] 
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

class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out


if __name__ =="__main__":

    train_csv_path = '.\\src\\regression\\traindata.csv'
    # train_csv_path = '.\\src\\regression\\augmented_data.csv'
    valid_csv_path = '.\\src\\regression\\validdata.csv'
    save_folder = '.\\src\\regression\\result'
    os.makedirs(save_folder, exist_ok=True)

    train_x, train_y = dataset_geeration(train_csv_path)
    valid_x, valid_y = dataset_geeration(valid_csv_path)

    train_dataloader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)

    input_size = 10 # Feature의 개수
    hidden_size = 50
    output_size = 1  # 예측할 값 (Age)

    model = RegressionModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 학습을 위한 설정
    num_epochs = 10000

    # 학습 루프
    for epoch in range(1, num_epochs):
        for inputs, targets in train_dataloader:
            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch) % 100 == 0:
            model.eval()
            with torch.no_grad():
                predictions = model(valid_x)
                valid_loss = criterion(predictions, valid_y)

                predictions = predictions.cpu().numpy().astype(int)
                true_values = valid_y.cpu().numpy()

                predictions_diff = np.abs(true_values.flatten() - predictions.flatten())  # 절대값으로 변환
                ave_diff = np.sum(predictions_diff) / len(predictions_diff)
                df = pd.DataFrame({'True Values': true_values.flatten(), 'Predictions': predictions.flatten(), 'Difference': predictions_diff})
                df.to_csv(os.path.join(save_folder, f'predictions_{epoch}_{ave_diff:.3f}.csv'), index=False)
            model.train()

            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}, Valid Loss: {valid_loss.item():.4f}, Avg diff: {ave_diff:.4f}')