import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

data = pd.read_csv('traindata.csv')  # 'data.csv' 파일 경로를 입력하세요.

# Feature와 Target을 분리합니다.
train_x = data.iloc[:, 1:].values  # Feature: 1열부터 끝까지
train_y = data.iloc[:, 0].values   # Target: 0열 (Age)

# 데이터를 float32 타입의 numpy array로 변환합니다.
train_x = train_x.astype(np.float32)
train_y = train_y.astype(np.float32).reshape(-1, 1)  # y를 2D 배열로 변환

# 데이터를 Tensor로 변환합니다.
x_tensor = torch.tensor(train_x)
y_tensor = torch.tensor(train_y)

# TensorDataset과 DataLoader를 사용하여 데이터를 로드합니다.
dataset = TensorDataset(x_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    

# 모델, 손실 함수, 최적화 도구를 정의합니다.
input_size = 10  # Feature의 개수
hidden_size = 500
output_size = 1  # 예측할 값 (Age)

model = RegressionModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 학습을 위한 설정
num_epochs = 10000

# 학습 루프
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # 순전파
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # model.eval()
        # with torch.no_grad():
        #     predications = model(valid_x_tensor)


        # model.train()