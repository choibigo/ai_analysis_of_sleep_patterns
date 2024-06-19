import pandas as pd
import numpy as np
import torch

def fill_ages(csv_path, model_path, output_path):
    # CSV 파일을 읽어옵니다.
    data = pd.read_csv(csv_path)

    # Feature와 Target을 분리합니다.
    features = data.iloc[:, 1:].values  # Feature: 1열부터 끝까지
    features = features.astype(np.float32)

    # Torch 모델을 로드합니다.
    model = torch.jit.load(model_path).to('cpu')

    # Feature를 텐서로 변환합니다.
    features_tensor = torch.tensor(features)

    # 모델을 통해 Age를 예측합니다.
    with torch.no_grad():
        predicted_ages = model(features_tensor).numpy()

    # 예측된 Age를 int로 변환하여 데이터프레임에 채웁니다.
    data['Age'] = predicted_ages.astype(int)

    # 수정된 데이터를 새로운 CSV 파일로 저장합니다.
    data.to_csv(output_path, index=False)

if __name__ == "__main__":

    csv_path = '.\\data\\regression_testset.csv'
    model_path = '.\\src\\regression\\artifacts\\model_900.pt'
    output_path = '.\\result\\regression\\filled_regression_testset.csv'

    fill_ages(csv_path, model_path, output_path)
