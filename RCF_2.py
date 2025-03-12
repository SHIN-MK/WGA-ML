# # AWS CloudTrail 로그 이상치 탐지를 위한 Random Cut Forest 클러스터링
# 
# 이 노트북에서는 여러 JSON 파일로 저장된 AWS CloudTrail 로그를 읽어와서
# 1. 전처리 (필요한 필드 추출 및 수치형 데이터 변환)
# 2. PyTorch를 이용해 데이터를 GPU로 옮기는 예제 (전처리 단계)
# 3. rrcf 라이브러리를 이용한 Random Cut Forest를 통해 이상치 탐지
# 4. TSNE를 이용한 2차원 시각화 및 클러스터링 결과 확인
# 5. 이상치 로그의 세부 이벤트 정보를 사용자에게 출력
# 
# 순서에 따라 코드를 실행하면 로그 데이터 내 이상치(Anomaly) 이벤트를 확인할 수 있습니다.

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# anomaly detection용 라이브러리
import rrcf

# 전처리 및 시각화를 위한 라이브러리
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# GPU 사용을 위한 PyTorch (전처리 단계에서 GPU 메모리 활용 예제)
import torch

# Step 1. 여러 JSON 파일로 저장된 AWS CloudTrail 로그 불러오기 및 DataFrame 생성
def load_cloudtrail_logs(log_dir):
    """
    log_dir 내의 모든 .json 파일을 읽어와서, 
    각 파일 내 'Records' 항목을 추출하여 DataFrame으로 반환합니다.
    """
    records = []
    json_files = glob.glob(os.path.join(log_dir, '*.json'))
    for file in json_files:
        with open(file, 'r') as f:
            try:
                data = json.load(f)
                # CloudTrail 로그는 보통 'Records' 키에 이벤트 리스트를 담고 있음
                if 'Records' in data:
                    records.extend(data['Records'])
            except Exception as e:
                print(f"Error reading {file}: {e}")
    return pd.DataFrame(records)

# JSON 로그 파일들이 저장된 디렉토리 경로 (사용자에 맞게 수정)
log_directory = "./flaws_cloudtrail_logs"
df_logs = load_cloudtrail_logs(log_directory)

print(f"불러온 로그 건수: {df_logs.shape[0]}")

# Step 2. 로그 데이터 전처리 및 feature 생성
# 예제에서는 eventTime, eventName, eventSource 3가지를 이용합니다.
# eventTime은 타임스탬프(초)로 변환하고, eventName과 eventSource는 카테고리 인코딩합니다.
try:
    # eventTime을 datetime으로 변환 후, Unix timestamp(초)로 변환
    df_logs['timestamp'] = pd.to_datetime(df_logs['eventTime']).astype(np.int64) // 10**9
except Exception as e:
    print("eventTime 변환 중 오류:", e)

# eventName과 eventSource는 카테고리 코드로 변환 (문자열을 수치값으로)
if 'eventName' in df_logs.columns:
    df_logs['eventName_code'] = df_logs['eventName'].astype('category').cat.codes
else:
    df_logs['eventName_code'] = 0

if 'eventSource' in df_logs.columns:
    df_logs['eventSource_code'] = df_logs['eventSource'].astype('category').cat.codes
else:
    df_logs['eventSource_code'] = 0

# 최종 feature matrix 생성 (추가 feature가 필요하면 여기에 추가)
features = df_logs[['timestamp', 'eventName_code', 'eventSource_code']].values

# 스케일링 (정규화) 수행
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 2-1. GPU를 사용한 전처리 예제 (PyTorch를 사용하여 GPU 메모리로 전송)
# (실제 RCF 알고리즘은 CPU 기반이지만, 대용량 데이터 전처리 시 GPU를 활용할 수 있습니다.)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
print("features_tensor device:", features_tensor.device)

# rrcf 알고리즘은 numpy array를 필요로 하므로, GPU tensor를 다시 CPU로 옮겨 numpy array로 변환
features_cpu = features_tensor.cpu().numpy()

# Step 3. Random Cut Forest를 이용한 이상치 탐지
# RRCF 라이브러리를 이용하여 여러 개의 트리를 생성하고,
# 각 데이터 포인트에 대해 평균 코디스플레이션(codisp) 점수를 계산합니다.
num_trees = 100  # 생성할 트리 개수 (파라미터 조정 가능)
forest = []      # forest에 각 트리(RCTree) 저장

# 각 트리는 독립적으로 데이터를 삽입합니다.
for i in range(num_trees):
    tree = rrcf.RCTree()
    forest.append(tree)

# 각 데이터 포인트를 모든 트리에 삽입
num_points = features_cpu.shape[0]
for index, point in enumerate(features_cpu):
    for tree in forest:
        tree.insert_point(point, index=index)

# 각 포인트의 이상치 점수를 계산
anomaly_scores = np.zeros(num_points)
for tree in forest:
    for index in tree.leaves:
        anomaly_scores[index] += tree.codisp(index)

# 평균 anomaly score (트리 수로 나눔)
anomaly_scores /= num_trees

# DataFrame에 anomaly score 컬럼 추가
df_logs['anomaly_score'] = anomaly_scores

# Step 4. TSNE를 이용한 2차원 시각화 (클러스터링 및 이상치 시각화)
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(features_scaled)

# TSNE 결과를 DataFrame에 추가
df_logs['tsne_x'] = tsne_results[:, 0]
df_logs['tsne_y'] = tsne_results[:, 1]

plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_logs['tsne_x'], df_logs['tsne_y'], c=df_logs['anomaly_score'],
                      cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Anomaly Score')
plt.title('t-SNE Visualize: AWS CloudTrail Logs')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()

# Step 5. 이상치 로그 확인 및 출력
# 예제에서는 anomaly_score가 높은 상위 5건의 로그를 이상치로 판단합니다.
top_anomalies = df_logs.nlargest(5, 'anomaly_score')
print("이상치(Anomaly)로 판단된 로그:")
print(top_anomalies[['eventTime', 'eventName', 'eventSource', 'anomaly_score']])


