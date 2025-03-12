import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import rrcf  # RRCF 라이브러리: pip install rrcf

import torch


# GPU 장치 설정 (PyTorch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 로그 데이터 로드 및 전처리 (CPU에서 진행)
def load_cloudtrail_logs(data_dir):
    for file in glob.glob(os.path.join(data_dir, "*.json")):
        with open(file, "r") as f:
            data = json.load(f)
            records = data.get("Records", [])
            df = pd.json_normalize(records)
    return pd.DataFrame(df)

def preprocess_logs(df):
    df.fillna("unknown", inplace=True)
    # eventName, eventSource 피처에 대해 one-hot encoding (pandas 이용)
    features = pd.get_dummies(df[["userIdentity.type", "eventName", "userAgent", "requestParameters", "responseElements"]])
    features = features.astype(np.float32)
    return features, df

# 2. GPU를 활용한 데이터 준비: numpy -> PyTorch 텐서로 변환
def prepare_data(features):
    X_np = features.values  # (n_samples, n_features)
    # PyTorch 텐서로 변환하여 GPU로 옮김 (필요시 후속 연산에 활용)
    X_tensor = torch.tensor(X_np).to(device)
    # 여기서는 RCF에 넘겨주기 위해 numpy 배열로 다시 가져오지만,
    # GPU 상에서 추가 전처리나 deep 모델 연산에 활용할 수 있습니다.
    return X_tensor.cpu().numpy()

# 3. RCF (Random Cut Forest) 구축 (CPU 기반)
def build_rcf_forest(X, num_trees=40, tree_size=256):
    n_points = X.shape[0]
    forest = []
    for i in range(num_trees):
        tree = rrcf.RCTree()
        indices = np.random.choice(n_points, size=min(tree_size, n_points), replace=False)
        for idx in indices:
            tree.insert_point(X[idx], index=idx)
        forest.append(tree)
    return forest

def compute_anomaly_scores(X, forest):
    n_points = X.shape[0]
    scores = np.zeros(n_points)
    for tree in forest:
        for leaf in tree.leaves:
            scores[leaf] += tree.codisp(leaf)
    scores /= len(forest)
    return scores

# 4. GPU 활용 TSNE (cuML TSNE 사용 시; 없으면 sklearn TSNE 사용)
def visualize_tsne(X, scores, threshold, save_path="tsne_rcf.png"):
    try:
        from cuml import TSNE as cuTSNE
        print("Using cuML TSNE on GPU.")
        # cuML TSNE는 입력으로 cupy 배열을 요구할 수 있음
        import cupy as cp
        X_cp = cp.asarray(X)
        tsne = cuTSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(X_cp)
        embeddings = cp.asnumpy(embeddings)
    except ImportError:
        print("cuML TSNE not found, falling back to sklearn TSNE on CPU.")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(X)
    
    plt.figure(figsize=(8,6))
    anomalies = scores >= threshold
    plt.scatter(embeddings[~anomalies, 0], embeddings[~anomalies, 1], 
                c='blue', label='Normal', alpha=0.6)
    plt.scatter(embeddings[anomalies, 0], embeddings[anomalies, 1], 
                c='red', label='Anomaly', alpha=0.8)
    plt.legend()
    plt.title("t-SNE visualization of CloudTrail logs (RCF)")
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")

# 5. 전체 파이프라인 실행
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU-enhanced RCF-based Anomaly Detection for AWS CloudTrail Logs")
    parser.add_argument("--data_dir", type=str, default="./flaws_cloudtrail_logs", help="AWS CloudTrail JSON 로그 파일들이 저장된 디렉터리")
    parser.add_argument("--num_trees", type=int, default=100, help="RCF forest 내 트리 개수")
    parser.add_argument("--tree_size", type=int, default=512, help="각 트리에 넣을 최대 데이터 수")
    parser.add_argument("--anomaly_percentile", type=float, default=99, 
                        help="이상치 판별 기준: anomaly score의 상위 몇 퍼센트 이상이면 이상치로 간주")
    args = parser.parse_args()
    
    for i in range(10):
        # 1. 로그 로드
        print(f"Logs loading Start")
        df_logs = load_cloudtrail_logs(args.data_dir)
        print(f"Data loaded from {args.data_dir}")
        print(f"Loaded {len(df_logs)} log entries.")
        
        # 2. 전처리: 피처 벡터 및 원본 DataFrame 확보
        print(f"Preprocessing logs...")
        features, df_original = preprocess_logs(df_logs)
        print(f"Preprocessing complete.")
        # 3. GPU 활용 데이터 준비: one-hot 인코딩된 numpy 배열을 GPU 텐서 처리 후 numpy 배열로 변환
        print(f"Preparing data for RCF...")
        X = prepare_data(features)
        print(f"Feature matrix shape: {X.shape}")
        
        # 4. RCF forest 구축 (CPU)
        print(f"RCF forest counstruct Start.")
        forest = build_rcf_forest(X, num_trees=args.num_trees, tree_size=args.tree_size)
        print("RCF forest constructed.")
        
        # 5. 이상치 점수 계산
        scores = compute_anomaly_scores(X, forest)
        threshold = np.percentile(scores, args.anomaly_percentile)
        print(f"Anomaly threshold (at {args.anomaly_percentile} percentile): {threshold:.4f}")
        
        # 6. 이상치 인덱스 및 로그 출력
        file_path = "./anomalies_rcf.txt"
        anomaly_indices = np.where(scores >= threshold)[0]
        print(f"Detected {len(anomaly_indices)} anomalies.")
        with open(file=file_path, mode='w', encoding='utf-8') as file:
            for idx in anomaly_indices:
                log_entry = df_original.iloc[idx].to_dict()
                file.write(f"Index: {idx}, userIdentity.type: {log_entry.get('userIdentity.type')}, eventName: {log_entry.get('eventName')}, userAgent: {log_entry.get('userAgent')}, requestParameters: {log_entry.get('requestParameters')}, responseElements: {log_entry.get('responseElements')}\n")
            
        # 7. t-SNE를 통한 시각화 (GPU 활용 시 cuML 사용 가능)
        visualize_tsne(X, scores, threshold, save_path=f"tsne_rcf{i}.png")
        
        print(f"Anomaly detection pipeline completed. Results saved to {file_path} and tsne_rcf{i}.png")
