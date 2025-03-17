import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# GPU 사용 여부에 따라 RAPIDS cuML 라이브러리 사용 여부 결정 (설치되어 있다면)
try:
    from cuml.cluster import HDBSCAN as GPUHDBSCAN
    from cuml.manifold import UMAP as GPUUMAP
    import cupy as cp
    gpu_enabled = True
    print("GPU 환경: RAPIDS cuML 사용")
except ImportError:
    import hdbscan  # CPU 기반 HDBSCAN
    import umap.umap_ as umap  # CPU 기반 UMAP
    gpu_enabled = False
    print("GPU 환경이 아니거나 RAPIDS 라이브러리가 설치되어 있지 않습니다. CPU 기반으로 진행합니다.")


def load_logs_from_directory(directory):
    all_logs = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                    # AWS CloudTrail 로그가 "Records" 키에 리스트 형태로 있을 수도 있음
                    if isinstance(data, dict) and "Records" in data:
                        all_logs.extend(data["Records"])
                    else:
                        if isinstance(data, list):
                            all_logs.extend(data)
                        else:
                            all_logs.append(data)
                except json.JSONDecodeError:
                    print(f"JSON decode error in file: {filename}")
    return all_logs


# 1. 로그 로드
logs = load_logs_from_directory("./logs")
df = pd.DataFrame(logs)
print("불러온 로그 개수:", len(df))
print("로그 필드 예시:", df.columns)


# 2. 전처리 및 피처 생성: userIdentity.type + 기타 관심 필드
def combine_fields(row):
    fields = []
    
    # userIdentity.type 추출
    user_identity = row.get('userIdentity', {})
    if isinstance(user_identity, dict):
        user_type = user_identity.get('type', 'unknown')
        fields.append(user_type)
    
    # 기타 필드
    for col in ['eventName', 'eventSource', 'sourceIPAddress', 'userAgent']:
        if col in row and pd.notnull(row[col]):
            fields.append(str(row[col]))
    
    return " ".join(fields)

df['log_text'] = df.apply(combine_fields, axis=1)

# 3. TF-IDF 벡터화
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['log_text'])
X_array = X.toarray()  # (N, 1000)
feature_names = vectorizer.get_feature_names_out()

# 4. UMAP 차원 축소
if gpu_enabled:
    X_gpu = cp.asarray(X_array)
    umap_model = GPUUMAP(n_components=2, init='spectral', random_state=42)
    embedding = umap_model.fit_transform(X_gpu)
    embedding = cp.asnumpy(embedding)
else:
    umap_model = umap.UMAP(n_components=2, init='spectral', random_state=42)
    embedding = umap_model.fit_transform(X_array)


# 그리드 서치: min_cluster_size를 10부터 200까지 10씩 증가
from sklearn.ensemble import RandomForestClassifier

for cluster_num in range(10, 210, 10):
    # 5. HDBSCAN 클러스터링
    if gpu_enabled:
        clusterer = GPUHDBSCAN(min_cluster_size=cluster_num)
        embedding_gpu = cp.asarray(embedding)
        cluster_labels = clusterer.fit_predict(embedding_gpu)
        cluster_labels = cp.asnumpy(cluster_labels)
    else:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_num)
        cluster_labels = clusterer.fit_predict(embedding)

    df['cluster'] = cluster_labels

    # ===== 분류 모델 기반 피처 중요도 분석 =====
    # -1(이상치) 제외 후, 랜덤 포레스트로 "군집 라벨"을 예측
    valid_mask = (cluster_labels != -1)
    X_valid = X_array[valid_mask]
    y_valid = cluster_labels[valid_mask]

    # 군집 라벨이 아예 없거나(전부 -1) 클래스가 1개뿐이면 스킵
    unique_clusters = np.unique(y_valid)
    if len(unique_clusters) < 2:
        print(f"[min_cluster_size={cluster_num}] 유효 군집 라벨이 1개 이하이므로 분류 모델 생략.")
        # 시각화 및 파일 기록은 계속 진행
    else:
        # 분류 모델 학습
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_valid, y_valid)
        
        # 피처 중요도
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # 분류 정확도 (참고용)
        acc = rf.score(X_valid, y_valid)

        # 결과 저장
        imp_file_path = f"./rf_importance_{cluster_num}.txt"
        with open(imp_file_path, 'w', encoding='utf-8') as f_imp:
            f_imp.write(f"[RandomForest Feature Importance] min_cluster_size={cluster_num}\n")
            f_imp.write(f"유효 클러스터 수(=-1 제외): {len(unique_clusters)}\n")
            f_imp.write(f"클래스 라벨들: {unique_clusters}\n")
            f_imp.write(f"학습 샘플 수: {len(X_valid)}\n")
            f_imp.write(f"분류 정확도(Train set): {acc:.4f}\n\n")

            f_imp.write("상위 20개 피처 중요도:\n")
            top_n = min(20, len(importances))
            for i in range(top_n):
                feat_idx = indices[i]
                feat_name = feature_names[feat_idx]
                imp_val = importances[feat_idx]
                f_imp.write(f"{i+1:2d}. {feat_name:20s}: {imp_val:.4f}\n")
        print(f"[min_cluster_size={cluster_num}] 랜덤 포레스트 분류 모델 학습 완료. 결과: {imp_file_path}")

    # ========== 클러스터별 상위 TF-IDF 단어 (기존 코드) ==========
    unique_labels = sorted(set(cluster_labels) - {-1})
    top_n = 10
    file_path = f"./hyparam_{cluster_num}_relation.txt"
    with open(file=file_path, mode='w', encoding='utf-8') as file:
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_vectors = X_array[cluster_indices]
            cluster_mean = cluster_vectors.mean(axis=0)
            top_feature_indices = np.argsort(cluster_mean)[::-1][:top_n]
            top_features = [feature_names[i] for i in top_feature_indices]
            top_values = cluster_mean[top_feature_indices]

            file.write(f"=== Cluster {label} ===\n")
            for feature, val in zip(top_features, top_values):
                file.write(f"{feature:20s} : {val:.4f}\n")

    # ========== 시각화: 각 클러스터마다 색 다르게, 이상치는 빨간 X ==========
    plt.figure(figsize=(10, 7))

    outlier_mask = (cluster_labels == -1)
    plt.scatter(
        embedding[outlier_mask, 0],
        embedding[outlier_mask, 1],
        c='red',
        marker='x',
        s=50,
        alpha=0.8,
        label='Outliers'
    )

    unique_labels = sorted(set(cluster_labels) - {-1})
    cmap = plt.cm.get_cmap('Spectral', len(unique_labels))

    for i, label in enumerate(unique_labels):
        points = embedding[cluster_labels == label]
        color = cmap(i)
        
        plt.scatter(points[:, 0], points[:, 1],
                    color=color,
                    s=50,
                    alpha=0.8,
                    label=f'Cluster {label}')
        
        if len(points) >= 3:
            hull = ConvexHull(points)
            hull_vertices = np.append(hull.vertices, hull.vertices[0])
            plt.plot(points[hull_vertices, 0], points[hull_vertices, 1],
                     c=color, lw=2)

    plt.title(f"AWS CloudTrail Logs - HDBSCAN Clustering ({cluster_num})")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend()
    plt.savefig(f"hdbscan_clusters_{cluster_num}.png")
    plt.close()

    # ========== 이상치 로그 통계 기록 ==========
    outliers = df[df['cluster'] == -1]
    file_path = f"./hyparam_{cluster_num}.txt"
    with open(file=file_path, mode='w', encoding='utf-8') as file:
        file.write(f"min_cluster_size: {cluster_num}\n")
        file.write(f"탐지된 이상치 개수: {len(outliers)}\n")
        file.write(f"이상치 비율: {len(outliers) / len(df) * 100:.2f}%\n")
        file.write(f"클러스터 개수(=-1 제외): {len(unique_labels)}\n")
