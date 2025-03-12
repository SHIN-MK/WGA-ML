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

print("로그 필드 예시:")
print(df.columns)

# 2. 전처리 및 피처 생성
def combine_fields(row):
    fields = []
    for col in ["userIdentity", "eventName", "userAgent", "requestParameters", "responseElements"]:
        if col in row and pd.notnull(row[col]):
            fields.append(str(row[col]))
    return " ".join(fields)

df['log_text'] = df.apply(combine_fields, axis=1)


# 3. 텍스트 벡터화
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['log_text'])
X_array = X.toarray()  # (N, 1000)
feature_names = vectorizer.get_feature_names_out()

# 4. 차원 축소 (UMAP)
if gpu_enabled:
    X_gpu = cp.asarray(X_array)
    umap_model = GPUUMAP(n_components=2, random_state=42)
    embedding = umap_model.fit_transform(X_gpu)
    embedding = cp.asnumpy(embedding)
else:
    umap_model = umap.UMAP(n_components=2, random_state=42)
    embedding = umap_model.fit_transform(X_array)


# 5. HDBSCAN 클러스터링
if gpu_enabled:
    clusterer = GPUHDBSCAN(min_cluster_size=10)
    embedding_gpu = cp.asarray(embedding)
    cluster_labels = clusterer.fit_predict(embedding_gpu)
    cluster_labels = cp.asnumpy(cluster_labels)
else:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    cluster_labels = clusterer.fit_predict(embedding)

df['cluster'] = cluster_labels


## 피쳐
unique_labels = sorted(set(cluster_labels) - {-1})  # -1(이상치) 제외
top_n = 10  # 각 클러스터마다 표시할 상위 단어 개수

for label in unique_labels:
    # label 클러스터에 속한 데이터 인덱스
    cluster_indices = np.where(cluster_labels == label)[0]
    
    # 해당 클러스터 내 TF-IDF 벡터들
    cluster_vectors = X_array[cluster_indices]
    
    # 클러스터 내 평균 TF-IDF 계산 (shape: (1000,))
    cluster_mean = cluster_vectors.mean(axis=0)
    
    # 평균값이 큰 순으로 정렬해 상위 top_n 피처 인덱스 추출
    top_feature_indices = np.argsort(cluster_mean)[::-1][:top_n]
    
    # 실제 단어(피처) 이름
    top_features = [feature_names[i] for i in top_feature_indices]
    
    # 해당 단어들의 평균 TF-IDF 값
    top_values = cluster_mean[top_feature_indices]
    
    print(f"\n=== Cluster {label} ===")
    for feature, val in zip(top_features, top_values):
        print(f"{feature:20s} : {val:.4f}")

# 6. 시각화: 각 정상 클러스터마다 색을 다르게, 이상치는 빨간색 X
plt.figure(figsize=(10, 7))

# 먼저 이상치(-1)만 따로 표시
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

# 정상 클러스터 라벨들
unique_labels = sorted(set(cluster_labels) - {-1})

# 색상 팔레트(예: Spectral)에서 len(unique_labels)개 색을 추출
cmap = plt.cm.get_cmap('Spectral', len(unique_labels))

for i, label in enumerate(unique_labels):
    points = embedding[cluster_labels == label]
    color = cmap(i)  # 팔레트에서 i번째 색 추출
    
    # 클러스터 산점도
    plt.scatter(points[:, 0], points[:, 1],
                color=color,
                s=50,
                alpha=0.8,
                label=f'Cluster {label}')
    
    # 7. 클러스터 범위를 Convex Hull로 표시
    if len(points) >= 3:
        hull = ConvexHull(points)
        hull_vertices = np.append(hull.vertices, hull.vertices[0])
        plt.plot(points[hull_vertices, 0], points[hull_vertices, 1],
                 c=color, lw=2)
        
        # 헐 내부 중심(centroid)에 레이블 텍스트 표시
        # centroid = np.mean(points, axis=0)
        # plt.text(centroid[0], centroid[1], f"Cluster {label}",
        #          color=color, fontsize=12, ha='center', va='center',
        #          bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

plt.title("AWS CloudTrail Logs - HDBSCAN Clustering (Distinct Colors)")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend()
plt.show()

# 8. 이상치 로그 확인
outliers = df[df['cluster'] == -1]
print(f"탐지된 이상치 개수: {len(outliers)}")

if not outliers.empty:
    print("이상치에 해당하는 로그 예시:")
    display_cols = ["userIdentity", "eventName", "userAgent", "requestParameters", "responseElements"]
    print(outliers[display_cols].head(10))
else:
    print("이상치로 판단된 로그가 없습니다.")
