import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GPU / CPU 분기
try:
    import cupy as cp
    from cuml.cluster import HDBSCAN as GPUHDBSCAN
    from cuml.ensemble import RandomForestClassifier as cumlRF
    gpu_enabled = True
    print("GPU 환경: RAPIDS cuML 사용")
except ImportError:
    import hdbscan
    from sklearn.ensemble import RandomForestClassifier as skRF
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


def main():
    # 1) 로그 로드
    logs = load_logs_from_directory("./logs")
    df = pd.DataFrame(logs)
    print("불러온 로그 개수:", len(df))

    # 2) userIdentity.type, eventName, eventSource, sourceIPAddress, userAgent 컬럼만 추출
    #    없으면 'unknown'으로 채우는 식으로 처리
    def safe_get(row, col):
        val = row.get(col, 'unknown')
        if pd.isnull(val):
            return 'unknown'
        return str(val)

    # userIdentity.type은 별도 추출
    def get_user_identity_type(row):
        ui = row.get('userIdentity', {})
        if isinstance(ui, dict):
            return ui.get('type', 'unknown')
        return 'unknown'

    df['userIdentity_type'] = df.apply(get_user_identity_type, axis=1)
    df['eventName_'] = df.apply(lambda r: safe_get(r, 'eventName'), axis=1)
    df['eventSource_'] = df.apply(lambda r: safe_get(r, 'eventSource'), axis=1)
    df['sourceIPAddress_'] = df.apply(lambda r: safe_get(r, 'sourceIPAddress'), axis=1)
    df['userAgent_'] = df.apply(lambda r: safe_get(r, 'userAgent'), axis=1)

    # 3) 원-핫 인코딩
    #    userAgent나 IP가 너무 다양하면 메모리 폭발 가능 → 필요 시 필터링/샘플링 권장
    selected_cols = ['userIdentity_type','eventName_','eventSource_','sourceIPAddress_','userAgent_']
    df_selected = df[selected_cols].fillna('unknown')

    # pd.get_dummies로 원-핫 인코딩
    features = pd.get_dummies(df_selected, columns=selected_cols, dummy_na=False)
    print("원-핫 인코딩 후 shape:", features.shape)

    X_array = features.values  # (n_samples, n_encoded_features)

    # 4) HDBSCAN
    min_cluster_size = 50  # 예시로 50
    if gpu_enabled:
        X_gpu = cp.asarray(X_array)
        clusterer = GPUHDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(X_gpu)
        cluster_labels = cp.asnumpy(cluster_labels)
    else:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(X_array)

    df['cluster'] = cluster_labels
    print("HDBSCAN 완료. 군집 라벨 분포:", pd.Series(cluster_labels).value_counts())

    # 5) 랜덤 포레스트 분류 모델 (클러스터 라벨 예측)
    valid_mask = (cluster_labels != -1)
    X_valid = X_array[valid_mask]
    y_valid = cluster_labels[valid_mask]

    unique_clusters = np.unique(y_valid)
    if len(unique_clusters) < 2:
        print("유효 군집 라벨이 1개 이하이므로 분류 모델 생략.")
        return

    if gpu_enabled:
        X_valid_gpu = cp.asarray(X_valid)
        y_valid_gpu = cp.asarray(y_valid)
        rf = cumlRF(n_estimators=100, random_state=42)
        rf.fit(X_valid_gpu, y_valid_gpu)

        preds_gpu = rf.predict(X_valid_gpu)
        preds = cp.asnumpy(preds_gpu)
        acc = (preds == y_valid).mean()

        importances = rf.feature_importances_
        importances = cp.asnumpy(importances)
    else:
        rf = skRF(n_estimators=100, random_state=42)
        rf.fit(X_valid, y_valid)
        preds = rf.predict(X_valid)
        acc = (preds == y_valid).mean()
        importances = rf.feature_importances_

    print(f"랜덤 포레스트 분류 정확도: {acc:.4f}")

    # 6) 피처 중요도 출력 (상위 20개)
    indices = np.argsort(importances)[::-1]
    top_n = min(20, len(importances))
    feature_cols = features.columns  # get_dummies 후 컬럼명
    print("\n상위 20개 피처 중요도:")
    for i in range(top_n):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_cols[idx]:50s} : {importances[idx]:.4f}")


if __name__ == "__main__":
    main()
