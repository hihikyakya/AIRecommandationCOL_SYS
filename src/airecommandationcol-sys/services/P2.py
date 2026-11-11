"""사용자 잔차에 기반한 아이템 협업 필터링."""

import numpy as np
import pandas as pd


def _item_similarity(ratings: pd.DataFrame) -> pd.DataFrame:
    """공통 평가 사용자가 2명 이상인 아이템 쌍의 피어슨 유사도를 계산한다."""
    items = ratings.columns
    similarity = pd.DataFrame(0.0, index=items, columns=items)
    for item in items:
        similarity.loc[item, item] = 1.0
        for other in items:
            if str(item) >= str(other):
                continue
            common = ratings[[item, other]].dropna()
            if len(common) >= 2:
                value = common[item].corr(common[other])
                similarity.loc[item, other] = similarity.loc[other, item] = (
                    0.0 if pd.isna(value) else value
                )
    return similarity


def _baseline_components(ratings):
    global_mean = float(ratings.stack().mean())
    user_bias = ratings.mean(axis=1).sub(global_mean).fillna(0.0)
    item_bias = ratings.mean(axis=0).sub(global_mean).fillna(0.0)
    return global_mean, user_bias, item_bias


def _predict(ratings, similarity, user, item, k=3, baseline_residual=False):
    user_mean = ratings.loc[user].mean()
    global_mean, user_bias, item_bias = _baseline_components(ratings)
    baseline = global_mean + user_bias[user] + item_bias[item]
    center = baseline if baseline_residual else user_mean
    if pd.isna(center):
        center = global_mean

    neighbors = []
    for other, rating in ratings.loc[user].dropna().items():
        if other == item:
            continue
        weight = float(similarity.loc[item, other])
        if weight > 0:
            neighbors.append((other, float(rating), weight))
    neighbors.sort(key=lambda row: (-row[2], str(row[0])))
    neighbors = neighbors[:k]
    if neighbors:
        if baseline_residual:
            residual = sum(
                weight * (rating - (global_mean + user_bias[user] + item_bias[other]))
                for other, rating, weight in neighbors
            )
        else:
            residual = sum(weight * (rating - user_mean) for _, rating, weight in neighbors)
        scale = sum(abs(weight) for _, _, weight in neighbors)
        if scale:
            return float(np.clip(center + residual / scale, 1, 5))
    return float(np.clip(center, 1, 5))


def solve(df_problem: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[문제 2] Item-based CF")
    similarity = _item_similarity(df_problem)
    print("\n[2-1] 10×10 아이템 피어슨 유사도 행렬")
    print(similarity.round(3).to_string())

    predictions = df_problem.copy()
    baseline_predictions = df_problem.copy()
    for user in df_problem.index:
        for item in df_problem.columns[df_problem.loc[user].isna()]:
            predictions.loc[user, item] = _predict(df_problem, similarity, user, item)
            baseline_predictions.loc[user, item] = _predict(
                df_problem, similarity, user, item, baseline_residual=True
            )
    print("\n[2-2] 기본 Item-CF 예측 행렬 (k=3, 양의 유사도 이웃)")
    print(predictions.round(3).to_string())
    print("\n[2-3] 베이스라인 보정 잔차 Item-CF 예측 행렬")
    print(baseline_predictions.round(3).to_string())

    new_users = ["U09", "U10", "U11", "U12"]
    print("\n[2-4] New User 기본 예측의 동점 원인 및 보정 결과")
    for user in new_users:
        missing = df_problem.loc[user].isna()
        basic = predictions.loc[user, missing]
        corrected = baseline_predictions.loc[user, missing]
        print(
            f"  {user}: 기본 고유 예측값={basic.nunique()}개, "
            f"보정 고유 예측값={corrected.nunique()}개; "
            f"기본={basic.round(2).to_dict()}, 보정={corrected.round(2).to_dict()}"
        )
    return baseline_predictions
