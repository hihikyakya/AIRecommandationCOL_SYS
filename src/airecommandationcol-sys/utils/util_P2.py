import pandas as pd
import numpy as np


def item_similarity(ratings: pd.DataFrame) -> pd.DataFrame:
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


def predict(ratings, similarity, user, item, k=3, baseline_residual=False):
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