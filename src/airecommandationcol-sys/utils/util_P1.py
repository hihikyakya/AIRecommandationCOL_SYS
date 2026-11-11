import pandas as pd
import numpy as np

def pearson_similarity(ratings: pd.DataFrame) -> pd.DataFrame:
    """공통 평가가 2개 이상인 사용자 쌍의 피어슨 상관계수를 계산한다."""
    users = ratings.index
    sim = pd.DataFrame(0.0, index=users, columns=users)
    for u in users:
        sim.loc[u, u] = 1.0
        for v in users:
            if u >= v:
                continue
            common = ratings.loc[[u, v]].dropna(axis=1)
            if common.shape[1] >= 2:
                value = common.loc[u].corr(common.loc[v])
                sim.loc[u, v] = sim.loc[v, u] = 0.0 if pd.isna(value) else value
    return sim


def _baseline(ratings: pd.DataFrame):
    global_mean = float(ratings.stack().mean())
    user_bias = ratings.mean(axis=1).sub(global_mean).fillna(0.0)
    item_bias = ratings.mean(axis=0).sub(global_mean).fillna(0.0)
    return global_mean, user_bias, item_bias


def predict_cell(ratings, similarities, user, item, k=3, beta=0.0, use_baseline=False):
    mean_u = ratings.loc[user].mean()
    global_mean, user_bias, item_bias = _baseline(ratings)
    base = (global_mean + user_bias[user] + item_bias[item]) if use_baseline else mean_u
    if pd.isna(base):
        base = global_mean

    candidates = ratings[item].dropna().drop(user, errors="ignore")
    neighbors = []
    for neighbor, rating in candidates.items():
        overlap = int((ratings.loc[user].notna() & ratings.loc[neighbor].notna()).sum())
        raw_sim = float(similarities.loc[user, neighbor])
        adjusted = raw_sim * overlap / (overlap + beta) if overlap + beta else 0.0
        if adjusted > 0:
            neighbors.append((neighbor, rating, adjusted, overlap))
    neighbors.sort(key=lambda row: (-row[2], str(row[0])))
    neighbors = neighbors[:k]

    if neighbors:
        numerator = sum(
            weight * (rating - ratings.loc[neighbor].mean())
            for neighbor, rating, weight, _ in neighbors
        )
        denominator = sum(abs(weight) for _, _, weight, _ in neighbors)
        if denominator:
            return float(np.clip(base + numerator / denominator, 1, 5)), neighbors
    return float(np.clip(base, 1, 5)), []


def hop2_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """미평가 셀을 1-hop 이웃의 평점으로 확장해 2-hop 이웃을 만든다."""
    sim = pearson_similarity(ratings)
    expanded = ratings.copy()
    for user in ratings.index:
        for item in ratings.columns[ratings.loc[user].isna()]:
            value, _ = predict_cell(ratings, sim, user, item)
            expanded.loc[user, item] = value
    return expanded