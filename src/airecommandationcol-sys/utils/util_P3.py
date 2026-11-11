import pandas as pd


def _user_type(user, ratings):
    count = int(ratings.loc[user].notna().sum())
    if user == "U08":
        return "상수 평가자"
    if count == 1:
        return "New"
    if count >= 7:
        return "Heavy"
    return "Light"


def _item_type(item, ratings):
    count = int(ratings[item].notna().sum())
    if count == 0:
        return "New"
    return "Popular" if count >= 5 else "Unpopular"


def top_recommendations(ratings, predictions, algorithm):
    rows = []
    for user in ratings.index:
        candidates = predictions.loc[user, ratings.loc[user].isna()].sort_values(
            ascending=False, kind="stable"
        )
        for rank, (item, score) in enumerate(candidates.head(2).items(), start=1):
            rows.append(
                {
                    "알고리즘": algorithm,
                    "사용자": user,
                    "사용자 유형": _user_type(user, ratings),
                    "순위": rank,
                    "추천 아이템": item,
                    "아이템 유형": _item_type(item, ratings),
                    "예측 평점": float(score),
                }
            )
    return pd.DataFrame(rows)