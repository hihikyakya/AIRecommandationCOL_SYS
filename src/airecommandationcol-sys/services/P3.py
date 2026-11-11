"""사용자와 아이템 유형에 따른 추천 결과 비교."""

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


def _top_recommendations(ratings, predictions, algorithm):
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


def solve(df_problem: pd.DataFrame, df_userCF: pd.DataFrame, df_itemCF: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[문제 3] 사용자·아이템 유형별 추천 결과 비교")
    user_results = _top_recommendations(df_problem, df_userCF, "User-CF")
    item_results = _top_recommendations(df_problem, df_itemCF, "Item-CF")
    results = pd.concat([user_results, item_results], ignore_index=True)
    print("\n[3-1] 사용자별 미평가 아이템 Top-2 추천")
    print(results.to_string(index=False, formatters={"예측 평점": "{:.3f}".format}))

    print("\n[3-2] 사용자 유형별 추천 평점 요약")
    user_summary = results.groupby(["알고리즘", "사용자 유형"], observed=True)["예측 평점"].agg(
        추천수="count", 평균="mean", 표준편차="std"
    )
    print(user_summary.round(3).to_string())
    print("\n[3-3] 아이템 유형별 추천 평점 요약")
    item_summary = results.groupby(["알고리즘", "아이템 유형"], observed=True)["예측 평점"].agg(
        추천수="count", 평균="mean", 표준편차="std"
    )
    print(item_summary.round(3).to_string())

    print("\n[3-4] 작동 특성 해석")
    print(
        "  - User-CF는 공통으로 평가한 아이템에서 계산한 양의 피어슨 이웃이 있어야 "
        "개인화된 보정이 가능하며, 이웃이 없으면 사용자 평균으로 폴백합니다."
    )
    print(
        "  - Item-CF는 사용자 자신의 평점 잔차를 쓰므로 평가 기록이 있는 사용자에게 "
        "적용할 수 있지만, 상수 평가자 U08의 잔차는 모두 0입니다."
    )
    print(
        "  - 아이템 유사도 계산에 공통 평가자가 필요하므로 신규 아이템 I09·I10은 "
        "유사 이웃이 없습니다. 베이스라인 보정은 이 경우 사용자·아이템 평균 편향으로 폴백합니다."
    )
    print(
        "  - Heavy 사용자는 User-CF의 이웃 근거가 상대적으로 풍부하고, Light/New 사용자는 "
        "이웃 부족으로 폴백에 더 의존합니다. Popular 아이템은 관측 근거가 많고, "
        "Unpopular/New 아이템은 유사도 추정이 불안정하거나 불가능합니다."
    )
    return results
