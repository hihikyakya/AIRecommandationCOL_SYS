"""사용자 잔차에 기반한 아이템 협업 필터링."""

import pandas as pd
from ..utils.util_P2 import item_similarity, predict

def solve(df_problem: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[문제 2] Item-based CF")
    similarity = item_similarity(df_problem)
    print("\n[2-1] 10×10 아이템 피어슨 유사도 행렬")
    print(similarity.round(3).to_string())

    predictions = df_problem.copy()
    baseline_predictions = df_problem.copy()
    for user in df_problem.index:
        for item in df_problem.columns[df_problem.loc[user].isna()]:
            predictions.loc[user, item] = predict(df_problem, similarity, user, item)
            baseline_predictions.loc[user, item] = predict(
                df_problem, similarity, user, item, baseline_residual=True
            )
    print("\n[2-2] 기본 Item-CF 예측 행렬 (k=3, 양의 유사도 이웃)")
    print(predictions.round(3).to_string())
    print("\n[2-3] 베이스라인 보정 잔차 Item-CF 예측 행렬")
    print(baseline_predictions.round(3).to_string())

    new_users = ["U09", "U10", "U11", "U12"]
    print("\n[2-4] New User 기본 예측의 동점 원인 및 보정 결과")
    print(
        "  원인: 기본 Item-CF는 이웃 아이템 평점에서 사용자 평균을 뺀 잔차를 합산합니다. "
        "평점이 1개뿐인 사용자는 그 한 평점이 곧 사용자 평균이므로 관측 잔차가 0입니다. "
        "따라서 이웃이 있어도 보정항이 0이고 모든 미평가 아이템 예측이 동일한 사용자 평균이 됩니다. "
        "이웃이 없을 때도 사용자 평균으로 폴백하므로 같은 동점이 발생합니다."
    )
    for user in new_users:
        observed = df_problem.loc[user].dropna()
        missing = df_problem.loc[user].isna()
        basic = predictions.loc[user, missing]
        corrected = baseline_predictions.loc[user, missing]
        print(
            f"  {user}: 관측 평점={observed.to_dict()}, 사용자 평균={observed.mean():.3f}, "
            f"단일 관측 잔차={observed.iloc[0] - observed.mean():.3f}; "
            f"기본 고유 예측값={basic.nunique()}개, "
            f"보정 고유 예측값={corrected.nunique()}개; "
            f"기본={basic.round(2).to_dict()}, 보정={corrected.round(2).to_dict()}"
        )
    return baseline_predictions
