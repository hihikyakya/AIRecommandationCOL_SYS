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
