"""피어슨 기반 사용자 협업 필터링."""
import pandas as pd
from ..utils.util_P1 import pearson_similarity, predict_cell, hop2_ratings 


def solve(df_problem: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[문제 1] User-based CF")
    similarity = pearson_similarity(df_problem)
    print("\n[1-1] 12×12 사용자 피어슨 유사도 행렬")
    print(similarity.round(3).to_string())

    predictions = df_problem.copy()
    for user in df_problem.index:
        for item in df_problem.columns[df_problem.loc[user].isna()]:
            value, _ = predict_cell(df_problem, similarity, user, item)
            predictions.loc[user, item] = value
    print("\n[1-2] 미평가 셀을 채운 예측 평점 행렬 (k=3, 양의 유사도 이웃); 실제 서비스에서는 정수로 변환하여 최종 판단에 사용하기")
    print(predictions.round(3).to_string())

    user, item = "U05", "I07"
    hop2 = hop2_ratings(df_problem)
    print("\n[1-3] Hop-2 이웃 예측 (이웃 평점을 1-hop 예측으로 보강)")
    for beta in (0, 3):
        value, neighbors = predict_cell(df_problem, similarity, user, item, beta=beta)
        names = [f"{name}(s={weight:.3f}, n={overlap})" for name, _, weight, overlap in neighbors]
        print(f"  U05-I07, β={beta}: 이웃={names or '없음'}, 예측={value:.3f}")
    hop_sim = pearson_similarity(hop2)
    hop_value, hop_neighbors = predict_cell(hop2, hop_sim, user, item)
    print(f"  Hop-2 U05-I07: 이웃={[name for name, *_ in hop_neighbors]}, 예측={hop_value:.3f}")
    return predictions
