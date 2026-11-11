import numpy as np
nan = np.nan
USERS = [f"U{i:02d}" for i in range(1, 13)]
ITEMS = [f"I{j:02d}" for j in range(1, 11)]
R = np.array([
    [ 5.0, 5.0, 3.0, 2.0, 5.0, 5.0, 1.0, nan, nan, nan], 
    [ 5.0, 4.0, 3.0, 2.0, 5.0, 5.0, 1.0, nan, 5.0, nan], 
    [ 3.0, 2.0, 5.0, 5.0, 1.0, 1.0, 5.0, 4.0, nan, nan], 
    [ 3.0, 2.0, 5.0, 5.0, 1.0, 2.0, 4.0, 4.0, nan, nan], 
    [ 5.0, 4.0, 3.0, 1.0, 5.0, nan, nan, nan, nan, nan], 
    [ 2.0, 2.0, 5.0, 5.0, nan, nan, nan, 4.0, nan, nan], 
    [ 4.0, nan, 2.0, nan, nan, 4.0, 5.0, nan, nan, nan],
    [ 2.0, 2.0, 2.0, 2.0, nan, nan, nan, 2.0, nan, nan], 
    [ 5.0, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [ nan, nan, 5.0, nan, nan, nan, nan, nan, nan, nan], 
    [ nan, 5.0, nan, nan, nan, nan, nan, nan, nan, nan], 
    [ nan, nan, nan, 5.0, nan, nan, nan, nan, nan, nan]
], dtype=float)

[문제 1] User-based CF 구현 

12×12 사용자 유사도 행렬과 미평가 셀을 채운 전체 예측 평점 행렬을 구하시오.
U05의 I07 예측에서 Shrinkage 적용 전(β=0)과 후(β=3)의 선택 이웃 및 예측값 차이를 비교하시오.
[문제 2] Item-based CF 구현

10×10 아이템 유사도 행렬과 미평가 셀을 채운 전체 예측 평점 행렬을 구하시오.
평가가 1개뿐인 New User(U09~U12)에서 기본 Item-CF 예측값이 모두 동점으로 나오는 원인을 설명하고, 강의안의 베이스라인 보정 잔차 방식을 적용해 이를 해결하시오.
[문제 3] 사용자·아이템 유형별 추천 결과 비교

12명 사용자 각각의 미평가 아이템 중 Top-2 추천 아이템과 예측 평점을 User-CF와 Item-CF별로 출력하시오.
결과를 바탕으로 User 유형(Heavy / Light / New 및 상수 평가자 U08)과 Item 유형(Popular / Unpopular / New I09, I10)에 따라 두 알고리즘의 작동 여부와 유불리가 어떻게 달라지는지 비교 분석하시오.