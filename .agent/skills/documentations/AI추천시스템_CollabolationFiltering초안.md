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
- 유저 기반 협업 필터링 예측 공식: \hat{r} = \mean_u + \frac{ \sum_v\in N_k(u,i) {s(u,v) (r_vi - \mean_v)} }{\sum_v {|s(u,v)|}} ; 유사도 함수는 피어슨 유사도
[아래 방식도 참고하여 예측하기]
- 폴백 사다리: 기존 CF 예측 공식 사용 -실패시->\mean_u대신 baseline(b=\mean+b_u+b_i) 사용 -실패시->전체 평균으로 채우기
- hop-2: 예측하려는 값의 이웃의 이웃 유저의 값을 모아서 유저기반 CF진행하기.
U05의 I07 예측에서 Shrinkage 적용 전(β=0)과 후(β=3)의 선택 이웃 및 예측값 차이를 비교하시오.
- 축소 보정 공식: adjusted_sim(u,v) = (\frac{N}{N+\beta}) \times sim(u,v)

[문제 2] Item-based CF 구현

10×10 아이템 유사도 행렬과 미평가 셀을 채운 전체 예측 평점 행렬을 구하시오.
- 아이템 CF 예측 공식: \hat{r_ui} = \mean_u + \frac{ \sum_{j \in N_k(i)\cap I_u}{s(i,j)(r_uj-\mean_u)} }{\sum_j |s(i,j)|} ; 유사도 계산만 item을 기반으로 한다. + 유저 자신의 잔차 이용
평가가 1개뿐인 New User(U09~U12)에서 기본 Item-CF 예측값이 모두 동점으로 나오는 원인을 설명하고, 강의안의 베이스라인 보정 잔차 방식을 적용해 이를 해결하시오.
- 강의자료 안의 베이스라인 보정 잔차 방식: \mean_u 대신 baseline(b=\mean+b_u+b_i)을 사용하여 다시 빈칸들을 채워서 해결하기.
- 동점 원인: 한개일때에 대하여 공식에 대입을 하여 계산을 해보면, mean_u끼리 상쇄되어서 r_ui자신이 나오기 때문이다. => mean_u대신 baseline을 넣으면 r_ui + b_i - b_j가 되어 다른 값으로 채워질 수 있다. <=이 이유 설명을 코드에 담을것!

[문제 3] 사용자·아이템 유형별 추천 결과 비교

12명 사용자 각각의 미평가 아이템 중 Top-2 추천 아이템과 예측 평점을 User-CF와 Item-CF별로 출력하시오.
- 앞에서 User-CF방식으로 채운 table, Item-CF방식으로 채운 테이블을 이용하여, 각각에서 Top-2 추천 아이템과 예측 평점 출력하기 
결과를 바탕으로 User 유형(Heavy / Light / New 및 상수 평가자 U08)과 Item 유형(Popular / Unpopular / New I09, I10)에 따라 두 알고리즘의 작동 여부와 유불리가 어떻게 달라지는지 비교 분석하시오.
- 말그대로 그룹잘 나누고, 각 user와 item 찾고, 시각화나 describe등의 통계화 방법을 이용하여 확인한 후에 그걸 기반으로 User 유형과 Item유형에 따른 차이를 비교 분석하면 됨.