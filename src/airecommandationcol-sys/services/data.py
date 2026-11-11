"""문제에서 제공한 작은 평점 행렬을 표 형태로 구성한다."""

import numpy as np
import pandas as pd


USERS = [f"U{i:02d}" for i in range(1, 13)]
ITEMS = [f"I{i:02d}" for i in range(1, 11)]


def make_table():
    """행은 사용자, 열은 아이템인 관측 평점 DataFrame을 반환한다."""
    ratings = np.array(
        [
            [5, 5, 3, 2, 5, 5, 1, np.nan, np.nan, np.nan],
            [5, 4, 3, 2, 5, 5, 1, np.nan, 5, np.nan],
            [3, 2, 5, 5, 1, 1, 5, 4, np.nan, np.nan],
            [3, 2, 5, 5, 1, 2, 4, 4, np.nan, np.nan],
            [5, 4, 3, 1, 5, np.nan, np.nan, np.nan, np.nan, np.nan],
            [2, 2, 5, 5, np.nan, np.nan, np.nan, 4, np.nan, np.nan],
            [4, np.nan, 2, np.nan, np.nan, 4, 5, np.nan, np.nan, np.nan],
            [2, 2, 2, 2, np.nan, np.nan, np.nan, 2, np.nan, np.nan],
            [5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=float,
    )
    return pd.DataFrame(ratings, index=USERS, columns=ITEMS)
