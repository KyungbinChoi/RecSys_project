# 카테고리 기준 인기도 (클릭수, 사용자 수) 추천

import pandas as pd
from typing import Literal, Optional, Dict, List

def build_category_popularity(
    train_df: pd.DataFrame,
    item_df: pd.DataFrame,
    method: Literal["clicks", "users"] = "clicks",
    min_count: int = 1,
) -> pd.DataFrame:
    """
    학습 데이터(train_df) + 아이템 메타(item_df)를 사용해
    카테고리별 인기 아이템 순위를 계산합니다.

    Parameters
    ----------
    train_df : DataFrame
        전처리된 학습 인터랙션. 필수 컬럼: ['item_id','clicked'].
        method='users'일 때는 'user_id'가 필요(없으면 'user_session_id'로 대체).
    item_df : DataFrame
        아이템 메타. 필수 컬럼: ['item_id','category'] (있으면 'subcategory','title'도 유지).
    method : {'clicks','users'}
        'clicks'  -> 카테고리별 (clicked==1) 합으로 인기 점수 산정
        'users'   -> 카테고리별 (clicked==1) 한 고유 사용자 수(nunique)로 인기 점수 산정
    min_count : int
        이 값 미만의 인기 점수인 아이템은 제외(드문 아이템 제거용)

    Returns
    -------
    pop_df : DataFrame
        컬럼: ['category','item_id','score','rank', (optional) 'subcategory','title']
        - score: 선택한 method 기준의 인기 점수
        - rank : 카테고리별 내림차순 랭크(1 = 최고 인기)
    """
    df = train_df.merge(
        item_df[["item_id", "category", "subcategory", "title"]].drop_duplicates("item_id"),
        on="item_id",
        how="left",
    )

    # 클릭된 행만 카운트(인기도는 클릭 기반이 일반적)
    clicked = df[df["clicked"] == 1].copy()

    if method == "clicks":
        agg = (
            clicked.groupby(["category", "item_id"], as_index=False)["clicked"]
            .sum()
            .rename(columns={"clicked": "score"})
        )
    elif method == "users":
        user_col = "user_id" if "user_id" in clicked.columns else (
            "user_session_id" if "user_session_id" in clicked.columns else None
        )
        if user_col is None:
            raise ValueError("method='users'를 쓰려면 train_df에 'user_id' 또는 'user_session_id'가 필요합니다.")
        agg = (
            clicked.groupby(["category", "item_id"], as_index=False)[user_col]
            .nunique()
            .rename(columns={user_col: "score"})
        )
    else:
        raise ValueError("method는 'clicks' 또는 'users'만 허용됩니다.")

    # 희소 아이템 컷오프
    agg = agg[agg["score"] >= min_count].copy()

    # 메타 병합 및 카테고리 내 랭킹
    pop_df = agg.merge(
        item_df[["item_id", "subcategory", "title"]],
        on="item_id",
        how="left",
    )
    pop_df["rank"] = pop_df.groupby("category")["score"].rank(method="first", ascending=False).astype(int)
    pop_df = pop_df.sort_values(["category", "rank", "item_id"]).reset_index(drop=True)

    return pop_df[["category", "item_id", "score", "rank", "subcategory", "title"]]
