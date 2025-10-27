import pandas as pd
import numpy as np
from scipy import sparse

def build_id_maps(train_df: pd.DataFrame):
    """연속 인덱스 매핑(user↔uid, item↔iid) 반환"""
    users = train_df["user_id"].astype(str).unique()
    items = train_df["item_id"].astype(str).unique()
    user2idx = {u:i for i,u in enumerate(users)}
    item2idx = {it:i for i,it in enumerate(items)}
    idx2user = np.array(users)
    idx2item = np.array(items)
    return user2idx, item2idx, idx2user, idx2item

def build_user_item_csr(train_df: pd.DataFrame,
                        user2idx: dict,
                        item2idx: dict,
                        use_bm25: bool = True):
    """
    사용자-아이템 클릭 기반 implicit matrix 생성 (CSR: shape [n_users, n_items])
    clicked==1만 신호로 사용. 필요하면 가중치로 변형 가능.
    """
    df = train_df.loc[train_df["clicked"] == 1, ["user_id","item_id"]].astype(str)
    ui = df.assign(
        u=lambda d: d["user_id"].map(user2idx),
        i=lambda d: d["item_id"].map(item2idx),
        v=1.0
    ).dropna(subset=["u","i"])
    rows = ui["u"].astype(int).to_numpy()
    cols = ui["i"].astype(int).to_numpy()
    data = ui["v"].to_numpy(dtype=np.float32)
    n_users = len(user2idx); n_items = len(item2idx)
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

    if use_bm25:
        # implicit 제공 BM25 가중과 동일 로직을 간단히 적용(근사): idf 성분 강화
        # 정확한 bm25_weight를 쓰고 싶다면 implicit.preprocessing.bm25_weight 사용
        try:
            from implicit.preprocessing import bm25_weight
            mat = bm25_weight(mat).tocsr()
        except Exception:
            # fallback: 단순 행/열 정규화
            item_norm = np.sqrt(mat.power(2).sum(axis=0)).A1 + 1e-6
            mat = mat @ sparse.diags(1.0 / item_norm)
    return mat
