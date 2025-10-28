import pandas as pd
import numpy as np
from scipy import sparse
from implicit.nearest_neighbours import CosineRecommender
from implicit.preprocessing import bm25_weight

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
            
            mat = bm25_weight(mat).tocsr()
        except Exception:
            # fallback: 단순 행/열 정규화
            item_norm = np.sqrt(mat.power(2).sum(axis=0)).A1 + 1e-6
            mat = mat @ sparse.diags(1.0 / item_norm)
    return mat



def fit_itemknn_cosine(user_item_csr: sparse.csr_matrix,
                       K: int = 100) -> CosineRecommender:
    """
    implicit의 아이템기반 코사인 KNN 모델 학습.
    implicit은 item-user 행렬을 기대하므로 전치 필요.
    """
    model = CosineRecommender(K=K)
    item_user_csr = user_item_csr.T.tocsr()  # shape [n_items, n_users]
    model.fit(item_user_csr)  # 유사도 인덱스 구축
    return model

def rank_by_itemknn(valid_df: pd.DataFrame,
                    train_df: pd.DataFrame,
                    K: int = 100,
                    use_bm25: bool = True,
                    tie_break: str = "position"):
    """
    item-based CF로 valid_df 후보들에 점수/랭크 부여.
    반환: pred_df[impression_id, user_id, item_id, position, clicked, score, rank]
    """
    # 1) 매핑/행렬/모델
    user2idx, item2idx, idx2user, idx2item = build_id_maps(train_df)
    UI = build_user_item_csr(train_df, user2idx, item2idx, use_bm25=use_bm25)
    model = fit_itemknn_cosine(UI, K=K)

    # user별 학습 히스토리(희소 벡터) 접근을 위해 유지
    # UI[u] : shape (1, n_items)
    # implicit 모델에는 rank_items가 있으며 candidate 리스트 점수 계산에 유용
    has_rank_items = hasattr(model, "rank_items")

    # 2) valid_df에서 (impression, user, [candidates]) 그룹화
    v = valid_df[["impression_id","user_id","item_id","position","clicked"]].copy()
    v["user_id"] = v["user_id"].astype(str)
    v["item_id"] = v["item_id"].astype(str)
    v["position"] = v["position"].astype(int)

    rows = []
    for imp_id, grp in v.groupby("impression_id", sort=False):
        uid = grp["user_id"].iloc[0]
        uidx = user2idx.get(uid, None)
        # cold-start user: 스코어 0으로 처리(혹은 popularity 백필 로직 삽입 가능)
        cand_items = grp["item_id"].tolist()
        cand_iidx = np.array([item2idx.get(it, -1) for it in cand_items], dtype=int)

        scores = np.zeros(len(grp), dtype=np.float32)

        if uidx is not None:
            # 후보 중 train에 등장한 아이템만 점수 계산
            valid_mask = cand_iidx >= 0
            if valid_mask.any():
                if has_rank_items:
                    # implicit의 rank_items 활용 (ids, scores) 반환
                    # 유저 uidx의 학습 히스토리를 UI에서 전달
                    _, sc = model.rank_items(uidx, UI, cand_iidx[valid_mask])
                    scores[valid_mask] = sc.astype(np.float32)
                else:
                    # fallback: 사용자 프로파일 × (아이템 유사도) 근사
                    # 유사 아이템 상위K만 반영되므로 빠름
                    user_vec = UI.getrow(uidx)                   # (1, n_items)
                    # model.similar_items(iid, N=K)로 각 후보의 근접 이웃을 찾아
                    # 사용자 보유 아이템과 겹치는 유사도를 합산하는 간략 스코어
                    for j, iid in enumerate(cand_iidx):
                        if iid < 0: 
                            continue
                        sims = model.similar_items(iid, N=K)     # (ids, sims)
                        # 사용자 보유 아이템과 교집합만 점수에 반영
                        nn_idx, nn_sim = sims[0], sims[1]
                        # 사용자 벡터의 해당 아이템 인덱스 값(클릭 가중)을 곱해 합산
                        # 희소 접근: user_vec[:, nn_idx]는 csr 고급 인덱싱 비효율 → 개별 추출
                        s = 0.0
                        for ii, w in zip(nn_idx, nn_sim):
                            val = user_vec[0, ii]
                            if val != 0:
                                s += float(val) * float(w)
                        scores[j] = s

        # 3) 그룹 내 랭킹(내림차순), tie-break
        # 점수 동일하면 position(원본 순서) 또는 item_id로 안정화
        order_cols = ["score"]
        ascending = [False]
        if tie_break == "position":
            order_cols += ["position"]
            ascending += [True]
        elif tie_break == "item_id":
            order_cols += ["item_id"]
            ascending += [True]

        g2 = grp.copy()
        g2["score"] = scores
        g2 = g2.sort_values(order_cols, ascending=ascending, kind="mergesort")
        g2["rank"] = np.arange(1, len(g2) + 1, dtype=int)
        # 최종 출력은 원본 후보 순서(position)로 되돌려 두면 파일 포맷 생성이 쉬움
        g2 = g2.sort_values(["position"], kind="mergesort")
        rows.append(g2)

    pred_df = pd.concat(rows, axis=0, ignore_index=True)
    return pred_df[["impression_id","user_id","item_id","position","clicked","score","rank"]]
