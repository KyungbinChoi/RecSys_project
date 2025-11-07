import numpy as np
import pandas as pd
from typing import Optional
from scipy import sparse

from implicit.nearest_neighbours import CosineRecommender,bm25_weight

def build_id_maps(train_df: pd.DataFrame):
    users = train_df["user_id"].astype(str).unique()
    items = train_df["item_id"].astype(str).unique()
    user2idx = {u:i for i,u in enumerate(users)}
    item2idx = {it:i for i,it in enumerate(items)}
    return user2idx, item2idx

def build_user_item_csr(train_df: pd.DataFrame, user2idx, item2idx, use_bm25=True):
    df = train_df.loc[train_df["clicked"] == 1, ["user_id","item_id"]].astype(str)
    ui = df.assign(u=lambda d: d["user_id"].map(user2idx),
                   i=lambda d: d["item_id"].map(item2idx),
                   v=1.0).dropna(subset=["u","i"])
    rows = ui["u"].astype(int).to_numpy()
    cols = ui["i"].astype(int).to_numpy()
    data = ui["v"].to_numpy(np.float32)
    UI = sparse.csr_matrix((data,(rows,cols)),
                           shape=(len(user2idx), len(item2idx)),
                           dtype=np.float32)
    if use_bm25:
        UI = bm25_weight(UI).tocsr()
    return UI

def rank_impressions_with_itemknn_fixed(
    impress_df: pd.DataFrame,
    train_df: pd.DataFrame,
    K: int = 200,
    use_bm25: bool = True,
    tie_break: str = "position",
    pop_backfill: Optional[pd.Series] = None,  # item_id -> pop score
    alpha: float = 0.9,                        # 혼합비(0~1)
    filter_seen_to_zero: bool = True           # 본 아이템은 0점으로 강등
) -> pd.DataFrame:
    """
    반환: pred_df[impression_id,user_id,item_id,position,(clicked),score,rank]
    """
    user2idx, item2idx = build_id_maps(train_df)
    UI = build_user_item_csr(train_df, user2idx, item2idx, use_bm25=use_bm25)
    model = CosineRecommender(K=K)
    model.fit(UI.T.tocsr())  # item-user

    has_clicked = "clicked" in impress_df.columns
    cols = ["impression_id","user_id","item_id","position"] + (["clicked"] if has_clicked else [])
    v = impress_df[cols].copy()
    v["user_id"] = v["user_id"].astype(str)
    v["item_id"] = v["item_id"].astype(str)
    v["position"] = v["position"].astype(int)

    # (선택) 인기도 맵
    pop_map = None
    if pop_backfill is not None:
        pop_map = pop_backfill.astype(float)

    outs = []
    for imp_id, grp in v.groupby("impression_id", sort=False):
        uid = grp["user_id"].iloc[0]
        uidx = user2idx.get(uid, None)

        cand_items = grp["item_id"].tolist()
        cand_iidx = np.array([item2idx.get(it, -1) for it in cand_items], dtype=int)
        scores = np.zeros(len(grp), dtype=np.float32)

        if uidx is not None:
            valid_mask = cand_iidx >= 0
            cand_iidx_valid = cand_iidx[valid_mask]
            if cand_iidx_valid.size > 0:
                # ⬇️ 핵심1: 반환 ids ↔ scores를 "후보 인덱스"에 정확히 매핑
                ret_ids, sc = model.rank_items(
                    uidx, UI, cand_iidx_valid,
                    # 일부 버전에서 매개변수 지원: 필요시 False로 지정
                    # filter_already_liked_items=False
                )
                sc = sc.astype(np.float32)
                # ⬇️ 핵심2: -inf / inf / nan 정리
                sc[~np.isfinite(sc)] = 0.0

                # ret_ids는 "아이템 내부 인덱스"이므로,
                # cand_iidx_valid에서 ret_ids 위치를 찾아 점수 대입
                # (id -> 위치) 매핑
                id2pos = {iid: pos for pos, iid in enumerate(cand_iidx_valid)}
                for iid, s in zip(ret_ids, sc):
                    pos = id2pos.get(iid, None)
                    if pos is not None:
                        # valid_mask 내 상대 위치 → 전체 grp 내 절대 위치로 변환
                        absolute_pos = np.flatnonzero(valid_mask)[pos]
                        scores[absolute_pos] = s

        # (선택) 이미 본 아이템 0점 강등
        if filter_seen_to_zero and (uidx is not None):
            seen = set(UI.getrow(uidx).indices)
            for j, iid in enumerate(cand_iidx):
                if iid in seen:
                    scores[j] = 0.0  # 또는 아주 작은 값으로

        # (선택) 인기도 백필 혼합
        if pop_map is not None:
            pop = np.array([pop_map.get(it, 0.0) if hasattr(pop_map, "get")
                            else pop_map.loc.get(it, 0.0) for it in cand_items],
                           dtype=np.float32)
            scores = alpha * scores + (1.0 - alpha) * pop

        g = grp.copy()
        g["score"] = scores

        # 그룹 내 랭킹(동점 안정화)
        sort_cols, ascending = (["score"], [False])
        if tie_break == "position":
            sort_cols += ["position"]; ascending += [True]
        elif tie_break == "item_id":
            sort_cols += ["item_id"];  ascending += [True]

        g = g.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        g["rank"] = np.arange(1, len(g) + 1, dtype=int)
        g = g.sort_values(["position"], kind="mergesort")
        outs.append(g)

    pred_df = pd.concat(outs, ignore_index=True)
    keep = ["impression_id","user_id","item_id","position","score","rank"]
    if has_clicked:
        keep.insert(4, "clicked")
    return pred_df[keep]




# 4. DuckDB로 truth/pred 라인 생성
def build_truth_with_position_duckdb(
    validation_df: pd.DataFrame,
    imp_col: str = "impression_id",
    pos_col: str = "position",
    label_col: str = "clicked"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    truth.txt 포맷용 DF 생성
      - truth_df: [impression_id, clicked_label(list[int])]
      - truth_lines: [line]  => "impid [0,1,0,...]"
    """
    df = validation_df[[imp_col, pos_col, label_col]].copy()
    df[imp_col] = df[imp_col].astype(str)
    df[pos_col] = df[pos_col].astype(int)
    df[label_col] = df[label_col].astype(int)

    con = ddb.connect()
    con.register("tbl", df)
    truth_df = con.execute(f"""
        SELECT
          {imp_col} AS impression_id,
          list({label_col} ORDER BY {pos_col}) AS clicked_label
        FROM tbl
        GROUP BY {imp_col}
        ORDER BY {imp_col}
    """).df()
    con.close()

    truth_lines = truth_df.assign(
        line=lambda d: d.apply(lambda r: f"{r['impression_id']} " + json.dumps([int(x) for x in r["clicked_label"]]), axis=1)
    )[["line"]]
    return truth_df, truth_lines


def build_prediction_lines_with_position_duckdb(
    pred_df: pd.DataFrame,
    imp_col: str = "impression_id",
    pos_col: str = "position",
    rank_col: str = "rank"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    prediction.txt 포맷용 DF 생성
      - pred_fmt_df: [impression_id, ranks(list[int])]
      - pred_lines  : [line]  => "impid [4,1,3,...]"
    """
    df = pred_df[[imp_col, pos_col, rank_col]].copy()
    df[imp_col] = df[imp_col].astype(str)
    df[pos_col] = df[pos_col].astype(int)
    df[rank_col] = df[rank_col].astype(int)

    con = ddb.connect()
    con.register("tbl", df)
    pred_fmt_df = con.execute(f"""
        SELECT
          {imp_col} AS impression_id,
          list(CAST({rank_col} AS BIGINT) ORDER BY {pos_col}) AS ranks
        FROM tbl
        GROUP BY {imp_col}
        ORDER BY {imp_col}
    """).df()
    con.close()

    pred_lines = pred_fmt_df.assign(
        line=lambda d: d.apply(lambda r: f"{r['impression_id']} " + json.dumps([int(x) for x in r["ranks"]]), axis=1)
    )[["line"]]
    return pred_fmt_df, pred_lines