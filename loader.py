from __future__ import annotations
import re
import os
import gc
import json
import argparse
import pandas as pd
from typing import Optional, Tuple, List
from pathlib import Path
from src.utils.logging_utils import get_logger

logger = get_logger("recsys.data")

# ------------------ 공통 유틸 ------------------
def _to_datetime(series: pd.Series) -> pd.Series:
    # MIND는 "%m/%d/%Y %I:%M:%S %p" 형태가 일반적 (예: 11/15/2019 12:31:21 AM)
    # 다양한 포맷을 자동 파싱하도록 errors="coerce" 사용
    return pd.to_datetime(series, errors="coerce", utc=True)

def _save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # pyarrow가 없는 환경을 대비하여 fallback
        df.to_pickle(str(path).replace(".parquet", ".pkl"))

# ------------------ MIND: 로더 ------------------
def load_mind_news(news_path: str | Path) -> pd.DataFrame:
    """
    news.tsv 스키마(탭 구분)
    columns: [news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities]
    """
    usecols = [0,1,2,3,4]
    names   = ["item_id","category","subcategory","title","abstract"]
    df = pd.read_csv(news_path, sep="\t", header=None, names=names, usecols=usecols, dtype={
        "item_id":"string", "category":"string","subcategory":"string","title":"string","abstract":"string"
    }, quoting=3, encoding="utf-8")
    df.drop_duplicates(subset=["item_id"], inplace=True)
    return df

def _parse_impressions(imps: str) -> list[tuple[str,int]]:
    """
    'N12345-0 N67890-1 ...' 형태를 [(item_id, clicked), ...]로 변환
    """
    pairs = []
    if not isinstance(imps, str) or not imps.strip():
        return pairs
    for tok in imps.strip().split():
        # item-click 형식
        if "-" in tok:
            item, clk = tok.rsplit("-", 1)
            try:
                pairs.append((item, int(clk)))
            except ValueError:
                continue
    return pairs

def load_mind_behaviors(behaviors_path: str | Path) -> pd.DataFrame:
    """
    behaviors.tsv 스키마(탭 구분)
    columns: [impression_id, user_id, time, history, impressions]
    """
    names = ["impression_id","user_id","time","history","impressions"]
    df = pd.read_csv(behaviors_path, sep="\t", header=None, names=names, dtype={
        "impression_id":"int64","user_id":"string","time":"string","history":"string","impressions":"string"
    }, quoting=3, encoding="utf-8")
    # 시간 파싱
    df["timestamp"] = _to_datetime(df["time"])
    # 세션 기준으로 활용할 수 있도록 user_id, timestamp 만 우선 정리
    df.sort_values(["user_id","timestamp"], inplace=True, kind="mergesort")
    return df

def build_mind_interactions(behaviors_df: pd.DataFrame, news_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    MIND behaviors에서 노출-클릭 로그를 row 단위로 전개
    반환:
      interactions: [user_id, item_id, timestamp, impression_id, position, clicked, history_len]
      items(optional): news와 머지된 아이템 메타 (item_id, category, subcategory, title(옵션))
    """
    rows = []
    # impressions explode
    for _, row in behaviors_df.iterrows():
        pairs = _parse_impressions(row.get("impressions",""))
        if not pairs:
            continue
        for pos, (item_id, clicked) in enumerate(pairs, start=1):
            rows.append({
                "user_id": row["user_id"],
                "item_id": item_id,
                "timestamp": row["timestamp"],
                "impression_id": row["impression_id"],
                "position": pos,
                "clicked": int(clicked),
                "history_len": 0 if pd.isna(row.get("history")) else len(str(row.get("history")).split())
            })
    inter = pd.DataFrame(rows)
    # 표준 스키마 정리
    inter["clicked"] = inter["clicked"].astype("int8")
    inter["position"] = inter["position"].astype("int16")
    inter["user_id"] = inter["user_id"].astype("string")
    inter["item_id"] = inter["item_id"].astype("string")
    # 아이템 메타
    if news_df is None:
        items = pd.DataFrame({"item_id": inter["item_id"].drop_duplicates()})
    else:
        items = news_df.copy()
        # 표준화
        items["item_id"] = items["item_id"].astype("string")
        for col in ["category","subcategory","title","abstract"]:
            if col in items.columns:
                items[col] = items[col].astype("string")
        # interactions에 없는 아이템은 남겨둬도 무방
    return inter, items

# ------------------ 세션화(옵션) ------------------
def sessionize_interactions(interactions: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    """
    동일 user 내 연속 클릭 간격이 gap_minutes를 넘으면 세션 분리
    """
    df = interactions.sort_values(["user_id","timestamp","position"]).copy()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df["prev_ts"] = df.groupby("user_id")["ts"].shift(1)
    gap = pd.Timedelta(minutes=gap_minutes)
    df["new_session"] = (df["prev_ts"].isna()) | ((df["ts"] - df["prev_ts"]) > gap)
    df["session_id"] = df.groupby("user_id")["new_session"].cumsum().astype("int64")
    df.drop(columns=["ts","prev_ts","new_session"], inplace=True)
    return df

# ------------------ CLI 엔트리포인트 ------------------
def build_mind_dataset(behaviors_path: str, news_path: str, outdir: str, session_gap_minutes: int = 30) -> Tuple[str, str]:
    logger.info("MIND 데이터 로딩 시작")
    news = load_mind_news(news_path)
    logger.info(f"news rows: {len(news):,}")
    behv = load_mind_behaviors(behaviors_path)
    logger.info(f"behaviors rows: {len(behv):,}")
    inter, items = build_mind_interactions(behv, news)
    logger.info(f"interactions rows: {len(inter):,}")
    inter = sessionize_interactions(inter, gap_minutes=session_gap_minutes)
    # 저장
    outdir = Path(outdir)
    inter_path = outdir / "interactions.parquet"
    items_path = outdir / "items.parquet"
    _save_parquet(inter, inter_path)
    _save_parquet(items, items_path)
    logger.info(f"저장 완료: {inter_path} / {items_path}")
    return str(inter_path), str(items_path)

def main():
    parser = argparse.ArgumentParser(description="MIND 데이터 로더/전처리")
    parser.add_argument("--dataset", type=str, default="mind", choices=["mind"], help="현재는 mind만 지원")
    parser.add_argument("--behaviors", type=str, required=True, help="behaviors.tsv 경로")
    parser.add_argument("--news", type=str, required=True, help="news.tsv 경로")
    parser.add_argument("--outdir", type=str, default="data/processed", help="출력 디렉토리")
    parser.add_argument("--session_gap_minutes", type=int, default=30, help="세션 분리 간격(분)")
    args = parser.parse_args()

    if args.dataset == "mind":
        build_mind_dataset(args.behaviors, args.news, args.outdir, args.session_gap_minutes)
    else:
        raise NotImplementedError("지원하지 않는 데이터셋")

if __name__ == "__main__":
    main()