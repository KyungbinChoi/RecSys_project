import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

def create_interaction_matrix(df: pd.DataFrame, user_col: str, item_col: str) -> csr_matrix:
    """
    Create an interaction matrix from a DataFrame.
    """
    # Create a user-item interaction matrix
    interaction_matrix = csr_matrix((df['rating'], (df[user_col], df[item_col])))
    return interaction_matrix

def check_cold_start_users(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list:
    """
    Check cold start users.
    """
    ucol = 'user_id'
    # 고유 유저 추출 (벡터화)
    train_users = train_df[ucol].astype(str).unique()
    test_users  = test_df[ucol].astype(str).unique()
    # 차집합 (정렬된 결과)
    cold_users = np.setdiff1d(test_users, train_users, assume_unique=False)
    return cold_users.tolist()

def check_cold_start_items(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list:
    """
    Check cold start items.
    """
    train_items = train_df["item_id"].astype(str).unique()
    test_items  = test_df["item_id"].astype(str).unique()
    cold_items = np.setdiff1d(test_items, train_items, assume_unique=False)
    return cold_items.tolist()
