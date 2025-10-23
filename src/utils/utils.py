import pandas as pd
from scipy.sparse import csr_matrix

def create_interaction_matrix(df: pd.DataFrame, user_col: str, item_col: str) -> csr_matrix:
    """
    Create an interaction matrix from a DataFrame.
    """
    # Create a user-item interaction matrix
    interaction_matrix = csr_matrix((df['rating'], (df[user_col], df[item_col])))
    return interaction_matrix