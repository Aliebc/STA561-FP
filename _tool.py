import os
import pandas as pd

def load_source_data(path) -> pd.DataFrame:
    """
    Load the source data from the specified path.
    """
    path2 = os.path.join('data', path)
    if not os.path.exists(path2):
        os.system(f'cd data && gzip -k -d {path}.gz')
    
    df = pd.read_stata(path2)
    return df

def load_cleaned_data(path) -> pd.DataFrame:
    """
    Load the cleaned data from the specified path.
    """
    path2 = os.path.join('clean', path)
    if not os.path.exists(path2):
        os.system(f'cd clean && gzip -k -d {path}.gz')
    df = pd.read_stata(path2)
    return df

def save_cleaned_data(df: pd.DataFrame, path: str):
    """
    Save the cleaned data to the specified path.
    """
    path2 = os.path.join('clean', path)
    df.to_stata(path2, version=118)
    os.system(f'cd clean && gzip -kf {path}')