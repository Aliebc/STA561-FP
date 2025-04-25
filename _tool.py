import os
import pandas as pd
import re

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

def save_cleaned_data(df: pd.DataFrame, path: str, labels: dict = None):
    """
    Save the cleaned data to the specified path.
    """
    path2 = os.path.join('clean', path)
    df.to_stata(path2, version=118, variable_labels=labels)
    os.system(f'cd clean && gzip -kf {path}')
    
def is_in_target_columns(column, target_columns):
    is_in = False
    for target_column in target_columns:
        if re.search(target_column, column):
            is_in = True
            break
    return is_in

def get_target_columns(df: pd.DataFrame, target_columns):
    """
    Get the target columns from the DataFrame based on the target columns.
    """
    cols = df.columns
    target_columns = [col for col in cols if is_in_target_columns(col, [target_columns])]
    return target_columns

def filter_columns(df: pd.DataFrame, target_columns: list) -> pd.DataFrame:
    """
    Filter the columns of the DataFrame based on the target columns.
    """
    filtered_columns = [col for col in df.columns if is_in_target_columns(col, target_columns)]
    return df[filtered_columns]

def classify_income_level(monthly_income):
    """
    根据2017年中国收入分档标准,输入月收入,返回所属的收入等级(1-6)
    等级越高，收入越高。
    """
    if pd.isna(monthly_income):
        return pd.NA
    annual_income = monthly_income * 12

    if annual_income <= 10000:
        return 1  # 低收入群体
    elif annual_income <= 20000:
        return 2  # 较低收入群体
    elif annual_income <= 30000:
        return 3  # 中低收入群体
    elif annual_income <= 45000:
        return 4  # 中等收入群体
    elif annual_income <= 70000:
        return 5  # 中高收入群体
    else:
        return 6  # 高收入群体
