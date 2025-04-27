import os
import pandas as pd
import re
import shutil
import lzma

def load_source_data(path) -> pd.DataFrame:
    """
    Load the source data from the specified path. Supports automatic decompression of .xz files.
    Works on both Windows and Unix-like systems.
    """
    path2 = os.path.join('data', path)

    # If .dta file doesn't exist, try to decompress from .xz
    if not os.path.exists(path2):
        xz_path = path2 + '.xz'
        if os.path.exists(xz_path):
            with lzma.open(xz_path, 'rb') as f_in:
                with open(path2, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            raise FileNotFoundError(f"Neither {path2} nor {xz_path} found.")

    df = pd.read_stata(path2)
    return df

def load_cleaned_data(path) -> pd.DataFrame:
    """
    Load the cleaned data from the specified path. Supports automatic decompression of .xz files.
    Works on both Windows and Unix-like systems.
    """
    path2 = os.path.join('clean', path)

    # If .dta file doesn't exist, try to decompress from .xz
    if not os.path.exists(path2):
        xz_path = path2 + '.xz'
        if os.path.exists(xz_path):
            with lzma.open(xz_path, 'rb') as f_in:
                with open(path2, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            raise FileNotFoundError(f"Neither {path2} nor {xz_path} found.")

    df = pd.read_stata(path2)
    return df

def save_cleaned_data(df: pd.DataFrame, path: str, labels: dict = None):
    """
    Save the cleaned data to the specified path.
    """
    path2 = os.path.join('clean', path)
    if labels is None:
        labels = {}
    df.to_stata(path2, version=118, variable_labels=labels, write_index=False)
    os.system(f'cd clean && xz -T8 -kf {path}')
    
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

def classify_income_level6(monthly_income):
    """
    根据2017年中国收入分档标准,输入月收入,返回所属的收入等级(1-6)
    等级越高，收入越高。
    """
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

def classify_income_level(monthly_income):
    """
    根据2017年中国收入分档标准,输入月收入,返回所属的收入等级(1-6)
    等级越高，收入越高。
    """
    annual_income = monthly_income * 12

    if annual_income <= 60000:
        return 3  # 中高收入群体
    else:
        return 4  # 高收入群体
    
def get_cityid(countyid):
    municipalities = {'11', '12', '31', '50'}
    countyid_str = str(countyid)
    if countyid_str[:2] in municipalities:
        return countyid_str[:2] + '0000'
    else:
        return countyid_str[:4] + '00'

def get_stata_labels() -> dict:
    """
    Get the variable labels from the DataFrame.
    """
    reader = pd.io.stata.StataReader('data/chfs2017_ind_202104.dta')
    original_labels = reader.variable_labels()
    original_labels['age'] = '年龄'
    original_labels['countyid'] = '县级行政区划代码'
    original_labels['cityid'] = '市级行政区划代码'
    original_labels['cincome'] = '市级收入水平'
    new_labels = {}

    for k, v in original_labels.items():
        new_labels[k] = v
        new_labels[k + '_father'] = v + '(父亲)'
        new_labels[k + '_mother'] = v + '(母亲)'
    return new_labels