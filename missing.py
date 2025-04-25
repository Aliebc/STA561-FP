## Cleaning CHFS 2017 Individual Data
from sklearn.impute import KNNImputer
import pandas as pd
from _tool import (
    load_cleaned_data, 
    filter_columns, 
    get_target_columns,
    classify_income_level
)

df_income = load_cleaned_data('chfs2017_income.dta')

USED_COLUMNS = [
    'a3109(_[a-z]*r)?$',
    'a2022a',
    'a3106$'
]
df_income = filter_columns(df_income, USED_COLUMNS)

df_income.update(df_income[get_target_columns(df_income, 'a3109')].map(lambda x: classify_income_level(x)))

df_income.update(
    df_income[
        get_target_columns(df_income, 'a3109')
    ].fillna(df_income['a3109_father'].mode()[0])
)

#print(df_income['a3109'].value_counts())

df_income.update(df_income[get_target_columns(df_income, 'a2022a')].fillna(0))
df_income.update(df_income[get_target_columns(df_income, 'a3106')].fillna(7777))

print(df_income)
print(df_income.columns)