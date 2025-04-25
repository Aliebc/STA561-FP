## Cleaning CHFS 2017 Individual Data
from sklearn.impute import KNNImputer
import pandas as pd
from _tool import (
    load_cleaned_data, 
    filter_columns, 
    get_target_columns,
    classify_income_level,
    save_cleaned_data
)

df_income = load_cleaned_data('chfs2017_income.dta')

USED_COLUMNS = [
    'a3109(_[a-z]*r)?$',
    'a2022a',
    'a3106$',
    'age',
    'a2019_prov_code(_[a-z]*r)?$'
]
df_income = filter_columns(df_income, USED_COLUMNS)

df_income.update(df_income[get_target_columns(df_income, 'a3109')].map(
    lambda x: classify_income_level(x), na_action='ignore')
)

for col in get_target_columns(df_income, 'a3109'):
    df_income[col] = df_income[col].fillna(df_income[col].mode()[0])

for col in get_target_columns(df_income, 'age'):
    df_income[col] = df_income[col].fillna(df_income[col].mode()[0])

for col in get_target_columns(df_income, 'a2019_prov_code_'):
    print(col)
    #df_income[col] = df_income[col].replace('', pd.NA)
    df_income[col] = df_income[col].fillna(df_income['a2019_prov_code'])


#print(df_income['a3109'].value_counts())

df_income.update(df_income[get_target_columns(df_income, 'a2022a')].fillna(0))
df_income.update(df_income[get_target_columns(df_income, 'a3106')].fillna(7777))

print(df_income)

save_cleaned_data(
    df_income,
    'chfs2017_income2.dta',
    None
)