## Cleaning CHFS 2017 Individual Data
from sklearn.impute import KNNImputer
import pandas as pd
from _tool import (
    load_cleaned_data, 
    load_source_data,
    filter_columns, 
    get_target_columns,
    classify_income_level,
    save_cleaned_data
)

df_income = load_cleaned_data('chfs2017_income.dta')

#df_income = df_income[df_income['a3109'] < 75000/12]

USED_COLUMNS = [
    'a3109(_[a-z]*r)?$',    #核心收入
    'a2022a',
    'a2012a_',
    'a2012_',
    'a2015_',
    'a2028',
    'a2029',
    'f1001a_[a-z]*r$',
    'a2022k_[a-z]*r$',
    'a3106$',
    'age',
    'cincome',
    'a2019_prov_code(_[a-z]*r)?$',
    'a3118_\d{1,4}_mc_',
    'a2003$'
]
df_income = filter_columns(df_income, USED_COLUMNS)

df_income.update(df_income[get_target_columns(df_income, 'a3109')].map(
    lambda x: classify_income_level(x), na_action='ignore')
)

for col in get_target_columns(df_income, 'a3109'):
    df_income[col] = df_income[col].fillna(10)

for col in get_target_columns(df_income, 'age'):
    df_income[col] = df_income[col].fillna(df_income[col].mean())
    
for col in get_target_columns(df_income, 'cincome'):
    df_income[col] = df_income[col].fillna(df_income[col].mean())
    
for col in get_target_columns(df_income, 'a3106'):
    df_income[col] = df_income[col].fillna(df_income[col].mode()[0])
    print(f"Mode of {col}: {df_income[col].mode()[0]}")
    
for col in get_target_columns(df_income, 'a3118'):
    df_income[col] = df_income[col].fillna(0)

for col in get_target_columns(df_income, 'a2019_prov_code_'):
    print(col)
    #df_income[col] = df_income[col].replace('', pd.NA)
    df_income[col] = df_income[col].fillna(df_income['a2019_prov_code'])


#print(df_income['a3109'].value_counts())

df_income.update(df_income[get_target_columns(df_income, 'a2022a')].fillna(0))
#df_income.update(df_income[get_target_columns(df_income, 'a3106')].fillna(7777))
df_income.update(df_income[get_target_columns(df_income, 'a2012a')].fillna(2))
df_income.update(df_income[get_target_columns(df_income, 'a2015')].fillna(2))
df_income.update(df_income[get_target_columns(df_income, 'a2028')].fillna(0))
df_income.update(df_income[get_target_columns(df_income, 'a2029')].fillna(0))
df_income.update(df_income[get_target_columns(df_income, 'f1001')].fillna(7777))
df_income.update(df_income[get_target_columns(df_income, 'a2022k')].fillna(2))
df_income.update(df_income[get_target_columns(df_income, 'a2012')].fillna(3))

print(df_income)
df_income.info()

save_cleaned_data(
    df_income,
    'chfs2017_income2.dta',
    None
)