## Cleaning CHFS 2017 Individual Data
from sklearn.impute import KNNImputer
import pandas as pd
from _tool import load_cleaned_data

df_income = load_cleaned_data('chfs2017_income.dta')

print(df_income)
print(df_income.columns)