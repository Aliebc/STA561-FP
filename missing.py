## Cleaning CHFS 2017 Individual Data
from sklearn.impute import KNNImputer
import pandas as pd

df_income = pd.read_stata('clean/chfs2017_income.dta')

print(df_income)