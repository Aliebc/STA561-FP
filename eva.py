from models import get_models
import pandas as pd
from _tool import (
    load_cleaned_data
)
from sklearn.model_selection import train_test_split
# accuracy matric
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def one_hot_column(df, col):
    if col not in df.columns:
        return df
    dummies = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df.drop(columns=col), dummies], axis=1)
    return df

def apply_original_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        'a3106',
        'a3109',
        'a2019_prov_code',
        'f1001',
        'a2012',
        'a2015'
    ]:
        for suffix in ['_father', '_mother']:
            if col + suffix in df.columns:
                df = one_hot_column(df, col + suffix)
        df = one_hot_column(df, col)
    return df

if __name__ == "__main__":
    df_income = load_cleaned_data('chfs2017_income2.dta')
    df_income_applied = df_income
    models = get_models()
    Y = df_income_applied['a3109'].astype('int')
    Y = Y - Y.min()
    X = df_income_applied.drop(columns=['a3109'])
    X = apply_original_df(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    for model in models:
        print("-" * 40)
        print(f"Model Name: {model.model_name}")
        ins = model.model_class()
        ins.fit(X_train, y_train)
        y = ins.predict(X_test)
        print(classification_report(y_test, y))
        print("-" * 40)