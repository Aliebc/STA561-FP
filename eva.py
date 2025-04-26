from models import get_models
import pandas as pd
from _tool import (
    load_cleaned_data
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
# accuracy matric
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import os

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
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
    df_res = pd.DataFrame()
    result_list = []
    for model in models:
        print("-" * 40)
        print(f"Model Name: {model.model_name}")
        ins = model.model_class()
        ins.fit(X_train, y_train)
        y_pred = ins.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        acc = report['accuracy']
        precision_1 = report['1']['precision']
        recall_1 = report['1']['recall']
        f1_1 = report['1']['f1-score']

        result_list.append({
            'Model Name': model.model_name,
            'Accuracy': acc,
            'Precision (1)': precision_1,
            'Recall (1)': recall_1,
            'F1-Score (1)': f1_1
        })

        print(classification_report(y_test, y_pred, zero_division=0))
        print("-" * 40)

    df_res = pd.DataFrame(result_list)
    df_res = df_res.sort_values(by='F1-Score (1)', ascending=False)
    df_res = df_res.reset_index(drop=True)
    print(df_res)
    os.makedirs('output', exist_ok=True)
    df_res.to_csv('output/eva.csv', index=False)