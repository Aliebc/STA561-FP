from models import get_models
import pandas as pd
from _tool import (
    load_cleaned_data
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
)
from sklearn.metrics import roc_auc_score
import os
import sys
import numpy as np
from scipy.stats import pearsonr
import json

def one_hot_column(df, col):
    if col not in df.columns:
        return df
    dummies = pd.get_dummies(df[col], prefix=col)
    # 把True和False转换为1和0
    dummies = dummies.astype(int)
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

def select_columns(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    columns = df.columns
    new_columns = []
    for col in columns:
        # 选择相关系数显著性 < 0.1 的列
        print(df[col])
        corref = pearsonr(df[col], y)
        if corref[1] < 0.1:
            new_columns.append(col)
    return df[new_columns]

if __name__ == "__main__":
    df_income = load_cleaned_data('chfs2017_income2.dta')
    df_income_applied = df_income
    Y = df_income_applied['a3109'].astype('int')
    Y = Y - Y.min()
    X = df_income_applied.drop(columns=['a3109'])
    X = apply_original_df(X)
    X = select_columns(X, Y)
    print(X.columns)
    json.dump(X.columns.tolist(), open('output/columns.json', 'w'), indent=4)
    X = PCA(n_components=8).fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=41)
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
    df_res = pd.DataFrame()
    result_list = []
    if len(sys.argv) > 1 and sys.argv[1] == 'lazy':
        from lazypredict.Supervised import LazyClassifier
        def custom_metric_f1(y_test, y_pred):
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            return report['1']['f1-score']
        clf = LazyClassifier(
            ignore_warnings=True,
            custom_metric=custom_metric_f1,
        )
        models_lz, predictions = clf.fit(X_train, X_test, y_train, y_test)
        models_df = pd.DataFrame(models_lz).reset_index()
        models_df = models_df.sort_values(by='ROC AUC', ascending=False)
        print(models_df)
        exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        df_res = pd.read_csv('output/eva.csv')
        df_res = df_res.sort_values(by='ROC AUC', ascending=False)
        df_res = df_res.reset_index(drop=True)
        print(df_res)
        df_res['Model Name'] = df_res['Model Name'].str.replace('_', R'\_')
        df_res.to_latex('output/eva.tex', index=False, float_format='%.3f')
        exit(0)
    for model in get_models():
        print("-" * 40)
        print(f"Model Name: {model.model_name}")
        ins = model.model_class()
        ins.fit(X_train, y_train)
        y_pred = ins.predict(X_test)
        y_pred_proba = ins.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        acc = report['accuracy']
        precision_1 = report['1']['precision']
        recall_1 = report['1']['recall']
        f1_1 = report['1']['f1-score']

        result_list.append({
            'Model Name': model.model_name,
            'Accuracy': acc,
            'Precision (1)': precision_1,
            'Recall (1)': recall_1,
            'F1-Score (1)': f1_1,
            'ROC AUC': roc_auc,
        })

        print(classification_report(y_test, y_pred, zero_division=0))
        print("-" * 40)

    df_res = pd.DataFrame(result_list)
    df_res = df_res.sort_values(by='ROC AUC', ascending=False)
    df_res = df_res.reset_index(drop=True)
    print(df_res)
    os.makedirs('output', exist_ok=True)
    df_res.to_csv('output/eva.csv', index=False)