from flask import Flask, request, jsonify
from models import find_model
from flask_cors import CORS
import json
import pandas as pd
from eva import (
    one_hot_column,
    apply_original_df,
    select_columns,
    PCA,
    SMOTE,
    train_test_split,
    get_models,
    load_cleaned_data
)

app = Flask(__name__)
CORS(app)

ODTA = load_cleaned_data('chfs2017_income2.dta')
INCOMEDTA = load_cleaned_data('INCOME2017C.dta')
INCOMEDTA.dropna(subset=['cincome'], inplace=True)

NCOMPONENTS = 12

@app.route('/eva', methods=['POST'])
def eva():
    data = request.get_json()
    # 这里留空逻辑，稍后可以补充
    df = ODTA.copy()
    cols = df.columns
    new_row = pd.Series(index=cols)
    for role in ['self', 'father', 'mother']:
        if role in data:
            if role == 'self':
                rolen = ''
            else:
                rolen = '_' + role
            if 'urbanization' in data[role]:
                new_row[f'a2022a{rolen}'] = 1 if data[role]['urbanization'] == 'yes' else 2
            else:
                new_row[f'a2022a{rolen}'] = 2
            if 'hukouTransfer' in data[role]:
                new_row[f'a2022k{rolen}'] = 1 if data[role]['hukouTransfer'] == 'yes' else 2
            else:
                new_row[f'a2022k{rolen}'] = 2
            if 'gender' in data[role]:
                new_row[f'a2003{rolen}'] = 1 if data[role]['gender'] == 'male' else 2
            else:
                if role == 'self':
                    new_row[f'a2003{rolen}'] = 1
            if 'year' in data[role]:
                new_row[f'age{rolen}'] = 2025 - int(data[role]['year'])
            if 'city' in data[role]:
                new_row[f'a2019_prov_code{rolen}'] = int(data[role]['city'][:2])
                cityid = data[role]['city'][:4] + '00'
                cincome = INCOMEDTA[INCOMEDTA['cityid'] == cityid]['cincome']
                new_row[f'cincome{rolen}'] = cincome.iloc[0] if not cincome.empty else ODTA['cincome'].mean()
            if 'brothers' in data[role]:
                new_row[f'a2028{rolen}'] = int(data[role]['brothers'])
            else:
                new_row[f'a2028{rolen}'] = 0
            if 'sisters' in data[role]:
                new_row[f'a2029{rolen}'] = int(data[role]['sisters'])
            else:
                new_row[f'a2029{rolen}'] = 0
            if 'income' in data[role]:
                new_row[f'a3109{rolen}'] = int(data[role]['income'])
            else:
                new_row[f'a3109{rolen}'] = ODTA['a3109'].mean()
            if 'pension' in data[role]:
                new_row[f'f1001a{rolen}'] = int(data[role]['pension'])
            else:
                new_row[f'f1001a{rolen}'] = 7777
            if 'workUnit' in data[role]:
                new_row[f'a3106{rolen}'] = int(data[role]['workUnit'])
            else:
                new_row[f'a3106{rolen}'] = 7777
            if 'commute' in data[role] and role != 'self':
                for j in list(range(1, 9)) + ['7777']:
                    if str(j) in data[role]['commute']:
                        new_row[f'a3118_{j}_mc{rolen}'] = 1
                    else:
                        new_row[f'a3118_{j}_mc{rolen}'] = 0
            elif role != 'self':
                for j in list(range(1, 9)) + ['7777']:
                    new_row[f'a3118_{j}_mc{rolen}'] = 0
            if 'education' in data[role]:
                new_row[f'a2012{rolen}'] = int(data[role]['education'])
            else:
                new_row[f'a2012{rolen}'] = 3
            if 'overseas' in data[role]:
                new_row[f'a2012a{rolen}'] = 1 if data[role]['overseas'] == 'yes' else 2
            else:
                new_row[f'a2012a{rolen}'] = 2
            if 'party' in data[role]:
                new_row[f'a2015{rolen}'] = 1 if data[role]['party'] == 'yes' else 2
            else:
                new_row[f'a2015{rolen}'] = 2
    new_row = new_row[cols]
    with pd.option_context('display.max_rows', None):                
        print(new_row)
    if new_row.isnull().any():
        print("Missing values in new row:", new_row[new_row.isnull()])
        return jsonify({"message": "Missing values in input data"}), 400
    df_new = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    X_new = df_new.drop(columns=['a3109'])
    X_new = apply_original_df(X_new)
    
    X_new = PCA(n_components=NCOMPONENTS).fit_transform(X_new)
    X_train = X_new[:-1]
    y_train = df_new['a3109'].astype('int')[:-1]
    y_train = y_train - y_train.min()
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
    model = find_model('GradientBoosting').model_class()
    model.fit(X_train, y_train)
    # 取出最后一行
    X_new = X_new[-1].reshape(1, -1)
    # 预测
    y_pred = model.predict(X_new)
    y_pred_proba = model.predict_proba(X_new)
    print("Predicted class:", y_pred)
    print("Predicted probabilities:", y_pred_proba)
    ret_data = {
        "predicted_class": int(y_pred[0]),
        "predicted_probabilities": y_pred_proba[0].tolist()
    }
    print("Received data:", data)
    return jsonify({"message": "Success", "data": ret_data}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)