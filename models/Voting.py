from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from _tool import load_cleaned_data

# 加载你的数据
df = load_cleaned_data('chfs2017_income2.dta')
X = df.drop(columns=['a3109'])
y = df['a3109'].astype(int) - df['a3109'].min()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

param_grid = {
    'n_estimators': [100, 300],
    'learning_rate': [0.1, 0.03, 0.01],
    'max_depth': [6, 8, 10]
}

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
