# STA561 - Machine Learning Final Project

---

**Duke University - STA 561**

**Spring 2025**

---

## Project Overview

In this project, we developed and trained a machine learning model using data from the China Household Finance Survey (CHFS). The dataset comprises detailed socioeconomic information from over 40,000 households and 120,000 individuals across mainland China. 

Our primary objective was to predict the probability that a young individual will enter the top 10% income bracket in the future. The model leverages a wide range of features, including educational background, family wealth, geographic location, and early career choices, to make accurate and interpretable predictions.

Through careful preprocessing, feature engineering, and model selection, we aimed to build a robust tool that could provide valuable insights into the factors influencing upward income mobility among Chinese youth.

---

## Model Evaluation

Below are the evaluation results for different machine learning models, assessed based on key metrics such as Accuracy, Precision, Recall, and F1-Score:

| Model Name                             |   Accuracy |   Precision (1) |   Recall (1) |   F1-Score (1) |   ROC AUC |
|:---------------------------------------|-----------:|----------------:|-------------:|---------------:|----------:|
| Logistic                               |      0.722 |           0.268 |        0.566 |          0.364 |     0.723 |
| RandomForest                           |      0.742 |           0.264 |        0.469 |          0.338 |     0.722 |
| GradientBoosting                       |      0.725 |           0.273 |        0.575 |          0.37  |     0.713 |
| CatBoost                               |      0.734 |           0.274 |        0.54  |          0.363 |     0.713 |
| Stacking\_LightGBM\_MLP                |      0.75  |           0.261 |        0.425 |          0.323 |     0.711 |
| Stacking\_All                          |      0.749 |           0.254 |        0.407 |          0.313 |     0.683 |
| Stacking\_LGBM\_CatBoost\_MLP          |      0.744 |           0.259 |        0.442 |          0.327 |     0.682 |
| Stacking\_RF\_LGBM\_CatBoost\_XGB\_MLP |      0.75  |           0.256 |        0.407 |          0.314 |     0.682 |
| LightGBM                               |      0.74  |           0.239 |        0.389 |          0.296 |     0.68  |
| DecisionTree                           |      0.653 |           0.227 |        0.611 |          0.331 |     0.674 |
| XGBoost\_Tuned                         |      0.77  |           0.26  |        0.345 |          0.297 |     0.673 |
| KNN                                    |      0.733 |           0.245 |        0.434 |          0.313 |     0.648 |
| MLP                                    |      0.725 |           0.213 |        0.354 |          0.266 |     0.622 |
| Dummy                                  |      0.86  |           0     |        0     |          0     |     0.5   |



