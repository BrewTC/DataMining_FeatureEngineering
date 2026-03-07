import ML_classification_model, ML_hyperparameter_tuning, ML_sampler_model
from collections import Counter
import pandas as pd
import os, glob

# 呼叫方法:
# 檔名 (.) def name (X_train, y_train, X_test, y_test)
path = os.getcwd()
os.chdir(path+'\\Using Data')

df_X = pd.read_csv('0219_rf_25.csv')
X = df_X
df_y = pd.read_csv('y.csv')
y = df_y['y']
print(Counter(y))

# TomekLinks # 'auto'對少數類以外重採樣，'all'對所有類別重採樣
X, y = ML_sampler_model.TomekLinks_sampler(X, y, sampling_strategy='all')

# 1. 清除資料中標籤有重疊的資料
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 28)

# 2. 訓練集的資料平衡
# SMOTENC_Sampler # 數據集包含連續特徵和分類特徵的混合
X_res, y_res = ML_sampler_model.SMOTENC_sampler(X_train, y_train, categorical_features=[0,1,2], k_neighbors=5)

# 3. 測試集的資料平衡(下採樣)
# RandomUnderSampler
# X_rus, y_rus = ML_sampler_model.RUS_sampler(X_test, y_test)

# 4. 進行模型的訓練和預測
# ML_hyperparameter_tuning.RF_GS_hyper_tuning(X_res, y_res, X_rus, y_rus)
ML_classification_model.Decision_Tree_Classifier(X_res, y_res, X_test, y_test)


'''
# ---------- 針對資料數據類別進行重採樣方法 ---------- #
# SMOTEN_Sampler # 數據集僅由分類特徵組成
X_res, y_res = ML_sampler_model.SMOTEN_sampler(X, y, k_neighbors=5)

# SMOTENC_Sampler # 數據集包含連續特徵和分類特徵的混合
X_res, y_res = ML_sampler_model.SMOTENC_sampler(X, y, , categorical_features=[0,1,2], k_neighbors=5)

# SMOTETomek # 'auto'對少數類以外重採樣，'all'對所有類別重採樣
X_res, y_res = ML_sampler_model.SMOTETomek_sampler(X, y, sampling_strategy='auto')

# SMOTEENN # 'auto'對少數類以外重採樣，'all'對所有類別重採樣
X_res, y_res = ML_sampler_model.SMOTETomek_sampler(X, y, sampling_strategy='auto')

# ---------- OverSampler ---------- #
# RandomOverSampler
X_res, y_res = ML_sampler_model.ROS_sampler(X, y)

# ROSE_Sampler
X_res, y_res = ML_sampler_model.ROSE_sampler(X, y, shrinkage=1)

# SMOTE_Sampler
X_res, y_res = ML_sampler_model.SMOTE_sampler(X, y, k_neighbors=5)

# AdasynSampler # 可能只關注異常值
X_res, y_res = ML_sampler_model.SMOTE_sampler(X, y, n_neighbors=5)

# ---------- UnderSampler ---------- #
# RandomUnderSampler
X_ros, y_ros = ML_sampler_model.RUS_sampler(X, y)

# TomekLinks # 'auto'對少數類以外重採樣，'all'對所有類別重採樣
X_res, y_res = ML_sampler_model.TomekLinks_sampler(X, y, sampling_strategy='auto')
# ---------- ----------- ---------- #
'''


