# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 23:54:35 2022

@author: User
"""
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np

data = pd.read_csv('thesis_close_final_134.csv')

# Use iloc and select all rows (:) against the last column (-1):
X, y = data, data.iloc[:,-1:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lda = LinearDiscriminantAnalysis(n_components='mle')
X_r2 = lda.fit(X, y).transform(X)

# The number of samples must be more than the number of classes.
# 意思是說最後的答案分類(last column)要小於樣本數，因為我拿來測試的都是獨立的id_num
# 例如答案分類是0和1



# LDA.explained_variance_ratio_

# np.cumsum(LDA.explained_variance_ratio_)