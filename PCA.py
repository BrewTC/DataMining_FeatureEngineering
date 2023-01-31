# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

X = pd.read_csv('thesis_close_final_134.csv')


# n_components代表我們希望留下的特徵維度，我們可以給定任意數值。
# 另外，我們也可以給定'mle'讓演算法用最大概似法幫我們決定合適的components數量。
pca=PCA(n_components=10) 
pca.fit(X)

# n_components='mle' is only supported if n_samples >= n_features

print(pca.n_components_)

print(pca.explained_variance_ratio_)


cumsum_pca = np.cumsum(pca.explained_variance_ratio_)
print(cumsum_pca)

