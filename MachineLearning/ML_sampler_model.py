from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN, SMOTENC, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter

# ---------- 針對資料數據類別進行重採樣方法 ---------- #
def SMOTEN_sampler(X, y, k_neighbors): # 數據集僅由分類特徵組成
    sampler = SMOTEN(random_state=28, k_neighbors=k_neighbors, n_jobs=-1)
    X_res, y_res = sampler.fit_resample(X, y)
    print('SmotenSampler :', Counter(y_res))
    return X_res, y_res

def SMOTENC_sampler(X, y, categorical_features, k_neighbors): # 數據集包含連續特徵和分類特徵的混合
    smote_nc = SMOTENC(categorical_features=categorical_features, k_neighbors=k_neighbors, n_jobs=-1)
    X_res, y_res = smote_nc.fit_resample(X, y)
    print('SMOTENCSampler :', Counter(y_res))
    return X_res, y_res

def SMOTETomek_sampler(X, y, sampling_strategy):
    smt = SMOTETomek(random_state=28, sampling_strategy=sampling_strategy, n_jobs=-1)
    X_res, y_res = smt.fit_resample(X, y)
    print('SMOTETomek_Sampler :', Counter(y_res))
    return X_res, y_res

def SMOTEENN_sampler(X, y, sampling_strategy):
    sme = SMOTEENN(random_state=28, sampling_strategy=sampling_strategy, n_jobs=-1)
    X_res, y_res = sme.fit_resample(X, y)
    print('SMOTEENN_Sampler :', Counter(y_res))
    return X_res, y_res

# ---------- OverSampler ---------- #
def ROS_sampler(X, y):
    rus = RandomOverSampler(random_state=28)
    X_res, y_res = rus.fit_resample(X, y)
    print('RandomOverSampler :', Counter(y_res))
    return X_res, y_res

def ROSE_sampler(X, y, shrinkage):
    rose = RandomOverSampler(random_state=28, shrinkage=shrinkage)
    X_res, y_res = rose.fit_resample(X, y)
    print('RoseSampler :', Counter(y_res))
    return X_res, y_res

def SMOTE_sampler(X, y, k_neighbors):
    sm = SMOTE(random_state=28, k_neighbors=k_neighbors, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X, y)
    print('SmoteSampler :', Counter(y_res))
    return X_res, y_res

def ADASYN_sampler(X, y, n_neighbors): # 可能只關注異常值
    ada = ADASYN(random_state=28, n_neighbors=n_neighbors, n_jobs=-1)
    X_res, y_res = ada.fit_resample(X, y)
    print('AdasynSampler :', Counter(y_res))
    return X_res, y_res

# ---------- UnderSampler ---------- #
def RUS_sampler(X, y):
    rus = RandomUnderSampler(random_state=28)
    X_res, y_res = rus.fit_resample(X, y)
    print('RandomUnderSampler :', Counter(y_res))
    return X_res, y_res

def TomekLinks_sampler(X, y, sampling_strategy):
    tl = TomekLinks(sampling_strategy=sampling_strategy, n_jobs=-1)
    X_res, y_res = tl.fit_resample(X, y)
    print('TomekLinks_Sampler before :', Counter(y))
    print('TomekLinks_Sampler after:', Counter(y_res))
    return X_res, y_res



