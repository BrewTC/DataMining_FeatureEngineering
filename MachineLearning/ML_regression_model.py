from sklearn import metrics
from sklearn.model_selection import cross_val_score
import numpy as np

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# define models to test
def get_models():
    models, names = list(), list()
    # LR
    models.append(LinearRegression(n_jobs=-1))
    names.append('LinearRegression')
    # LDA
    models.append(LinearDiscriminantAnalysis())
    names.append('LinearDiscriminantAnalysis')
    # NB
    models.append(GaussianNB())
    names.append('GaussianNB')
    # RANSAC
    models.append(RANSACRegressor(base_estimator=LinearRegression(), random_state=28))
    names.append('RANSACRegressor')
    # Ridge Regression
    models.append(Ridge(max_iter=-1, random_state=28))
    names.append('Ridge')
    # LASSO Regression
    models.append(Lasso(random_state=28))
    names.append('LASSO')
    # ElasticNet
    models.append(ElasticNet(random_state=28))
    names.append('ElasticNet')
    # Polynomial Regression
    models.append(PolynomialFeatures(degree=2))
    names.append('PolynomialRegression')
    # Stochastic Gradient Descent
    models.append(SGDRegressor(random_state=28))
    names.append('SGDRegressor')
    # Random Forest Regressor
    models.append(RandomForestRegressor(random_state=28, n_jobs=-1))
    names.append('RandomForestRegressor')
    # SVM
    models.append(SVR())
    names.append('SVM') 
    
    return models, names