
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

	
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
### 注意 ###
#  多分類標籤下要更改參數 average=None is only implemented for multi_class='ovo' and average='micro' is only implemented for multi_class='ovr'
	

def RF_GS_hyper_tuning(X_train, y_train, X_test, y_test):
	# GridSearch score with F1-score
	# https://stackoverflow.com/questions/56084591/how-to-do-gridsearchcv-for-f1-score-in-classification-problem-with-scikit-learn
	# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
	# https://scikit-learn.org/stable/modules/model_evaluation.html
	# from sklearn.metrics import f1_score, make_scorer
	# f1 = make_scorer(f1_score , average='macro')

	# Random forest
	# n_jobs=-1 to allow run it on all cores
	params = {
			'n_estimators': [100,200,300,400,500],
			'criterion': ['gini', 'entropy'],
			'min_samples_split': [2,4,5,6,8],
			'min_samples_leaf': [2,4,5,6,8],
			'max_leaf_nodes': [4,10,20,50,None]
			}

	gs = GridSearchCV(
						RandomForestClassifier(random_state=28, n_jobs=8), 
						params, 
						n_jobs=8, 
						cv=KFold(n_splits=3), 
						scoring='f1'
					) # scoring='roc_auc', 'f1_weighted'
					
	gs.fit(X_train, y_train)

	print('Best score:', gs.best_score_)
	print('Best score:', gs.best_params_)

	gsbp = gs.best_params_

	rf = RandomForestClassifier(
								n_estimators=gsbp['n_estimators'], 
								criterion=gsbp['criterion'], 
								min_samples_split=gsbp['min_samples_split'], 
								min_samples_leaf=gsbp['min_samples_leaf'],
								max_leaf_nodes=gsbp['max_leaf_nodes'],
								random_state=28,
								n_jobs=8
							)

	rf.fit(X_train,y_train)
	y_train_hat =rf.predict(X_train)
	y_test_hat =rf.predict(X_test)

	print(rf)
	print('Train performance')
	print('-------------------------------------------------------')
	print(classification_report(y_train,y_train_hat))

	print('Test performance')
	print('-------------------------------------------------------')
	print(classification_report(y_test,y_test_hat))

	# print('Roc_auc score')
	# print('-------------------------------------------------------')
	# print(roc_auc_score(y_test,y_test_hat))
	# print('')

	print('Confusion matrix')
	print('-------------------------------------------------------')
	print(confusion_matrix(y_test,y_test_hat))

def RF_Randomized_Search_hyper_tuning(X_train, y_train, X_test, y_test):

	params = {
				'n_estimators': [100,200,300,400,500],
				'criterion': ['gini', 'entropy'],
				'bootstrap': [True,False],
				'max_depth': [3,5,10,20,50,75,100,None],
				'min_samples_split': [2,4,5,6,8],
				'min_samples_leaf': [2,4,5,6,8],
				'max_leaf_nodes': [4,10,20,50,None]
			}

	gs = RandomizedSearchCV( 
							RandomForestClassifier(random_state=28, n_jobs=8), 
							param_distributions=params, 
							n_iter = 100, 
							n_jobs=8, 
							cv=KFold(n_splits=3), 
							verbose = 0, 
							scoring='f1'
						) # scoring='roc_auc', 'f1_weighted'

	gs.fit(X_train, y_train)

	print('Best score:', gs.best_score_)
	print('Best score:', gs.best_params_)

	gsbp = gs.best_params_
	rf = RandomForestClassifier(n_estimators=gsbp['n_estimators'], 
								criterion=gsbp['criterion'], 
								bootstrap=gsbp['bootstrap'],
								max_depth=gsbp['max_depth'],
								min_samples_split=gsbp['min_samples_split'], 
								min_samples_leaf=gsbp['min_samples_leaf'],
								max_leaf_nodes=gsbp['max_leaf_nodes'],
								random_state=28,
								n_jobs=8
							)

	rf.fit(X_train,y_train)
	y_train_hat =rf.predict(X_train)
	y_test_hat =rf.predict(X_test)

	print(rf)
	print('Train performance')
	print('-------------------------------------------------------')
	print(classification_report(y_train,y_train_hat))

	print('Test performance')
	print('-------------------------------------------------------')
	print(classification_report(y_test,y_test_hat))

	# print('Roc_auc score')
	# print('-------------------------------------------------------')
	# print(roc_auc_score(y_test,y_test_hat))
	# print('')

	print('Confusion matrix')
	print('-------------------------------------------------------')
	print(confusion_matrix(y_test,y_test_hat))

def XGBoost_hyperparameter_tuning(X_train, y_train, X_test, y_test):

	params = {
				'n_estimators': [100], # ,200,300,400,500
				'learning_rate': [0.01,0.02,0.03,0.04,0.05,0.1], # [*range(0.01, 1.1, 0.01)],  
				'booster': ['gbtree', 'gblinear'],
				'gamma': [0, 0.5, 1],
				'reg_alpha': [0, 0.5, 1],
				'reg_lambda': [0.5, 1, 5],
				'base_score': [0.2, 0.5, 1]
			}

	gs = GridSearchCV(
						XGBClassifier(random_state=28, n_jobs=8), 
						params, 
						n_jobs=8, 
						cv=KFold(n_splits=3), 
						scoring='f1'
					)

	gs.fit(X_train, y_train)

	print('Best score:', gs.best_score_)
	print('Best score:', gs.best_params_)

	gsbp = gs.best_params_
	xgb = XGBClassifier(
		n_estimators=gsbp['n_estimators'], 
		learning_rate=gsbp['learning_rate'], 
		booster=gsbp['booster'], 
		gamma=gsbp['gamma'],
		reg_alpha=gsbp['reg_alpha'],
		reg_lambda=gsbp['reg_lambda'],
		base_score=gsbp['base_score'],
		random_state=28,
		n_jobs=8,
	)

	xgb.fit(X_train, y_train)
	y_train_hat = xgb.predict(X_train)
	y_test_hat = xgb.predict(X_test)

	print(xgb)
	print('Train performance')
	print('-------------------------------------------------------')
	print(classification_report(y_train, y_train_hat))

	print('Test performance')
	print('-------------------------------------------------------')
	print(classification_report(y_test, y_test_hat))

	# print('Roc_auc score')
	# print('-------------------------------------------------------')
	# print(roc_auc_score(y_test, y_test_hat)) 
	# # 多分類標籤下要更改參數 average=None is only implemented for multi_class='ovo' and average='micro' is only implemented for multi_class='ovr'
	# print('')

	print('Confusion matrix')
	print('-------------------------------------------------------')
	print(confusion_matrix(y_test, y_test_hat))

def Extra_Trees_hyperparameter_tuning(X_train, y_train, X_test, y_test):
	
	params = {
				'n_estimators': [100, 200, 500],
				'criterion': ['gini', 'entropy'],
				'min_samples_split': [1,2,4,5],
				'min_samples_leaf': [1,2,4,5],
				'max_leaf_nodes': [4,10,20,50,None]
			}
	
	gs = GridSearchCV(
						ExtraTreesClassifier(random_state=28, n_jobs=8), 
						params, 
						n_jobs=8, 
						cv=KFold(n_splits=3), 
						scoring='f1'
					) # scoring='roc_auc', 'f1_weighted'
	
	gs.fit(X_train, y_train)
	
	print('Best score:', gs.best_score_)
	print('Best score:', gs.best_params_)
	
	gsbp = gs.best_params_

	etc = ExtraTreesClassifier(			
								n_estimators=gsbp['n_estimators'], 
								criterion=gsbp['criterion'], 
								min_samples_split=gsbp['min_samples_split'], 
								min_samples_leaf=gsbp['min_samples_leaf'],
								max_leaf_nodes=gsbp['max_leaf_nodes'],
								random_state=28,
								n_jobs=8
							)

	etc.fit(X_train, y_train)
	y_train_hat = etc.predict(X_train)
	y_test_hat = etc.predict(X_test)
	
	print(etc)
	print('Train performance')
	print('-------------------------------------------------------')
	print(classification_report(y_train, y_train_hat))
	
	print('Test performance')
	print('-------------------------------------------------------')
	print(classification_report(y_test, y_test_hat))
	
	# print('Roc_auc score')
	# print('-------------------------------------------------------')
	# print(roc_auc_score(y_test, y_test_hat))
	# print('')
	
	print('Confusion matrix')
	print('-------------------------------------------------------')
	print(confusion_matrix(y_test, y_test_hat))
	
def K_Neighbors_Classifier_hyperparameter_tuning(X_train, y_train, X_test, y_test):

    params = {
                'n_neighbors' : [3,5,7,9,11,13,15],
                'weights' : ['uniform', 'distance'],
                'algorithm' : ['auto', 'ball_tree','kd_tree'],
                'p' : [1,2]
            }

    gs = GridSearchCV(KNeighborsClassifier(n_jobs=8), params, cv=KFold(n_splits=3), n_jobs=8)
    gs.fit(X_train, y_train)
        
    print('Best score:', gs.best_score_)
    print('Best score:', gs.best_params_)

    gsbp = gs.best_params_

    Knn = KNeighborsClassifier(
								n_neighbors=gsbp['n_neighbors'], 
								weights=gsbp['weights'], 
								algorithm=gsbp['algorithm'],
								p=gsbp['p'],
								n_jobs=8
   							)

    Knn.fit(X_train,y_train)

    y_train_hat = Knn.predict(X_train)
    y_test_hat = Knn.predict(X_test)

    Knn.fit(X_train,y_train)
    y_train_hat = Knn.predict(X_train)
    y_test_hat = Knn.predict(X_test)

    print(Knn)
    print('Train performance')
    print('-------------------------------------------------------')
    print(classification_report(y_train,y_train_hat))

    print('Test performance')
    print('-------------------------------------------------------')
    print(classification_report(y_test,y_test_hat))

    # print('Roc_auc score')
    # print('-------------------------------------------------------')
    # print(roc_auc_score(y_test,y_test_hat))
    # print('')

    print('Confusion matrix')
    print('-------------------------------------------------------')
    print(confusion_matrix(y_test,y_test_hat))

def Support_Vector_Classification(X_train, y_train, X_test, y_test):
    # GridSearch score with F1-score
	# https://stackoverflow.com/questions/56084591/how-to-do-gridsearchcv-for-f1-score-in-classification-problem-with-scikit-learn
	# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
	# https://scikit-learn.org/stable/modules/model_evaluation.html
    # from sklearn.metrics import f1_score, make_scorer
    # f1_weighted = make_scorer(f1_score , average='f1_weighted')
    
    params = {
				'C': [10, 15, 20, 25, 30, 35],  # 0.1, 1, 10, 20, 30, 50, 1000, 100
				'kernel': ['linear','rbf'], # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
				'gamma': ['scale', 'auto'], # , 'auto'
				'max_iter':[-1]
			}
	
    gs = GridSearchCV(
						SVC(random_state=28, max_iter=-1), 
						params, 
						n_jobs=8, 
						cv=KFold(n_splits=3), 
						scoring='f1'
					) # f1_weighted

    gs.fit(X_train, y_train)

    print('Best score:', gs.best_score_)
    print('Best score:', gs.best_params_)

    gsbp = gs.best_params_

    svc = SVC(
				C=gsbp['C'], 
				kernel=gsbp['kernel'],
				gamma=gsbp['gamma'], 
				max_iter=gsbp['max_iter'],
				random_state=28
			)
    
    svc.fit(X_train,y_train)

    y_train_hat =svc.predict(X_train)
        
    y_test_hat =svc.predict(X_test)
	
    print(svc)
    print('Accuracy score')
    print(accuracy_score(y_test_hat,y_test))
    print('Train performance')
    print('-------------------------------------------------------')
    print(classification_report(y_train,y_train_hat))

    print('Test performance')
    print('-------------------------------------------------------')
    print(classification_report(y_test,y_test_hat))

    # print('Roc_auc score')
    # print('-------------------------------------------------------')
    # print(roc_auc_score(y_test,y_test_hat))
    # print('')
    
    print('Confusion matrix')
    print('-------------------------------------------------------')
    print(confusion_matrix(y_test,y_test_hat))





