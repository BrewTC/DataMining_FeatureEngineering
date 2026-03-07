from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def Decision_Tree_Classifier(X_train, y_train, X_test, y_test):
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state=28)
    dt.fit(X_train,y_train)
    y_train_hat = dt.predict(X_train)
    y_test_hat = dt.predict(X_test)

    print(dt)
    print('Train performance')
    print('-------------------------------------------------------')
    print(classification_report(y_train,y_train_hat))

    print('Test performance')
    print('-------------------------------------------------------')
    print(classification_report(y_test,y_test_hat))
    
    print('Confusion matrix')
    print('-------------------------------------------------------')
    print(confusion_matrix(y_test,y_test_hat))

def Random_Forest_Classifier(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=28, n_jobs=-1)
    rf.fit(X_train,y_train)
    y_train_hat = rf.predict(X_train)
    y_test_hat = rf.predict(X_test)

    print(rf)
    print('Train performance')
    print('-------------------------------------------------------')
    print(classification_report(y_train,y_train_hat))

    print('Test performance')
    print('-------------------------------------------------------')
    print(classification_report(y_test,y_test_hat))
    
    print('Confusion matrix')
    print('-------------------------------------------------------')
    print(confusion_matrix(y_test,y_test_hat))

def K_Neighbors_Classifier(X_train, y_train, X_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    y_train_hat = knn.predict(X_train)
    y_test_hat = knn.predict(X_test)

    print(knn)
    print('Train performance')
    print('-------------------------------------------------------')
    print(classification_report(y_train, y_train_hat))

    print('Test performance')
    print('-------------------------------------------------------')
    print(classification_report(y_test, y_test_hat))

    print('Confusion matrix')
    print('-------------------------------------------------------')
    print(confusion_matrix(y_test, y_test_hat))

def XGBoost_Classifier(X_train, y_train, X_test, y_test):
    from xgboost import XGBClassifier
    xgb = XGBClassifier(random_state=28, n_jobs=-1)
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

    print('Confusion matrix')
    print('-------------------------------------------------------')
    print(confusion_matrix(y_test, y_test_hat))

def Support_Vector_Classifier(X_train, y_train, X_test, y_test):
    from sklearn.svm import SVC
    svc=SVC(random_state=28, max_iter=-1)
    svc.fit(X_train,y_train)
    y_train_hat = svc.predict(X_train)
    y_test_hat = svc.predict(X_test)

    print(svc)
    print('Train performance')
    print('-------------------------------------------------------')
    print(classification_report(y_train,y_train_hat))

    print('Test performance')
    print('-------------------------------------------------------')
    print(classification_report(y_test,y_test_hat))
    
    print('Confusion matrix')
    print('-------------------------------------------------------')
    print(confusion_matrix(y_test,y_test_hat))

def Logistic_Regression(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=28, n_jobs=-1)
    model.fit(X_train, y_train)
    y_train_hat = model.predict(X_train)
    y_test_hat = model.predict(X_test)

    print(model)
    print('Train performance')
    print('-------------------------------------------------------')
    print(classification_report(y_train, y_train_hat))

    print('Test performance')
    print('-------------------------------------------------------')
    print(classification_report(y_test, y_test_hat))

    print('Confusion matrix')
    print('-------------------------------------------------------')
    print(confusion_matrix(y_test, y_test_hat))






