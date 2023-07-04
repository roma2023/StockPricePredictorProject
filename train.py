import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from finta import TA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
def precisionSMP(values, prediction):
    good = 0
    total = 0
    if(np.sum(prediction) == 0):
        return 0
    for it in range(len(values)):
        if prediction[it] == 1 and values[it]==1:
                good+=1
    return good/np.sum(prediction)


def _train_KNN(X_train, y_train, X_test, y_test):

    knn = KNeighborsClassifier()
    # Create a dictionary of all values we want to test for n_neighbors
    params_knn = {'n_neighbors': np.arange(1, 25)}
    
    # Use gridsearch to test all values for n_neighbors
    knn_gs = GridSearchCV(knn, params_knn, cv=5)
    
    # Fit model to training data
    knn_gs.fit(X_train, y_train)
    
    # Save best model
    knn_best = knn_gs.best_estimator_
     
    # Check best n_neigbors value
    #print(knn_gs.best_params_)
    
    prediction = knn_best.predict(X_test)
    
    #cm = confusion_matrix(y_test, prediction)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #disp.plot(cmap=plt.cm.Blues)
    #plt.title("Confusion Matrix")
    #plt.show()

    #print(classification_report(y_test, prediction))
    #print(confusion_matrix(y_test, prediction))
    
    return knn_best
    
def train_svm(X_train, y_train, X_test, y_test):
    svm = SVC()
    # Create a dictionary of all values we want to test for C and gamma
    params_svm = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 'scale', 'auto']}
    
    # Use gridsearch to test all values for C and gamma
    svm_gs = GridSearchCV(svm, params_svm, cv=5)
    
    # Fit model to training data
    svm_gs.fit(X_train, y_train)
    
    # Save best model
    svm_best = svm_gs.best_estimator_
     
    # Check best C and gamma values
    #print(svm_gs.best_params_)
    
    prediction = svm_best.predict(X_test)
    
    #cm = confusion_matrix(y_test, prediction)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #disp.plot(cmap=plt.cm.Blues)
    #plt.title("Confusion Matrix")
    #plt.show()

    #print(classification_report(y_test, prediction))
    #print(confusion_matrix(y_test, prediction))
    
    return svm_best

def train_rf(X_train, y_train, X_test, y_test):

    """
    Function that uses random forest classifier to train the model
    :return:
    """
    
    # Create a new random forest classifier
    rf = RandomForestClassifier()
    
    # Dictionary of all values we want to test for n_estimators
    params_rf = {'n_estimators': [110,130,140,150,160,180,200]}
    
    # Use gridsearch to test all values for n_estimators
    rf_gs = GridSearchCV(rf, params_rf, cv=5)
    
    # Fit model to training data
    rf_gs.fit(X_train, y_train)
    
    # Save best model
    rf_best = rf_gs.best_estimator_
    
    # Check best n_estimators value
    #print(rf_gs.best_params_)
    
    prediction = rf_best.predict(X_test)
    
    #cm = confusion_matrix(y_test, prediction)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #disp.plot(cmap=plt.cm.Blues)
    #plt.title("Confusion Matrix")
    #plt.show()

    #print(classification_report(y_test, prediction))
    #print(confusion_matrix(y_test, prediction))
    
    return rf_best
def train_dt(X_train, y_train, X_test, y_test):

    """
    Function that uses random forest classifier to train the model
    :return:
    """
    
    # Create a new random forest classifier
    dt = tree.DecisionTreeClassifier()
    
    # Dictionary of all values we want to test for n_estimators
    params_grid = {'max_depth':range(1,11), 'min_samples_split':range(1,11)}
    
    # Use gridsearch to test all values for n_estimators
    dt_gs = GridSearchCV(dt, params_grid, cv=5)
    
    # Fit model to training data
    dt_gs.fit(X_train, y_train)
    
    # Save best model
    dt_best = dt_gs.best_estimator_
    
    # Check best n_estimators value
    #print(rf_gs.best_params_)
    
    prediction = dt_best.predict(X_test)
    
    #cm = confusion_matrix(y_test, prediction)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #disp.plot(cmap=plt.cm.Blues)
    #plt.title("Confusion Matrix")
    #plt.show()

    #print(classification_report(y_test, prediction))
    #print(confusion_matrix(y_test, prediction))
    
    return dt_best

def _ensemble_model(svm_model, dt_model, X_train, y_train, X_test, y_test):
    
    # Create a dictionary of our models
    estimators=[('svm', svm_model),('ds', dt_model)]
    
    # Create AdaBoost classifier and fit it to the ensemble model
    adaboost = AdaBoostClassifier(n_estimators= 700, random_state=42)
    adaboost.fit(X_train, y_train)
    
    # Test our model on the test data
    #print(adaboost.score(X_test, y_test))
    
    prediction = adaboost.predict(X_test)
    
    #cm = confusion_matrix(y_test, prediction)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #disp.plot(cmap=plt.cm.Blues)
    #plt.title("Confusion Matrix")
    #plt.show()

    #print(classification_report(y_test, prediction))
    #print(confusion_matrix(y_test, prediction))
    
    return adaboost

def cross_Validation_ADA(data):

    # Split data into equal partitions of size len_train
    
    num_train = 5 # Increment of how many starting points (len(data) / num_train  =  number of train-test sets)
    len_train = 50 # Length of each train-test set
    
    # Lists to store the results from each model
    rf_RESULTS = []
    knn_RESULTS = []
    svm_RESULTS = []
    dt_RESULTS = []
    ensemble_RESULTS = []
    
    i = 0
    printer = 0
    Best_Model = 0
    Best_Model_rf = 0
    Best_Model_knn = 0
    Best_Model_svm = 0
    Best_Model_dt = 0
    Best_Model_ensemble = 0
    Best_precision = 0
    Best_precision_rf = 0
    Best_precision_knn = 0
    Best_precision_svm = 0
    Best_precision_dt = 0
    Best_precision_ensemble = 0
    while True:
        
        # Partition the data into chunks of size len_train every num_train days
        df = data.iloc[i * num_train : (i * num_train) + len_train]
        i += 1
        
        
        
        if len(df) < 25:
            break
        
        y = df['pred']
        features = [x for x in df.columns if x not in ['pred']]
        X = df[features]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 7 * len(X) // 10,shuffle=False)
        try:
            rf_model = train_rf(X_train, y_train, X_test, y_test)
            knn_model = _train_KNN(X_train, y_train, X_test, y_test)
            svm_model = train_svm(X_train, y_train, X_test, y_test)
            dt_model = train_dt(X_train, y_train, X_test, y_test)
            ensemble_model = _ensemble_model(svm_model, dt_model, X_train, y_train, X_test, y_test)

            rf_prediction = rf_model.predict(X_test)
            knn_prediction = knn_model.predict(X_test)
            svm_prediction = svm_model.predict(X_test)
            dt_prediction = dt_model.predict(X_test)
            ensemble_prediction = ensemble_model.predict(X_test)

            rf_precision = precisionSMP(y_test.values, rf_prediction)
            knn_precision = precisionSMP(y_test.values, knn_prediction)
            svm_precision = precisionSMP(y_test.values, svm_prediction)
            dt_precision = precisionSMP(y_test.values, dt_prediction)
            ensemble_precision = precisionSMP(y_test.values, ensemble_prediction)
        except:
            y_train[0] = 0
            y_train[1] = 1
            rf_model = train_rf(X_train, y_train, X_test, y_test)
            knn_model = _train_KNN(X_train, y_train, X_test, y_test)
            svm_model = train_svm(X_train, y_train, X_test, y_test)
            dt_model = train_dt(X_train, y_train, X_test, y_test)
            ensemble_model = _ensemble_model(svm_model, dt_model, X_train, y_train, X_test, y_test)

            rf_prediction = rf_model.predict(X_test)
            knn_prediction = knn_model.predict(X_test)
            svm_prediction = svm_model.predict(X_test)
            dt_prediction = dt_model.predict(X_test)
            ensemble_prediction = ensemble_model.predict(X_test)

            rf_precision = precisionSMP(y_test.values, rf_prediction)
            knn_precision = precisionSMP(y_test.values, knn_prediction)
            svm_precision = precisionSMP(y_test.values, svm_prediction)
            dt_precision = precisionSMP(y_test.values, dt_prediction)
            ensemble_precision = precisionSMP(y_test.values, ensemble_prediction)
        if Best_precision < rf_precision: 
            Best_Model = rf_model
            Best_precision = rf_precision
        if Best_precision < knn_precision: 
            Best_Model = knn_model
            Best_precision = knn_precision
        if Best_precision < svm_precision: 
            Best_Model = svm_model
            Best_precision = svm_precision
        if Best_precision < dt_precision: 
            Best_Model = dt_model
            Best_precision = dt_precision
        if Best_precision < ensemble_precision: 
            Best_Model = ensemble_model
            Best_precision = ensemble_precision
        if Best_precision_rf < rf_precision: 
            Best_Model_rf = rf_model
            Best_precision_rf = rf_precision
        if Best_precision_knn < knn_precision: 
            Best_Model_knn = knn_model
            Best_precision_knn = knn_precision
        if Best_precision_svm < svm_precision: 
            Best_Model_svm = svm_model
            Best_precision_svm = svm_precision
        if Best_precision_dt < dt_precision: 
            Best_Model_dt = dt_model
            Best_precision_dt = dt_precision
        if Best_precision_ensemble < ensemble_precision: 
            Best_Model_ensemble = ensemble_model
            Best_precision_ensemble = ensemble_precision
        #if (printer%10 == 0):
            #print(i * num_train, (i * num_train) + len_train)
            #print('rf prediction is ', rf_prediction)
            #print('knn prediction is ', knn_prediction)
            #print('svm prediction is ', svm_prediction)
            #print('ensemble prediction is ', ensemble_prediction)
            #print('truth values are ', y_test.values)
            #print(rf_precision, knn_precision, svm_precision, ensemble_precision)
        rf_RESULTS.append(rf_precision)
        knn_RESULTS.append(knn_precision)
        svm_RESULTS.append(svm_precision)
        dt_RESULTS.append(dt_precision)
        ensemble_RESULTS.append(ensemble_precision)
        printer = printer + 1
        
    print('SVM precision = ' + str( sum(svm_RESULTS) / len(svm_RESULTS)))
    print('RF precision = ' + str( sum(rf_RESULTS) / len(rf_RESULTS)))
    print('KNN precision = ' + str( sum(knn_RESULTS) / len(knn_RESULTS)))
    print('DT precision = ' + str( sum(dt_RESULTS) / len(dt_RESULTS)))
    print('Ensemble precision = ' + str( sum(ensemble_RESULTS) / len(ensemble_RESULTS)))
    if(Best_precision == Best_precision_ensemble):
        return Best_Model_ensemble, Best_precision_ensemble
    if(Best_precision == Best_precision_rf):
        return Best_Model_rf, Best_precision_rf
    if(Best_precision == Best_precision_knn):
        return Best_Model_knn, Best_precision_knn
    if(Best_precision == Best_precision_svm):
        return Best_Model_svm, Best_precision_svm
    if(Best_precision == Best_precision_dt):
        return Best_Model_dt, Best_precision_dt
    return Best_Model, Best_precision
