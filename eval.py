import pandas as pd 
from sklearn.grid_search import GridSearchCV
from sklearn.utils import check_random_state 
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR  
import numpy as np 
import sys 
 

rs = check_random_state(42)

def evaluate(train_name, test_name, model_name, model, params): 
    # load train data 
    data = pd.read_csv(train_name,sep=';')
    X_train = data[data.columns[:-1]]
    y_train = data[data.columns[-1]]
 
    # load test data 
    data = pd.read_csv(test_name,sep=';') 
    X_test = data[data.columns[:-1]]
    y_test = data[data.columns[-1]]

    kr = GridSearchCV(model, param_grid = params, cv=5, n_jobs = 10, pre_dispatch = "n_jobs")
    kr.fit(X_train, y_train)
    yy_train = kr.predict(X_train) 
    yy_test = kr.predict(X_test)
    diff = (y_train-yy_train)
    train_error = 100*sum(diff*diff)/len(yy_train)
    diff = (y_test-yy_test)
    test_error = 100*sum(diff*diff)/len(yy_test) 
    print(model_name, train_name,kr.best_params_,train_error,test_error)
    
    
            


setups = [ ("KernelRidge_rbf", KernelRidge(kernel = 'rbf', gamma=0.1), {"alpha": np.logspace(-7,0,100),
                                                                         "gamma": np.logspace(-3,0,100)}),
            ("KernelRidge_linear", KernelRidge(kernel = 'linear'), {"alpha": np.logspace(-7,0,100)}),
            ("SVR_rbf", SVR(kernel = 'rbf', gamma = 0.1), { "C": np.logspace(-7,0,100),
                                                            "gamma": np.logspace(-3,0,100)}), 
            ("SVR_linear", SVR(kernel = 'linear'), { "C": np.logspace(-7,0,100)}), 
            ("SVR_poly",  SVR(kernel = 'poly'), { "C": np.logspace(-7,0,100),
                                                  "degree": [2,3],
                                                  "coef0": np.linspace(-5,5,100)}), 
            ("SVR_sigmoid", SVR(kernel = 'sigmoid'),  { "C": np.logspace(-7,0,100),
                                                        "coef0": np.linspace(-5,5,100)}) ]
  
  

for s in setups:
    for x in ["CO","NOx","NO2"]:
        for i in range(1,6):
            evaluate("data/"+x+"-nrm-part"+str(i)+".train.csv",
                     "data/"+x+"-nrm-part"+str(i)+".test.csv",
                     s[0], s[1], s[2])
                 
                 
                 

