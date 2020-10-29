# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 17:19:40 2018

@author: paolo
"""

# TODO LIST:
# qs regressori  con le nuove features: https://stackoverflow.com/questions/49094242/svm-provided-a-bad-result-in-my-data-how-to-fix
# feature ranking
#http://scikit-learn.org/stable/modules/feature_selection.html 
#https://www.researchgate.net/publication/220637867_Feature_selection_for_support_vector_machines_with_RBF_kernel
#https://stats.stackexchange.com/questions/2179/variable-importance-from-svm

# hybrid lasso 
# prediction new reviews unknown of the same product
# https://stats.stackexchange.com/questions/47771/what-is-the-intuition-behind-beta-distribution/47782#47782
# https://stats.stackexchange.com/questions/69210/drawing-from-dirichlet-distribution
# https://stats.stackexchange.com/questions/244917/what-exactly-is-the-alpha-in-the-dirichlet-distribution

# ECONOMICS IDEA?


#import os
#import contextlib
from operator import itemgetter
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib


import gc
gc.collect()
# Importing the Training dataset
dataTrain = pd.read_csv(r'C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Result15112018\Training.csv')

X = dataTrain.iloc[:, 5:38].values
print(X)
print(X.shape)
X = np.delete(X,30,axis=1)        #column ProdID deleted
print(X)
print(X.shape)

y = dataTrain.iloc[:, 2:3].values             # HELPFUL VOTES
#y = dataTrain.iloc[3:34000:, 1:2].values            # % HELPFUL VOTES
#y = dataTrain.iloc[3:34000:, 3:4].values            # TOTAL VOTES
print(y)
print(y.shape)

# Importing the Testing dataset 
dataTest = pd.read_csv(r'C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Result15112018\Test.csv')

# Importing the X values of Testing dataset (few votes AND context few votes window 2)
X1_realTest = dataTest.iloc[1:50, 5:38].values
X1_realTest = np.delete(X1_realTest,30,axis=1)        #column ProdID deleted
# Importing  the Y values of the Testing dataset (isolated Low voted)
y1_realTest = dataTest.iloc[1:50, 2:3].values
#Y6=np.delete(Y6, 5, axis=0) # <------------------ THIS IS THE PROBLEM (the reviewer with 1000 reviews)
print(y1_realTest)
print(y1_realTest.shape)

# Importing the X values of Testing dataset (high votes AND context high votes window 2)
X2_realTest = dataTest.iloc[51:62, 5:38].values
X2_realTest = np.delete(X2_realTest,30,axis=1)        #column ProdID deleted
# Importing  the Y values of the Testing dataset (isolated Low voted)
y2_realTest = dataTest.iloc[51:62, 2:3].values
#Y6=np.delete(Y6, 5, axis=0) # <------------------ THIS IS THE PROBLEM (the reviewer with 1000 reviews)
print(y2_realTest)
print(y2_realTest.shape)

# Importing the X values of Testing dataset  (period : 1,2,6,12,24)
X3_realTest = dataTest.iloc[63:113, 5:38].values
X3_realTest = np.delete(X3_realTest,30,axis=1)        #column ProdID deleted
# Importing  the Y values of the Testing dataset (isolated Low voted)
y3_realTest = dataTest.iloc[63:113, 2:3].values
#Y6=np.delete(Y6, 5, axis=0) # <------------------ THIS IS THE PROBLEM (the reviewer with 1000 reviews)
print(y3_realTest)
print(y3_realTest.shape)

# Importing the X values of Testing dataset   (rating : 1,2,3,4,5)
X4_realTest = dataTest.iloc[114:164, 5:38].values
X4_realTest = np.delete(X4_realTest,30,axis=1)        #column ProdID deleted
# Importing  the Y values of the Testing dataset (isolated Low voted)
y4_realTest = dataTest.iloc[114:164, 2:3].values
#Y6=np.delete(Y6, 5, axis=0) # <------------------ THIS IS THE PROBLEM (the reviewer with 1000 reviews)
print(y4_realTest)
print(y4_realTest.shape)



#def get_data(path='.'):
#    p = Path(path)
#    kwargs = dict(delim_whitespace=True, header=None)
#    X_train = pd.read_csv(list(p.glob('trainX.txt*'))[0], **kwargs)
#    y_train = pd.read_csv(list(p.glob('trainY.txt*'))[0], **kwargs)
#    X_test = pd.read_csv(list(p.glob('testX.txt*'))[0], **kwargs)
#    y_test = pd.read_csv(list(p.glob('testY.txt*'))[0], **kwargs)
#    return (pd.concat([X_train, X_test], ignore_index=True),
#            pd.concat([y_train, y_test], ignore_index=True)[0])


def get_data_split():
    return train_test_split(X, y, test_size=0.3, random_state=0)
#X_train,X_test,y_train,y_test=train_test_split(X,Y,..)

# CROSS VALIDATION OF FITTED MODELS
def tune_models_hyperparams(X, y, models, **common_grid_kwargs):        #  **common_grid_kwargs  variable num of arg
    grids = {}                                                          # empty lists/dicts
    for model in models:
        print('{:-^70}'.format(' [' + model['name'] + '] '))
        pipe = Pipeline([                                           # sequence of operations Standar Scale transform your data in noemal distribution will have a mean value 0 and standard deviation of 1. G
                    ("scale", StandardScaler()),
                    (model['name'], model['model'])   ])
        grids[model['name']] = (GridSearchCV(pipe,                  # Exhaustive search over specified parameter values for an estimator
                                           param_grid=model['param_grid'],
                                           **common_grid_kwargs)
                                  .fit(X, y))                       # Run fit with all sets of parameters.    
        # saving single trained model ...
        joblib.dump(grids[model['name']], './{}.pkl'.format(model['name']))
    return grids


# Choose the best model with respect to "Mean squared error regression loss"
def get_best_model(grid, X_test, y_test,
                        metric_func=mean_squared_error):
    res = {name : round(metric_func(y_test, model.predict(X_test)), 3)
           for name, model in grid.items()}
    print('Mean Squared Error:', res)
    best_model_name = min(res, key=itemgetter(1))
    return grid[best_model_name]
#    return model


def test_dataset(grid, X_test, y_test):
    res = {}
    for name, model in grid.items():
        y_pred = model.predict(X_test)
        res[name] = {'MSE': mean_squared_error(y_test, y_pred), # average squared difference between the estimated values and what is estimated
                       'R2': r2_score(y_test, y_pred)       # R2 coefficient of determination is a statistical measure of how well the regression predictions approximate the real data points. An R2 of 1 indicates that the regression predictions perfectly fit the data.
                      }
    return res

def predict(grid, X_test, model_name):
    return grid[model_name].predict(X_test)


 # print best score of cross validation and best params, for each model , but not for all models (see get_best_model)
def print_grid_results(grids):
    for name, model in grids.items():
        print('{:-^70}'.format(' [' + name + '] '))
        print('Score:\t\t{:.2%}'.format(model.best_score_))     
        print('Parameters:\t{}'.format(model.best_params_))
        print('*' * 70)


def predict_save_results (X_realTest, y_realTest, X, y, regressor, name, typeTest):
    i = 0;
    #Y7=np.delete(Y7, np.s_[6:7], axis=1) #column votes of reviewer deleted
    #Y7=np.delete(Y7, 5, axis=0) 
    #print(X_realTest)
    regressor.fit(X,y);
    result = np.array(X_realTest)
    num_rows, num_cols = result.shape
    out = np.zeros(shape=(num_rows,2))
#    out[0] = ["predicted","real"]
    for p in result:
        p = p.reshape(1, -1)
    #    Y1_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(p)))   #Using Feature scaling
        y_item_pred = regressor.predict(p)
        print("predicted ", y_item_pred, "real ", y_realTest[i])
        out[i] = [y_item_pred,y_realTest[i]]    # format these data!!!!!!
        i = i+1   
    y_pred = regressor.predict(X_realTest)
    train_score=regressor.score(X,y)
    test_score=regressor.score(X_realTest,y_realTest)    
    MSE =  mean_squared_error(y_realTest, y_pred)
    R2 = r2_score(y_realTest, y_pred) 
    print (" Result final prediction on Test "+typeTest)
    print ("Train_score: ", train_score, " test_score: ", test_score, " MSE= ", MSE, " R2= ", R2)
    np.savetxt(r"C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Result15112018\Result"+name+"."+typeTest+".csv", out, delimiter=",")
  


models = [
    {   'name':     'SVR_rbf',
        'model':    SVR(kernel='rbf', cache_size=1000),
        'title':    "SVR_rbf",       
        'param_grid': {
            'SVR_rbf__C':           [100.0, 400.0, 800.0],
            'SVR_rbf__max_iter':    [2000],
            'SVR_rbf__gamma':           [0.1, 0.05, 0.02]
#            'SVR_rbf__cache_size':  [1000]
         } 
    },
    {   'name':     'SVR_linear',
        'model':      SVR(kernel='linear'),
        'title':    "SVR_rbf",
        'param_grid': {
            'SVR_linear__C':           [0.01, 0.1, 1, 5, 100.0, 400.0, 800.0],
            'SVR_linear__max_iter':    [5000]
        } 
    },
    {   'name':     'Ridge',
        'model':    Ridge(),
        'title':    "Ridge",
        'param_grid': {
            'Ridge__alpha':         [0.0001, 0.1, 0.5, 5, 10, 50, 100, 500],
            'Ridge__max_iter':      [1000]
        } 
    },
    {   'name':     'Lasso',
        'model':    Lasso(),
        'title':    "Lasso",
        'param_grid':  {
            'Lasso__alpha':         [0.0001, 0.001, 0.01, 0.1, 1, 10],
            'Lasso__max_iter':      [10e5],
            'Lasso__normalize':     [True]
         } 
    },
    {   'name':     'RandomForest',
        'model':    RandomForestRegressor(),
        'title':    "RandomForest",
        'param_grid':  {
            'RandomForest__n_estimators':   [50, 250, 500],
            'RandomForest__max_depth':      [5],
        } 
    },
]

# When a python program is executed, directly by the interpreter,  python interpreter starts executing code inside it
def main(): 
    X_train, X_test, y_train, y_test = \
        get_data_split()  # X_train, X_test, y_train, y_test of CROSS VALIDATION PROCESS (all data extracted from primitive training X,y)
    grid = tune_models_hyperparams(X_train, y_train, models, cv=3,  # cross-validation of 
                                   verbose=2, n_jobs=-1)
    print_grid_results(grid)                        # print best score of cross validation and best params..
    get_best_model(grid, X_test, y_test)    # Mean squared error regression loss
    
    predict_save_results (X1_realTest, y1_realTest, X, y, Lasso(alpha = 0.01, max_iter=1000000, normalize=True), 'Lasso', 'few votes AND context few votes')
    predict_save_results (X2_realTest, y2_realTest, X, y, Lasso(alpha = 0.01, max_iter=1000000, normalize=True), 'Lasso', 'high votes AND context high votes')
    predict_save_results (X3_realTest, y3_realTest, X, y, Lasso(alpha = 0.01, max_iter=1000000, normalize=True), 'Lasso', 'period on the platform')
    predict_save_results (X4_realTest, y4_realTest, X, y, Lasso(alpha = 0.01, max_iter=1000000, normalize=True), 'Lasso', 'rating')
    
    
    

    

#    df = pd.DataFrame({'predicted': model.predict(X7)})
#    print(df)
#    dr = pd.DataFrame({'real': y6})
#    print(dr)
#    df.to_csv('C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\predicted.csv', index=False)

#if __name__ == "__main__":
#    p =  Path(__file__).parent.resolve()
#    main(p)
