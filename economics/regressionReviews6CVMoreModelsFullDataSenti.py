# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 17:07:16 2019

INTERESTING: try to use LSTM Neural Network in order to evaluate features regression
see: 
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/



@author: paolo
"""

import itertools as IT
import argparse as argsp
import pandas as pd
import numpy as np
import statistics as stat
import random as rand
import scipy.stats as scstat
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from pandas import Series
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
from matplotlib.ticker import NullFormatter
from matplotlib.dates import MonthLocator, DateFormatter
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import math 
import operator 
from functools import reduce


# important products to set the new tests
# https://www.amazon.com/dp/B0041Q38NU 1.700 reviews NOW
# https://www.amazon.com/dp/B004G6002M   reviews
# https://www.amazon.com/dp/B004GF8TIK  reviews
# https://www.amazon.com/dp/B0044YU60M 
# https://www.amazon.com/B0043T7FXE 

import sys
sys.path.append('C:\\UniDatiSperimentali\\PAOLONE IDEAS\dottorato\\python e SIDE PG\\pythonCode')
for line in sys.path: 
    print (line)
    
from regressionReviews5CVMoreModelsFullDataSenti import core, coreFeatureExtraction, simpleOLS, All_pcc, \
list_sameLen_textReview, list_sameColumn, list_sameRating, list_sameProd, list_groupProds, list_LittleRating, list_HighRating, \
list_smallVotes, list_largeVotes, list_percentVotes, list_groupProdsCriteria, createUniqueProd, list_groupProdsPlusDate, plotMultiTrends, \
normalization, predict_results

import gc
gc.collect()

# Importing the Training dataset
dataTrain = pd.read_csv(r'C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Result10022019\Training.csv', error_bad_lines=False)
series = Series.from_csv(r'C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Result10022019\Training.csv', header=0)

def DataGenTest (numTest):   
        X00 = dataTrain.iloc[:, 5:46].values  # all columns
        print(X00)
        print(X00.shape)
        X0 = dataTrain.iloc[:, 5:45].values   # minus rewiewer id 
        print(X0)
        print(X0.shape)
        X_ = np.delete(X0,34,axis=1)        #column ProdID deleted
        print(X_)
        print(X_[0])
        print('X minus ProdID r',X_.shape)        
        X1_ = np.delete(X_,np.s_[6:9],axis=1)        #  3 columns reviewer reputation
        print(X1_[0])
        print('X1 minus3 features reviewer',X1_.shape)        
        X1A_ = np.delete(X_,np.s_[7],axis=1)        #  minus %Hvotes reviewer
        print(X1A_[0])
        print('X1 minus %Hvotes reviewer',X1A_.shape)        
        y_ = dataTrain.iloc[:, 1:2].values             # % HELPFUL VOTES
        y__ = dataTrain.iloc[:, 2:3].values             # HELPFUL VOTES
        y___ = dataTrain.iloc[:, 3:4].values            # TOTAL VOTES
        
        date = dataTrain.iloc[:, 4:5].values 
        
        print(y_)
        print(y_.shape)
        if (numTest == 1):
            return X_, y_
        if (numTest == 2):
            return X_, y_, X1_
        if (numTest == 6):
            return X_, y_, X1A_
        if (numTest == 4):
            return X0, X_, y_, y__, y___, date    # %v+, v+, v   # with prodID
        if (numTest == 5):
            return X0, X_, y_, y__     # with prodID  # with  # % HELPFUL VOTES, # HELPFUL VOTES
   
                   
        X0 = dataTrain.iloc[:, 5:45].values
        print(X0)
        print(X0.shape)
        print('\nORIGINAL ',X0[0])
        X_ = np.delete(X0,np.s_[26:35],axis=1)        #columns SENTIMENTS  deleted
        print('\nDEL SENTIMENT + prod ',X_[0])
        print(X_.shape)        
        #plotting_trends (y_, date, 'prova1')
        
        if (numTest == 3):        
            return X_, y_


#0, 5, 7, 21
            
def dirichlet_distr (best_model, X0, X,  y, y__, y___, dateCol, IndexRating, IndexRepu1Feature, Index2Repu2eature, IndexLenFeature, listProdIDs, listProdID2s,listProdID3s):
    
        colProduct = len(X0[0]) - 1
        newX, newy, ya, yb, newDateCol = list_groupProdsPlusDate (IT.zip_longest(X0, y, y__, y___),colProduct,listProdIDs,dateCol)
        newX = np.array(newX)    # leave the product column
        newy = np.array(newy)    
        print ('newy',newy)
        X_Test = newX
        y_Test = newy
        X_Test = np.delete(X_Test,colProduct,axis=1)        #column ProdID deleted
        y_pred = predict_results (X_Test, y_Test, X, y, best_model, "")
        newy = np.array(newy)        # v%+
        ya = np.array(ya)            # 1 - %v+  (Thre arent the %v-  but the diff in % betewwn the total and the v+) 
        yb = np.array(yb)            # v+
        y_pred = np.array(y_pred)     # v%+ PREDICT WITH SOME VARIATION TO MAIN FEATURES
        #print('newy',newy)
        newDateCol = np.array(newDateCol)        
        # return:  date, %v+, 1 - %v+, v+,  %v+ pred, rating (normal), len (normal), first feature (normal), second feature
        x1,y1, y1a, y1b, y1c, k1,j1,w11,w12  = createUniqueProd(list(zip(newX, newy, ya, yb, y_pred, newDateCol)),IndexRating, IndexRepu1Feature,Index2Repu2eature, IndexLenFeature,  '', 'lowRating', False)
        alpha = y1c
        distrib1 = dirichlet_pdf(y1, alpha)     # y1 =  %v+  , # alpha = y1c = %v+ pred
        print ('Dirichlet prior distrib about  %v+ of each review (alpha = predictions)', distrib1)       # plotting [range of dates] -> distrib1   [range of dates] -> y1  multinomial instances of votes in date = final date of range
        
        alpha = y1c
        distrib2 = dirichlet_pdf(y1b, alpha)     # y1b =  v+  , # alpha = y1c = %v+ pred
        print ('Dirichlet prior distrib about  v+ of each review  (alpha = predictions)', distrib2)       # plotting [range of dates] -> distrib2   [range of dates] -> y1b  multinomial instances of votes in date = final date of range

        alpha = w11
        distrib3 = dirichlet_pdf(y1b, alpha)     # y1b =  v+  , # alpha = Winner feature  -> reputation MEAN H VOTES x reviews) x REVIEWER
        print ('Dirichlet prior distrib about  v+ of each review (alpha = main feature)', distrib3)       # plotting [range of dates] -> distrib2   [range of dates] -> y1b  multinomial instances of votes in date = final date of range


#
# 1. Create an agregate list of products with respect to a list of name products 
# 2. predict_results x+%
# 3. changeValueColumns  (Repu Reviewer)  
# 4. predict_results x+% after increasing value features Repu Reviewer
# 5. changeValueColumns  (Context votes)  
# 6. predict_results x+% after increasing value features Context votes
# 7.         
#        
def part1 (factor, best_model, X0, X,  y, y__, y___, dateCol, IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature, listProdIDs):

        # NOTE: LOW votes less influenced by context , and also by reputation reviewer: the people vote more autonomously
        # return the data matrix X,y,date (syncro values) of the listProdIDs
        # LOW RATING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  THE lEVEL OF RATING IS SELECT BY THE listProdIDs previously chosen
        
        print('{:*^120}'.format(' LIST PROD IDS '))
        print('*',listProdIDs,'*')
        print('{:*^120}'.format(''),'\n')

        colProduct = len(X0[0]) - 1
        # create unique product by a list
        # X, %v+ , 1 - %v+, v+, date
        newX, newy, ya, yb, dateCol = list_groupProdsPlusDate (IT.zip_longest(X0, y, y__, y___),colProduct,listProdIDs,dateCol)
        newX = np.array(newX)    # leave the product column
        newy = np.array(newy)    
        print ('%v+ test',newy,'\n')
        X_Test = newX
        y_Test = newy
        X_Test = np.delete(X_Test,colProduct,axis=1)        #column ProdID deleted
        print ('colProduct',colProduct)
        print ('IndexRating',IndexRating,'rating',X[0][IndexRating])
        print ('IndexWinFeature',IndexWinFeature,'WinFeature',X[0][IndexWinFeature])
        print ('Index2WinFeature',Index2WinFeature,'2WinFeature',X[0][Index2WinFeature])
        print ('IndexLenFeature',IndexLenFeature,'LenFeature',X[0][IndexLenFeature])

        #print('X_Test[0]',X_Test[0])
        X_Test = X_Test.astype(np.float)
        print('X_Test[0]',X_Test[0])
        print('X_Training[0]',X[0])
        
        
        index1 = feature_names_Dimitri.index(' (MEAN H VOTES x reviews) x REVIEWER')
        index2 = feature_names_Dimitri.index('# REVIEWS x REVIEWER')    
        index3 = feature_names_Dimitri.index('MEAN H VOTES x CONTEXT (WIN = 4)')
        index4 = feature_names_Dimitri.index('MEAN H VOTES x CONTEXT (WIN = 2)')    
        print ('value index1: (MEAN H VOTES x reviews) x REVIEWER',X[0][index1])
        print ('value index2: # REVIEWS x REVIEWER',X[0][index2])
        print ('value index3: MEAN H VOTES x CONTEXT (WIN = 4)',X[0][index3])
        print ('value index4: MEAN H VOTES x CONTEXT (WIN = 2)',X[0][index4])

        
        if (factor > 0):     
            # change the column of X in the index of reputation and context
            y_pred0 = predict_results (X_Test, [], X, y, best_model, "")
 
            X_Test = changeValueColumns (X_Test, feature_names_Dimitri, factor, index1, index2 )
            print('factor',factor,X_Test[0])
            y_pred1 = predict_results (X_Test, [], X, y, best_model, "")
        
            X_Test = changeValueColumns (X_Test, feature_names_Dimitri, factor, index3, index4 )
            print('factor *',factor,X_Test[0])
            y_pred2 = predict_results (X_Test, [], X, y, best_model, "")
        
            df = pd.DataFrame({'y real' : y_Test, 'y pred before' : y_pred0, 'y pred after1' : y_pred1, 'y pred after2' : y_pred2 })
            print (df)
        else:
            y_pred2 = newy

        
        #changeValueColumnContext (IT.zip_longest(y_pred,X_Test,dateCol),listProdIDs,feature_names_Dimitri)  TOO COMPLEX!!!
        return newX, newy, ya, yb, y_pred2, dateCol
     
        # from best_model calculate prediction %v+ reviews of a date
        # given a review for prediction , changing the value of main feature (reputation) and predict (pred %v+) settiung changes 
        # and generate ypred' ... calculate the diff, and the (mean, sd) of diff , put this diff in a structure A
        # the same operation changing 5 times the value....
        # 3. step idem but referring to 2.o features and to 3.o features... crete the comparison of the best result
 


#        # taken from prods the below listProdIDs
#        listProdIDs = ['B0046BTK14', 'B0040IO1RQ', 'B004D4917W', 'B004477D0A' , 'B00456V6WG' ] # example LONG PRESENCE 'lowRating' 
#        listProdID2s = ['B0040IUI46' , 'B00427TAIK' 'B0042FZA1S' , 'B00462RMS6', 'B0041RSDXE'] # example LONG PRESENCE 'MediumRating' ' 
#        listProdID3s = ['B0049P6OTI', 'B0041VZN6U', 'B0045EOWLU' , 'B00422KZQG', 'B004D5GXOA'] # example LONG PRESENCE 'HighRating' 

 # X0 = all features plus prodID (use only in statistical methods: es curve, and NOT in Features Learning)
def AggregationProductsResult (best_model, X0, X, y, y__, y___, dateCol, IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature,  presence,typer,listProdIDs, listProdID2s,listProdID3s ): 
 
         # merge into a unique product the group listProdIDs and plot diferent curves
        # Drow a chart (long period) with 3 curves (low, medium, high rating)  x: date y: %votes+
        # QUESTION:     THE VOTES MUST BE INV CORRELATED TO AVERAGE RATING
        # Drow a chart (low rating, short period) with 3 curves x: date y1: %votes+  y2: reviewer reputation y3: context votes
        # Six type of chart like the previous:  QUESTION: are there a correlation among y1, y2, y3 ?  and (AN INVERSE CORRELATION) about y1 (or y2 or y3) and rating?
        
        #factor = 20.0
        factor = 0.0 # if DIMITRI SIMULATION -> factor > 0 otherwose it's = 0
        
        # LOW RATING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x1 = []
        y1 = y1a = y1b = y1c = []
        if (len(listProdIDs) > 0):
            print ('list1',listProdID2s)
            newX, newy, ya, yb, y_pred, newDateCol = part1 (factor, best_model, X0, X,  y, y__, y___, dateCol, IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature, listProdIDs)
            newy = np.array(newy)        # v%+
            ya = np.array(ya)            # 1 - %v+  (Thre arent the %v-  but the diff in % betewwn the total and the v+) 
            yb = np.array(yb)            # v+
            y_pred = np.array(y_pred)     # v%+ PREDICT WITH SOME VARIATION TO MAIN FEATURES
            #print('newy',newy)
            newDateCol = np.array(newDateCol)        
            # return:  date, %v+, 1 - %v+, v+,  %v+ pred, rating (normal), len (normal), first feature (normal), second feature
            x1,y1, y1a, y1b, y1c, k1,j1,w11,w12, nrev1 = createUniqueProd(list(zip(newX, newy, ya, yb, y_pred, newDateCol)),IndexRating,IndexWinFeature, Index2WinFeature, IndexLenFeature, presence, 'low'+typer, False)
        
        # return the data matrix X,y,date (syncro values) of the listProdIDs
        # MEDIUM RATING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x2 = []
        if (len(listProdID2s) > 0):
            print ('list2',listProdID2s)
            newX, newy, ya, yb, y_pred, newDateCol = part1 (factor, best_model, X0, X,  y, y__, y___, dateCol, IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature, listProdID2s)
            newX = np.array(newX)    
            newy = np.array(newy)  
            ya = np.array(ya)      # 1 - %v+  (Thre arent the %v-  but the diff in % betewwn the total and the v+) 
            yb = np.array(yb)      # v+
            y_pred = np.array(y_pred)     # v%+ PREDICT WITH SOME VARIATION TO MAIN FEATURES
            #print('newy',newy)
            newDateCol = np.array(newDateCol)        # 35
            # return:  date, %v+, 1 - %v+, v+,  %v+ pred, rating (normal), len (normal), first feature (normal), second feature
            x2,y2, y2a, y2b, y2c, k2,j2,w21,w22, nrev2 = createUniqueProd(list(zip(newX, newy, ya, yb, y_pred, newDateCol)),IndexRating,IndexWinFeature, Index2WinFeature, IndexLenFeature,  presence, 'medium'+typer, False)
        
        # return the data matrix X,y,date (syncro values) of the listProdIDs
        # HIGH RATING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x3 = []
        if (len(listProdID3s) > 0):
            print ('list3',listProdID3s)
            newX, newy, ya, yb, y_pred, newDateCol = part1 (factor, best_model, X0, X,  y, y__, y___, dateCol, IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature, listProdID3s)
            newX = np.array(newX)    
            newy = np.array(newy)  
            ya = np.array(ya)      # 1 - %v+  (Thre arent the %v-  but the diff in % betewwn the total and the v+) 
            yb = np.array(yb)     # v+
            y_pred = np.array(y_pred)     # v%+ PREDICT WITH SOME VARIATION TO MAIN FEATURES
            #print('newy',newy)
            newDateCol = np.array(newDateCol)  
            # return:  date, %v+, 1 - %v+, v+,  %v+ pred, rating (normal), len (normal), first feature (normal), second feature
            x3,y3, y3a, y3b, y3c, k3,j3,w31,w32, nrev3 = createUniqueProd(list(zip(newX, newy, ya, yb, y_pred, newDateCol)),IndexRating,IndexWinFeature, Index2WinFeature, IndexLenFeature, presence, 'high'+typer, False)
        
        # drawn dotted average rating
        # try to using the same x
        #print ('np.array(x1)',np.array(x1))
        
        
        lista = [x1, x2, x3]   # date
        maxLen =len(max (lista))    # len of higher range of dates of each date array
        
        mean1 = 0
        sd1 = 0
        if (len(y1) > 0):
            mean1 = sum(y1)/float(len(y1))      # %v+... 
            sd1 = stat.stdev(y1)
        mean2 = sum(y2)/float(len(y2))
        sd2 = stat.stdev(y2)
        mean3 = sum(y3)/float(len(y3))
        sd3 = stat.stdev(y3)
        
        mean1a = 0
        if (len(y1a) > 0):
            mean1a = sum(y1a)/float(len(y1a))      # 1 - %v+... 
        mean2a = sum(y2a)/float(len(y2a))
        mean3a = sum(y3a)/float(len(y3a))       
        
        mean1b = 0
        if (len(y1b) > 0):
            mean1b = sum(y1b)/float(len(y1b))      # v+... 
        mean2b = sum(y2b)/float(len(y2b))
        mean3b = sum(y3b)/float(len(y3b))       
        
            
        # %vote+        
        listAverages = ['low'+typer+'  mean %vote+ {0:.3f}'.format(mean1)+' (sd {0:.3f}'.format(sd1)+')',  ', medium'+typer+' mean %vote+  {0:.3f}'.format(mean2)+' (sd {0:.3f}'.format(sd2)+') ', ', high'+typer+' mean %vote+ {0:.3f}'.format(mean3)+' (sd {0:.3f}'.format(sd3)+') ']        
        print ('y1',y1,'y1c',y1c )
        print ('y2',y2,'y2c',y2c )
        print ('y3',y3,'y3c',y3c )
        if (y1 == y1c and y2 == y2c and y3 == y3c):  # without prediction
            if (len(y1) > 0):
                listCouples_xy = [list(zip(np.array(x1),np.array(y1))),list(zip(np.array(x2),np.array(y2))),list(zip(np.array(x3),np.array(y3)))]   
            else : 
                listCouples_xy = [list(zip(np.array(x2),np.array(y2))),list(zip(np.array(x3),np.array(y3)))]                    
        else :                                       # with prediction
            listCouples_xy = [list(zip(np.array(x1),np.array(y1))),list(zip(np.array(x2),np.array(y2))),list(zip(np.array(x3),np.array(y3))),list(zip(np.array(x1),np.array(y1c))),list(zip(np.array(x2),np.array(y2c))),list(zip(np.array(x3),np.array(y3c)))]               
        #plotMultiTrends ( listCouples_xy,  'Longtime '+presence+' (mean %vote+ x month)', maxLen, 0, ['%vote+ low'+typer+'','%vote+ medium'+typer+'','%vote+ high'+typer+'','pred %vote+ low'+typer+'','pred %vote+ medium'+typer+'','pred %vote+ high'+typer+''],'Chart relationship %vote+ Low/Medium/High '+typer+' Product', listAverages, '%v+')
        plotMultiTrends ( listCouples_xy,  ' '+presence+' (mean %vote+ x month)', maxLen, 0, ['%vote+ medium'+typer+'','%vote+ high'+typer+'','pred %vote+ medium'+typer+'','pred %vote+ high'+typer+''],'Chart relationship %vote+ Low/Medium/High '+typer+' Product', listAverages, '%v+', 0)

        # %vote-   could create confusion
#        listAverages2 = ['low'+typer+'  %vote- {0:.3f}'.format(mean1a)+')',  ', mediumRating   %vote- {0:.3f}'.format(mean2a)+')',  ', highRating  %vote- {0:.3f}'.format(mean3a)+') ']        
#        if (len(y1a) > 0):
#            listCouples_xy = [list(zip(np.array(x1),np.array(y1a))),list(zip(np.array(x2),np.array(y2a))),list(zip(np.array(x3),np.array(y3a)))]   
#        else :
#            listCouples_xy = [list(zip(np.array(x2),np.array(y2a))),list(zip(np.array(x3),np.array(y3a)))]               
#        plotMultiTrends ( listCouples_xy, 'Longtime '+presence+' ( %vote-  x month)', maxLen, 0, ['%vote- medium'+typer+'','%vote-  high'+typer+''],'Chart relationship %vote-  Low/Medium/High '+typer+' Product', listAverages2, '%v-', 0)

        #vote+
        listAverages3 = ['low'+typer+'  mean vote+ {0:.3f}'.format(mean1b)+')',  ', mediumRating  mean vote+  {0:.3f}'.format(mean2b)+')',  ', highRating  mean vote+  {0:.3f}'.format(mean3b)+') ']        
        if (len(y1b) > 0):
            listCouples_xy = [list(zip(np.array(x1),np.array(y1b))),list(zip(np.array(x2),np.array(y2b))),list(zip(np.array(x3),np.array(y3b)))]   
        else:
            listCouples_xy = [list(zip(np.array(x2),np.array(y2b))),list(zip(np.array(x3),np.array(y3b)))]               
        plotMultiTrends ( listCouples_xy, ' '+presence+' (mean vote+ x month)', maxLen, 0, ['vote+ medium'+typer+'','vote+ high'+typer+''],'Chart relationship vote+ Low/Medium/High '+typer+' Product', listAverages3,  'v+', 1)

        if (len(y1b) > 0):
            y1d = normalization (percentVotesWeighted (y1, y1b))
            nrev1 = normalization(nrev1)
        y2d = normalization (percentVotesWeighted (y2, y2b))
        nrev2 = normalization(nrev2)
        y3d = normalization (percentVotesWeighted (y3, y3b))
        nrev3 = normalization(nrev3)
        # IN THE plotMultiTrends AT THE BOTTOM, ADDING ALSO nrev2 nrev3 IN ORDER TO HAVE AN IDEA IF SOME MINIMUM
        # IS CAUSED BY 0 REVIEWS OR MINIMUM

        # %vote+ * vote+
        listAverages3 = ['low'+typer+'  mean (vote+) (normal max) {0:.3f}'.format(mean1b)+')',  ', mediumRating mean vote+ (normal max) {0:.3f}'.format(mean2b)+')',  ', highRating  mean vote+ (normal max)   {0:.3f}'.format(mean3b)+') ']        
        if (len(y1b) > 0):
            listCouples_xy = [list(zip(np.array(x1),np.array(y1d))),list(zip(np.array(x2),np.array(y2d))),list(zip(np.array(x3),np.array(y3d)))]   
        else:
            listCouples_xy = [list(zip(np.array(x2),np.array(y2d))),list(zip(np.array(x3),np.array(y3d)))]               
        plotMultiTrends ( listCouples_xy, ' '+presence+' (mean vote+ (normal max) x month)', maxLen, 0, ['(vote+) normal medium'+typer+'','vote+ normal high'+typer+''],'Chart relationship (vote+) per vote%+ Low/Medium/High '+typer+' Product', listAverages3,  'v+ normal', 0)

        if (len(y1) > 0):
                corrVotes (' LOW '+typer+'', y1, y1b)
        corrVotes (' MEDIUM '+typer+'', y2, y2b)
        corrVotes (' HIGH '+typer+'', y3, y3b)
        
        # feature!!!!!  LOW RATING
        if (len(y1) > 0):
            listCouples_xyA = [list(zip(np.array(x1),np.array(y1))),list(zip(np.array(x1),np.array(w11))),list(zip(np.array(x1),np.array(w12))),list(zip(np.array(x1),np.array(j1)))]   # date, vote, rating
            listCorr = [corr ('  corr (context vote+, repu reviewer)=',w12, w11),corr ('  corr (len rev, repu reviewer)=',j1, w11),corr ('  corr (%vote+, repu reviewer)=',y1, w11), corr (' corr (%vote+,context vote+)=',y1, w12), corr (' corr (%vote+, len review)=',y1, j1)]
            plotMultiTrends ( listCouples_xyA, ' '+presence+' low'+typer+'', maxLen, 2, ['%vote+ low'+typer+'','reviewer repu low'+typer+'','context vote+ low'+typer+'','lenRev low'+typer+''],'Chart low'+typer+' ( %vote+, Repu reviewer, Context vote, Len reviews)', listCorr, '%', 0)

        # feature!!!!!  MEDIUM RATING       for Dimitri substitute y2 with y2d 
        listCouples_xyB = [list(zip(np.array(x2),np.array(y2))),list(zip(np.array(x2),np.array(w21))),list(zip(np.array(x2),np.array(w22))),list(zip(np.array(x2),np.array(j2)))]   # date, vote, rating
        listCorr = [corr ('  corr (context vote+, repu reviewer)=',w22, w21),corr ('  corr (len rev, repu reviewer)=',j2, w21),corr ('  corr (%vote+, repu reviewer)=',y2, w21), corr (' corr (%vote+,  context vote+)=',y2, w22), corr (' corr (%vote+, len review)=',y2, j2)]
        plotMultiTrends ( listCouples_xyB, ' '+presence+' %v+ vs features medium'+typer+'', maxLen, 6, ['%vote+ medium'+typer+'','reviewer repu medium'+typer+'','context vote+ medium'+typer+'','lenRev medium'+typer+''],'Chart medium'+typer+' ( %vote+, Repu reviewer, Context vote, Len reviews)', listCorr, '%', 0)

        # feature!!!!!  HIGH RATING
        listCouples_xyC = [list(zip(np.array(x3),np.array(y3))),list(zip(np.array(x3),np.array(w31))),list(zip(np.array(x3),np.array(w32))),list(zip(np.array(x3),np.array(j3)))]   # date, vote, rating
        listCorr = [corr ('  corr (context vote+, repu reviewer)=',w32, w31),corr ('  corr (len rev, repu reviewer)=',j3, w31),corr ('  corr (%vote+, repu reviewer)=',y3, w31),  corr (' corr (%vote+, context vote+)=',y3, w32), corr (' corr (%vote+, len review)=',y3, j3)]
        plotMultiTrends ( listCouples_xyC, ' '+presence+' %v+ vs features high'+typer+'', maxLen, 10, ['%vote+ high'+typer+'','reviewer repu high'+typer+'','context vote+ high'+typer+'','lenRev high'+typer+''],'Chart high'+typer+' ( %vote+, Repu reviewer, Context vote, Len reviews)', listCorr, '%', 0)
         
        # Drow a chart (long period) with 3 curves (low, medium, high rating)  x: date y: %votes+
        # QUESTION:     THE VOTES MUST BE INV CORRELATED TO AVERAGE RATING
        # Drow a chart (low rating, short period) with 3 curves x: date y1: %votes+  y2: reviewer reputation y3: context votes
        # Six type of chart like the previous:  QUESTION: are there a correlation among y1, y2, y3 ?  and (AN INVERSE CORRELATION) about y1 (or y2 or y3) and rating?
        

def corr (text, ya, yb):
        #yb_T = np.array(yb.transpose())
        pearson_1 = scstat.pearsonr(ya,yb)
        print (text,pearson_1[0])
        return text + str('{0:.3f}'.format(pearson_1[0]))

        #  calculate y1d = y1 * y1b, y2d, y3d
def percentVotesWeighted (y1, y1b):
        y = []
        for i in range(0,len(y1)):
            y.append(y1[i]*y1b[i])
        y = np.array(y) 
        return y


# Map Original Features to reduced (without sentiment and pther informations)
        
def reduceMatrix (X0, X, listOriginalFeatures, listReducesFeauture):    
    newX = []
    newX0 = []
    listIndex = []
    #Z = IT.zip_longest(listOriginalFeatures,listReducesFeauture)
    for item in listReducesFeauture:
        print ('item in listReducesFeauture',item)
        for i in [i for i,x in enumerate(listOriginalFeatures) if x == item]:
            listIndex.append(i)
    print ('listIndex',listIndex)
    X = np.array(X)
    for i in range(0,len(X)): 
       x = X[i]
       row = []        
       for j in listIndex:
#            print ('j',j,'x[j]',x[j])
            row.append(x[j])
       row = np.array(row)
 #      print ('row',row)
       #newX = np.vstack([newX, row])
       newX.append(row)
    newX = np.array(newX)
    for i in range(0,len(newX)): 
       prod = X0[i][34]
       x = newX[i]
       x = x.tolist()
       x.append(prod)
       newX0.append(x)    
    newX0 = np.array(newX0)
    return newX0, newX

def changeValueColumns (X, listReducesFeauture, factor, index1, index2):  

    colAfter1 = []
    colPost1 = []
    colAfter2 = []
    colPost2 = []
    newX = X    
    for i in range(0,len(X)):
        colAfter1.append(X[i][index1])
        colAfter2.append(X[i][index2])
        newX[i][index1] = X[i][index1] + (factor*float(X[i][index1]))/float(100)   
        newX[i][index2] = X[i][index2] + (factor*float(X[i][index2]))/float(100) 
        colPost1.append(X[i][index1])
        colPost2.append(X[i][index2])
    
#    df1 = pd.DataFrame({'colAfter1' : np.array(colAfter1), 'colPost1' : np.array(colPost1) })
#    print (df1)
#    df2 = pd.DataFrame({'colAfter2' : np.array(colAfter2), 'colPost2' : np.array(colPost2) })
#    print (df2)
    return newX

#def changeValueColumnContext (Z,listProdIDs,listReducesFeauture):  # Z = IT.zip_longest(y_pred,X_Test,dateCol,prodCol)
#       
#    index1 = listReducesFeauture.index('MEAN H VOTES x CONTEXT (WIN = 2)')    
#    index2 = listReducesFeauture.index('MEAN H VOTES x CONTEXT (WIN = 4)')
#    prodCol = Z[3]
#    for p in listProdIDs:
#        for z in Z:
#            newx = z[1]
#            newx [index1]
#    
#    
#

# X IS THE VECTOR OF DISTRIBUTION OF VOTES, ALPHAS ARE THE IPER_PARAMS ABOUT THE DISTRIBUTION QITH THE SAME VECTOR LEN    
def dirichlet_pdf(x, alpha): 
   return (math.gamma(sum(alpha)) / 
   reduce(operator.mul, [math.gamma(a) for a in alpha]) * 
   reduce(operator.mul, [x[i]**(alpha[i]-1.0) for i in range(len(alpha))]))
   
def dirichlet_inductive(x, alpha): 
    return 0

    
def corrVotes (text, na, nb):    
   # correlazione v+% e v+ in general, and when v+ > 1,2,3,4,5.., 10.
        Z = zip(na, nb)
        values = [1,2,3,4,5,10]
        u1 = []
        u2 = []
        u3 = []
        for i in values:
            for z in Z:
                if (z[1] >= i):
                    u1.append(z[0])  #v+%
                    u2.append(z[1])  #v+
            u1 = np.array(u1)        
            u2 = np.array(u2)   
            u3 = normalization(u2)  # normalize -> [0,1] v+
            print ('\nvote+ each item > ',i,' result =',u2,'\nresult normal =',u3,'\n')
            print (' when vote+ >',i)
            corr (text+'corr (%vote+, vote+)= ',u1, u2)
            corr (text+'corr (%vote+, normal(vote+))=, ',u1, u3)



feature_names_senti = ['RATING','SAMPLE ENT MAX1','RATIO USEFUL / ALL','1/2 RATIO USEFUL / ALL','DENS POS:NVJ','# REVIEWS x PROD','# REVIEWS x REVIEWER', '(MEAN %H VOTES x reviews) x REVIEWER', 
             ' (MEAN H VOTES x reviews) x REVIEWER', 'CURRENT LIFETIME PROD (month)', 'FULL LIFETIME PROD  (month)', '# H VOTES/# T VOTES x PROD', 'MEAN H VOTES x CONTEXT (WIN = 2)', 
             'COEFF VARIAT H VOTES x CONTEXT (WIN = 2)', 'DIFF H (CURRENT VOTE' 'MEAN VOTES) (WIN = 2)', 'MEAN H VOTES x CONTEXT (WIN = 2 LEFT)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 2 LEFT)', 
             'DIFF (CURRENT VOTE  MEAN VOTES) (WIN = 2 LEFT)', 'MEAN H VOTES x CONTEXT (WIN = 4)', 
             'COEFF VARIAT H VOTES x CONTEXT (WIN = 4)',' DIFF (CURRENT VOTE MEAN VOTES) (WIN = 4)', 'MEAN H VOTES x CONTEXT (WIN = 4 LEFT)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 4 LEFT)', 
             'DIFF (CURRENT VOTE MEAN VOTES) (WIN = 4 LEFT)', 'DENS. COMPOUND WORDS',	'LEN TITLE/LEN TEXT',	'SENT: POSNEGSUM',	
             'SENT: NUMPOS/ALL','SENT: NUMNEG/ALL',	'SENT: NUMCONTINUOUSPOS',	'SENT: NUMCONTINUOUSNEG',	'SENT 1/2: POSNEGSUM' ,	
             'SENT 1/2: NUMPOS/ALL' ,'SENT 1/2: NUMNEG/ALL' ,'LENGTH REVIEW','LENGTH NORMAL REVIEW', 'POSITIVE WORDS', 'NEGATIVE WORDS' ,'ARI']

feature_names_sentiNoBiasReviewer = ['RATING','SAMPLE ENT MAX1','RATIO USEFUL / ALL','1/2 RATIO USEFUL / ALL','DENS POS:NVJ','# REVIEWS x PROD','# REVIEWS x REVIEWER',  
             ' (MEAN H VOTES x reviews) x REVIEWER', 'CURRENT LIFETIME PROD (month)', 'FULL LIFETIME PROD  (month)', '# H VOTES/# T VOTES x PROD', 'MEAN H VOTES x CONTEXT (WIN = 2)', 
             'COEFF VARIAT H VOTES x CONTEXT (WIN = 2)', 'DIFF H (CURRENT VOTE' 'MEAN VOTES) (WIN = 2)', 'MEAN H VOTES x CONTEXT (WIN = 2 LEFT)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 2 LEFT)', 
             'DIFF (CURRENT VOTE  MEAN VOTES) (WIN = 2 LEFT)', 'MEAN H VOTES x CONTEXT (WIN = 4)', 
             'COEFF VARIAT H VOTES x CONTEXT (WIN = 4)',' DIFF (CURRENT VOTE MEAN VOTES) (WIN = 4)', 'MEAN H VOTES x CONTEXT (WIN = 4 LEFT)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 4 LEFT)', 
             'DIFF (CURRENT VOTE MEAN VOTES) (WIN = 4 LEFT)', 'DENS. COMPOUND WORDS',	'LEN TITLE/LEN TEXT',	'SENT: POSNEGSUM',	
             'SENT: NUMPOS/ALL','SENT: NUMNEG/ALL',	'SENT: NUMCONTINUOUSPOS',	'SENT: NUMCONTINUOUSNEG',	'SENT 1/2: POSNEGSUM' ,	
             'SENT 1/2: NUMPOS/ALL' ,'SENT 1/2: NUMNEG/ALL' ,'LENGTH REVIEW','LENGTH NORMAL REVIEW', 'POSITIVE WORDS', 'NEGATIVE WORDS' ,'ARI']

feature_names_Dimitri = ['RATING','SAMPLE ENT MAX1','RATIO USEFUL / ALL','1/2 RATIO USEFUL / ALL','DENS POS:NVJ','# REVIEWS x PROD','# REVIEWS x REVIEWER',  
             ' (MEAN H VOTES x reviews) x REVIEWER', 'CURRENT LIFETIME PROD (month)', 'FULL LIFETIME PROD  (month)', '# H VOTES/# T VOTES x PROD', 
             'MEAN H VOTES x CONTEXT (WIN = 2)', 'MEAN H VOTES x CONTEXT (WIN = 4)', 
             'DENS. COMPOUND WORDS',	'LEN TITLE/LEN TEXT',	'SENT: POSNEGSUM',	
             'SENT: NUMPOS/ALL','SENT: NUMNEG/ALL',	'SENT: NUMCONTINUOUSPOS',	'SENT: NUMCONTINUOUSNEG',	'SENT 1/2: POSNEGSUM' ,	
             'SENT 1/2: NUMPOS/ALL' ,'SENT 1/2: NUMNEG/ALL' ,'LENGTH REVIEW','LENGTH NORMAL REVIEW', 'POSITIVE WORDS', 'NEGATIVE WORDS' ,'ARI']

feature_names_Dimitri_Chinese = ['RATING','SAMPLE ENT MAX1','RATIO USEFUL / ALL','1/2 RATIO USEFUL / ALL','DENS POS:NVJ','# REVIEWS x PROD','# REVIEWS x REVIEWER',  
             ' (MEAN H VOTES x reviews) x REVIEWER', 'CURRENT LIFETIME PROD (month)', 'FULL LIFETIME PROD  (month)', '# H VOTES/# T VOTES x PROD', 
             'DENS. COMPOUND WORDS',	'LEN TITLE/LEN TEXT',	'SENT: POSNEGSUM',	
             'SENT: NUMPOS/ALL','SENT: NUMNEG/ALL',	'SENT: NUMCONTINUOUSPOS',	'SENT: NUMCONTINUOUSNEG',	'SENT 1/2: POSNEGSUM' ,	
             'SENT 1/2: NUMPOS/ALL' ,'SENT 1/2: NUMNEG/ALL' ,'LENGTH REVIEW','LENGTH NORMAL REVIEW', 'POSITIVE WORDS', 'NEGATIVE WORDS' ,'ARI']

feature_names_sentiNoReviewer = ['RATING','SAMPLE ENT MAX1','RATIO USEFUL / ALL','1/2 RATIO USEFUL / ALL','DENS POS:NVJ','# REVIEWS x PROD', 
             'CURRENT LIFETIME PROD (month)', 'FULL LIFETIME PROD  (month)', '# H VOTES/# T VOTES x PROD', 'MEAN H VOTES x CONTEXT (WIN = 2)', 
             'COEFF VARIAT H VOTES x CONTEXT (WIN = 2)', 'DIFF H (CURRENT VOTE' 'MEAN VOTES) (WIN = 2)', 'MEAN H VOTES x CONTEXT (WIN = 2 LEFT)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 2 LEFT)', 
             'DIFF (CURRENT VOTE  MEAN VOTES) (WIN = 2 LEFT)', 'MEAN H VOTES x CONTEXT (WIN = 4)', 
             'COEFF VARIAT H VOTES x CONTEXT (WIN = 4)',' DIFF (CURRENT VOTE MEAN VOTES) (WIN = 4)', 'MEAN H VOTES x CONTEXT (WIN = 4 LEFT)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 4 LEFT)', 
             'DIFF (CURRENT VOTE MEAN VOTES) (WIN = 4 LEFT)', 'DENS. COMPOUND WORDS',	'LEN TITLE/LEN TEXT',	'SENT: POSNEGSUM',	
             'SENT: NUMPOS/ALL','SENT: NUMNEG/ALL',	'LENGTH REVIEW','LENGTH NORMAL REVIEW', 'POSITIVE WORDS', 'NEGATIVE WORDS' ,'ARI']

feature_names = ['RATING','SAMPLE ENT MAX1','RATIO USEFUL / ALL','1/2 RATIO USEFUL / ALL','DENS POS:NVJ','# REVIEWS x PROD','# REVIEWS x REVIEWER', '(% H VOTES x reviews) x REVIEWER', 
             ' (MEAN H VOTES x reviews) x REVIEWER', 'CURRENT LIFETIME PROD (month)', 'FULL LIFETIME PROD  (month)', '# H VOTES/# T VOTES x PROD', 'MEAN H VOTES x CONTEXT (WIN = 2)', 
             'COEFF VARIAT H VOTES x CONTEXT (WIN = 2)', 'DIFF H (CURRENT VOTE' 'MEAN VOTES) (WIN = 2)', 'MEAN H VOTES x CONTEXT (WIN = 2 LEFT)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 2 LEFT)', 
             'DIFF (CURRENT VOTE MEAN VOTES) (WIN = 2 LEFT)', 'MEAN H VOTES x CONTEXT (WIN = 4)', 
             'COEFF VARIAT H VOTES x CONTEXT (WIN = 4)',' DIFF (CURRENT VOTE MEAN VOTES) (WIN = 4)', 'MEAN H VOTES x CONTEXT (WIN = 4 LEFT)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 4 LEFT)', 
             'DIFF (CURRENT VOTE  MEAN VOTES) (WIN = 4 LEFT)', 'DENS. COMPOUND WORDS',	'LEN TITLE/LEN TEXT','LENGTH REVIEW','LENGTH NORMAL REVIEW', 'POSITIVE WORDS', 'NEGATIVE WORDS' ,'ARI']



def main(args) : 
    
    # PROVE  INPUT: 2 --opt_arg 3 --predict
#    print(args.opt_pos_arg)
#    print(args.opt_arg)
#    print(args.switch)
    
    #exit
    
    choice = args.input;
    print("Argument values:", choice )   

    if choice == 'general_test':    # INPUT:  + SENTIMENT
        X, y = DataGenTest (1) 
        simpleOLS(X,y,feature_names_senti)
         
        # TEST GENERAL
        X, y = DataGenTest (1) 
        dim = X.shape[0]
        print("dim:", dim)   
        best_model_ = core(X,y,'general_test',0.03,feature_names_senti)
        modelName = best_model_.best_estimator_.steps[1][1].__class__.__name__
        print('modelName Chosen: '+modelName)     
        coreFeatureExtraction(X, y, 'general_test'+' - Best features',modelName,3,feature_names_senti)      # chose : 1,2 or 3
        All_pcc(X,y,feature_names_senti)  
 
        # TEST GENERAL WITHOUT REVIEWER FEATURES
        X, y, X1 = DataGenTest (2) 
        dim = X1.shape[0]
        print("dim:", dim)   
        #newTrainX, newTrainy = list_HighRating  (IT.zip_longest(X1, y), 0) 
        best_model_ = core(X1, y,'general_test ',0.03,feature_names_sentiNoReviewer)        
        
        
        # TEST GENERAL WITHOUT REVIEWER %Hvotes
        X, y, X1 = DataGenTest (6) 
        dim = X1.shape[0]
        print("dim:", dim)   
        #newTrainX, newTrainy = list_HighRating  (IT.zip_longest(X1, y), 0) 
        best_model_ = core(X1, y,'general_test without reviewer %Hvotes',0.03,feature_names_sentiNoBiasReviewer)     
        
        modelName = best_model_.best_estimator_.steps[1][1].__class__.__name__
        print('modelName Chosen: '+modelName)     
        coreFeatureExtraction(X1, y, 'general_test'+' - Best features',modelName,1,feature_names_sentiNoReviewer)      # chose : 1,2 or 3
  
        newTrainX, newTrainy = list_sameRating (IT.zip_longest(X, y), 0, 5)     # SAME RATING
        All_pcc(newTrainX, newTrainy,feature_names_senti)
        best_model = core(newTrainX, newTrainy, 'test rating (1) ',0.03,feature_names_senti)       
        modelName = best_model.best_estimator_.steps[1][1].__class__.__name__
        print('modelName Chosen: ',modelName)     
        coreFeatureExtraction(newTrainX, newTrainy,'test rating (1)  '+' - Best features',modelName,2,feature_names_senti)      # chose : 1,2 or 3
        
        
#        FOR THE TEST OF ECONOMICS EXPERIMENT (CHESSA, ANNA)
#        DI OGNI CATEGORIA (MEDIA E ALTA, PERCHE’ LA BASSA E’ DIFF DA DIFFERENZIARE), PRENDERE UN GRUPPO DI 5 PROD SIMILI E DELLO STESSO PREZZO, E UN GRUPPO CON ALTA VARIANZA E VEDERE COSA SUCCEDE NEL GRAFICO V%+ E NEL GRAFICO V+. Credo impossibile trovare , dato un certo set di qualità fissata, gruppi di prodotti di prezzo basso/medio/alto perché si confronterebbero mele con pere. Qs dataset che uso sono orientati all’informazione e non ai prodotti o prezzi, che sono molto miscelati tra loro e non facilmente individuabili (la categ elettronica è molto di alto livello, quindi con una distribuzione elevatissima di prodotti, inoltre contiene  prodotti, accessori e case tutti insieme)  
#        METTERE UN SWITCHING PER TENERE LE CURVE CON O SENZA TRATTEGGIO (OSSIA ESCLUDERE LE PREDICTION AL VARIARE DELLE FEATURES).


######################################################################## EXPERIMENTS DIMITRI ##############################################################

        X0, X, y, y__, y___, date = DataGenTest (4) 
        # best_model_ = core(X1, y,'general_test without reviewer %Hvotes',0.03,feature_names_sentiNoBiasReviewer)         
        ## SEE HERE A WAY TO IMPROVE THE BEST FEATURE SELECTION IN RANDOM FOREST
        ## https://medium.com/turo-engineering/how-not-to-use-random-forest-265a19a68576

        X0, X = reduceMatrix (X0, X, feature_names_senti, feature_names_Dimitri)
        X = np.array(X)
        X0 = np.array(X0)
        print('X[0]',X[0],'\nX0[0]',X0[0])
        print('X0',X0)   
        best_model = None
        best_model = core(X, y, ' General test with reduct columns feature_names_Dimitri',0.03,feature_names_Dimitri)   # <---------------- FORCED! BECAUSE OTHERWISE IT'S NOT POSSIBLE SELECT FEATURES IN SIMPLE WAY


 ######################## EXPERIMENTS ANNA STEP1 (UNA TANTUM): COLLECT GROUP OF PRODs FOR RATING (quality) ###########################################################
 ##############  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
 
# "01/01/2009"
# 'short', 'mediumRating',  lowRating, highRating
 
        # Return (product, rating) of group of prod which match with criteria: 
        #  (long or short period) AND (low, medium, high rating)
        prods, ratings = list_groupProdsCriteria (IT.zip_longest(X0, y),28, 0, 9, 'long', 'highRating', date, "01/01/2006" )  #dd/mm/yyyy  "01/01/2009"  long time > 2.5 y, "01/01/2012"  0 < Short time < 2.5 , 
        prods = np.array(prods)    
        ratings = np.array(ratings)    
        print(prods)  # LIST OF prodS  (which satisfy above CRITERIA)
        print(ratings)  # list OF RATING (which satisfy above CRITERIA)
        print('prod shape',prods.shape)
        print('rating shape',ratings.shape)
        
        
        # matrix A = [product, rating, presence of platform, date reviews, %hvotes]
        # extract 5 high quality and short time [2010-2014] -> matrix A1, same len
        # grouping into a unique prod , where y = (mean) reviews %hv  or y = (sd) reviews %hv  and date = range of date [01-month-yy]    see plotting_trends (y_, date, 'prova1')
 
    
    ######### EXPERIMENTS ANNA STEP2: LAUNCH DIOFFERENT GROUPS FOR RATING AND PRICE ###########################################################


        X0, X, y, y__, y___, date = DataGenTest (4)     
        X0, X = reduceMatrix (X0, X, feature_names_senti, feature_names_Dimitri)
        X = np.array(X)
        X0 = np.array(X0)
        print('X[0]',X[0],'\nX0[0]',X0[0])
        print('X0',X0)   

        best_model = None

        # taken from prods the below listProdIDs
        #GROUP A WITH PRICE LEVEL = High  price
        listProdID2s = ['B004GK0GKO' , 'B0041RSPR8', 'B004G8QO5C' , 'B004FLL5AY', 'B004FLL5AY']       # example LONG PRESENCE 'Low Rating' ' High  pricing' ' 505, 450, 270, 660, 660
        listProdID3s = ['B004BQKQ8A', 'B0040JHVC2', 'B003ZSHNEA' , 'B003ZYF3LO', 'B0041RSPRS']      # example LONG PRESENCE 'High Rating' ' High  pricing' ' 660  540  850 440 600
        # IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature
        price = 'High price '
        AggregationProductsResult (best_model, X0, X, y, y__, y___, date, 0, 7, 12, 23, price, 'Quality', [], listProdID2s,listProdID3s)

        # taken from prods the below listProdIDs
        #GROUP A WITH PRICE LEVEL = High-Medium  price
        listProdID2s = ['B0042SDDXM' , 'B0046BTK14', 'B00428N9OK' , 'B003ZX8AZQ', 'B0045371FU'] # example LONG PRESENCE 'Low Rating' ' High-Medium  pricing' ' 179, 174, 200, 170, 150
        listProdID3s = ['B004G8QZPG', 'B0041OSQ9I', 'B004EFUOY4' , 'B00429N160', 'B003ZSHNG8'] # example LONG PRESENCE 'High Rating' ' High-Medium  pricing' ' 160, 175 , 128, 159, 120
        # IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature
        price = 'High-Medium price '
        AggregationProductsResult (best_model, X0, X, y, y__, y___, date, 0, 7, 12, 23, price, 'Quality', [], listProdID2s,listProdID3s)
        
        
        # taken from prods the below listProdIDs
        #GROUP A WITH PRICE LEVEL = Low-Medium  price
        listProdID2s = ['B0041LYY6K' , 'B00426C55G', 'B00426C56U' , 'B004E10KFG', 'B0042X8NT6'] # example LONG PRESENCE 'Low Rating' ' Low-Medium  pricing' '42, 87,59,79, 67
        listProdID3s = ['B004ABO7QI', 'B0044DEDC0', 'B0041OUA38' , 'B00434UCDE', 'B004071ZXA'] # example LONG PRESENCE 'High Rating' ' Low-Medium  pricing' '77, 39, 40, 90, 35 
        # IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature
        price = 'Low-Medium price '
        AggregationProductsResult (best_model, X0, X, y, y__, y___, date, 0, 7, 12, 23, price, 'Quality', [], listProdID2s,listProdID3s)


        # taken from prods the below listProdIDs
        #GROUP A WITH PRICE LEVEL = Low price
        listProdID2s = ['B0042BUXG4' , 'B0040IO1RQ' 'B0041NFIBS' , 'B0043EV20Q', 'B004BFZHO4'] # example LONG PRESENCE 'Low Rating' 'Low pricing' '10, 13,13,25, 25
        listProdID3s = ['B0043T7FXE', 'B003ZSP0WW', 'B0041Q38NU' , 'B0049P6OTI', 'B0041Q38N0'] # example LONG PRESENCE 'High Rating' 'Low pricing' '28, 27, 10, 25, 15 
        # IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature
        price = 'Low price '
        AggregationProductsResult (best_model, X0, X, y, y__, y___, date, 0, 7, 12, 23, price, 'Quality', [], listProdID2s,listProdID3s)

        
  ######## NEW EXPERIMENTS ANNA STEP2: LAUNCH DIOFFERENT GROUPS FOR RATING AND PRICE ###########################################################


        
        # taken from prods the below listProdIDs
        # ANCHE CON TRE PROD VA BENE !!!!!!!!!!!!!!!!!!!!!
        # PREPARARE UN PO' DI SERIE PER MICHELA (5 SERIE DA TRE PER OGNI CLASSE DI PREZZO)
        #GROUP A WITH PRICE LEVEL = Low price
        listProdID2s = ['B0041NFIBS' , 'B0043EV20Q', 'B004BFZHO4'] # example LONG PRESENCE 'Low Rating' 'Low pricing' '10, 13,13,
        listProdID3s = ['B0043T7FXE',  'B003ZSP0WW', 'B0041Q38N0'] # example LONG PRESENCE 'High Rating' 'Low pricing' '28, 27, 10, 
        # IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature
        price = 'low price lr17 hr20 '
        AggregationProductsResult (best_model, X0, X, y, y__, y___, date, 0, 7, 12, 23, price, 'Quality', [], listProdID2s,listProdID3s)

       #GROUP A WITH PRICE LEVEL = Low-Medium  price
        listProdID2s = ['B0041LYY6K' , 'B00426C55G', 'B00426C56U'] # example LONG PRESENCE 'Low Rating' ' Low-Medium  pricing' '42, 87,59
        listProdID3s = ['B004ABO7QI', 'B0044DEDC0', 'B0041OUA38'] # example LONG PRESENCE 'High Rating' ' Low-Medium  pricing' '77, 39, 40+
        # IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature
        price = 'Low-Medium price  lr65 hr70 '
        AggregationProductsResult (best_model, X0, X, y, y__, y___, date, 0, 7, 12, 23, price, 'Quality', [], listProdID2s,listProdID3s)

       # taken from prods the below listProdIDs
        #GROUP A WITH PRICE LEVEL = High-Medium  price
        listProdID2s = ['B0042SDDXM' , 'B0046BTK14', 'B0045371FU'] # example LONG PRESENCE 'Low Rating' ' High-Medium  pricing' ' 179, 174, 150, 
        listProdID3s = ['B004G8QZPG', 'B0041OSQ9I', 'B004EFUOY4'] # example LONG PRESENCE 'High Rating' ' High-Medium  pricing' ' 160, 175 , 128
        # IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature
        price = 'High-Medium price lr165 hr150 '
        AggregationProductsResult (best_model, X0, X, y, y__, y___, date, 0, 7, 12, 23, price, 'Quality', [], listProdID2s,listProdID3s)
        
        # taken from prods the below listProdIDs
        #GROUP A WITH PRICE LEVEL = High  price
        listProdID2s = ['B004GK0GKO' , 'B0041RSPR8', 'B004FLL5AY']       # example LONG PRESENCE 'Low Rating' ' High  pricing' ' 505, 450, 660
        listProdID3s = ['B004BQKQ8A', 'B0040JHVC2', 'B003ZSHNEA']      # example LONG PRESENCE 'High Rating' ' High  pricing' ' 660  540  850 
        # IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature
        price = 'High price lr530 hr630  '
        AggregationProductsResult (best_model, X0, X, y, y__, y___, date, 0, 7, 12, 23, price, 'Quality', [], listProdID2s,listProdID3s)
    

####################################################################################################################################################


        
        # taken from prods the below listProdIDs
        #GROUP A WITH PRICE LEVEL = Low price
        listProdIDs = ['B0042BUXG4', 'B0040IO1RQ', 'B004HBK4T0', 'B004BJLXAM' , 'B0043862N4' ] # example LONG PRESENCE 'lowRating' 'Low pricing' '6, 5, 10, 8, 7'
        listProdID2s = ['B0041D0K1Q' , 'B00426C57O' 'B0041NFIBS' , 'B0043EV20Q', 'B004BFZHO4'] # example LONG PRESENCE 'MediumRating' 'Low pricing' '10, 13,13,25, 25
        listProdID3s = ['B0043T7FXE', 'B003ZSP0WW', 'B0041Q38NU' , 'B0049P6OTI', 'B0041Q38N0'] # example LONG PRESENCE 'HighRating' 'Low pricing' '28, 27, 10, 25, 15 
        # IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature
        price = 'low price '
        AggregationProductsResult (best_model, X0, X, y, y__, y___, date, 0, 7, 12, 23, price, 'Quality', listProdIDs, listProdID2s,listProdID3s)

        # taken from prods the below listProdIDs
        #GROUP B WITH PRICE LEVEL!!!!!! = High price
        listProdIDs = ['B0042SDDXM', 'B00429N16A', 'B0046BTK14', 'B00428N9OK' , 'B0040QE98O' ] # example LONG PRESENCE 'lowRating' 'high pricing' '180, 300, 174, 200, 150'
        listProdID2s = ['B004FLJVXM' , 'B003ZX8AZQ', 'B004GK0GKO' , 'B0041OSAZ8', 'B0040702HA '] # example LONG PRESENCE 'MediumRating' 'high pricing' '120, 170,180,127, 230
        listProdID3s = ['B003ZSHNGS', 'B0040JHVC2', 'B0042X9LC4' , 'B003ZSHNEA', 'B003ZYF3LO '] # example LONG PRESENCE 'HighRating' 'high pricing' '126, 540, 890 , 850, 440
        # IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature
        price = 'high price '
        AggregationProductsResult (best_model, X0, X, y, y__, y___, date, 0, 7, 12, 23, price, 'Quality', listProdIDs, listProdID2s,listProdID3s)

       # taken from prods the below listProdIDs
        #GROUP C WITHOUT QUALITY!!!!!! = Different prices
        listProdIDs = ['B00429N16A', 'B0040QE98O', 'B0040JHVC2', 'B0040702HA' , 'B003ZSHNEA' ]          # High price
        listProdID2s = ['B004FLJVXM' , 'B0040IO1RQ', 'B003ZSHNGS' , 'B004BFZHO4', 'B0041OSAZ8 ']        # Mixed price
        listProdID3s = ['B0049P6OTI', 'B0041D0K1Q', 'B0043EV20Q' , 'B0043862N4', 'B0041NFIBS ']         # Low price
        # IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature
        price = 'high-mixed-low price '
        AggregationProductsResult (best_model, X0, X, y, y__, y___, date, 0, 7, 12, 23, price, 'Price',listProdIDs, listProdID2s,listProdID3s)


####################################################################################################################################################


        # taken from prods the below listProdIDs
        #GROUP 1   WITHOUT PRICE DISTINTION
        listProdIDs = ['B0046BTK14', 'B0040IO1RQ', 'B004D4917W', 'B004477D0A' , 'B00456V6WG' ] # example LONG PRESENCE 'lowRating' 
        listProdID2s = ['B0040IUI46' , 'B00427TAIK' 'B0042FZA1S' , 'B00462RMS6', 'B0041RSDXE'] # example LONG PRESENCE 'MediumRating' ' 
        listProdID3s = ['B0049P6OTI', 'B0041VZN6U', 'B0045EOWLU' , 'B00422KZQG', 'B004D5GXOA'] # example LONG PRESENCE 'HighRating' 
        # IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature
        AggregationProductsResult (best_model, X0, X, y, y__, y___, date, 0, 7, 12, 23, 'mixed (0) prices', 'Quality', listProdIDs, listProdID2s,listProdID3s)


        #GROUP 2   WITHOUT PRICE DISTINTION
        listProdIDs = ['B004HBK4T0', 'B004BJLXAM', 'B004D4917W', 'B00413PEZS' , 'B00456V6WG' ] # example LONG PRESENCE 'lowRating' 
        listProdID2s = ['B0041D0K1Q' , 'B00427TAIK' 'B0041OSAZ8' , 'B00462RMS6', 'B0040702HA'] # example LONG PRESENCE 'MediumRating' ' 
        listProdID3s = ['B0045KGZOG', 'B0041Q38N0', 'B00483WRZ6' , 'B004EBUXHQ', 'B004ALXSTU'] # example LONG PRESENCE 'HighRating' 
        AggregationProductsResult (best_model, X0, X, y, y__, y___, date,  0, 7, 12, 23, 'mixed (1) prices', 'Quality',listProdIDs, listProdID2s,listProdID3s)



        #GROUP 3   WITHOUT PRICE DISTINTION
        listProdIDs = ['B0048HUNSK', 'B00475WJEY', 'B004D4917W', 'B0043862N4' , 'B00449U3K0' ] # example LONG PRESENCE 'lowRating' 
        listProdID2s = ['B0045IIZKU' , 'B004FLL53Q' 'B004COIBFG' , 'B00462RMS6', 'B004ABO7QI'] # example LONG PRESENCE 'MediumRating' ' 
        listProdID3s = ['B0043XZEEC', 'B0041Q38N0', 'B0049KV50G' , 'B004ASY5ZY', 'B004C9P9TM'] # example LONG PRESENCE 'HighRating' 
        AggregationProductsResult (best_model, X0, X, y, y__, y___, date,  0, 7, 12, 23, 'mixed (2) prices', 'Quality', listProdIDs, listProdID2s,listProdID3s)

        
#        listProdIDs = ['B004D4917W' , 'B004FG16MG', 'B004HT6TS2', 'B004HB2X4Y','B0045DMA42' ] # example short PRESENCE 'lowRating' 
#        listProdID2s = ['B004G8QO5C','B004G8QO5C' , 'B0041I8UAO', 'B0049MPQGI', 'B004E5J61G' ] # example short PRESENCE 'MediumRating' 
#        listProdID3s = ['B0049VGHOO', 'B0043CG3QG', 'B004HKIB6E', 'B004AB70US', 'B004861K1A' ] # example short PRESENCE 'HighRating' 
#        AggregationProductsResult (X0, y, dateCol, 0, 8, 21, 35, 'short', listProdIDs, listProdID2s,listProdID3s)
        
        
        # CHINESE MODEL !!!!!
        X0, X, y, y__, y___, date = DataGenTest (4) 
        X0c, Xc = reduceMatrix (X0, X, feature_names_senti, feature_names_Dimitri_Chinese)
        Xc = np.array(Xc)
        X0c = np.array(X0c)
        print('Xc[0]',Xc[0],'\nX0c[0]',X0c[0])
        
        best_modelc = core(Xc, y, ' General test with reduct columns feature_names_Dimitri',0.03,feature_names_Dimitri_Chinese)   # <---------------- FORCED! BECAUSE OTHERWISE IT'S NOT POSSIBLE SELECT FEATURES IN SIMPLE WAY
#0, 5, 7, 21
        dirichlet_distr (best_modelc, X0c, Xc,  y, y__, y___, date, 0, 7, 5, 21, listProdIDs, listProdID2s,listProdID3s)




        
        
        X0, X, y, y__, y___, date = DataGenTest (4) 
        prodID = 'B003ZUIHY8' 
        newX, newy = list_sameProd (IT.zip_longest(X0, y),34,prodID)
        newX = np.array(newX)    
        newy = np.array(newy)    
        #newTrainX, newTrainy = list_sameRating (IT.zip_longest(newX, newy), 0, 1)     # SAME RATING
        newTrainX, newTrainy = list_LittleRating  (IT.zip_longest(newX, newy), 0) 
        #newTrainX, newTrainy = list_HighRating  (IT.zip_longest(newX, newy), 0) #list_HighRating
        newTrainX = np.delete(newX,34,axis=1)        #column ProdID deleted
        print(newTrainX[0])
        print(newTrainX.shape)
        newTrainy = newy            
        best_model = core(newTrainX, newTrainy, 'test rating (1) + PROD: '+prodID,0.03,feature_names_senti)   # <---------------- FORCED! BECAUSE OTHERWISE IT'S NOT POSSIBLE SELECT FEATURES IN SIMPLE WAY
        
        modelName = best_model.best_estimator_.steps[1][1].__class__.__name__
        print('--------- modelName Chosen: '+modelName)     
        coreFeatureExtraction(newTrainX, newTrainy, 'test rating (1) + PROD: '+prodID + ' - Best features',modelName,3,feature_names_senti)  
       # All_pcc(newTrainX, newTrainy,feature_names_senti)
              
       
       # SAME OR similar RATING and SAME PROD
        X0, X, y, y__, y___, date = DataGenTest (4) 
        prodID = 'B004G6002M' 
        newX, newy = list_sameProd (IT.zip_longest(X0, y),34,prodID)
        newX = np.array(newX)    
        newy = np.array(newy)    
        #newTrainX, newTrainy = list_sameRating (IT.zip_longest(newX, newy), 0, 1)     # SAME RATING
        newTrainX, newTrainy = list_LittleRating  (IT.zip_longest(newX, newy), 0) 
        #newTrainX, newTrainy = list_HighRating  (IT.zip_longest(newX, newy), 0) #list_HighRating
        newTrainX = np.delete(newTrainX,34,axis=1)        #column ProdID deleted
        print(newTrainX[0])
        print(newTrainX.shape)
        best_model = core(newTrainX, newTrainy, 'test rating (1) + PROD: '+prodID,0.03,feature_names_senti)   # <---------------- FORCED! BECAUSE OTHERWISE IT'S NOT POSSIBLE SELECT FEATURES IN SIMPLE WAY

        
        # GREY ZONE: little rating and large votes and Opposite
        listProdIDs = ['B0041Q38NU','B004G6002M', 'B004GF8TIK', 'B0044YU60M', 'B0043T7FXE'] 
        X0, X, y, y__ = DataGenTest (5)   # y = % HELPFUL VOTES, y__ # HELPFUL VOTES
        newX, newy = list_smallVotes (IT.zip_longest(X0, y, y__), 5)       # large votes
        newX, newy = list_groupProds (IT.zip_longest(newX, newy),34,listProdIDs)    # group of meaningful prods
        newX = np.array(newX)    
        newy = np.array(newy) 
        newTrainX, newTrainy = list_HighRating  (IT.zip_longest(newX, newy), 0)   # little rating
        newTrainX = np.delete(newTrainX,34,axis=1)        #column ProdID deleted
        print(newTrainX[0])
        print(newTrainX.shape)
        best_model = core(newTrainX, newTrainy, 'test rating (1) + PROD: '+prodID,0.03,feature_names_senti)   # <---------------- FORCED! BECAUSE OTHERWISE IT'S NOT POSSIBLE SELECT FEATURES IN SIMPLE WAY

        # GREY ZONE: little rating and percentage of votes = K
        #listProdIDs = ['B0041Q38NU','B004G6002M', 'B004GF8TIK', 'B0044YU60M', 'B0043T7FXE'] 
        X0, X, y, y__ = DataGenTest (5)   # y = % HELPFUL VOTES, y__ # HELPFUL VOTES
        newX, newy = list_percentVotes (IT.zip_longest(X0, y, y__), 0.8)       # percentagevotes
        #newX, newy = list_groupProds (IT.zip_longest(newX, newy),34,listProdIDs)    # group of meaningful prods
        newX = np.array(newX)    
        newy = np.array(newy) 
        newTrainX, newTrainy = list_LittleRating  (IT.zip_longest(newX, newy), 0)   # little rating
        newTrainX = np.delete(newTrainX,34,axis=1)        #column ProdID deleted
        print(newTrainX[0])
        print(newTrainX.shape)
        best_model = core(newTrainX, newTrainy, 'test rating (1) + PROD: '+prodID,0.03,feature_names_senti)   # <---------------- FORCED! BECAUSE OTHERWISE IT'S NOT POSSIBLE SELECT FEATURES IN SIMPLE WAY
        
        
        newTrainX, newTrainy = list_sameColumn (IT.zip_longest(X, y), 2, 0.6, True, feature_names_senti)
        All_pcc(newTrainX, newTrainy,feature_names_senti)
        best_model = core(newTrainX, newTrainy, 'test High Usefull/All (0.6) ',0.03,feature_names_senti)
        modelName = best_model.best_estimator_.steps[1][1].__class__.__name__
        print('modelName Chosen: ',modelName)     
        coreFeatureExtraction(newTrainX, newTrainy,'test High Usefull/All (0.6) '+' - Best features',modelName,2,feature_names_senti)      # chose : 1,2 or 3

        newTrainX, newTrainy = list_sameColumn (IT.zip_longest(X, y), 4, 0.6, True, feature_names_senti)
        All_pcc(newTrainX, newTrainy,feature_names_senti)
        best_model = core(newTrainX, newTrainy, 'test DENS POS:NVJ (0.6) ',0.03,feature_names_senti)
        modelName = best_model.best_estimator_.steps[1][1].__class__.__name__
        print('modelName Chosen: ',modelName)     
        coreFeatureExtraction(newTrainX, newTrainy,'test DENS POS:NVJ (0.6)  '+' - Best features',modelName,2,feature_names_senti)      # chose : 1,2 or 3

        newTrainX, newTrainy = list_sameLen_textReview(IT.zip_longest(X, y), 3, 27, 200, 0)           # TRAINING FILE SAME lEN
        All_pcc(newTrainX, newTrainy,feature_names_senti)  

 