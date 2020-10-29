# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:46:51 2019

@author: paolo
"""



import itertools as IT
import argparse as argsp
import pandas as pd
import numpy as np
import statistics as stat
import scipy.stats as scstat
#from scipy import stats
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from scipy.stats import poisson
from pandas import Series
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d, UnivariateSpline, splrep, splev

#import datetime
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
from matplotlib.ticker import NullFormatter
from matplotlib.dates import MonthLocator, DateFormatter
from statsmodels.tsa.arima_model import ARIMA
from random import seed
from random import random
import statsmodels.api as sm
import pylab as pl
import math 
from functools import reduce



import sys
sys.path.append('C:\\UniDatiSperimentali\\PAOLONE IDEAS\dottorato\\python e SIDE PG\\pythonCode')
for line in sys.path: 
    print (line)
    
from regressionReviews6CVMoreModelsFullDataSenti import reduceMatrix, AggregationProductsResult, feature_names_senti, feature_names_Dimitri
from regressionReviews5CVMoreModelsFullDataSenti import normalization, normalization1, plotting_trends2_dot, plotting_trends3_formulaLinear, extractInfoDemand, plotting_utility, plotting_utilities
from regressionReviews4CVMoreModelsFullData import listValuesProdPlusDate, statisticSimple, statisticSimple1


import gc
gc.collect()

# Importing the Training dataset
dataTrain = pd.read_csv(r'C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Result10022019\Training.csv', error_bad_lines=False)
series = Series.from_csv(r'C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Result10022019\Training.csv', header=0)

def DataGenTest (numTest):   
        X00 = dataTrain.iloc[:, 5:46].values  # all columns
        print(X00)
        print(X00.shape)
        X0 = dataTrain.iloc[:, 5:45].values   # X0 all column minus rewiewer id 
        print(X0)
        print(X0.shape)
        X_ = np.delete(X0,34,axis=1)        # X = X0 minus column ProdID 
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
            return X0, X_, y_, y__, y___, date    # %v+, v+, v, date list
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

# mean v+%
def ComputeQuality1 (y,r,rand):
#    y_norm = normalization (y)  #it's already normalized
    mean_vote, sd_vote =  statisticSimple (y, 'mean %v+')
    mean_voteR = mean_vote - mean_vote*0.2*(random())
    if (rand):
        q = mean_voteR * r
    else:    
        q = mean_vote * r
    print ('QUAL',q)
    return q

# y(array):  cumulative v+% / num values , r: rating
def ComputeQuality2 (y,r):
    q = 0
    print ('%v+',y)
    for i in range(1,len(y)):
        partial = (y[i]+y[i-1])/2
        q = q + partial
        #print ('%q+ CUMULATIVE',q)
    q = (q/len(y)) * r
    print ('QUAL CUMULATIVE ',q)
    return q

# y(array): cumulative v+% / num values, rep(array): reputation reviewer, winCont(array): context windows r: rating
def ComputeQuality3A (y,rep,winCont,r):
    
    q = qPlus1 = qPlus2 = 0
    print ('%v+',y)
    for i in range(1,len(y)):
        partial = (y[i]+y[i-1])/2
        if (float(rep[i]) > 10):
            repuRev = 10
        else:
            repuRev = float(rep[i])
        partialPlus1 = repuRev*partial                 #math.log
        partialPlus2 = float(winCont[i])*2*partial
        q = q + partial
        qPlus1 = qPlus1 + partialPlus1
        qPlus2 = qPlus2 + partialPlus2
#        print ('%q+ CUMULATIVE',q)
    q = (q/len(y)) * r
    qPlus1 = (qPlus1/len(y)) * r
    qPlus2 = (qPlus2/len(y)) * r
    df = pd.DataFrame({'vote+%' : y, 'rep' : rep, 'winCont' : winCont})
    print (df)
    print ('\nQUAL CUMULATIVE ',q)
    print ('\nQUAL CUMULATIVE Wheight1',qPlus1)  # reputation 
    print ('\nQUAL CUMULATIVE Wheight2',qPlus2)  # context vote
    return qPlus1

# y(array): cumulative v+% / num values, rep(array): reputation reviewer, r: rating
def ComputeQuality3 (y,rep,r):
    
    q = qPlus1 = 0
    print ('%v+',y)
    for i in range(1,len(y)):
        partial = (y[i]+y[i-1])/2
        if (float(rep[i]) >= 4):
            repuRev = 4
        else:
            repuRev = float(rep[i])
        partialPlus1 = repuRev*partial                 #math.log
        q = q + partial
        qPlus1 = qPlus1 + partialPlus1
#        print ('%q+ CUMULATIVE',q)
    q = (q/len(y)) * r
    qPlus1 = (qPlus1/len(y)) * r
    #df = pd.DataFrame({'vote+%' : y, 'rep' : rep})
    #print (df)
    print ('\nQUAL CUMULATIVE ',q)
    print ('\nQUAL CUMULATIVE Wheight1',qPlus1)  # reputation 
    return qPlus1


# TODO create another fitting curve function that generate the curve (price, quality)  levelRating
    
def FittingCurveLevelPrice (listProdIDs, priceListLog, priceList, X0, y, y_l, cost, p_h, y__, y___,date, info,path):    
    
        print('{:*^120}'.format(' LIST PROD IDS '))
        print('*',listProdIDs,'*')
        print('{:*^120}'.format(''),'\n')

        rand = False
        colProduct = len(X0[0]) - 1
        colRating = 0
        colRepuRev = 7
        colWinCont = 12
        qualityList = []     # q_vec  v+%
#        qualityListb = []    # q_vec v+
#        qualityListc = []    # q_vec v+%  z rating
        ratingList = []     # r_vec
#        ratingListz  = [] 
        voteList = []
#        voteListR = []
        voteListb = []
                # seed random number generator
        seed(1)
        for product in listProdIDs:
            # X, %v+ , 1 - %v+, v+, date
            newX, newy, ya, yb, dateCol, r, w1, w2 = listValuesProdPlusDate (IT.zip_longest(X0, y, y__, y___),colProduct,colRating,colRepuRev,colWinCont,product,date)
            newy = np.array(newy)    # %v+
            yb = np.array(yb)        # v+
            mean_rating, sd_rating, z_rating = statisticSimple1 (r, 'rating')
            z_rating5 =normalization1 (z_rating, (0, 5))
            print ('mean_rating',mean_rating,'sd_rating',sd_rating,'\nrating',r,'\nz_rating',z_rating5, '\n')
            mean_zrating, sd_zrating = statisticSimple (z_rating5, 'z_rating')
            #q = ComputeQuality2 (newy,mean_rating)   # cumulative v+% / num values
            q = ComputeQuality3 (newy, w1, mean_rating)   
            ComputeQuality1 (newy,mean_rating,rand)  # qual on %v+
            #qb = ComputeQuality1 (yb,mean_rating)   # qual on v+
            #qc = ComputeQuality1 (newy,mean_zrating)   # qual on %v+, z score rating
#            qualityListb.append(qb)
#            qualityListc.append(qc)
            qualityList.append(q)
            mean_vote, sd_vote = statisticSimple (newy, '%vote+')
            mean_voteb, sd_voteb = statisticSimple (yb, 'vote+')
            ratingList.append(mean_rating)
#            ratingListz.append(mean_zrating)
            voteList.append (mean_vote)
#            mean_voteR = mean_vote - mean_vote*0.2*(random())
#            voteListR.append (mean_voteR)            
            voteListb.append (mean_voteb)
        qualityList = np.array(qualityList)
        ratingList = np.array(ratingList)
#        ratingListz = np.array(ratingListz)
#        qualityListb = np.array(qualityListb)
#        qualityListc = np.array(qualityListc)
        voteList = np.array(voteList)
#        voteListR = np.array(voteListR)
        qualityListNormal = normalization (qualityList)
#        if (rand):
#            print ('voteList',voteList,'\nvoteListRandom',voteListR)
        voteListb = np.array(voteListb)
#        qualityListbNormal = normalization (qualityListb)
#        qualityListcNormal = normalization (qualityListc)
        print ('qualityList',qualityList, '\nqualityListNormal_%v',qualityListNormal,  '\nrating List',ratingList, '\n\n' )
        df = pd.DataFrame({'rating' : ratingList, 'qualityListNormal_%v' : qualityListNormal, 'qualityList' : qualityList, 'listProdIDs' : listProdIDs, '%voteList' : voteList, 'voteList' : voteListb })
#        df = df.sort_values(by=['rating'])
        print (df)
        print ('\n\n')
#        print (df['rating'].tolist(),'\n') 
#        print (df['qualityListNormal_%v'].tolist(),'\n')
#        print (df['qualityListNormal_v'].tolist(),'\n')
        ratingListOrd = df['rating'].tolist()
        qualityListNormalOrdPosPerc = df['qualityListNormal_%v'].tolist()  # qual on %v+ Blue - GOOD
#        qualityListNormalOrdPos = df['qualityListNormal_v'].tolist()        # qual on %v - GREEN
#        qualityListzNormalOrdPosPerc = df['qualityListNormal_%v-z'].tolist()  # qual on %v+, ORANGE z score rating -> makes stable the variance
        steps = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]  
        if (y_l==0):
            plotting_trends2_dot (ratingListOrd, qualityListNormalOrdPosPerc, info, ' %v+',2, 'rating')    # Using Spline
        else:
#            if(linear):
                plotting_trends3_formulaLinear (steps, qualityListNormalOrdPosPerc,  y_l, cost, p_h, ratingListOrd, priceList, info+ ' (first '+str(max(steps))+ ' steps)', ' %v+',0,path) 
#            else:
#                plotting_trends3_formulaRootSquare (steps, qualityListNormalOrdPosPerc, cost, p_h, ratingListOrd, priceList, info+ ' (first '+str(max(steps))+ ' steps)', ' %v+',0,path) 


        
        # to fit a Cobb-Douglas see http://sphaerula.com/legacy/R/cobbDouglas.html and https://pdfs.semanticscholar.org/029d/05c41ef2ea24092cb879bcd6d4e56b242a04.pdf
        
        
def AnalysisDemand2 (listProdIDs, priceListLog, priceList, X0, y, y_q, cost,  y__, y___,date,x_size):    
    
        print('{:*^120}'.format(' LIST PROD IDS '))
        print('*',listProdIDs,'*')
        print('{:*^120}'.format(''),'\n')

        rand = False
        colProduct = len(X0[0]) - 1
        colRating = 0
        colRepuRev = 7
        colWinCont = 12
        qualityList = []     # q_vec  v+%
        ratingList = []     # r_vec
        voteList = []
        voteListb = []
        for product in listProdIDs:
            # X, %v+ , 1 - %v+, v+, date
            newX, newy, ya, yb, dateCol, r, w1, w2 = listValuesProdPlusDate (IT.zip_longest(X0, y, y__, y___),colProduct,colRating,colRepuRev,colWinCont,product,date)
            newy = np.array(newy)    # %v+
            yb = np.array(yb)        # v+
            mean_rating, sd_rating, z_rating = statisticSimple1 (r, 'rating')
            z_rating5 =normalization1 (z_rating, (0, 5))
            print ('mean_rating',mean_rating,'sd_rating',sd_rating,'\nrating',r,'\nz_rating',z_rating5, '\n')
            mean_zrating, sd_zrating = statisticSimple (z_rating5, 'z_rating')
            q = ComputeQuality3 (newy, w1, mean_rating)   
            ComputeQuality1 (newy,mean_rating,rand)  # qual on %v+
            qualityList.append(q)
            mean_vote, sd_vote = statisticSimple (newy, '%vote+')
            mean_voteb, sd_voteb = statisticSimple (yb, 'vote+')
            ratingList.append(mean_rating)
            voteList.append (mean_vote)
            voteListb.append (mean_voteb)
        qualityList = np.array(qualityList)
        ratingList = np.array(ratingList)
        voteList = np.array(voteList)
        qualityListNormal = normalization (qualityList)
        voteListb = np.array(voteListb)
       # print ('qualityList',qualityList, '\nqualityListNormal_%v',qualityListNormal,  '\nrating List',ratingList, '\n\n' )
        df = pd.DataFrame({'rating' : ratingList, 'qualityListNormal_%v' : qualityListNormal, 'qualityList' : qualityList, 'listProdIDs' : listProdIDs, 'priceList' : priceList, '%voteList' : voteList, 'voteList' : voteListb })
        print (df)
        print ('\n\n\n')
        ratingListOrd = df['rating'].tolist()
        qualityListNormalOrdPosPerc = df['qualityListNormal_%v'].tolist()  # qual on %v+ Blue - GOOD
        
        steps = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]       
        #x_size = 30
        utilityList = []
        utilityListSR = []
        yList = []
        yListSR = []
        bc = []
        maxPrice = max(priceList)
        startPrice = min(priceList)
        step = (maxPrice-startPrice)/x_size                     # maxPrice IN THIS CASEIS THE MAX BUDGET CONSTRAINT 
        for i in range(3,x_size):
                bc.append(startPrice + int((step/3)*(i-2)**1.3))
        bc = np.array(bc)
        print ('bc[]=',bc,'len bc[]',len(bc))
        for i in range(0,len(bc)):
            outSR_u, outSR_y, out_u, out_y  = extractInfoDemand (steps, qualityListNormalOrdPosPerc,  y_q, cost, bc[i], ratingListOrd, priceList)
            utilityList.append(out_u)
            utilityListSR.append(outSR_u)
            yList.append(out_y)
            yListSR.append(outSR_y)                      
        utilityList=np.array(utilityList)
        utilityListSR=np.array(utilityListSR)
        yList=np.array(yList)
        yListSR=np.array(yListSR)   
       #uAll = np.zeros(2*len(utilityList)) 
        uAll = np.concatenate((utilityList,utilityListSR))        
        print('uAll',uAll,'\nutilityList',utilityList,'\nutilityListSR',utilityListSR)
        #minUL = min(uAll)
        for i in range(0,len(uAll)):
#               utilityList[i] = utilityList[i] + abs(minUL)
#               utilityListSR[i] = utilityListSR[i] + abs(minULSR)
                if (uAll[i] > 0):   
                    uAll[i] = uAll[i]
                else:
                    uAll[i] = 0
        #print('minUL',minUL,'uAll',uAll)
        #uAll = normalization (uAll)        
        utilityList = uAll[:len(utilityList)]       
        utilityListSR = uAll[len(utilityList):len(uAll)]      
        #print ('utilityList=',utilityList,len(utilityList))
        #print ('utilityListSR=',utilityListSR,len(utilityListSR))
        return utilityList, utilityListSR, yList, yListSR
        

def AnalysisDemand1 (listProdIDs, priceListLog, priceList, X0, y,  y_q, cost, y__, y___,date,x_size):    
    
        print('{:*^120}'.format(' LIST PROD IDS '))
        print('*',listProdIDs,'*')
        print('{:*^120}'.format(''),'\n')

        rand = False
        colProduct = len(X0[0]) - 1
        colRating = 0
        colRepuRev = 7
        colWinCont = 12
        qualityList = []     # q_vec  v+%
        voteList = []
        ratingList =  []
        for product in listProdIDs:
            # X, %v+ , 1 - %v+, v+, date
            newX, newy, ya, yb, dateCol, r, w1, w2 = listValuesProdPlusDate (IT.zip_longest(X0, y, y__, y___),colProduct,colRating,colRepuRev,colWinCont,product,date)
            newy = np.array(newy)    # %v+
            mean_rating, sd_rating, z_rating = statisticSimple1 (r, 'rating')
            #print ('mean_rating',mean_rating,'sd_rating',sd_rating,'\nrating',r, 'product',product,'\n')
            q = ComputeQuality3 (newy, w1, mean_rating)   
            ComputeQuality1 (newy,mean_rating,rand)  # qual on %v+
            qualityList.append(q)
            mean_vote, sd_vote = statisticSimple (newy, '%vote+')
            ratingList.append(mean_rating)
            voteList.append (mean_vote)
        qualityList = np.array(qualityList)
        priceListLog = np.array(priceListLog)
        ratingList = np.array(ratingList)
        voteList = np.array(voteList)
        qualityListNormal = normalization (qualityList)
        print ('qualityList',qualityList, '\nqualityListNormal_%v',qualityListNormal, '\nrating List',ratingList, '\n\n' )
        df = pd.DataFrame({'rating' : ratingList, 'qualityListNormal_%v' : qualityListNormal, 'priceLog' : priceListLog, 'price' : priceList, 'qualityList' : qualityList, 'listProdIDs' : listProdIDs, '%voteList' : voteList })
        print ('BY RATING\n',df)
        print ('\n\n')
        priceList = df['price'].tolist()
        priceListLog = df['priceLog'].tolist()
        qualityListNormalOrdPosPerc = df['qualityListNormal_%v'].tolist()  # qual on %v+ Blue - GOOD
         # normalize rating by sd in order to reduce the variance   
         
        steps = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        #x_size = 25
        utilityList = []
        utilityListSR = []
        yList = []
        yListSR = []
        bc = []
        maxPrice = max(priceList)
        startPrice = min(priceList)
        step = (maxPrice-startPrice)/x_size                     # maxPrice IN THIS CASEIS THE MAX BUDGET CONSTRAINT 
        for i in range(3,x_size):
                bc.append(startPrice + int((step/3)*(i-2)**1.3))
        bc = np.array(bc)
        print ('bc[]=',bc,'len bc[]',len(bc))
        for i in range(0,len(bc)):
            outSR_u, outSR_y, out_u, out_y  = extractInfoDemand (steps, qualityListNormalOrdPosPerc,  y_q, cost, bc[i], ratingList, priceList)
            utilityList.append(out_u)
            utilityListSR.append(outSR_u)
            yList.append(out_y)
            yListSR.append(outSR_y)                        
        utilityList=np.array(utilityList)
        utilityListSR=np.array(utilityListSR)
        yList=np.array(yList)
        yListSR=np.array(yListSR)
        #uAll = np.zeros(2*len(utilityList)) 
        uAll = np.concatenate((utilityList,utilityListSR))        
        print('uAll',uAll,'\nutilityList',utilityList,'\nutilityListSR',utilityListSR)
        minUL = min(uAll)
        for i in range(0,len(uAll)):
#               utilityList[i] = utilityList[i] + abs(minUL)
#               utilityListSR[i] = utilityListSR[i] + abs(minULSR)
                if (uAll[i] > 0):   
                    uAll[i] = uAll[i]
                else:
                    uAll[i] = 0
        print('minUL',minUL,'uAll',uAll)
        #uAll = normalization (uAll)        
        utilityList = uAll[:len(utilityList)]       
        utilityListSR = uAll[len(utilityList):len(uAll)]      
        #print ('utilityList=',utilityList,len(utilityList))
        #print ('utilityListSR=',utilityListSR,len(utilityListSR))
        return utilityList, utilityListSR, yList, yListSR

        #plotting_trends3_formulaLinear (steps, qualityListNormalOrdPosPerc,  y_l, cost, p_h, ratingList, priceList, info+ ' (first '+str(max(steps))+ ' steps)', ' %v+',0,path) 



def DataFrameData (listProdIDs, priceListLog, priceList, X0, y, y__, y___,date):    
    
        print('{:*^120}'.format(' LIST PROD IDS '))
        print('*',listProdIDs,'*')
        print('{:*^120}'.format(''),'\n')

        rand = False
        colProduct = len(X0[0]) - 1
        colRating = 0
        colRepuRev = 7
        colWinCont = 12
        qualityList = []     # q_vec  v+%
        voteList = []
#        voteListR = []
        ratingList =  []
        # seed random number generator
        seed(1)
        for product in listProdIDs:
            # X, %v+ , 1 - %v+, v+, date
            newX, newy, ya, yb, dateCol, r, w1, w2 = listValuesProdPlusDate (IT.zip_longest(X0, y, y__, y___),colProduct,colRating,colRepuRev,colWinCont,product,date)
            newy = np.array(newy)    # %v+
            mean_rating, sd_rating, z_rating = statisticSimple1 (r, 'rating')
            print ('mean_rating',mean_rating,'sd_rating',sd_rating,'\nrating',r, 'product',product,'\n')
            q = ComputeQuality3 (newy, w1, mean_rating)   
            ComputeQuality1 (newy,mean_rating,rand)  # qual on %v+
            qualityList.append(q)
            mean_vote, sd_vote = statisticSimple (newy, '%vote+')
            ratingList.append(mean_rating)
            #mean_vote = np.random.rand(100)*10
#            mean_voteR = mean_vote - mean_vote*0.2*(random())
            voteList.append (mean_vote)
#            voteListR.append (mean_voteR)            
        qualityList = np.array(qualityList)
        priceListLog = np.array(priceListLog)
        ratingList = np.array(ratingList)
        voteList = np.array(voteList)
#        voteListR = np.array(voteListR)
        qualityListNormal = normalization (qualityList)
#        if (rand):
#            print ('voteList',voteList,'\nvoteListRandom',voteListR)
        print ('qualityList',qualityList, '\nqualityListNormal',qualityListNormal, '\nrating List',ratingList, '\n\n' )
        df = pd.DataFrame({'rating' : ratingList, 'qualityListNormal' : qualityListNormal, 'priceLog' : priceListLog, 'price' : priceList, 'qualityList' : qualityList, 'listProdIDs' : listProdIDs, '%voteList' : voteList })
#        df = df.sort_values(by=['rating'])
        print ('BY RATING\n',df)
        return df
        

def PlotDistribution(h,indexColor):        
       
    #h = sorted([186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,      187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 159,      161, 178, 175, 185, 175, 162, 173, 172, 177, 175, 172, 177, 180])  #sorted

        # https://python-graph-gallery.com/python-colors/
    colors = ['green', 'red', 'cyan' , 'orange', 'magenta', 'navy', 'green', 'brown', 'pink','olive','red','navy', 'slategray','coral','aquamarine','slategray']
#    plt.scatter(x, u, marker='x', color=colors[indexColor+2], label = 'Utility')   # Utility point
 

    fit = scstat.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed        
    print ('normal',fit,'\n\n');
    n, min_max, mean, var, skew, kurt = scstat.describe(fit)
    print("Number of elements: {0:d}".format(n))
    print("Minimum: {0:8.6f} Maximum: {1:8.6f}".format(min_max[0], min_max[1]))
    print("Mean: {0:8.6f}".format(mean))
    print("Variance: {0:8.6f}".format(var))
    print("Skew : {0:8.6f}".format(skew))
    print("Kurtosis: {0:8.6f}".format(kurt))
   # print(fit.stats(moments="mvsk"))
    pl.plot(h,fit,'x', color=colors[5])
    pl.hist(h,normed=True,color=colors[indexColor])      #use this to draw h
    pl.show()    


def Plot1(x,y,indexColor):        
       
    #h = sorted([186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,      187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 159,      161, 178, 175, 185, 175, 162, 173, 172, 177, 175, 172, 177, 180])  #sorted

        # https://python-graph-gallery.com/python-colors/
    colors = ['green', 'red', 'cyan' , 'orange', 'magenta', 'navy', 'green', 'brown', 'pink','olive','red','navy', 'slategray','coral','aquamarine','slategray']
#    plt.scatter(x, u, marker='x', color=colors[indexColor+2], label = 'Utility')   # Utility point
    pl.plot(x,y,  color=colors[indexColor])
    pl.show()    
    
    
def PlotBar(x,y,indexColor):        
       
    #h = sorted([186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,      187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 159,      161, 178, 175, 185, 175, 162, 173, 172, 177, 175, 172, 177, 180])  #sorted

        # https://python-graph-gallery.com/python-colors/
    colors = ['green', 'red', 'cyan' , 'orange', 'magenta', 'navy', 'green', 'brown', 'pink','olive','red','navy', 'slategray','coral','aquamarine','slategray']
    fig = plt.figure()
    ax = fig.add_axes([0,0,2,1])
#    langs = ['C', 'C++', 'Java', 'Python', 'PHP']
#    students = [23,17,35,29,12]

    ax.bar(x,y, color=colors[indexColor],width = 0.20)
    plt.show()
    



def PlotSpline(x,y,indexColor):        
       
    #h = sorted([186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,      187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 159,      161, 178, 175, 185, 175, 162, 173, 172, 177, 175, 172, 177, 180])  #sorted

        # https://python-graph-gallery.com/python-colors/
    colors = ['green', 'red', 'cyan' , 'orange', 'magenta', 'navy', 'green', 'brown', 'pink','olive','red','navy', 'slategray','coral','aquamarine','slategray']
#    fig = plt.figure()
#    ax = fig.add_axes([0,0,3,1])
#    langs = ['C', 'C++', 'Java', 'Python', 'PHP']
#    students = [23,17,35,29,12]
    xx = np.arange(1, max(x) ,0.5)      # range xmin, xmax, precision
    s2 = interpolate.UnivariateSpline (x, y, k=3) # smoothing_factor 0.1
    plt.plot(xx, s2(xx), 'r', label = 'Spline fitted') # SPLINE   # see https://stackoverflow.com/questions/17913330/fitting-data-using-univariatespline-in-scipy-python

    plt.bar(x,y, color=colors[indexColor],width = 0.20)
    plt.show()
    

def FittingCurveLevelRating (listProdIDs, priceListLog, priceList, X0, y,  y_l, cost, p_h, y__, y___,date, info, path):    
    
        print('{:*^120}'.format(' LIST PROD IDS '))
        print('*',listProdIDs,'*')
        print('{:*^120}'.format(''),'\n')

        rand = False
        colProduct = len(X0[0]) - 1
        colRating = 0
        colRepuRev = 7
        colWinCont = 12
        qualityList = []     # q_vec  v+%
        voteList = []
#        voteListR = []
        ratingList =  []
        # seed random number generator
        seed(1)
        for product in listProdIDs:
            # X, %v+ , 1 - %v+, v+, date
            newX, newy, ya, yb, dateCol, r, w1, w2 = listValuesProdPlusDate (IT.zip_longest(X0, y, y__, y___),colProduct,colRating,colRepuRev,colWinCont,product,date)
            newy = np.array(newy)    # %v+
            mean_rating, sd_rating, z_rating = statisticSimple1 (r, 'rating')
            print ('mean_rating',mean_rating,'sd_rating',sd_rating,'\nrating',r, 'product',product,'\n')
            q = ComputeQuality3 (newy, w1, mean_rating)   
            ComputeQuality1 (newy,mean_rating,rand)  # qual on %v+
            qualityList.append(q)
            mean_vote, sd_vote = statisticSimple (newy, '%vote+')
            ratingList.append(mean_rating)
            #mean_vote = np.random.rand(100)*10
#            mean_voteR = mean_vote - mean_vote*0.2*(random())
            voteList.append (mean_vote)
#            voteListR.append (mean_voteR)            
        qualityList = np.array(qualityList)
        priceListLog = np.array(priceListLog)
        ratingList = np.array(ratingList)
        voteList = np.array(voteList)
#        voteListR = np.array(voteListR)
        qualityListNormal = normalization (qualityList)
#        if (rand):
#            print ('voteList',voteList,'\nvoteListRandom',voteListR)
        print ('qualityList',qualityList, '\nqualityListNormal_%v',qualityListNormal, '\nrating List',ratingList, '\n\n' )
        df = pd.DataFrame({'rating' : ratingList, 'qualityListNormal_%v' : qualityListNormal, 'priceLog' : priceListLog, 'price' : priceList, 'qualityList' : qualityList, 'listProdIDs' : listProdIDs, '%voteList' : voteList })
#        df = df.sort_values(by=['rating'])
        print ('BY RATING\n',df)
#        df = df.sort_values(by=['price'])
#        print ('BY PRICE\n',df)
        print ('\n\n')
        priceList = df['price'].tolist()
        priceListLog = df['priceLog'].tolist()
        qualityListNormalOrdPosPerc = df['qualityListNormal_%v'].tolist()  # qual on %v+ Blue - GOOD
         # normalize rating by sd in order to reduce the variance   
         
        steps = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]  
        if (y_l==0):
            plotting_trends2_dot (priceListLog, qualityListNormalOrdPosPerc,   info, ' %v+',0, 'price')    # Using Spline
        else:
#            if(linear):
                plotting_trends3_formulaLinear (steps, qualityListNormalOrdPosPerc,  y_l, cost, p_h, ratingList, priceList, info+ ' (first '+str(max(steps))+ ' steps)', ' %v+',0,path) 
#            else:
#                plotting_trends3_formulaRootSquare (steps, qualityListNormalOrdPosPerc, cost, p_h, ratingList, priceList, info+ ' (first '+str(max(steps))+ ' steps)', ' %v+',0,path) 


#
#   EXTRACT THE RATING LIST AND ORDER BY RATING AND PRICES
#                
def CreateListPr_AND_Rat (listProdIDs, priceList, X0, y, y__, y___,date):
        colProduct = len(X0[0]) - 1
        colRating = 0
        colRepuRev = 7
        colWinCont = 12
        priceListLog = CreateLogFunction (priceList)  
        ratingList =  []
        for product in listProdIDs:
            newX, newy, ya, yb, dateCol, r, w1, w2 = listValuesProdPlusDate (IT.zip_longest(X0, y, y__, y___),colProduct,colRating,colRepuRev,colWinCont,product,date)
            mean_rating, sd_rating, z_rating = statisticSimple1 (r, 'rating')
            print ('mean_rating',mean_rating,'sd_rating',sd_rating,'\nrating',r, 'product',product,'\n')
            ratingList.append(mean_rating)
        priceListLog = np.array(priceListLog)
        ratingList = np.array(ratingList)
        df = pd.DataFrame({'rating' : ratingList, 'priceLog' : priceListLog, 'price' : priceList, 'listProdIDs' : listProdIDs })
        df = df.sort_values(by=['rating'], ascending=False)
        listProdxRatIDs =  df['listProdIDs'].tolist()
        priceListLogxRatIDs = df['priceLog'].tolist()
        priceListxRatIDs = df['price'].tolist()
        print ('BY RATING\n',df)
        df = df.sort_values(by=['price'])
        listProdxPriceIDs =  df['listProdIDs'].tolist()
        priceListLogxPriceIDs = df['priceLog'].tolist()
        priceListxPriceIDs = df['price'].tolist()
        print ('BY price\n',df)
        statisticSimple1 (ratingList, 'ratingList')
        statisticSimple1 (priceList, 'priceList')
        return listProdxRatIDs, priceListLogxRatIDs, priceListxRatIDs, listProdxPriceIDs, priceListLogxPriceIDs, priceListxPriceIDs
               #DataFrame.listProdIDs order by Rating, DataFrame.priceLog order by Rating, DataFrame.price order by Rating,  
               #DataFrame.listProdIDs order by price, DataFrame.priceLog order by price, DataFrame.price order by price,  



def CreateLogFunction (priceList):   
    priceListLowRLog =  []
    for pr in priceList:
        priceListLowRLog.append(np.log(pr))
    priceListLowRLog = np.array(priceListLowRLog)
    print (priceListLowRLog)
    return priceListLowRLog

def formula4 (q,c):
    result = 0.5 * (2*q + 1 - math.sqrt(8*q*c + 1))
    return result

# Distribution of qty result by price level
    
def QtyinPrices (yList,prices):
        cont = np.zeros(len(prices))  #price level segmentation 
        for y in yList:         # price of winner result
            for i in range(0,len(prices)):
                if (i < len(prices)-1):
                    if (prices[i] < y <=  prices[i+1]):
                         cont[i] += 1
                         print('prices[i]:',prices[i],'y:',y,'i=',i,'cont=',cont[i])
                else:
                    if (prices[i] == y):
                        cont[i] += 1         
                        print('prices[i]:',prices[i],'y:',y,'i=',i,'cont=',cont[i])
        print ('cont',cont,'\nwin price result:',yList)
        return cont

# Distribution of qty result by price level
#def IDProdsinPrices (Z,prices):   # ---------------> Z = IT.zip_longest(listProds, priceLists)
#        #cont = np.zeros(len(prices))  #price level segmentation 
#        for y in range(0,len(listProds)):         # price of winner result
#            for i in range(3,len(prices)):
#                if (i < len(prices)-1):
#                    if (prices[i] < listProds[y] <=  prices[i+1]):
#                        print (i,listProds[y])
#                else:
#                    if (prices[i] == listProds[y]):
#                        print (i,listProds[y])                   
        

#segment the list of prices
def distrPrices_old (priceLists, degree):
        #x_size = 40
        maxPrice = max (priceLists)
        x_size = int(maxPrice**(float(1)/degree))              
        prices = np.zeros(x_size)           # price level distrib
        for i in range(3,len(prices)):
            prices[i] = int(i**degree)                # quadratic distribution of prices
        print ('x_size: ',x_size, 'max: ',maxPrice, 'prices',prices,'\n')
        return prices

#segment the list of prices
def distrPrices (priceLists, x_size):
        prices = []
        maxPrice = max(priceLists)
        startPrice = min(priceLists)
        step = (maxPrice-startPrice)/x_size                     # maxPrice IN THIS CASEIS THE MAX BUDGET CONSTRAINT 
        for i in range(3,x_size):
                prices.append(startPrice + int((step/3)*(i-2)**1.3))
        prices = np.array(prices)
        print ('x_size: ',x_size, 'max: ',maxPrice, 'prices, len',prices,len(prices),'\n')
        return prices



def pmfPoisson (priceLists, index_lambda, x_size):   #  es  pmfPoisson (priceLists, 1, 30):
        # x_size = 30
        arr = []
        lambdaList = [100,200,400,800]  # mean of bc of the population
        rv = poisson(lambdaList(index_lambda))     # lambda(1) = 200
        maxPrices = max(priceLists)
        #for num in range(0, max (priceLists)):
        for num in np.arange(0., maxPrices, maxPrices/x_size):
            arr.append(rv.pmf(num))
        arr = np.array(arr)
        plt.grid(True)
        plt.plot(arr, linewidth=2.0)
        plt.show()  
        return arr


def main(args) : 
    
    # PROVE  INPUT: 2 --opt_arg 3 --predict
#    print(args.opt_pos_arg)
#    print(args.opt_arg)
#    print(args.switch)
#    exit;
    
        choice = args.input;
        print("Argument values:", choice )   

    
    ######### EXPERIMENTS ANNA STEP2: LAUNCH DIFFERENT GROUPS FOR RATING AND PRICE ###########################################################

        # X0 all column minus rewiewer id 
        # X = X0 minus column ProdID  
        # y, y__, y___, date =  %v+, v+, v, date list
        
        
        X0, X, y, y__, y___, date = DataGenTest (4)     
        X0, X = reduceMatrix (X0, X, feature_names_senti, feature_names_Dimitri)
        X = np.array(X)
        X0 = np.array(X0)
        print('X[0]',X[0],'\nX0[0]',X0[0])
        print('X0',X0)   
        best_model = None

        
  ########  EXPERIMENTS ANNA STEP2: LAUNCH DIFFERENT GROUPS FOR RATING AND PRICE ###########################################################


        
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
    

######## NEW EXPERIMENTS ANNA STEP3: Product by Product : Quality per Rating at different prices ######################################################


        # X0 all column minus rewiewer id 
        # X = X0 minus column ProdID  
        # y, y__, y___, date =  %v+, v+, v, date list
        
        
        X0, X, y, y__, y___, date = DataGenTest (4)     
        X0, X = reduceMatrix (X0, X, feature_names_senti, feature_names_Dimitri)
        X = np.array(X)
        X0 = np.array(X0)
        print('X[0]',X[0],'\nX0[0]',X0[0])
        print('X0',X0)   
 
 
 
        listProdLowRIDs =  ['B0043EV20Q', 'B004BFZHO4', 'B0041LYY6K', 'B00426C55G' ,'B00426C56U','B004E10KFG','B0042X8NT6','B0042SDDXM', 'B0046BTK14', 'B00428N9OK' , 'B003ZX8AZQ', 'B0045371FU', 'B004HBK4T0' , 'B0043862N4', 'B0042BUXG4']
        priceListLowR =   [25, 26, 42, 87, 59, 79, 67, 179, 174, 200, 170, 150, 10, 7, 6] # 16
        listProdHigRIDs =  ['B0043T7FXE', 'B0041Q38NU' ,'B0049P6OTI','B0041Q38N0', 'B004CLYEDC','B0044DEDC0','B004BFXBXI', 'B004071ZXA', 'B00429N160', 'B003ZSHNG8', 'B004EFUOY4', 'B0041OSQ9I', 'B004FA8NOQ', 'B0044UHJWY']
        priceListHighR  = [28, 10, 25, 15, 11, 200, 39, 40, 159, 120, 128, 175, 454, 750] # 14
        priceListLowRLog = CreateLogFunction (priceListLowR)  
        priceListHighRLog = CreateLogFunction (priceListHighR)  


        FittingCurveLevelRating (listProdLowRIDs, priceListLowRLog, priceListLowR, X0, y, 0,0,0, y__, y___,date, 'low star 4.2down')
        FittingCurveLevelRating (listProdHigRIDs, priceListHighRLog, priceListHighR, X0, y, 0,0,0, y__, y___,date , 'high star 4.2up')



#######################################################################################################################################################################################################################        
########################################################################  EXPERIMENTS TO CONFIRM THE CHANGE OF SHAPE FOR LOW PRICE (WHEN THE RATING INCREASES THE VOTES, AND THEN THE QUALITY DECREASES  #########################################################################################################        
        
             # TO DO: 
             # 1) FOR each prod take the list of v%+ , then the list of qualities, using 2 formulas : average vote within time 
             # and area under the cumulative curve  within time . Normalize them . We have 2 lists [ratings =[r1,...,rn] , qualities  =[q1,...,qn]]  list(zip(np.array(rating),np.array(qualities)))
             # 2) make a trivial tracing of dot points (blue) and a simple interpolation (red curve). 
             # 3) repeat this test for the 4 type of prices: Low-  price, Low-Medium, High-Medium, High              
             # 4) write a list zip(prodIDs,price) low rating and high rating ... all code rimane the same, but substitute the price 
             
                   # List product  'several rating from 3 to 5 'High pricing'      
   #     listCProdIDs = ['B004GK0GKO' , 'B0041RSPR8', 'B004G8QO5C' , 'B004FLL5AY', 'B004BQKQ8A', 'B0040JHVC2', 'B003ZSHNEA' , 'B003ZYF3LO', 'B0041RSPRS']      
 
    
    # List product  'several rating from 3 to 5 'Low pricing' 
    # 'B0042BUXG4' , 'B0040IO1RQ', ...
        listProdIDs =  ['B0041G62TW', 'B00413PEZS','B004HBK4T0', 'B00428N9OK', 'B0041NFIBS' , 'B0043EV20Q', 'B004BFZHO4', 'B0043T7FXE', 
                        'B0041Q38NU' , 'B0049P6OTI', 'B0041Q38N0',  'B004CLYEDC', 'B004EBX5GW']
    # List product  'several rating from 3 to 5 'Medium pricing' 
        listAProdIDs = ['B0041LYY6K' , 'B00426C55G', 'B00426C56U' , 'B004E10KFG', 'B0042X8NT6', 'B004ABO7QI', 'B0044DEDC0', 'B0041OUA38' , 'B00434UCDE', 'B004071ZXA']
   # List product  'several rating from 3 to 5 'High pricing'      
        listBProdIDs = ['B0042SDDXM' , 'B0046BTK14', 'B00428N9OK' , 'B003ZX8AZQ', 'B0045371FU','B004G8QZPG', 'B0041OSQ9I', 'B004EFUOY4' , 'B00429N160', 'B003ZSHNG8'] 


               # the products of listProdIDs was been aggregate by price level 
        qualityList = FittingCurveLevelPrice (listProdIDs, X0, y, 0,0,0, y__, y___,date, ' low price' )
        qualityListA = FittingCurveLevelPrice (listAProdIDs, X0, y, 0,0,0,  y__, y___,date , ' medium price')
        qualityListB = FittingCurveLevelPrice (listBProdIDs, X0, y, 0,0,0, y__, y___,date , ' high price')
        # the products of listProdIDs was been aggregate by price level  ---> TO DO AGGREGATE THE PROD BY 2 LEVEL RATING AND ORDER BY PRICE

        
        qualityListTot = qualityList.extend(qualityListA).extend(qualityListB)
        print (qualityListTot, max(qualityListTot))
        
       


        
###################################################### EXPERIMENTS FOR EXTRACTING RESULTS OF STOPPING RULE (RANDOM, BY PRICE, BY RATING) ###########################################################################################################################    

        listProdRandomIDs =  ['B0041Q38NU' ,'B0043EV20Q', 'B004BFZHO4',  'B00426C55G' ,'B0042X8NT6','B0042SDDXM', 'B00428N9OK' , 'B004FA8NOQ',  'B0041Q38N0', \
                              'B0046BTK14' , 'B0041OSQ9I' ,'B0043T7FXE' , 'B004071ZXA' , 'B004CLYEDC', 'B0044UHJWY']        
        priceListRandomRIDs = [10, 25, 26, 87, 67, 179, 200, 454, 15, 174, 175, 28, 40, 11, 750] # 15
        listProdxPriceIDs =  ['B0042BUXG4' , 'B0043862N4', 'B0041Q38NU',  'B004CLYEDC', 'B0041Q38N0','B0043EV20Q', 'B004BFZHO4', 'B0041LYY6K', 'B00426C56U', 'B0042X8NT6', \
                              'B004E10KFG', 'B00426C55G' ,  'B003ZSHNG8', 'B004EFUOY4', 'B0045371FU' ]
        priceListxPriceIDs = [6, 7, 10, 11, 15, 25, 26, 42, 59, 67, 79, 87, 120, 128, 150] # 15   ASC            
#        listProdxRatIDs =  ['B0043EV20Q', 'B004BFZHO4', 'B0041LYY6K', 'B00426C55G' ,'B00426C56U','B004E10KFG','B0042X8NT6','B0042SDDXM', 'B0046BTK14', 'B00428N9OK' , 'B003ZX8AZQ', 'B0045371FU', 'B0043862N4', 'B0042BUXG4', 'B0043T7FXE', 'B0041Q38NU' ,'B0049P6OTI','B0041Q38N0', 'B004CLYEDC','B0044DEDC0','B004BFXBXI', 'B004071ZXA', 'B00429N160', 'B003ZSHNG8', 'B004EFUOY4', 'B0041OSQ9I', 'B004FA8NOQ', 'B0044UHJWY']
#        priceListxRatIDs = [25, 26, 42, 87, 59, 79, 67, 179, 174, 201, 170, 150, 7, 6, 28, 10, 24, 15, 11, 200, 39, 40, 159, 120, 128, 175, 454, 750]  # 10  -> take first 10!!!!        
        listProdxRatIDs =  ['B004FA8NOQ' , 'B004071ZXA', 'B00429N160', 'B0044UHJWY' ,'B004CLYEDC', 'B004EFUOY4', 'B0043EV20Q', 'B0041Q38N0', 'B0049P6OTI', 'B004BFXBXI', \
                             'B00426C56U', 'B004BFZHO4', 'B003ZSHNG8', 'B0044DEDC0', 'B0041Q38NU']
        priceListxRatIDs = [454 , 40, 159 , 750, 11, 128, 25, 15, 24, 39, 59, 26, 120, 200, 10 ]  # 15   DESC   
       
        priceListRandomRIDsLog = CreateLogFunction (priceListRandomRIDs)  
        priceListxPriceIDsLog = CreateLogFunction (priceListxPriceIDs)  
        priceListxRatIDsLog = CreateLogFunction (priceListxRatIDs)  
         
#        np.log
#        np.log10
         

        qual_max = 1.0
        clow = 0.2  #cost of step
#        clow = 0.2      #cost of step
        chigh = 0.3     #cost of step
        y_vl =formula4 (qual_max,clow)
#        y_l = formula4 (qual_max,clow)
        y_h = formula4 (qual_max,chigh)
        
        p_h = 100 # max price
        
        FittingCurveLevelRating (listProdRandomIDs, priceListRandomRIDsLog, priceListRandomRIDs, X0, y,  y_vl, y_h, p_h, y__, y___,date , '15 items random')
#        FittingCurveLevelRating (listProdxPriceIDs, priceListxPriceIDsLog,  priceListxPriceIDs, X0, y,  y_vl, y_l, y_h, p_h, y__, y___,date , 'price order x1=Rating,x2=Price')
#        FittingCurveLevelRating (listProdxRatIDs, priceListxRatIDsLog, priceListxRatIDs, X0, y,  y_vl, y_l, y_h, p_h, y__, y___,date , 'rating order x1=Rating,x2=Price')        
        FittingCurveLevelPrice (listProdxPriceIDs, priceListxPriceIDsLog,  priceListxPriceIDs, X0, y,  y_vl, y_h, p_h, y__, y___,date , 'price order x1=Rating,x2=Price limit p')
        FittingCurveLevelPrice (listProdxRatIDs, priceListxRatIDsLog, priceListxRatIDs, X0, y,  y_vl, y_h, p_h, y__, y___,date , 'rating order x1=Rating,x2=Price limit p')
    
        print ('\n\nc very low=',clow,'formula 4',y_vl)
        #print ('c low=',clow,'formula 4',y_l)
        print ('c high=',chigh,'formula 4',y_h)

    ###################################################### THE SAME EXPERIMENT LIKE BEFORE BUT STARTING BY 4 SUBSET (LITTLE, MEDIUM, HIGH PRICE, RAMDOM PRICE) ###################################################
    ###################################################   FOR EXTRACTING RESULTS OF STOPPING RULE (RANDOM, BY PRICE, BY RATING) #################################################################################
    
    
        X0, X, y, y__, y___, date = DataGenTest (4)     
        X0, X = reduceMatrix (X0, X, feature_names_senti, feature_names_Dimitri)
        X = np.array(X)
        X0 = np.array(X0)
        print('X[0]',X[0],'\nX0[0]',X0[0])
        print('X0',X0)   

    
        qual_max = 1
        cost = 0.1  #cost of step
        y_l =formula4 (qual_max,cost) 
        
        #linear = True # Use Linear STOPPING RULE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        linear = True # Use ROOT SQUARE STOPPING RULE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        if (linear):
            path= 'C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result-Reputation-Linear05122019\\'
        else:
            path= 'C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result-Reputation-RSquare05122019\\'            
        
#        y_h = formula4 (qual_max,chigh)
        print ('\n\nc low=',cost,'q>',y_l)
#        print ('c high=',chigh,'q>',y_h)
        
 # tolto 'B0041686QY' , $10.1
    
        listProdLPID0s  =  ['B0041Q38NU' ,'B0043EV20Q', 'B004BFZHO4',   'B0041Q38N0', 'B0043T7FXE' ,  'B004CLYEDC','B0041OMIV0' , 'B0041G62WE', 'B0041NFIBS', \
                               'B003ZSP0WW' , 'B0049P6OTI', 'B0041Q38N0', 'B0041VZN6U', 'B00461E7JE',  'B00454YCWY'  , 'B0049V0ZX8' , 'B004EBX5GW', 'B00425S1H8' , 'B0040JHMIU',  \
                               'B0042BUXG4' ,'B0040IO1RQ', 'B004HBK4T0' ,'B0044O3PZK', 'B004D8NZ52'  ]        
        priceListLPID0s  = [10, 25, 25.1, 15,  28,  11,  10.2, 13, 13.1, 27, 25, 15.1, 18, 28, 19, 21, 11.2, 15.3, 22, \
                                6, 5, 10, 5.1, 9 ] # 25
        priceListLPID0sLog  = CreateLogFunction (priceListLPID0s)  

        
        listProdLPxRatID1s, priceListLogLPxRatID1s,  priceListLPxRatID1s, listProdLPxPriceID1s, priceListLogLPxPriceID1s, priceListLPxPriceID1s = CreateListPr_AND_Rat (listProdLPID1s, priceListLPID1s, X0, y, y__, y___,date)
        
        p_h = 20 # BUDGET CONSTRAINT
        title = '(Low Price) cost='+str(cost)+' Max qual='+str(qual_max)
        bc = ' budget const= 20$'
        print ('\n***************************** RANDOM ********************** '+title+bc)
        FittingCurveLevelRating (listProdLPID1s, priceListLPID1sLog, priceListLPID1s, X0, y,  y_l, cost, p_h, y__, y___,date , title+' 20 items random  x1=Rating,x2=Price',path)
        print ('\n***************************** BY PRICE  ********************** '+title+bc)
        FittingCurveLevelPrice (listProdLPxPriceID1s, priceListLogLPxPriceID1s,  priceListLPxPriceID1s, X0, y,  y_l, cost, p_h, y__, y___,date , title+' price order x1=Rating,x2=Price'+bc,path)
        print ('\n***************************** BY RATING   ********************** '+title+bc)
        FittingCurveLevelPrice (listProdLPxRatID1s, priceListLogLPxRatID1s, priceListLPxRatID1s, X0, y,  y_l, cost, p_h, y__, y___,date , title+' rating order x1=Rating,x2=Price'+bc,path)
        
        print ('\n\nc  low=',cost,'formula 4',y_l)
#        print ('c high=',chigh,'formula 4',y_h)
        
        
        
        

        # 'B004H912FC',  46.2    
        listProdLPID1s =  ['B00426C55G' ,'B0042X8NT6', 'B004ABO7QI', 'B00475XLOG', 'B0040VNDKO', 'B004BFXBXI' ,'B004AA4E8K' ,'B0043M668G', 'B004ALYGI2', 'B00434UCDE' , \
                               'B004071ZXA', 'B0049S6ZUS' , 'B00413PHDM', 'B0049I48JI', 'B004AM624C',  'B004FJV8EE' ,  'B0040723AO', 'B004CRSM4I', 'B004GW25WY', \
                               'B0041LYY6K', 'B00426C56U', 'B004E10KFG', 'B0040702HU', 'B0044YU60M'   ]
        priceListLPID1s = [87, 67, 50, 40, 90, 39, 35, 35.1, 49, 77, 39.1, 31, 99, 55, 38, 38.1, 30, 42, 33, \
                                42, 59, 79, 88, 65 ] # 25

        priceListLPID1sLog =  CreateLogFunction (priceListLPID1s)  
        
        listProdLPxRatID1s, priceListLogLPxRatID1s,  priceListLPxRatID1s, listProdLPxPriceID1s, priceListLogLPxPriceID1s, priceListLPxPriceID1s = CreateListPr_AND_Rat (listProdLPID1s, priceListLPID1s, X0, y, y__, y___,date)
        
        p_h = 70  # BUDGET CONSTRAINT
        title = '(Medium Low Price) cost='+str(cost)+' Max qual='+str(qual_max)
        bc = ' bc= 70$'
        FittingCurveLevelRating (listProdLPID1s, priceListLPID1sLog, priceListLPID1s, X0, y,  y_l, cost, p_h, y__, y___,date , title+' 20 items random  x1=Rating,x2=Price',path)
        FittingCurveLevelPrice (listProdLPxPriceID1s, priceListLogLPxPriceID1s,  priceListLPxPriceID1s, X0, y,  y_l, cost, p_h, y__, y___,date , title+' price order x1=Rating,x2=Price'+bc,path)
        FittingCurveLevelPrice (listProdLPxRatID1s, priceListLogLPxRatID1s, priceListLPxRatID1s, X0, y,  y_l, cost, p_h, y__, y___,date , title+' rating order x1=Rating,x2=Price'+bc,path)
        
        print ('\n\nc low=',cost,'formula 4',y_l)
#        print ('c high=',chigh,'formula 4',y_h)
        
        
        
        
        
        
        
                                
        listProdLPID1s =   ['B0042SDDXM', 'B00428N9OK' , 'B0046BTK14' , 'B0041OSQ9I', 'B0044DEDC0', 'B004G8QZPG', 'B0041OSAZ8' ,'B003ZSHNG8', 'B00429N160', 'B0044779G8' ,\
                               'B00471EYUU' , 'B004FEEY9A' , 'B004ASY5ZY' , 'B0042FZ51I' , 'B004HKIB6E', 'B0049KV50G', 'B0045371FU', 'B0040QMB8O' , 'B004A88RM6' , \
                               'B00427ZLS8' , 'B003ZX8B0U' , 'B0041OSQB6' , 'B00427ZLRO' , 'B004EPV7TK'  ]
        priceListLPID1s =  [179, 200, 174, 175, 200.1, 160, 127, 120, 159, 145, 115, 157,199, 149, 110, 165.9, 150, 140, 120, \
                                150.2, 220, 170, 249, 200.1 ] # 24

        priceListLPID1sLog = CreateLogFunction (priceListLPID1s)  
        
        listProdLPxRatID1s, priceListLogLPxRatID1s,  priceListLPxRatID1s, listProdLPxPriceID1s, priceListLogLPxPriceID1s, priceListLPxPriceID1s = CreateListPr_AND_Rat (listProdLPID1s, priceListLPID1s, X0, y, y__, y___,date)
        
        p_h = 200  # BUDGET CONSTRAINT
        title = '(Medium High Price) cost='+str(cost)+' Max qual='+str(qual_max)
        bc = ' bc= 200$'
        FittingCurveLevelRating (listProdLPID1s, priceListLPID1sLog, priceListLPID1s, X0, y, y_l, cost, p_h, y__, y___,date , title+' 20 items random  x1=Rating,x2=Price',path)
        FittingCurveLevelPrice (listProdLPxPriceID1s, priceListLogLPxPriceID1s,  priceListLPxPriceID1s, X0, y,  y_l, cost,  p_h, y__, y___,date , title+' price order x1=Rating,x2=Price'+bc,path)
        FittingCurveLevelPrice (listProdLPxRatID1s, priceListLogLPxRatID1s, priceListLPxRatID1s, X0, y,  y_l, cost,  p_h, y__, y___,date , title+' rating order x1=Rating,x2=Price'+bc,path)
        
        print ('\n\nc low=',cost,'formula 4',y_l)
#        print ('c high=',chigh,'formula 4',y_h)
                                
        
        
        
        
        
        listProdLPID1s =  ['B004FA8NOQ', 'B0044UHJWY', 'B004BLK24S', 'B0042X9LC4', 'B003ZSHNEA', 'B003ZYF3LO', 'B0041RSPRS', 'B0044UHJWY', 'B0040X4PQI', 'B0040JHVC2', \
                               'B004BQKQ8A' ,'B004EBUXHQ', 'B0047ZGIUK', 'B003ZSHNCC', 'B004G8QO8O', 'B003ZSHNE0', 'B0043M6F14', 'B004H0MQYW', 'B004FLL5AY', 'B004G8QO5C', \
                               'B0041RSPR8', 'B004FLJVXM', 'B004GK0GKO', 'B00429N16A', 'B004HO58OI' ]
        priceListLPID1s = [454 , 750, 540, 890, 850, 440, 598, 750, 1350, 541, \
                                660, 305, 314, 454.1, 330, 1600, 370, 299, 660, 270, \
                                450, 450.1, 505, 300, 599]  # 25
 
        priceListLPID1sLog = CreateLogFunction (priceListLPID1s)  
        
        listProdLPxRatID1s, priceListLogLPxRatID1s,  priceListLPxRatID1s, listProdLPxPriceID1s, priceListLogLPxPriceID1s, priceListLPxPriceID1s =  CreateListPr_AND_Rat (listProdLPID1s, priceListLPID1s, X0, y, y__, y___,date)
        
        p_h = 700 # BUDGET CONSTRAINT
        title = '(High Price)  cost='+str(cost)+' Max qual='+str(qual_max)
        bc = ' bc= 700$'
        FittingCurveLevelRating (listProdLPID1s, priceListLPID1sLog, priceListLPID1s, X0, y,  y_l, cost,  p_h, y__, y___,date , title+' 20 items random  x1=Rating,x2=Price',path)
        FittingCurveLevelPrice (listProdLPxPriceID1s, priceListLogLPxPriceID1s,  priceListLPxPriceID1s, X0, y,  y_l, cost, p_h, y__, y___,date , title+' price order x1=Rating,x2=Price'+bc,path)
        FittingCurveLevelPrice (listProdLPxRatID1s, priceListLogLPxRatID1s, priceListLPxRatID1s, X0, y,   y_l, cost,  p_h, y__, y___,date , title+' rating order x1=Rating,x2=Price'+bc,path)
        
        print ('\n\nc low=',cost,'formula 4',y_l)
#        print ('c high=',chigh,'formula 4',y_h)
        


        
        
       
#        print ('c high=',chigh,'formula 4',y_h)
       
        
        
    ###################################################### EXPERIMENTS ABOUT DEMAND CURVE, UTILITIES AND ELASTICITY ########################################################
  
    
        listProdLPID0s  =  ['B0041Q38NU' ,'B0043EV20Q', 'B004BFZHO4',   'B0041Q38N0', 'B0043T7FXE' ,  'B004CLYEDC','B0041OMIV0' , 'B0041G62WE', 'B0041NFIBS', \
                               'B003ZSP0WW' , 'B0049P6OTI', 'B0041Q38N0', 'B0041VZN6U', 'B00461E7JE',  'B00454YCWY'  , 'B0049V0ZX8' , 'B004EBX5GW', 'B00425S1H8' , 'B0040JHMIU',  \
                               'B0042BUXG4' ,'B0040IO1RQ', 'B004HBK4T0' ,'B0044O3PZK', 'B004D8NZ52'  ]        
        priceListLPID0s  = [10, 25, 25.1, 15,  28,  11,  10.2, 13, 13.1, 27, 25, 15.1, 18, 28, 19, 21, 11.2, 15.3, 22, \
                                6, 5, 10, 5.1, 9 ] # 25
        priceListLPID0sLog  = CreateLogFunction (priceListLPID0s)  

        listProdLPID2s =  ['B00426C55G' ,'B0042X8NT6', 'B004ABO7QI', 'B00475XLOG', 'B0040VNDKO', 'B004BFXBXI' ,'B004AA4E8K' ,'B0043M668G', 'B004ALYGI2', 'B00434UCDE' , \
                               'B004071ZXA', 'B0049S6ZUS' , 'B00413PHDM', 'B0049I48JI', 'B004AM624C',  'B004FJV8EE' ,  'B0040723AO', 'B004CRSM4I', 'B004GW25WY', \
                               'B0041LYY6K', 'B00426C56U', 'B004E10KFG', 'B0040702HU', 'B0044YU60M'   ]
        priceListLPID2s = [87, 67, 50, 40, 90, 39, 35, 35.1, 49, 77, 39.1, 31, 99, 55, 38, 38.1, 30, 42, 33, \
                                42, 59, 79, 88, 65 ] # 25

        priceListLPID2sLog =  CreateLogFunction (priceListLPID2s)  

        listProdLPID3s =   ['B0042SDDXM', 'B00428N9OK' , 'B0046BTK14' , 'B0041OSQ9I', 'B0044DEDC0', 'B004G8QZPG', 'B0041OSAZ8' ,'B003ZSHNG8', 'B00429N160', 'B0044779G8' ,\
                               'B00471EYUU' , 'B004FEEY9A' , 'B004ASY5ZY' , 'B0042FZ51I' , 'B004HKIB6E', 'B0049KV50G', 'B0045371FU', 'B0040QMB8O' , 'B004A88RM6' , \
                               'B00427ZLS8' , 'B003ZX8B0U' , 'B0041OSQB6' , 'B00427ZLRO' , 'B004EPV7TK'  ]
        priceListLPID3s =  [179, 200, 174, 175, 200.1, 160, 127, 120, 159, 145, 115, 157,199, 149, 110, 165.9, 150, 140, 120, \
                                150.2, 220, 170, 249, 200.1 ] # 24

        priceListLPID3sLog = CreateLogFunction (priceListLPID3s)  

        listProdLPID4s =  ['B004FA8NOQ', 'B0044UHJWY', 'B004BLK24S', 'B0042X9LC4', 'B003ZSHNEA', 'B003ZYF3LO', 'B0041RSPRS', 'B0044UHJWY', 'B0040X4PQI', 'B0040JHVC2', \
                               'B004BQKQ8A' ,'B004EBUXHQ', 'B0047ZGIUK', 'B003ZSHNCC', 'B004G8QO8O', 'B003ZSHNE0', 'B0043M6F14', 'B004H0MQYW', 'B004FLL5AY', 'B004G8QO5C', \
                               'B0041RSPR8', 'B004FLJVXM', 'B004GK0GKO', 'B00429N16A', 'B004HO58OI' ]
        priceListLPID4s = [454 , 750, 540, 890, 850, 440, 598, 750, 1350, 541, \
                                660, 305, 314, 454.1, 330, 1600, 370, 299, 660, 270, \
                                450, 450.1, 505, 300, 599]  # 25
 
        priceListLPID4sLog = CreateLogFunction (priceListLPID4s)  
        
        
   ###################################################### TRACE CURVES CURVES CURVES  ########################################################
        

        
        X0, X, y, y__, y___, date = DataGenTest (4)     
        X0, X = reduceMatrix (X0, X, feature_names_senti, feature_names_Dimitri)
        X = np.array(X)
        X0 = np.array(X0)
        print('X[0]',X[0],'\nX0[0]',X0[0])
        print('X0',X0)   

        path= 'C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result-Reputation-24122019\\'

        
        listProdIDs = listProdLPID3s  + listProdLPID2s + listProdLPID4s + listProdLPID0s   # FULL LIST
        priceLists = priceListLPID3s + priceListLPID2s + priceListLPID4s + priceListLPID0s
        priceListLogs = CreateLogFunction (priceLists) 
                
        print ('listProd', listProdIDs, 'len', len(listProdIDs), '\npriceList', priceLists, 'len', len(priceLists),'\npriceListLog', priceListLogs, 'len',  len(priceListLogs), '\n')

        #listProdsxRat, priceListLogsxRat,  priceListsxRat,  listProdsxPr, priceListLogsxPr,  priceListsxPr =     CreateListPr_AND_Rat (listProdIDs, priceLists, X0, y, y__, y___,date)            
        df = DataFrameData (listProdIDs, priceListLogs, priceLists, X0, y, y__, y___,date)    
        
        df = df.sort_values(by=['price'])
        prices = df['price'].tolist()   
        PlotDistribution(prices, 0)

        df = df.sort_values(by=['priceLog'])
        priceListLog = df['priceLog'].tolist()             
        PlotDistribution(priceListLog, 1)
        
        df = df.sort_values(by=['qualityListNormal'])
        qualityListNormal = df['qualityListNormal'].tolist()             
        PlotDistribution(qualityListNormal, 2)
        
        df = df.sort_values(by=['priceLog'])
        qualityListNormal = df['qualityListNormal'].tolist()             
        PlotBar(priceListLog,qualityListNormal,7)
        Plot1(priceListLog,qualityListNormal,7)
        print('priceListLog:',priceListLog,'\n\n')
        print('prices:',prices,'\n\n')
        print('qualityListNormal:',qualityListNormal,'\n\n')

         
#                pd.DataFrame({'rating' : ratingList, 'qualityListNormal_%v' : qualityListNormal, 'priceLog' : priceListLog, 'price' : priceList, 'qualityList' : qualityList, 'listProdIDs' : listProdIDs, '%voteList' : voteList })

        
        
         # The 6 outputs are: DataFrame.listProdIDs order by Rating, DataFrame.priceLog order by Rating, DataFrame.price order by Rating,  
         # DataFrame.listProdIDs order by price, DataFrame.priceLog order by price, DataFrame.price order by price,  

        qual_max = 1
        x_size = 23
#        cost_l = 0.1
#        cost_h = 0.15 
        cost_l = 0.08
        cost_h = 0.15 
        
        
        y_l =formula4 (qual_max,cost_l) 
        y_h =formula4 (qual_max,cost_h) 
        print ('\n\nc low=',cost_l,'formula 4',y_l)
        print ('\n\nc high=',cost_h,'formula 4',y_h)
        title = ''
        
        #degree = 1.3 # 2 OR 3 IS OK
        prices = distrPrices (priceLists, x_size);
        qty_price_level = QtyinPrices (priceLists,prices)          # Qty of product x price level, try different degree 
        print ('listProd', listProds, 'len', len(listProds), '\npriceList', priceLists, 'len', len(priceLists),'\nprice_level', prices, \
               'len',  len(prices),'\nqty_price_level', qty_price_level, 'len',  len(qty_price_level), '\n')
        #----------------------->  FARLA MEGLIO IDProdsinPrices (listProds,prices)  
        
#        priceDistr = pd.DataFrame({'listProds' : listProds, 'priceLists' : priceLists, 'price_level' : prices, 'qty_price_level' : qty_price_level })
#        priceDistr = priceDistr.sort_values(by=['priceLists'])
#        print ('priceDistr order by priceLists ',priceDistr,'\n')
#        priceDistr = priceDistr.sort_values(by=['price_level'])
#        print ('priceDistr order by price_level ',priceDistr,'\n')
#        priceDistr = priceDistr.sort_values(by=['qty_price_level'])
#        print ('priceDistr order by priceLists ',priceDistr,'\n')
#

        utilityList_ran, utilityListSR_ran, yList_ran, yListSR_ran = AnalysisDemand1 (listProds, priceListLogs, priceLists, X0, y, y_l, cost_l, y__, y___,date,x_size )  # RANDOM
        utilityList_lp, utilityListSR_lp, yList_lp, yListSR_lp = AnalysisDemand2 (listProdsxPr, priceListLogsxPr,  priceListsxPr, X0, y,  y_l, cost_l,  y__, y___,date,x_size )  #oRDER BY PRICE
        utilityList_lr, utilityListSR_lr, yList_lr, yListSR_lr = AnalysisDemand2 (listProdsxRat, priceListLogsxRat,  priceListsxRat, X0, y,  y_l, cost_l, y__, y___,date,x_size )  #oRDER BY rating
       
        utilityList_hp, utilityListSR_hp, yList_hp, yListSR_hp = AnalysisDemand2 (listProdsxPr, priceListLogsxPr,  priceListsxPr, X0, y,  y_h, cost_h,  y__, y___,date,x_size )  #oRDER BY PRICE
        utilityList_hr, utilityListSR_hr, yList_hr, yListSR_hr = AnalysisDemand2 (listProdsxRat, priceListLogsxRat,  priceListsxRat, X0, y,  y_h, cost_h, y__, y___,date,x_size )  #oRDER BY rating
        
        print ('utilityList_ran=',utilityList_ran,len(utilityList_ran),'cost=',cost_l)
        print ('utilityListSR_ran=',utilityListSR_ran,len(utilityListSR_ran),'cost=',cost_l)
        print ('utilityList_lp=',utilityList_lp,len(utilityList_lp),'cost=',cost_l)
        print ('utilityListSR_lp=',utilityListSR_lp,len(utilityListSR_lp),'cost=',cost_l)
        print ('utilityList_lr=',utilityList_lr,len(utilityList_lp),'cost=',cost_l)
        print ('utilityListSR_lr=',utilityListSR_lr,len(utilityListSR_lp),'cost=',cost_l)
        print ('utilityList_hp=',utilityList_hp,len(utilityList_hp),'cost=',cost_h)
        print ('utilityListSR_hp=',utilityListSR_hp,len(utilityListSR_hp),'cost=',cost_h)
        print ('utilityList_hr=',utilityList_hr,len(utilityList_hr),'cost=',cost_h)
        print ('utilityListSR_hr=',utilityListSR_hr,len(utilityListSR_hr),'cost=',cost_h)
        totUtiList = np.zeros(len(utilityList_lp))
        totUtiListSR = np.zeros(len(utilityList_lp))
        for i in range(0,len(utilityList_lp)):
            totUtiList[i] = (utilityList_lp[i]+utilityList_lr[i]+utilityList_hp[i]+utilityList_hr[i]+utilityList_ran[i])/5
            totUtiListSR[i] = (utilityListSR_lp[i]+utilityListSR_lr[i]+utilityListSR_hp[i]+utilityListSR_hr[i]+utilityListSR_ran[i])/4
        print ('totUtiList',totUtiList,'\n')
        print ('totUtiListSR',totUtiListSR,'\n')
        # Utilities distribution
        # chart: x-axis=x_size (segmentation of bc),  y-axis=utility

        plotting_utilities(prices,totUtiList, totUtiListSR, 2, 2,50, x_size, 'UTILITIES high prices c=0.08 0.15 spline=2', 1, path)


       
        
        ##################################################################################################################################
        # FARE SOLO UTILITY PLOT E ELASTICITY WITH RESPECT TO SOME PRODUCT
        ##################################################################################################################################
        
        # Utilities distribution
        # chart: x-axis=x_size (segmentation of bc),  y-axis=totContNormal
   
        print ('**************************  Inv Demand medium-high prices c=0.08 0.15 spline=2  ******************\n\n')
        cont_ran = QtyinPrices (yList_ran,prices)
        print ('cont_ran: ',cont_ran,'\n','yList_ran: ',yList_ran,'\n',)
        cont_SRran = QtyinPrices (yListSR_ran,prices)
        print ('cont_SRran: ',cont_SRran,'\n','yListSR_ran: ',yListSR_ran,'\n',)
        cont_lp = QtyinPrices (yList_lp,prices)
        print ('cont_lp: ',cont_lp,'\n','yList_lp: ',yList_lp,'\n',)
        cont_lr = QtyinPrices (yList_lr,prices)
        print ('cont_lR: ',cont_lr,'\n','yList_lR: ',yList_lr,'\n',)
        cont_hp = QtyinPrices (yList_hp,prices)
        cont_hr = QtyinPrices (yList_hr,prices)
        cont_SRlp = QtyinPrices (yListSR_lp,prices)
        print ('cont_SRlp: ',cont_SRlp,'\n','yListSR_lp: ',yListSR_lp,'\n',)
        cont_SRlr = QtyinPrices (yListSR_lr,prices)
        print ('cont_SRlr: ',cont_SRlr,'\n','yListSR_lr: ',yListSR_lr,'\n',)
        cont_SRhp = QtyinPrices (yListSR_hp,prices)
        cont_SRhr = QtyinPrices (yListSR_hr,prices)
        
        totCont = np.zeros(len(cont_lp))
        totContSR = np.zeros(len(cont_lp))
        for i in range(0,len(cont_lp)):
            totCont[i] = float(cont_lp[i]+cont_hp[i]+cont_lr[i]+cont_hr[i]+cont_ran[i])/5
            totContSR[i] = float(cont_SRlp[i]+cont_SRhp[i]+cont_SRlr[i]+cont_SRhr[i]+cont_SRran[i])/5
        print ('totCont',totCont,'\n')
        print ('totContSR',totContSR,'\n')
        
        # UNIFORM DISTRIBUTION
        # chart: x-axis=prices,  y-axis=totContNormal
        totContNormal = normalization(totCont)
        totContSRNormal = normalization(totContSR)
        plotting_utilities(prices,totContNormal, totContSRNormal, 2, 2, 10,  x_size, 'Inv Demand medium-high prices c=0.08 0.15 spline=2', 1, path)

         
#        bc = np.zeros(x_size)
#        maxPrice = max(priceList)
#        step = maxPrice/x_size                     # maxPrice IN THIS CASEIS THE MAX BUDGET CONSTRAINT 

       
        poisson = pmfPoisson(priceLists, 1, len(totCont))      # pOISSON DISTRIBUTION OF CONSUMERS
        for i in poisson:
            totCont[i] = totCont[i]*poisson[i]
            totContSR[i] = totContSR[i]*poisson[i]
            
  
       # POISSON DISTRIBUTION
        # chart: x-axis=prices,  y-axis=totContNormal
        totContNormal = normalization(totCont)
        totContSRNormal = normalization(totContSR)
      





 
   ###################################################   FOR EXTRACTING RESULTS OF STOPPING RULE (selection RANDOM) #################################################################################
       
        
        
        
        listProdLPID1s =  ['B0041Q38NU' ,'B0043EV20Q', 'B004BFZHO4',  'B00426C55G' ,'B0042X8NT6','B0042SDDXM', 'B00428N9OK' , 'B004FA8NOQ',  'B0041Q38N0', \
                              'B0046BTK14' , 'B0041OSQ9I' ,'B0043T7FXE' , 'B004071ZXA' , 'B004CLYEDC', 'B0044UHJWY' ,'B0040IO1RQ', 'B004HBK4T0' ,'B0044O3PZK', 'B004D8NZ52', 'B004HO58OI', \
                              'B00427ZLS8' , 'B003ZX8B0U' , 'B0041OSQB6' , 'B00427ZLRO' , 'B004EPV7TK'  ]        
        priceListLPID1s = [10, 25, 26, 87, 67, 179, 200, 454, 15, 174, 175, 28, 40, 11, 750,  5, 10, 5.1, 9, 599, \
                                150.2, 220, 170, 249, 200.1 ] # 25
 
        priceListLPID1sLog = CreateLogFunction (priceListLPID1s)  
        
        listProdLPxRatID1s, priceListLogLPxRatID1s,  priceListLPxRatID1s, listProdLPxPriceID1s, priceListLogLPxPriceID1s, priceListLPxPriceID1s = CreateListPr_AND_Rat (listProdLPID1s, priceListLPID1s, X0, y, y__, y___,date)
        
        p_h = 250 # max price
        title = '(All Price)  cost='+str(cost)+' Max qual='+str(qual_max)
        bc = ' bc= 250$'
        FittingCurveLevelRating (listProdLPID1s, priceListLPID1sLog, priceListLPID1s, X0, y,   y_l, cost,  p_h, y__, y___,date , title+' 20 items random  x1=Rating,x2=Price',path)
        FittingCurveLevelPrice (listProdLPxPriceID1s, priceListLogLPxPriceID1s,  priceListLPxPriceID1s, X0, y,  y_l, cost,  p_h, y__, y___,date , title+' price order x1=Rating,x2=Price'+bc,path)
        FittingCurveLevelPrice (listProdLPxRatID1s, priceListLogLPxRatID1s, priceListLPxRatID1s, X0, y,  y_l, cost,  p_h, y__, y___,date , title+' rating order x1=Rating,x2=Price'+bc,path)
        
        print ('\n\nc  low=',cost,'formula 4',y_l)

        
        