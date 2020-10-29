# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 00:41:27 2018

@author: paolo
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 17:19:40 2018

@author: paolo
"""

# TODO LIST:

# Such as the Training Set, get the training data with same range temporal, similar number or reviews, same neam of vote and same sd
# qs regressori  con le nuove features: https://stackoverflow.com/questions/49094242/svm-provided-a-bad-result-in-my-data-how-to-fix
# feature ranking
#http://scikit-learn.org/stable/modules/feature_selection.html 
#https://www.researchgate.net/publication/220637867_Feature_selection_for_support_vector_machines_with_RBF_kernel
#https://stats.stackexchange.com/questions/2179/variable-importance-from-svm

# https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/  ABOUT LASSO AND RIDGE + PLOTTING

# hybrid lasso 
# prediction new reviews unknown of the same product
# https://stats.stackexchange.com/questions/47771/what-is-the-intuition-behind-beta-distribution/47782#47782
# https://stats.stackexchange.com/questions/69210/drawing-from-dirichlet-distribution
# https://stats.stackexchange.com/questions/244917/what-exactly-is-the-alpha-in-the-dirichlet-distribution

# ECONOMICS IDEA?


#import os
#import contextlib
#from operator import itemgetter
#from pathlib import Path
import sys, traceback
import argparse as argsp
import pandas as pd
import numpy as np
import itertools as IT
import statistics as stat
import random as rand
import datetime
from scipy import stats
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from matplotlib.dates import MonthLocator, DateFormatter


import gc
gc.collect()
# Importing the Training dataset


dataTrain = pd.read_csv(r'C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Result15112018\Training.csv')


# Importing the Testing dataset 
dataTest = pd.read_csv(r'C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Result15112018\Test.csv')
# Importing the Testing dataset 
dataTest4 = pd.read_csv(r'C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Result15112018\Test4.csv')


def DataGenTest (numTest):
    
    if (numTest >= 1):
        X0 = dataTrain.iloc[:, 5:38].values
        print(X0)
        print(X0.shape)
        X = np.delete(X0,30,axis=1)        #column ProdID deleted
        print(X)
        print(X.shape)        
        y = dataTrain.iloc[:, 2:3].values             # HELPFUL VOTES
        #y = dataTrain.iloc[:, 1:2].values            # % HELPFUL VOTES
        #y = dataTrain.iloc[3:34000:, 3:4].values            # TOTAL VOTES
        print(y)
        print(y.shape)
        return X0, X, y
   
    if (numTest >= 2):
        # Importing the X values of Testing dataset (few votes AND context few votes window 2)
        X1_Test = dataTest.iloc[1:50, 5:38].values
        X1_Test = np.delete(X1_Test,30,axis=1)        #column ProdID deleted
        # Importing  the Y values of the Testing dataset (isolated Low voted)
        y1_Test = dataTest.iloc[1:50, 2:3].values
        #Y6=np.delete(Y6, 5, axis=0) # <------------------ THIS IS THE PROBLEM (the reviewer with 1000 reviews)
        print(y1_Test)
        print(y1_Test.shape)       
        # Importing the X values of Testing dataset (high votes AND context high votes window 2)
        X2_Test = dataTest.iloc[51:62, 5:38].values
        X2_Test = np.delete(X2_Test,30,axis=1)        #column ProdID deleted
        # Importing  the Y values of the Testing dataset (isolated Low voted)
        y2_Test = dataTest.iloc[51:62, 2:3].values
        #Y6=np.delete(Y6, 5, axis=0) # <------------------ THIS IS THE PROBLEM (the reviewer with 1000 reviews)
        print(y2_Test)
        print(y2_Test.shape)
        return X0, X, y, X1_Test, y1_Test, X2_Test, y2_Test

    
    if (numTest >= 3):
        # Importing the X values of Testing dataset  (period : 1,2,6,12,24)
        X3_Test = dataTest.iloc[63:113, 5:38].values
        X3_Test = np.delete(X3_Test,30,axis=1)        #column ProdID deleted
        # Importing  the Y values of the Testing dataset (isolated Low voted)
        y3_Test = dataTest.iloc[63:113, 2:3].values
        print(y3_Test)
        print(y3_Test.shape)
        return X0, X, y, X1_Test, y1_Test, X2_Test, y2_Test, X3_Test, y3_Test
   
    if (numTest >= 4):
        # Importing the X values of Testing dataset   (rating : 1,2,3,4,5)
        X4_Test = dataTest.iloc[114:164, 5:38].values
        X4_Test = np.delete(X4_Test,30,axis=1)        #column ProdID deleted
        # Importing  the Y values of the Testing dataset (isolated Low voted)
        y4_Test = dataTest.iloc[114:164, 2:3].values
        print(y4_Test)
        print(y4_Test.shape)

    if (numTest >= 5):    
        # Importing the X values of Testing dataset   (big Brand/little Brand)
        X5_Test = dataTest4.iloc[:, 5:38].values
        X5_Test = np.delete(X5_Test,30,axis=1)        #column ProdID deleted
        print(X5_Test)
        print(X5_Test.shape)
        # Importing  the Y values of the Testing dataset 
        y5_Test = dataTest4.iloc[:, 2:3].values
        print(y5_Test)
        print(y5_Test.shape)
    



# NO GOOD THIS ONE!!!!
#def filter_sameLen_textReview (X,y,lenRev):
#       # result = np.array(X)        # p[30]   len  , p[31]  len normal
#        lenRevfuzzy =  0.2*lenRev
#        print (lenRevfuzzy)
#        #num_rows = len(X)
#        #out = np.zeros(shape=([],32))    # numeric
#        out1 = []
#        out2 = []
#        i = 0
#        for k, h in IT.zip_longest(X, y):
##            if ((lenRev - 20) <= p[30] <= (lenRev + 20)):            
##            print ("j =", j, 'x[j] =',p, 'y[j] =',y[j])
##                k = pair[0] # X
##                h = pair[1] # y
#                #print (" k[30] ",k[30])
#                #print (" i , pair",i,pair[0],pair[1])
#                p = k[30]
#                if ((lenRev - lenRevfuzzy) <= p <= (lenRev + lenRevfuzzy)):
#                    print ('X[i][30]  =',p )
#                    #out[i] = X[i]
#                    out1.append(k)   # X
#                    out2.append(h)
#                    print ("i =", i, 'x[i] =',k, 'y[i] =',h)
#                    i=i+1
#                if i == 5:
#                   break;
#        print ("num cycle: ", i)
#        return out1, out2

# BETTER THIS ONE!!!!
# if usefulAll = 0 -> only test about the len reviews, 
# if usefulAll > 0 -> both tests: len reviews Or usefull / all > minUsefulAll
def list_sameLen_textReview(Z, indexUsAll, indexLen, lenRev, minUsefulAll): #  Z = zip_longest(X, y)
       # result = np.array(X)        # p[30]   len  , p[31]  len normal
        lenRevfuzzy =  0.25*lenRev
        print ('lenRevfuzzy =',lenRevfuzzy)
        #num_rows = len(X)
        #out = np.zeros(shape=([],32))    # numeric
        out1 = []
        out2 = []
        outstat = []
        for z in Z:
            #print ('z  =',z )
            # z = Z[i] = X[i] conc Y[i]
            p1 = z[0][indexLen]      # review len
            p2 = z[0][indexUsAll]    # 'RATIO USEFUL / ALL'
            if (minUsefulAll == 0 and ((lenRev - lenRevfuzzy) <= p1  <= (lenRev + lenRevfuzzy))):
                    #print ('X[i][indexLen]  =',p1 )
                    out1.append(z[0])   # X
                    out2.append(z[1])
                    outstat.append(p1)
            elif ((minUsefulAll > 0. and ((minUsefulAll - 0.05) <= p2  <= (minUsefulAll + 0.05))) and ((lenRev - lenRevfuzzy) <= p1  <= (lenRev + lenRevfuzzy))):
                    print ('X[i][indexLen]  =',p1 )
                    if (minUsefulAll > 0.):
                        print ('X[i][indexUsAll]  =',p2 )
                    out1.append(z[0])   # X
                    out2.append(z[1])
                    outstat.append(p1)
    #                print ('x[i] =',z[0], 'y[i] =',z[1])
    #               i=i+1
    #               if len(out1) == 5:
    #                       break;
        print ("Num lines Out X: ", len(out1))
        #print ("RATIO USEFUL / ALL: ", out2stat)  # too long
        mean = sum(outstat)/float(len(outstat))
        sd = stat.stdev(outstat)
        print ('dim Len: mean=', mean, 'sd=',sd)
        return out1, out2




def list_sameProd (Z, IndexProd, product):    #  Z = zip_longest(X0, y)  where X0 = X + 'prod column'
        out1 = []
        out2 = []
        for z in Z:
            #print ('z  =',z )
            p = z[0][IndexProd]     # old 30
            if (p == product):
                print ('product = ',p)
                out1.append(z[0])   # X
                out2.append(z[1])
                #print ('x[i] =',z[0], 'y[i] =',z[1])
        print ("num cycle: ", len(out1))
        statistic (out2)
        return out1, out2

def list_groupProds (Z, IndexProd, productlist):    #  Z = zip_longest(X0, y)  where X0 = X + 'prod column'
        out1 = []
        out2 = []
        for z in Z:
            #print ('z  =',z )
            p = z[0][IndexProd]     
            for product in productlist:
                if (p == product):
                    print ('product = ',p)
                    out1.append(z[0])   # X
                    out2.append(z[1])
        print ("num cycle: ", len(out1))
        statistic (out2)
        return out1, out2
    
# Create an agregate list of products extracting a list of name products    
def list_groupProdsPlusDate (Z, IndexProd, productlist, dateCol):    #  Z = zip_longest(X0, y)  where X0 = X + 'prod column'
        out1 = []    # X
        out2 = []    # %v+ 
        out2a = []   # 1 - %v+  (Thre arent the %v-  but the diff in % betewwn the total and the v+) 
        out2b = []   # v+
        out3 = []    # date
        row = 0
        for z in Z:
            p = z[0][IndexProd]     
            for product in productlist:
                #print ('p  =',p,'product = ',product )
                if (p == product):
                    currentDate = datetime.datetime.strptime(dateCol[row][0], '%d/%m/%Y')
                    out1.append(z[0])       # X
                    out2.append(z[1][0])    # %v+
                    if (z[3][0] == 0):      # vtot
                        vminus = 0.             # %v-
                    else:
                        vminus = 1 - z[1][0]    # %v- = 1 - %v+
                    out2a.append(vminus)   
                    out2b.append(z[2][0])   # v+
                    #print ('product = ',p, '%v+',z[1][0], '1-%v+',vminus,'currentDate ', currentDate)
                    out3.append(dateCol[row])   # date
            row = row +1
        print ("num cycle: ", len(out1))
        #statisticSimple (out2, '%vote+')
        print ('productlist',productlist)
        statisticSimple (out2,'')
        return out1, out2, out2a, out2b, out3   # X, %v+ , 1 - %v+, v+, date

# index win feature: 7, index win2 feature 12, ...
        
def listValuesProdPlusDate (Z, IndexProd, IndexRating, IndexReviewer, IndexWinCont, product, dateCol):    #  Z = zip_longest(X0, y)  where X0 = X + 'prod column'
        out1 = []    # X
        out2 = []    # %v+ 
        out2a = []   # 1 - %v+  (Thre arent the %v-  but the diff in % betewwn the total and the v+) 
        out2b = []   # v+
        out3 = []    # date
        out4 = []    # rating
        out5 = []    # repu reviewer
        out6 = []    # win context
        row = 0
        for z in Z:
            p = z[0][IndexProd]     
            #print ('p  =',p,'product = ',product )
            if (p == product):
                currentDate = datetime.datetime.strptime(dateCol[row][0], '%d/%m/%Y')
                if (z[3][0] > 0):          # ONLY WHEN vtot > 0
                     vminus = 1 - z[1][0]    # %v- = 1 - %v+
                     out1.append(z[0])           # X
                     out2.append(z[1][0])        # %v+
                     out2a.append(vminus)   
                     out2b.append(z[2][0])   # v+
                     #print ('product = ',p, '%v+',z[1][0], '1-%v+',vminus,'currentDate ', currentDate)
                     out3.append(dateCol[row])   # date
                     r = z[0][IndexRating]     
                     w1 = z[0][IndexReviewer]     
                     w2 = z[0][IndexWinCont]     
                     out4.append(float(r))
                     out5.append(float(w1))
                     out6.append(float(w2))
                row = row +1
        print ("num cycle: ", len(out1))
        #statisticSimple (out2, '%vote+')
        print ('product',product)
        statisticSimple (out2,'')
        return out1, out2, out2a, out2b, out3, out4, out5, out6   # X, %v+ , 1 - %v+, v+, date, rating, repu reviewer, win context


def list_sameRating (Z, IndexRating, ratingValue):    #  Z = zip_longest(X, y)  
        out1 = []
        out2 = []
        for z in Z:
            #print ('z  =',z )
            r = z[0][IndexRating]     
            if (r == ratingValue):
                print ('ratingValue = ',r)
                out1.append(z[0])   # X
                out2.append(z[1])
 #               print ('x[i] =',z[0], 'y[i] =',z[1])
        print ("num cycle: ", len(out1))
        return out1, out2

def list_LittleRating (Z, IndexRating):    #  Z = zip_longest(X, y)  
        out1 = []
        out2 = []
        for z in Z:
            #print ('z  =',z )
            p = z[0][IndexRating]     
            if (p <= 2):
                print ('ratingValue = ',p, '%Hvotes = ', z[1])
                out1.append(z[0])   # X
                out2.append(z[1])
 #               print ('x[i] =',z[0], 'y[i] =',z[1])
        print ("num cycle: ", len(out1))
        statistic (out2)
        return out1, out2
    
    
def list_HighRating (Z, IndexRating):    #  Z = zip_longest(X, y)  
    out1 = []
    out2 = []
    for z in Z:
        #print ('z  =',z )
        p = z[0][IndexRating]     
        if (p >= 4):
            #print ('ratingValue = ',p)
            out1.append(z[0])   # X
            out2.append(z[1])
 #               print ('x[i] =',z[0], 'y[i] =',z[1])
    print ("num cycle: ", len(out1))
    return out1, out2

    
def list_smallVotes (Z, maxVote):    #  Z = zip_longest(X, y, y__)  where  # y = % HELPFUL VOTES, y__ # HELPFUL VOTES
    out1 = []
    out2 = []
    outstat = []
    for z in Z:
        v = z[2]       # HELPFUL VOTES
        if (v <= maxVote):
            print ('helpfVote = ',v)
            out1.append(z[0])   # X
            out2.append(z[1])   # y
            outstat.append(z[1])
    print ("num cycle: ", len(out1))
    statistic (outstat)
    return out1, out2
    
def list_percentVotes (Z, percVote):    #  Z = zip_longest(X, y, y__)  where  # y = % HELPFUL VOTES, y__ # HELPFUL VOTES
    out1 = []
    out2 = []
    outstat = []
    for z in Z:
        v = z[1]       # %HELPFUL VOTES
        if (percVote - 0.05 <= v <= percVote + 0.05):
            print ('%helpfVote = ',v)
            out1.append(z[0])   # X
            out2.append(z[1])   # y
            outstat.append(z[1])
    print ("num cycle: ", len(out1))
    statistic (outstat)
    return out1, out2
    
    
def list_largeVotes (Z, minVote):    #  Z = zip_longest(X, y, y__)  where  # y = % HELPFUL VOTES, y__ # HELPFUL VOTES
    out1 = []
    out2 = []
    outstat = []
    for z in Z:
        v = z[2]       # HELPFUL VOTES
        if (v >= minVote):
            print ('helpfVote = ',v, '%helpfVote = ',z[1])
            out1.append(z[0])   # X
            out2.append(z[1])   # y
            outstat.append(z[1])
    print ("num cycle: ", len(out1))
    statistic (outstat)
    return out1, out2

def statistic (outstatArray):
    outstat = np.array([i[0] for i in outstatArray])
    print ('len(outstat):',len(outstat),'outstat: ', outstat)
    mean = sum(outstat)/float(len(outstat))
    sd = stat.stdev(outstat)
    print ('*********************** %helpfVote : mean=', '{0:.3f}'.format(mean), 'sd=','{0:.3f}'.format(sd),'**********************\n\n')

def statisticSimple (outstat, name):
    #print ("outstat: ", outstat)
    mean = sum(outstat)/float(len(outstat))
    sd = stat.stdev(outstat)
    print ('*********************** '+name+ ' : mean=', '{0:.3f}'.format(mean), 'sd=','{0:.3f}'.format(sd),'**********************\n\n')
    return mean, sd

def statisticSimple1 (outstat, name):
    #print ("outstat: ", outstat)
    mean = sum(outstat)/float(len(outstat))
    sd = stat.stdev(outstat)
    zscore = stats.zscore(outstat)
    print ('*********************** '+name+ ' : mean=', '{0:.3f}'.format(mean), ' sd=','{0:.3f}'.format(sd),'**********************\n\n')
    return mean, sd, zscore

def list_sameColumn (Z, IndexColumn, colValue, toleranceYN, feature_names):   
        out1 = []
        out2 = []
        tolerance =  0.15*colValue
        print ('value =',colValue,' tolerance =',tolerance, ' of ',feature_names[IndexColumn] )
        for z in Z:
            p = z[0][IndexColumn]     
            if (toleranceYN):
                if ((colValue - tolerance) <= p <= (colValue + tolerance)):                
                    out1.append(z[0])   # X
                    out2.append(z[1])
#                    print ('# x[i] =',z[0][IndexColumn], 'y[i] =',z[1])
            else:
                if (p == colValue): 
                    out1.append(z[0])   # X
                    out2.append(z[1])
                    print ('# x[i] =',z[0][IndexColumn], 'y[i] =',z[1])
#        print ("num cycle: ", len(out1))
        print ("num cycle: ", len(out1))
        return out1, out2




def mean_sd_ofX_i (Z, i):   
        out1 = []
        out2 = []      
        j = 0
        for z in Z:
            p = z[0]
            if (j == i):
                print ("i,j prod Rif",i,j,p)
                mean = sum(p)/float(len(p))
                sd = stat.stdev(p)
                print ("mean = ",mean, "sd =",sd)
                break
            j+=1           
        return out1, out2


# Test: mean_sd_ofX_i (IT.zip_longest(X, y), 3)

def iterative_Index (test):
        i1 = 0
        index1 = []
        for t in test:
            index1.append(i1)
            i1+=1
        return index1

def randomize (U,rangeOut):
    rand.shuffle(U)
    u = U[:rangeOut]
    return u
    

def normal (x, typeNorm):
    if typeNorm == 1:
        norm = x / np.linalg.norm(x)
    elif typeNorm == 2:
        norm = normalize(x[:,np.newaxis], axis=0).ravel()
    return norm


def similar_reviews (X, y, meanThis, sdThis, PeaksPercentThis, rangeOut, normalYN):   
        out1 = []
        out2 = []      
        test = []      
        threshold0 = PeaksPercentThis[0]
        threshold1 = PeaksPercentThis[1]
        Z = IT.zip_longest(X, y)
        for z in Z:
            test.append(z[1][0])
        j = 0
        indexes = iterative_Index (test)   # all index training set     
        PeakValues = [(meanThis, 3)[meanThis >= 5], 20, 50]
#        rangeOut = 200
        if (normalYN):
            indexesNorm = normal (indexes, 1)
        lookUp = dict(zip(indexesNorm,indexes))
        while True:
#           k = randomize (test,rangeOut)
#           print ("k : ",k)
            if (normalYN):
                kindexesNorm = randomize (indexesNorm,rangeOut)     # subset randomize of dim rangOut
                #print ("kindexesNorm : ",kindexesNorm)
                kindexes = []
                for key in kindexesNorm:
                    kindexes.append(lookUp[key])    
            else:
                kindexes = randomize (indexes,rangeOut)     # subset randomize of dim rangOut
            print ("kindexes : len=", len(kindexes), kindexes)
            k = []
            for kindex in kindexes:               
#                print ('kindex',kindex)
#                print ('y[kindex]',y[kindex])
#                print ('X[kindex]',X[kindex])
                k.append(y[kindex][0])              
            print ('k',k)
            mean = sum(k)/float(len(k))
            sd = stat.stdev(k)
            k1 = [i for i in k if ( i <= (PeakValues[0]+(PeakValues[0]*0.2)))]
            k1lenPerct = len(k1)/float(rangeOut)
            k2 = [i for i in k if ((PeakValues[1]+(PeakValues[1]*0.2)) >= i >= (PeakValues[1]-(PeakValues[1]*0.2)))]
            k3 = [i for i in k if i >= PeakValues[2]]
            k3lenPerct = len(k3)/float(rangeOut)
            print ("PeakValues : ",PeakValues)
            print ("mean = ",mean, "sd =",sd, "low values =",k1,"medium values =",k2,"high values =",k3,"k1lenPerct ",k1lenPerct, ">=",threshold0," k3lenPerct ",k3lenPerct, ">=",threshold1)
            j+=1
            if ((mean - (0.1*mean) <= meanThis <= mean + (0.1*mean)) and (sd  <= sdThis ) and (k1lenPerct >= threshold0) and (k3lenPerct >= threshold1)):
                for kindex in kindexes:               
                    print ('kindex',kindex)
                    print ('y[kindex]',y[kindex])
                    print ('X[kindex]',X[kindex])
                    out1.append(X[kindex])   # X              
                    out2.append(y[kindex])   #y
                print ("number cycles = ",j)
                return out1, out2



def numPeaks_reviews (X, y, PeaksPercentThis, rangeOut, normalYN):   
        out1 = []
        out2 = []      
        test = []      
        threshold1 = PeaksPercentThis[1]    # only high values
        Z = IT.zip_longest(X, y)
        for z in Z:
            test.append(z[1][0])
        j = 0
        indexes = iterative_Index (test)   # all index training set     
        PeakValues = [3, 20, 50]
#        rangeOut = 200
        if (normalYN):
            indexesNorm = normal (indexes, 1)
        lookUp = dict(zip(indexesNorm,indexes))
        while True:
#           k = randomize (test,rangeOut)
#           print ("k : ",k)
            if (normalYN):
                kindexesNorm = randomize (indexesNorm,rangeOut)     # subset randomize of dim rangOut
                #print ("kindexesNorm : ",kindexesNorm)
                kindexes = []
                for key in kindexesNorm:
                    kindexes.append(lookUp[key])    
            else:
                kindexes = randomize (indexes,rangeOut)     # subset randomize of dim rangOut
            print ("kindexes : len=", len(kindexes), kindexes)
            k = []
            for kindex in kindexes:               
#                print ('kindex',kindex)
#                print ('y[kindex]',y[kindex])
#                print ('X[kindex]',X[kindex])
                k.append(y[kindex][0])              
            print ('k',k)
            mean = sum(k)/float(len(k))
            sd = stat.stdev(k)
            k3 = [i for i in k if i >= PeakValues[2]]
            k3lenPerct = len(k3)/float(rangeOut)
            print ("PeakValues : ",PeakValues)
            print ("mean = ",mean, "sd =",sd, "high values peak =",k3, " k3lenPerct ",k3lenPerct, ">=",threshold1)
            j+=1
            if (k3lenPerct >= threshold1):
                for kindex in kindexes:               
                    print ('kindex',kindex)
                    print ('y[kindex]',y[kindex])
                    print ('X[kindex]',X[kindex])
                    out1.append(X[kindex])   # X              
                    out2.append(y[kindex])   #y
                print ("number cycles = ",j)
                return out1, out2




def get_data_split(X,y,t_size):
    return train_test_split(X, y, test_size=t_size, random_state=0)
#X_train,X_test,y_train,y_test=train_test_split(X,Y,..)
    


# CROSS VALIDATION OF FITTED MODELS
def tune_models_hyperparams(X, y, X_test, y_test, models, **common_grid_kwargs):        #  **common_grid_kwargs  variable num of arg
    grids = {}                                                          # empty lists/dicts
    for model in models:
        print('{:-^70}'.format(' [' + model['name'] + '] '))
        pipe = Pipeline(steps=[                                           # sequence of operations Standar Scale transform your data in noemal distribution will have a mean value 0 and standard deviation of 1. G
                    ("scale", StandardScaler()),
                    (model['name'], model['model'])   ])   
        clf = GridSearchCV(pipe, param_grid=model['param_grid'], **common_grid_kwargs)
        try:
            print (' feature_importances_ = ',pipe.steps[1][1].best_estimator_.feature_importances_ )
            #print (' 1111 feature_importances_ = ',modelObject.best_estimator_.feature_importances_)
        except AttributeError:
            #print (' No feature_importances_  for ',model['name'])       
            print (' find a correct way to extract feature_importances_  for ',model['name'])       
        grids[model['name']] = clf.fit(X, y)
        train_score=clf.score(X,y)
        test_score=clf.score(X_test, y_test)
        best_params = clf.best_params_        
        print (' train_score = ',train_score,' test_score = ',test_score, ' best_params= ',best_params)
# sys.exit(0) # DEBUG!!!
#        grids[model['train_score']] = train_score
#        grids[model['test_score']] = test_score
        model['train_score'] = train_score
        model['test_score'] = test_score
        for key, value in best_params.items():
            #print (key,value)
            model['best_params'][key] = value
        print ('model[best_params]',model['best_params'])       
        #break;  # DEBUG!!!       
        # saving single trained model ...
        joblib.dump(grids[model['name']], './{}.pkl'.format(model['name']))
    return grids


# Choose the best model with respect to "Mean squared error regression loss"
def get_best_model(typeExp, grid, X_test, y_test, metric_func=mean_squared_error):
    
    res = {name : round(metric_func(y_test, model.predict(X_test)), 3)
           for name, model in grid.items()}       
    best_model_name = min(res, key=res.get)         # <------------------- good! THIS IS TNE MIN
    print('best_model_name ',best_model_name, 'typeExp: ',typeExp)
    print('*' * 70)
    print('Mean Squared Error:', res, 'typeExp: ',typeExp, ' dim Test Set', X_test.shape)
#    best_model_name = min(res, key=itemgetter(1))  # <------------------ WRONG !!!! IT'S NOT THE MIN 
    for name, model in grid.items():
        if name == best_model_name:
            print ('model chosen: ',model)
            return model
#    return best_model_name
#    return model

def locate_min(a):
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a) 
                      if smallest == element]
    
    
def test_dataset(typeExp, grid, X_test, y_test, X_train):
    res = {}
    for name, model in grid.items():
        y_pred = model.predict(X_test)
        res[name] = {'typeExp:': typeExp, 'name model: ':name, 'MSE': '{0:.3f}'.format(mean_squared_error(y_test, y_pred)), # average squared difference between the estimated values and what is estimated
                       'R2': '{0:.3f}'.format(r2_score(y_test, y_pred)), 'dim Train Set (CV split)': X_train.shape       # R2 coefficient of determination is a statistical measure of how well the regression predictions approximate the real data points. An R2 of 1 indicates that the regression predictions perfectly fit the data.
                      }
        print (res[name]) 
    return res

def predict(grid, X_test, model_name):
    return grid[model_name].predict(X_test)


 # print best score of cross validation and best params..
def print_grid_results(grids, typeExp):
    #typeExp = 'prova'
    print('{:#^120}'.format(' ' + typeExp + ' '))
    for name, model in grids.items():
        print('{:-^70}'.format(' [' + name + '] '))
        print('Total Score:\t\t{:.2%}'.format(model.best_score_))
#        print('Train Score:\t\t{:.2%}'.format(model['train_score']))
#        print('Test Score:\t\t{:.2%}'.format(model['test_score']))
        print('Parameters:\t{}'.format(model.best_params_))
        print('*' * 70)



# feature extraction
#Statistical tests can be used to select those features that have the strongest relationship with the output variable.
#The example below uses the chi squared (chi^2) statistical test for non-negative features to select 'numExtract' of the best features 

def featureExtract1(X,y,numExtract, feature_names):
    print('{:#^120}'.format(' FEATURE EXTRACT: chi squared (chi^2) statistical test '))
    b1 = SelectKBest(score_func=chi2, k=numExtract)    # es k=4
    featureExtract_corpus(X,y,b1,feature_names,'score_func=chi2')
    b2 = SelectKBest(f_classif, k=numExtract)    # es k=4
    featureExtract_corpus(X,y,b2,feature_names,'ANOVA')
       
    
def featureExtract_corpus(X_train,y_old,b,feature_names,typeSelect):
#    X_train = b.fit_transform(X, y)
#    print('1) Dim feature selection matrix X ',np.array(X_train).shape)
#    mask = b.get_support() #list of booleans
#    new_features = [] # The list of your K best features
#    for bool, feature in zip(mask, feature_names):
#        if bool:
#            new_features.append(feature)
#    print(' Selected features SelectKBest ('+typeSelect+')', new_features)    
    #X_new = b.fit_transform(X, y)
    
    print (' y = ',y_old)
#        y = np.array(y_old)  
    y = np.asarray(y_old, dtype="|S6")
    X_new = b.fit_transform(X_train,y)
    #X_newTest = b.transform(X_test,y)
    X_new =np.array(X_new);
    print (' X_new[0] = ',X_new[0])
    print('Dim feature selection matrix X_new ',X_new.shape)
#    if (alreadyFit):    # SelectKBest has no attribute 'threshold_'
#        print(' threshold= ',b.threshold_)
    mask = b.get_support() #list of booleans
    new_features = [] # The list of your K best features
    origin_features = [] 
    for bool, feature in zip(mask, feature_names):
        origin_features.append(feature)
        if bool:
            new_features.append(feature)
    print(' \nSelected features Select ('+typeSelect+')', new_features)
    print(' \nOriginal features ', origin_features)
    return X_new



    # see also SVM-RFE https://medium.com/@aneesha/recursive-feature-elimination-with-scikit-learn-3a2cbdf23fb7
def featureExtract2(X,y,model,numExtract, feature_names):   #TODO LIST Bug
    rfe = RFE(model, numExtract, step=1)
    fit = rfe.fit(X, y)
    print("Num Features: %d") % fit.n_features_
    print("Selected Features: %s") % fit.support_
    print("Feature Ranking: %s") % fit.ranking_

    pipeline = Pipeline([
        ('rfe_feature_selection', rfe), 
        ('clf', model)
        ])
    pipeline.fit(X, y)
    mask = pipeline.named_steps['rfe_feature_selection'].support_
    new_features = [] # The list of your K best features
    for bool, feature in zip(mask, feature_names):
        if bool:
            new_features.append(feature)
    print(' Selected features RFE ', new_features)
    print(' Ranking features RFE ', pipeline.named_steps['rfe_feature_selection'].ranking_)
    

def predict_save_results (X_Test, y_Test, X, y, best_model, typeTest, typeExp, pathfile):
    
#    X = np.array(X_Test)
#    y = np.array(y)    
    print(typeExp, 'Dim training matrix X', np.array(X).shape, 'Dim training matrix y', np.array(y).shape)
    print(typeExp, 'Dim test matrix X', np.array(X_Test).shape, 'Dim test matrix y', np.array(y_Test).shape)
    
    regressor = best_model
    print('best_estimator ',regressor.best_estimator_)
    print('best_params ',regressor.best_params_)
    name_model = regressor.best_estimator_.steps[1][1].__class__.__name__
    print ('name model '+name_model)   
    
    
    #regressor.set_params(**best_params)               # check this and adding total score (also in the csv name)
    #model = regressor.fit(X,y);
    result = np.array(X_Test)
    num_rows, num_cols = result.shape
    out = np.zeros(shape=(num_rows+4,2))    # numeric
    #out = np.chararray((num_rows+4, 2))       # string
    i = 0;
    #float_formatter = lambda x: "%.2f" % x
    for p in result:
        p = p.reshape(1, -1)
    #    Y1_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(p)))   #Using Feature scaling
        y_item_pred = regressor.predict(p)
        print (' predicted {:.6}'.format(str(y_item_pred)), ' real {:.4}'.format(str(y_Test[i])))
        out[i] = [y_item_pred,y_Test[i]]    
        i = i+1   
    y_pred = regressor.predict(X_Test)
    train_score=regressor.score(X,y)
    train_scoreStr = '{0:.2f}'.format(train_score);
    test_score=regressor.score(X_Test,y_Test)    
    regressor
    MSE =  mean_squared_error(y_Test, y_pred)
    R2 = r2_score(y_Test, y_pred) 
    print (" Result final prediction on Test "+typeTest)
    print ("Train_score: ", '{:.2f}'.format(train_score), " test_score: ", '{:.2f}'.format(test_score), ' MSE= {:.2f}'.format(MSE), ' R2= {:.2f}'.format(R2))
    #out[i+1] = [,]
    out[i+2] = [train_score,test_score]    
    out[i+3] = [MSE,R2]  # out[i+3] = ['MSE= {0:.2f}'.format(MSE),' R2= {0:.2f}'.format(R2)]    
    #print ('model1 '+regressor.best_estimator_.steps[0][1])
    #fmt='%s, %s'
    #with open(r"C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Result15112018\Result"+"."+name_model+"."+train_scoreStr+"."+typeTest+"."+typeExp+".csv", 'wb') as f:np.savetxt(f,out,delimiter=",",fmt='%.2f')  #
    with open(pathfile+"."+name_model+"."+train_scoreStr+"."+typeTest+"."+typeExp+".csv", 'wb') as f:np.savetxt(f,out,delimiter=",",fmt='%.3f')  #
 

def predict_results (X_Test, y_test, X, y,  best_model, typeExp):
    
#    X = np.array(X_Test)
#    y = np.array(y)    
    print(typeExp, 'Dim training matrix X', np.array(X).shape, 'Dim training matrix y', np.array(y).shape)
    print(typeExp, 'Dim test matrix X', np.array(X_Test).shape)    
    regressor = best_model
    print('best_estimator ',regressor.best_estimator_)
    print('best_params ',regressor.best_params_)
    name_model = regressor.best_estimator_.steps[1][1].__class__.__name__
    print ('name model '+name_model)   
    
    #regressor.set_params(**best_params)               # check this and adding total score (also in the csv name)
    #model = regressor.fit(X,y);
    result = np.array(X_Test)
    num_rows, num_cols = result.shape
#    i = 0;
#    for p in result:
#        p = p.reshape(1, -1)
##        Y1_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(p)))   #Using Feature scaling
#        y_item_pred = regressor.predict(p)
#        print ('y predicted',y_item_pred,' y_test',y_test[i])
#        i = i+1   
    y_pred = regressor.predict(X_Test)
    y_pred = np.array(y_pred)  
#    print ('y pred list',y_pred)
#    print ('y_test',y_test)    
    if (len(y_test) > 0):
        df = pd.DataFrame({'y pred list' : y_pred, 'y_test' : y_test })
        print (df)
    train_score=regressor.score(X,y)
    print ('train_score', '{0:.2f}'.format(train_score));
    return y_pred



def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]


print(float(1)/50000)
# ADD ADABOOST TO MODELS!!!!!!!!!!!!!!
# https://xavierbourretsicotte.github.io/AdaBoost.html

models = [
    {   'name':     'RandomForest',
        'model':    RandomForestRegressor(criterion="mse"),  # default criterion="mse", "friedman_mse", "mae" others: entropy or gini for Random Forest Classifier, Not for regression: it's need implementation 
        'title':    "RandomForestRegressor",
        'best_params': {     # best params chosen
#            'RandomForest__n_estimators':           0,
#            'RandomForest__max_depth':              0,
        },
        'train_score' : 0,     # output score with best params
        'test_score'  : 0,
        'param_grid':  {
#            'RandomForest__n_estimators':   [50, 250, 500], <----- TYPICAL PARAMS
#            'RandomForest__max_depth':      [5, 8, 10, 15]
            'RandomForest__n_estimators':   [500, 1000],
            'RandomForest__max_depth':      [15, 18]            
        } 
    },
    {   'name':     'SVR_rbf',
        'model':    SVR(kernel='rbf'),
        'title':    "SVR_rbf", 
        'best_params': {     # best params chosen
            'SVR_rbf__C':           0,
            'SVR_rbf__max_iter':    0,
            'SVR_rbf__gamma':       0,
            'SVR_rbf__cache_size':  1000
         },
        'train_score' : 0,     # output score with best params
        'test_score'  : 0,
        'param_grid': {
            'SVR_rbf__C':           [100.0, 400.0, 800.0],
            'SVR_rbf__max_iter':    [2000],
            'SVR_rbf__gamma':       [0.1, 0.05, 0.02],
            'SVR_rbf__cache_size':  [1000]
         } 
    },
    {   'name':     'SVR_linear',
        'model':      SVR(kernel='linear'),
        'title':    "SVR_rbf",
        'best_params': {     # best params chosen
#            'SVR_linear__C':           0,
#            'SVR_linear__max_iter':    5000,
#            'SVR_linear__cache_size':  1000
        },
        'train_score' : 0,     # output score with best params
        'test_score'  : 0,
        'param_grid': {
            'SVR_linear__C':           [0.01, 0.1, 1, 5, 100.0, 400.0, 800.0],
            'SVR_linear__max_iter':    [5000],
            'SVR_linear__cache_size':  [1000]
        } 
    },
    {   'name':     'Ridge',
        'model':    Ridge(),
        'title':    "Ridge",
        'best_params': {     # best params chosen
#            'Ridge__alpha':           0,
#            'Ridge__max_iter':        0,
#            'Ridge__normalize':       True
        },
        'train_score' : 0,     # output score with best params
        'test_score'  : 0,
        'param_grid':  {
#            'Ridge__alpha':         [1e-15, 1e-10, 1e-8, 1e-4, 0.1, 0.5, 5, 10, 50, 100],
            'Ridge__alpha':         [1e-15, 1e-10, 1e-8, 1e-4, 0.1, 0.5, 5],
            'Ridge__max_iter':      [10e4],
            'Ridge__normalize':     [True]
        } 
    },
    {   'name':     'Lasso',
        'model':    Lasso(),
        'title':    "Lasso",
        'best_params': {     # best params chosen
#            'Lasso__alpha':           0,
#            'Lasso__max_iter':        0,
#            'Lasso__normalize':       True
        },
        'train_score' : 0,     # output score with best params
        'test_score'  : 0,
        'param_grid':  {
#            'Lasso__alpha':         [1e-15, 1e-10, 1e-8, 1e-4, 0.1, 0.5, 5, 10, 50, 100],
            'Lasso__alpha':         [1e-15, 1e-10, 1e-8, 1e-4, 0.1, 0.5, 5],
            'Lasso__max_iter':      [10e4],
            'Lasso__normalize':     [True]
         } 
    },
    {   'name':     'ElasticNet',
        'model':    ElasticNet(),
        'title':    "ElasticNet",
        'best_params': {     # best params chosen
#            'Lasso__alpha':           0,
#            'Lasso__max_iter':        0,
#            'Lasso__normalize':       True
        },
        'train_score' : 0,     # output score with best params
        'test_score'  : 0,
        'param_grid':  {
#            'Lasso__alpha':         [1e-15, 1e-10, 1e-8, 1e-4, 0.1, 0.5, 5, 10, 50, 100],
            'ElasticNet__alpha':         [1, 0.1, 1e-4, float(1)/50000],
            'ElasticNet__l1_ratio':      [0.5, 0.7, 0.3],
            'ElasticNet__max_iter':      [10e4],
            'ElasticNet__normalize':     [True],
            'ElasticNet__fit_intercept': [True]
          } 
    },
]

feature_names = ['RATING','SAMPLE ENT MAX1','SAMPLE ENT MAX2','RATIO USEFUL / ALL','DENS POS:NVJ','# REVIEWS x PROD','# REVIEWS x REVIEWER', '% VOTES x REVIEWER', 
             '# (MEAN H VOTES x reviews) x REVIEWER', 'CURRENT LIFETIME PROD (month)', 'FULL LIFETIME PROD  (month)', '# H VOTES/# T VOTES x PROD', 'MEAN H VOTES x CONTEXT (WIN = 2)', 
             'COEFF VARIAT H VOTES x CONTEXT (WIN = 2)', 'DIFF H (CURRENT VOTE' , 'MEAN VOTES) (WIN = 2)', 'MEAN H VOTES x CONTEXT (WIN = 2 LEFT)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 2 LEFT)', 
             'DIFF (CURRENT VOTE , MEAN VOTES) (WIN = 2 LEFT)', 'MEAN H VOTES x CONTEXT (WIN = 3)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 3)', 'DIFF (CURRENT VOTE ', 'MEAN VOTES) (WIN = 3)', 
             'MEAN H VOTES x CONTEXT (WIN = 3 LEFT)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 3 LEFT)', 'DIFF (CURRENT VOTE , MEAN VOTES) (WIN = 3 LEFT)', 'MEAN H VOTES x CONTEXT (WIN = 4)', 
             'COEFF VARIAT H VOTES x CONTEXT (WIN = 4)',' DIFF (CURRENT VOTE , MEAN VOTES) (WIN = 4)', 'MEAN H VOTES x CONTEXT (WIN = 4 LEFT)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 4 LEFT)', 
             'DIFF (CURRENT VOTE , MEAN VOTES) (WIN = 4 LEFT)','LENGTH REVIEW','LENGTH NORMAL REVIEW']

def core(X,y,typeExp):
    X = np.array(X)
    y = np.array(y)    
    print(typeExp, 'Dim input matrix X', X.shape, 'Dim input matrix y', y.shape)
    X_train, X_test, y_train, y_test = \
        get_data_split(X,y,0.2)  # X_train, X_test, y_train, y_test of CROSS VALIDATION PROCESS (all data extracted from primitive training X,y)
    grid = tune_models_hyperparams(X_train, y_train, X_test, y_test, models, cv=3,  # cross-validation of 
                                   verbose=2, n_jobs=-1)
    print_grid_results(grid,typeExp)                        # print best score of cross validation and best params..
    best_model = get_best_model(typeExp, grid, X_test, y_test)    # best model name with respect to Mean squared error regression loss    
    test_dataset(typeExp, grid, X_test, y_test, X_train)
    featureExtract1(X_train,y_train,5,feature_names)
    featureExtract2(X_train,y_train,best_model,5,feature_names)
    return best_model


def whiteBlock ():
    return 0


# When a python program is executed, directly by the interpreter,  python interpreter starts executing code inside it
def main(args) : 
    
    # PROVE  INPUT: 2 --opt_arg 3 --predict
#    print(args.opt_pos_arg)
#    print(args.opt_arg)
#    print(args.switch)
#    exit;
    
    choice = args.input;
    print("Argument values:", choice )
    if choice == 'general_test':    # INPUT:  'general_test'--predict
       X0, X, y, X1_Test, y1_Test, X2_Test, y2_Test = DataGenTest (3)
       best_model_name = core(X,y,'general_test')
    elif choice == 'len_test':           # 1 Experiment : training Set with reviews same review length and others...
       X0, X, y = DataGenTest (2)   
       newX, newy = list_sameLen_textReview (IT.zip_longest(X, y), 3, 30, 40, 0)
       best_model_name = core(newX,newy,'len_test: 40 ')
        
#       newX, newy = list_sameLen_textReview (IT.zip_longest(X, y),200, 0.6)
#       best_model_name = core(newX,newy,'len_test: 200')
#       newX, newy = list_sameLen_textReview (IT.zip_longest(X, y),500, 0.6)
#       best_model_name = core(newX,newy,'len_test: 500')

    elif choice == 'product_test':      # 2 Experiment : training Set with reviews same product...
       prodID = 'B000A6PPOK' #  B000LRMS66 Garmin, B000A6PPOK microsoft, B000JE7GPY  Belkin
       X0, X, y = DataGenTest (2)   
       newX, newy = list_sameProd (IT.zip_longest(X0, y),30,prodID,feature_names)
       newX = np.delete(newX,30,axis=1)
       print(newX.shape)
       core(newX,newy,'product_test: '+prodID)

    elif choice == 'slope_test':        # 3 Experiment : training Set with reviews similar slope...
       X0, X, y = DataGenTest (2)   
       newX, newy = similar_reviews (X, y, 2, 5, [0.9, 0.0], 250, True)     # THE BEST mse
       core(newX,newy,'similar slope: mean=2.5 sd<7 200dim')
       newX, newy = similar_reviews (X, y, 3, 15, [0.9, 0.01], 300, True)
       core(newX,newy,'similar slope: mean = 3 sd<15 250dim')
       newX, newy = similar_reviews (X, y, 6, 25, [0.9, 0.01], 200, True)
       core(newX,newy,'similar slope: mean=6  sd<25 250dim')
       newX, newy = similar_reviews (X, y, 2.5, 20, [0.9, 0.01], 500, True)
       core(newX,newy,'similar slope: mean = 3 No constraint sd')

    else:                                # 4 Experiment : training Set with reviews similar slope.. restricted to a number of limited product
         X0, X, y = DataGenTest (2)    
#        bigX = []
#        bigy = []
        #newX, newy = list_sameLen_textReview (IT.zip_longest(X, y),40, 0.6)     #short review and high % useful words
         newX, newy = list_sameLen_textReview (IT.zip_longest(X, y), 3, 30, 200, 0.6)     #long review and high % useful words
#        bigX.append(newX)
#        bigy.append(newy)
#         newX1, newy1 = similar_reviews (newX, newy, 0.5, 5, [0.9, 0.0], 100, True)
         #newX1, newy1 = similar_reviews (newX, newy, 9, 30, [0.7, 0.05], 100, True)
         newX1, newy1 = numPeaks_reviews (newX, newy, [0.7, 0.06], 100, True)
         core(newX1,newy1, 'same number peaks and same len')

    
    pathfile = "C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result15112018\\Result"
    
    
    if (args.predict and choice == 'general_test'):
        # X_Test, y_Test, X, y, best_model, typeTest  
        predict_save_results (X1_Test, y1_Test, X, y, best_model_name, 'isolated Low voted', 'General', pathfile)
        predict_save_results (X2_Test, y2_Test, X, y, best_model_name, 'contextualized hig voted', 'General', pathfile)
#        predict_save_results (X3_Test, y3_Test, X, y, best_model_name, 'period on the platform', 'General', pathfile)
#        predict_save_results (X4_Test, y4_Test, X, y, best_model_name, 'rating', 'General', pathfile)
#        predict_save_results (X5_Test, y5_Test, X, y, best_model_name, 'brands', 'General', pathfile)
    else:        
        newTestX = np.array(newX)[1:10, :]
        newTesty = np.array(newy)[1:10, :]
        newTrainX = np.delete(newX,np.s_[1:10],axis=0)  # <-------- THIS IS THE CORRECT FORM
        newTrainy = np.delete(newy,np.s_[1:10],axis=0)
        predict_save_results (newTestX, newTesty, newTrainX, newTrainy, best_model_name, 'len=40', 'len_test %votes', pathfile)
     
    
    

    

if __name__ == "__main__":
    parser = argsp.ArgumentParser(description='Optional app description')
    
    # Optional positional argument
    parser.add_argument("--input", type=str, required=True)
    # Optional argument
#    parser.add_argument('--opt_arg', type=int,
#                    help='An optional integer argument')    
    # Switch
    parser.add_argument('--predict', default=False, action='store_true')
    args = parser.parse_args()
    main(args)                      # From the console it's enough launch this statement after set the arg in comman Run -> Config per File : Command Lines Options

 




# 1 Experiment : training Set with reviews same review length
#y = dataTrain.iloc[:, 2:3].values   
#newX, newy = list_sameLen_textReview (IT.zip_longest(X, y),200, 0)
#newX, newy = list_sameLen_textReview (IT.zip_longest(X, y),500, 0)
#newX, newy = list_sameLen_textReview (IT.zip_longest(X, y),1000, 0)
#
#newX, newy = list_sameLen_textReview (IT.zip_longest(X, y),200, 0.6)
#newX, newy = list_sameLen_textReview (IT.zip_longest(X, y),500, 0.6)
#newX, newy = list_sameLen_textReview (IT.zip_longest(X, y),1000, 0.6)
#
#print ('test last row X[0]=',newX[2469][0])
#print ('test last row X[29]=',newX[2469][29])
#print ('test last row X[30]=',newX[2469][30])
#print ('test last row X[31]=',newX[2469][31])
#print ('test last row y=',newy[2469])
#
## 2 Experiment : training Set with reviews same product
#y = dataTrain.iloc[:, 2:3].values   
#newX, newy = list_sameProd (IT.zip_longest(X0, y),'B00001P4ZH')
#newX = np.delete(newX,30,axis=1)
#print(newX)
#print(newX.shape)
#
## 3 Experiment : training Set with reviews similar slope
#newX, newy = similar_reviews (X, y, 5, 5, [0.9, 0.01], 100)
#newX, newy = similar_reviews (X, y, 5, 15, [0.9, 0.01], 100)
#newX, newy = similar_reviews (X, y, 3, 20, [0.9, 0.01], 200)
#newX, newy = similar_reviews (X, y, 2.5, 20, [0.9, 0.01], 500)

    