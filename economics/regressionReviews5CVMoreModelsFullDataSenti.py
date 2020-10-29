# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 09:20:37 2019

@author: paolo
"""

# from file_a import b, c
# where in file_a
#def b():
#  # Something
#  return 1
#
#def c():
#  # Something
#  return 2

import itertools as IT
import argparse as argsp
import pandas as pd
import numpy as np
import statistics as stat
import random as rand
import scipy.stats as scstat
import math
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression
from sklearn.feature_selection import chi2
from scipy import interpolate
from scipy.interpolate import interp1d, UnivariateSpline, splrep, splev
from pandas import Series
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
from matplotlib.ticker import NullFormatter
from matplotlib.dates import MonthLocator, DateFormatter
from statsmodels.tsa.arima_model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize

import statsmodels.api as sm

# Importing the Training dataset
#
#from sklearn.datasets import make_classification
#from sklearn.ensemble import ExtraTreesClassifier





import sys
sys.path.append('C:\\UniDatiSperimentali\\PAOLONE IDEAS\dottorato\\python e SIDE PG\\pythonCode')
for line in sys.path: 
    print (line)
    
from regressionReviews4CVMoreModelsFullData import list_sameLen_textReview,predict_save_results,get_data_split, \
tune_models_hyperparams, print_grid_results, get_best_model, test_dataset, models, list_sameRating, list_sameColumn, \
list_sameProd, list_groupProds, list_LittleRating, list_HighRating, list_smallVotes, list_largeVotes, statistic, list_percentVotes, \
statisticSimple, list_groupProdsPlusDate, predict_results

import gc
gc.collect()
# Importing the Training dataset
dataTrain = pd.read_csv(r'C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Result11012019\Training_5.860-1280k.json.csv')
series = Series.from_csv(r'C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Result11012019\Training_5.860-1280k.json.csv', header=0)


# Importing the Testing dataset HighVotes   47 lines
dataTest2 = pd.read_csv(r'C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Result11012019\TestHighVotes.csv')
# Importing the Testing dataset LowVotes    378 lines
dataTest1 = pd.read_csv(r'C:\UniDatiSperimentali\PRODUCT REVIEW RUGGIERO\VOTE-RATING-ENTROPY\OutNew\Result11012019\TestLowVotes.csv')

date = dataTrain.iloc[:, 4:5].values            # dates



def DataGenTest (numTest):   
        X0 = dataTrain.iloc[:, 5:42].values
        print(X0)
        print(X0.shape)
        X_ = np.delete(X0,34,axis=1)        #column ProdID deleted
        print(X_)
        print(X_.shape)        
        y__ = dataTrain.iloc[:, 2:3].values             # HELPFUL VOTES
        y_ = dataTrain.iloc[:, 1:2].values            # % HELPFUL VOTES
        #y_ = dataTrain.iloc[:, 3:4].values            # TOTAL VOTES
        print(y_)
        print(y_.shape)
        if (numTest == 1):
            return X_, y_
        if (numTest == 4):
            return X0, X_, y_, y__
   
        # Importing the X values of Testing dataset ( total votes over 400)
        X2_realTest = dataTest1.iloc[1:30, 5:42].values
        X2_realTest = np.delete(X2_realTest,34,axis=1)        #column ProdID deleted
        y2_realTest = dataTest1.iloc[1:30, 1:2].values
        #Y6=np.delete(Y6, 5, axis=0) # <------------------ THIS IS THE PROBLEM (the reviewer with 1000 reviews)
        print(y2_realTest)
        print(y2_realTest.shape)        
        # Importing the X values of Testing dataset ( total votes around 10-15)
        X3_realTest = dataTest1.iloc[350:378, 5:42].values
        X3_realTest = np.delete(X3_realTest,34,axis=1)        #column ProdID deleted
#        print(X3_realTest)
        y3_realTest = dataTest1.iloc[350:378, 1:2].values
        #Y6=np.delete(Y6, 5, axis=0) # <------------------ THIS IS THE PROBLEM (the reviewer with 1000 reviews)
        print(y3_realTest)
        print(y3_realTest.shape)       
        X1_realTest = dataTest2.iloc[:, 5:42].values
        X1_realTest = np.delete(X1_realTest,34,axis=1)        #column ProdID deleted
        y1_realTest = dataTest2.iloc[:, 1:2].values
        #Y6=np.delete(Y6, 5, axis=0) # <------------------ THIS IS THE PROBLEM (the reviewer with 1000 reviews)
        print(y1_realTest)
        print(y1_realTest.shape)   
        if (numTest == 2):        
            return X_, y_, X1_realTest, y1_realTest, X2_realTest, y2_realTest, X3_realTest, y3_realTest
                   
        X0 = dataTrain.iloc[:, 5:42].values
        print(X0)
        print(X0.shape)
        print('\nORIGINAL ',X0[0])
        X_ = np.delete(X0,np.s_[26:35],axis=1)        #columns SENTIMENTS  deleted
        print('\nDEL SENTIMENT + prod ',X_[0])
        print(X_.shape)        
        plotting_trends (y_, date, 'prova1')
        
        if (numTest == 3):        
            return X_, y_


        
        
#def plotting_trends (y_old,date, name):
#
#        x = []
#        y = []
#        #print (y_[1][0])
#        for i in range(0, len(y_old)):
#           y.append(y_old[i])
#           x.append(datetime.datetime.strptime(date[i][0], '%d/%m/%Y'))   #dd/mm/yyyy
#        print (x)
#        print (y)
#        df = pd.DataFrame({'date' : x, 'y' : y })
#        print (df)
#        df = df.sort_values(by=['date'])
##        print (df)
#        
#        #plt.scatter(x, y)
##        days = mdates.DayLocator()
##        datemin = datetime(2011, 1, 1)
##        datemax = datetime(2013, 12, 31) 
#        plt.rcParams['figure.figsize'] = (30, 5)
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
##        ax.xaxis.set_major_locator(days)
##        ax.set_xlim(datemin, datemax)
#        ax.set_ylabel('%vote+') 
#        fig.autofmt_xdate(ha='right', which='both')
#        plt.yticks(np.arange(0, 1, 0.1))
#        plt.gca().xaxis.set_major_locator(MonthLocator())
#        plt.gca().xaxis.set_minor_locator(mdates.YearLocator())
#        plt.gca().xaxis.set_minor_formatter(DateFormatter('%Y'))
#        plt.gca().xaxis.set_major_formatter(DateFormatter('%b   '))
#        fig.autofmt_xdate(ha='left', which='minor')
#        plt.plot(df.date, df.y,'r--')
#        plt.show()    
#        fig.savefig('C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result11012019\\'+name+'.jpg', dpi=200)
        
        
        
#   Plotting and Saving (date, y_) curve
#  return the dipendent var y

def plotting_trends (y_, date, name, color,sign,save,label):

        # line plot of dataset
        y = []
        x = []
        #print (y_[1][0])
        for i in list(range(y_.shape[0])):
           y.append(y_[i][0])
           x.append(datetime.datetime.strptime(date[i][0], '%d/%m/%Y'))   #dd/mm/yyyy
        print (x)
        print (y)
        df = pd.DataFrame({'date' : x, 'y' : y })
        print (df)
        df = df.sort_values(by=['date'])
#        print (df)
        
        #plt.scatter(x, y)
#        days = mdates.DayLocator()
#        datemin = datetime(2011, 1, 1)
#        datemax = datetime(2013, 12, 31) 
        x_size = len(y)/25
        plt.rcParams['figure.figsize'] = (x_size, 5)
        fig = plt.figure()
        ax = fig.add_subplot(111)
#        ax.xaxis.set_major_locator(days)
#        ax.set_xlim(datemin, datemax)
        ax.set_ylabel(label) 
        fig.autofmt_xdate(ha='right', which='both')
        #plt.yticks(np.arange(0, 1, 0.1))
        plt.gca().xaxis.set_major_locator(MonthLocator())
        plt.gca().xaxis.set_minor_locator(mdates.YearLocator())
        plt.gca().xaxis.set_minor_formatter(DateFormatter('%Y'))
        plt.gca().xaxis.set_major_formatter(DateFormatter('%b   '))
        fig.autofmt_xdate(ha='left', which='minor')
        plt.plot(df.date, df.y,color+sign)      # '-' continuous , '--' dotted
        plt.show()   
        if (save):
            fig.savefig('C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result11012019\\'+name+'.jpg', dpi=300)
        return y
 
    
    # Similar to plotting_trends but using directly the (x,y) value
    # ex.: plotting_trends1 (df, 'prodID=B003ZSP0WW','b','-', False,'%vote+')
def plotting_trends1 (x,y, name, color,sign,save,label):

        x_size = len(y)/2
        plt.rcParams['figure.figsize'] = (x_size, 5)
        fig = plt.figure()
        ax = fig.add_subplot(111)
#        ax.xaxis.set_major_locator(days)
#        ax.set_xlim(datemin, datemax)
        ax.set_ylabel(label) 
        fig.autofmt_xdate(ha='right', which='both')
        #plt.yticks(np.arange(0, 1, 0.1))
        plt.gca().xaxis.set_major_locator(MonthLocator())
        plt.gca().xaxis.set_minor_locator(mdates.YearLocator())
        plt.gca().xaxis.set_minor_formatter(DateFormatter('%Y'))
        plt.gca().xaxis.set_major_formatter(DateFormatter('%b   '))
        fig.autofmt_xdate(ha='left', which='minor')
        plt.plot(x,y,color+sign)      # '-' continuous , '--' dotted
        plt.show()   
        eol = ''
        name = name + eol
        if (save):
            fig.savefig('C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result10022019\\'+name+'.jpg', dpi=300)
       

# vedi https://stackoverflow.com/questions/37872171/how-can-i-perform-two-dimensional-interpolation-using-scipy per fare
#       un' interpolazione in 3D     
#  vedi https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html per spline   


##############################################################
#   USED FOR PLOTTING DOT AND SPLINES IN ORDER TO SHOW THE SHAPE OF QUALITIES
##############################################################
            
            

            
def plotting_trends2_dot (x,y, name, type_y, indexColor, namex):

        # https://python-graph-gallery.com/python-colors/
        colors = ['green', 'red', 'blue' , 'orange', 'magenta', 'navy', 'green', 'brown', 'pink','olive','red','navy', 'slategray','coral','aquamarine','slategray']
   
        c, r = len(x), 4;
        print('#row',r,'#col',c)
        dataSaved = [[0 for x in range(c)] for y in range(r)] 

        fig=plt.gcf()
        x_size =  max(x)   
        y_size = 4
        #x_size = 5   ## setting for rating, quality
        print ('x_size',x_size,'y_size',y_size)
        i = 0
        plt.rcParams['figure.figsize'] = (x_size, y_size)
        plt.title(name)
        plt.ylabel('qual'+type_y+' '+namex)       # level price
        plt.xlabel(namex)   
        #plt.xlabel('rating')   ## setting for rating, quality
        xx = np.arange(1,x_size,0.1)      # range xmin, xmax, precision
        s2 = interpolate.UnivariateSpline (x, y, k=3) # smoothing_factor 0.1
        #s2 = interp1d(x, y, kind='quadratic')   
        #plt.plot(x, y, indexColor+'o', label = 'Data')
        plt.scatter(x, y, color=colors[indexColor], label = 'Data')
        #plt.set_smoothing_factor(0.05)       # Change manually until get a best fit
        plt.plot(xx, s2(xx), 'r', label = 'Spline fitted') # SPLINE   # see https://stackoverflow.com/questions/17913330/fitting-data-using-univariatespline-in-scipy-python
        plt.yticks(np.arange( 0.5,1, 0.1))   ## setting for price, quality
        if (namex=='price'):
            plt.xticks(np.arange( 0,x_size, 0.5))   ## setting for price, quality
        else:
            plt.xticks(np.arange( 2,x_size, 0.5))       ## setting for rating, quality 
        plt.legend(loc='upper left')  ## setting for price, quality
        plt.show()
        eol = ''
        name = name + eol
        fig.savefig('C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result02102019\\'+name+type_y+'.jpg', dpi=200)
        for j in range(0,c-1): 
           dataSaved[i][j] = x[j]       # x values
        i = i+1       
        for j in range(0,c-1): 
           dataSaved[i][j] = y[j]       # y values
        i = i+1   
        print ('xx',xx)
        print ('s2(xx)',xx)
#        for j in range(1,len(xx)-1): 
#           dataSaved[i][j] = xx[j]      # spline x values
#        i = i+1       
#        for j in range(1,len(xx)-1): 
#           dataSaved[i][j] = s2(xx[j])   # spline y values
        np.savetxt('C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result02102019\\'+name+type_y+'.csv', dataSaved, delimiter=";",fmt='%.3f')


##############################################################
#   USED FOR PLOTTING DOT WITHOUT SPLINES IN ORDER TO SHOW THE THRESOLD QUALITY WITH RESPECT TO DIFFERENT UNIT COST OF SEARCH
##############################################################
        
        # x = vector num steps
        # y = vector quality
        # y_vl (value) threshold
        # cost effort of make  step
        # p_h (value) BUDGET CONSTRAINT
        # rating vector rating
        # price vector price
        # name is the title chart
        # type_y  IS THE CHART TYPE
        # indexColor  Color used


        

def plotting_trends3_formulaLinear (x,y, y_l, cost, p_h, rating, price, name, type_y, indexColor, path):

        # https://python-graph-gallery.com/python-colors/
        colors = ['green', 'red', 'blue' , 'orange', 'magenta', 'navy', 'green', 'brown', 'pink','olive','red','navy', 'slategray','coral','aquamarine','slategray']
   
        fig = plt.figure()
        ax = fig.add_subplot(111)
       # cost = 0.06  #cost of step


        x_size = max(x)   
        y_size = 8
        y = y[:x_size]
        u = np.zeros(x_size)
        for i in range(0,len(u)):
           u[i] = y[i] - (i+1)*0.5*cost
#           if u[i] < 0:
#              u[i] = 0
           
        rating = rating[:x_size]
        price = price[:x_size]
        print ('\nquality',y,'\nutility',u,'\nrating',rating,'\nprice',price)
        print ('x_size',x_size,'y_size',y_size)
        c, r = len(x), 4;
        print('#row',r,'#col',c)
        dataSaved = [[0 for x in range(c)] for y in range(r)] 


        i = 0
        plt.rcParams['figure.figsize'] = (x_size, y_size)
        plt.title(name)
        #plt.xlabel('rating')   ## setting for rating, quality
        #plt.plot(x, y, indexColor+'o', label = 'Data')
    
        y_l = y_l - 0.07
        plt.axhline(y=y_l, color=colors[indexColor+1], label = 'c ='+str(cost), linestyle='--')                   # horixontal line Q threshold
        plt.text(0.2, y_l, 'q >='+'{0:.2f}'.format(y_l), fontsize=8, va='center', ha='left', backgroundcolor=colors[indexColor+1])
        
        # removed the line of q low!
#        plt.axhline(y=y_h, color=colors[indexColor+4],  label = 'c high', linestyle='--')
#        plt.text(0.2, y_h+0.05, 'q >='+'{0:.2f}'.format(y_h), fontsize=10, va='center', ha='left', backgroundcolor=colors[indexColor+4])
        
#        if (namex == 'rating'):
#            plt.scatter(x, y, color=colors[indexColor], label = 'Data')
#            for z in IT.zip_longest(x,y,rating,price):
#                txt = ' ' + '{0:.2f}'.format(z[2]) + ',$' + str(z[3])
#                print (txt)
#                plt.annotate(txt, (z[0], z[1]))
# 
#
#        elif (namex == 'price'):        
        plt.scatter(x, u, marker='x', color=colors[indexColor+2], label = 'Utility')   # Utility point
        plt.scatter(x, y, color=colors[indexColor], label = 'Data (red (bc) < '+'{0:.1f}'.format(p_h)+'$)')  # quality point
        
        for z in IT.zip_longest(x,y,rating,price,u):
            txt1 = '  ' + '{0:.2f}'.format(z[4]) 
            plt.annotate(txt1, (z[0], z[4]), color=colors[indexColor+2],size=7)   # Utility value
            #txt = '  r=' + '{0:.1f}'.format(z[2]) + ',$' + '{0:.0f}'.format(z[3])+ ',q=' + '{0:.2f}'.format(z[1])  (r,p,q)
            txt = '  r=' + '{0:.1f}'.format(z[2]) + ',$' + '{0:.0f}'.format(z[3])    #  only (r,p)
            if (z[3] < p_h):
                plt.annotate(txt, (z[0], z[1]), color=colors[indexColor+1],size=7)     # (rating,price) value  # IF price < BUDGET CONSTRAINT THEN THE COUPLE (rating, price) are red
            else:
                plt.annotate(txt, (z[0], z[1]),size=7)

#        ax2 = ax.twiny()
#        ax3 = ax.twiny()       
        ax.set_xlim(0, max(x)+1)
        ax.set_xticks(np.arange( 0,max(x)+1, 1))   
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange( 0,1.1, 0.1))   
#        ax2.set_xlim(2, 5)  # rating
#        ax2.set_xticks(np.arange( 2, 5, 0.2)) 
#        ax3.set_xlim(0, 1000)  # price
#        ax3.set_xticks(np.arange( 0,1000, 50))   

        
        ax.set_ylabel("quality")
#        ax2.set_xlabel("rating")  # SCALE OF RATING
#        ax3.set_xlabel("price")    # SCALE OF PRICE
        ax.legend(loc='lower right')   
        
        ax.xaxis.set_ticks_position('bottom') # set the position of the  x-axis to bottom
        ax.set_xlabel("step ")
        ax.xaxis.set_label_position('bottom')

#        ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
#        ax2.spines['bottom'].set_position(('outward', 35)) 
#        ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
#
#        ax3.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
#        ax3.spines['bottom'].set_position(('outward', 70))
#        ax3.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom

        
        #plt.legend(loc='upper left')  ## setting for price, quality
        plt.show()
        eol = ''
        name = name + eol
#        fig.savefig('C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result-Reputation05122019\\'+name+type_y+'.jpg', dpi=300)
        fig.savefig(path+name+type_y+'.jpg', dpi=300)        
        for j in range(0,c-1): 
           dataSaved[i][j] = x[j]       # x values
        i = i+1       
        for j in range(0,c-1): 
           dataSaved[i][j] = y[j]       # y values
        i = i+1   
        np.savetxt(path+name+type_y+'.csv', dataSaved, delimiter=";",fmt='%.3f')



def formula4Linear (q,c):
    result = 0.5 * (2*q + 1 - math.sqrt(8*q*c + 1))
    return result

def formula4RootSq (q,c,n):
    result = 0.5 * (2*q + 1 - math.sqrt(8*q*(math.sqrt(n) - math.sqrt(n-1))*c))
    return result

 
def plotting_trends3_formulaRootSquare (x,y, cost, p_h, rating, price, name, type_y, indexColor, path):

        # https://python-graph-gallery.com/python-colors/
        colors = ['green', 'red', 'blue' , 'orange', 'magenta', 'navy', 'green', 'brown', 'pink','olive','red','navy', 'slategray','coral','aquamarine','slategray']
   
        fig = plt.figure()
        ax = fig.add_subplot(111)
       # cost = 0.06  #cost of step


        x_size = max(x)   
        y_size = 8
        y = y[:x_size]
        u = np.zeros(x_size)
        y_l = np.zeros(x_size)
        for i in range(0,len(u)):
           y_l[i] = formula4RootSq (1,cost,(i+1))
           u[i] = y[i] - math.sqrt(i+1)*cost
           if u[i] < 0:
              u[i] = 0

        print('y_l',y_l)
       
        rating = rating[:x_size]
        price = price[:x_size]
        print ('\nquality',y,'\nutility',u,'\nrating',rating,'\nprice',price)
        print ('x_size',x_size,'y_size',y_size)
        c, r = len(x), 4;
        print('#row',r,'#col',c)
        dataSaved = [[0 for x in range(c)] for y in range(r)] 


        i = 0
        plt.rcParams['figure.figsize'] = (x_size, y_size)
        plt.title(name)
        #plt.xlabel('rating')   ## setting for rating, quality
        #plt.plot(x, y, indexColor+'o', label = 'Data')
    
        plt.plot(x, y_l,color=colors[indexColor+4], label = 'c ='+str(cost), linestyle='--')       
        #plt.text(0.2, y_l, 'q >='+'{0:.2f}'.format(y_l), fontsize=10, va='center', ha='left', backgroundcolor=colors[indexColor+1])
        
        plt.scatter(x, u, marker='x', color=colors[indexColor+2], label = 'Utility')
        plt.scatter(x, y, color=colors[indexColor], label = 'Data (red (bc) < '+'{0:.1f}'.format(p_h)+'$)')
        for z in IT.zip_longest(x,y,rating,price,u):
            txt1 = '  ' + '{0:.2f}'.format(z[4]) 
            plt.annotate(txt1, (z[0], z[4]), color=colors[indexColor+2])
            txt = '  r=' + '{0:.2f}'.format(z[2]) + ',$' + str(z[3])
            if (z[3] < p_h):
                plt.annotate(txt, (z[0], z[1]), color=colors[indexColor+1])     # IF price < BUDGET CONSTRAINT THEN THE COUPLE (rating, price) are red
            else:
                plt.annotate(txt, (z[0], z[1]))

#        ax2 = ax.twiny()
#        ax3 = ax.twiny()       
        ax.set_xlim(0, max(x)+1)
        ax.set_xticks(np.arange( 0,max(x)+1, 1))   
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange( 0,1.1, 0.1))   
#        ax2.set_xlim(2, 5)  # rating
#        ax2.set_xticks(np.arange( 2, 5, 0.2)) 
#        ax3.set_xlim(0, 1000)  # price
#        ax3.set_xticks(np.arange( 0,1000, 50))   

        
        ax.set_ylabel("quality")
#        ax2.set_xlabel("rating")  # SCALE OF RATING
#        ax3.set_xlabel("price")    # SCALE OF PRICE
        ax.legend(loc='lower right')   
        
        ax.xaxis.set_ticks_position('bottom') # set the position of the  x-axis to bottom
        ax.set_xlabel("step ")
        ax.xaxis.set_label_position('bottom')

#        ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
#        ax2.spines['bottom'].set_position(('outward', 35)) 
#        ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
#
#        ax3.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
#        ax3.spines['bottom'].set_position(('outward', 70))
#        ax3.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom

        
        #plt.legend(loc='upper left')  ## setting for price, quality
        plt.show()
        eol = ''
        name = name + eol
#        fig.savefig('C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result-Reputation05122019\\'+name+type_y+'.jpg', dpi=300)
        fig.savefig(path+name+type_y+'.jpg', dpi=300)        
        for j in range(0,c-1): 
           dataSaved[i][j] = x[j]       # x values
        i = i+1       
        for j in range(0,c-1): 
           dataSaved[i][j] = y[j]       # y values
        i = i+1   
        np.savetxt(path+name+type_y+'.csv', dataSaved, delimiter=";",fmt='%.3f')


 

def extractInfoDemand (x,y, y_q, cost,bc, rating, price):


        x_size = max(x)   
        y = y[:x_size]
        u = np.zeros(x_size)
        for i in range(0,len(u)):
          # u[i] = y[i] - (i+1)*0.5*cost
          u[i] = y[i] - (i+1)*cost
           
        rating = rating[:x_size]
        price = price[:x_size]
 

        outSR_u = outSR_y = out_u = out_y = 0
        i = 0  
        y_q = y_q - 0.15                         # horixontal line Q threshold
        print ('****************************************  extractInfoDemand  **********************************************************')
        print ('\nquality',y,'len(qual)=',len(y),'\nutility',u,'\nrating',rating,'\nprice',price, '\n\ncost',cost,'\nbc',bc,'\ny_q',y_q)

        for i  in range(0,len(y)):
            if (price[i] <= bc and y[i] >= y_q):   # y[i] <= bc And > Q threshold  SR
               outSR_u = u[i]
               outSR_y = price[i]
               break
            
        for i  in range(0,len(y)):
            if (price[i] <= bc ):   # y[i] <= bc  No SR
                out_u = u[i]
                out_y = price[i]
                break
            
            
        print ('* outSR_u',outSR_u,'outSR_y',outSR_y)
        print ('** out_u',out_u,'out_y',out_y)
  
        return  outSR_u, outSR_y, out_u, out_y 
            
                        



def plotting_utility (x,y, k, factor_x_axis,  x_segment, name, indexColor,path):

        # https://python-graph-gallery.com/python-colors/
        colors = ['green', 'red', 'blue' , 'orange', 'magenta', 'navy', 'green', 'brown', 'pink','olive','red','navy', 'slategray','coral','aquamarine','slategray']
   
        frame = pd.DataFrame({'x' : x, 'y' : y})
        frame = frame.drop_duplicates(subset=['x'], keep='first')
        frame = frame.sort_values(by=['x'])
        print (frame,'\n')
        x = frame['x'].tolist()
        y = frame['y'].tolist()


        fig=plt.gcf()
        x_size =  max(x)   
        y_size = 5
        

        print ('x_size',x_size,'y_size',y_size,'x_size/x_segment',x_size/x_segment)
        plt.rcParams['figure.figsize'] = (x_size/factor_x_axis, y_size)
        plt.title(name)
        plt.ylabel(name)       # Utility or Num
        plt.xlabel('price level ($)')   
        #plt.xlabel('rating')   ## setting for rating, quality
        xx = np.arange(1,x_size,x_size/x_segment)      # range xmin, xmax, precision
        s2 = interpolate.UnivariateSpline (x, y, k=k) 
        plt.scatter(x, y, color=colors[indexColor], label = 'Data')
        plt.plot(xx, s2(xx), 'r', label = 'Spline fitted') # SPLINE   # see https://stackoverflow.com/questions/17913330/fitting-data-using-univariatespline-in-scipy-python
        plt.yticks(np.arange(min(y),max(y),0.1))   
        plt.xticks(np.arange(min(x) ,x_size,x_size/x_segment))       
        plt.legend(loc='upper right')  
        plt.show()
        eol = ''
        name = name + eol
        fig.savefig(path+name+'.jpg', dpi=200)

 
def plotting_utilities (x,y1,y2, k1, k2, factor_x_axis, x_segment, name, indexColor,path):

        # https://python-graph-gallery.com/python-colors/
        colors = ['green', 'red', 'blue' , 'orange', 'magenta', 'navy', 'green', 'brown', 'pink','olive','red','navy', 'slategray','coral','aquamarine','slategray']
 
        
        frame = pd.DataFrame({'x' : x, 'y1' : y1, 'y2' : y2})
        frame = frame.drop_duplicates(subset=['x'], keep='first')
        frame = frame.sort_values(by=['x'])
        print (frame,'\n')
        x = frame['x'].tolist()
        y1 = frame['y1'].tolist()
        y2 = frame['y2'].tolist()
        #print ('x',x)
        
        fig=plt.gcf()
        x_size =  max(x)
        y_size = 10


        #x_size = 5   ## setting for rating, quality
        print ('min x',min(x),'x_size',x_size,'y_size',y_size,'x_size/x_segment',x_size/x_segment)
        plt.rcParams['figure.figsize'] = (x_size/factor_x_axis, y_size)
        plt.title(name)
        plt.ylabel(name)       # Utility or Num
        plt.xlabel('price level ($)')   
        #plt.xlabel('rating')   ## setting for rating, quality
        xx = np.arange(min(x),max(x),max(x)/x_segment)      # range xmin, xmax, precision
        s21 = interpolate.UnivariateSpline (x, y1, k=k1)   # NO SR
        s22 = interpolate.UnivariateSpline (x, y2, k=k2)    # Yes SR
        plt.scatter(x, y1, color=colors[indexColor], label = 'Data NoSR')
        plt.scatter(x, y2, color=colors[indexColor+1], label = 'Data SR')
        plt.plot(xx, s21(xx), 'r', label = 'Spline fit NoSR') 
        plt.plot(xx, s22(xx), 'b', label = 'Spline fit SR') 
        plt.yticks(np.arange(min(y2),max(y2),0.1))   
        plt.xticks(np.arange(min(x) ,max(x),max(x)/x_segment))       
        plt.legend(loc='upper right')  
        plt.show()
        eol = ''
        name = name + eol
        fig.savefig(path+name+'.jpg', dpi=100)

 


def plotMultiTrends (listZ, label, lungh, init, tipoLegend, title, listInfo, unit, y_ax ):   

    tipoOriginal = tipoLegend
    print ('tipo: ', tipoOriginal)

    tipo1 = [s + ' cumul' for s in tipoLegend]
    tipo2 = [s for s in tipoLegend]

# https://python-graph-gallery.com/python-colors/
    colors = ['green', 'red', 'blue' , 'orange', 'magenta', 'navy', 'green', 'brown', 'pink','olive','red','navy', 'slategray','coral','aquamarine','slategray']
    linestyles = ['-', '--', '-.', ':']
    fig=plt.gcf()
    #fig.autofmt_xdate(ha='right', which='both')
    x_size = lungh/2
    print('x_size ',x_size, 'len listZ ', len(listZ) )
   
    items = ''
    if (len(listInfo) > 0):
        for item in listInfo:
#         title = title +'\n'+str(listInfo[0])+str(listInfo[1])+str(listInfo[2])
            items = items + str(item)
    title = label +' ' + title +'\n' + items +'\n'


    
##################################################
    #                                        #
    #       CHART PER MONTH                  #
    #                                        #
##################################################    

    plt.rcParams['figure.figsize'] = (x_size, 5)
    plt.title(title)
    plt.ylabel(unit+' of Unit x Month')
    plt.xlabel('date ')       

    i = init
    i1 = 0
    listA = np.array([])
    listB = np.array([])

    #c, r = len(listZ[0]), len(listZ);
    c, r = lungh, len(listZ)*2;
    print('row',r,'col',c)
    dataSaved = [[0 for x in range(c)] for y in range(r)] 

    for Z in listZ:
       x_plot = []
       x_cumul_plot = []
       v_plot = []
       v_cumul_plot = []
       vMean_plot = []
       year_plot = []
       print('firstDate',Z[0][0])
       firstdate = Z[0][0]
       firstYear = '{:04d}'.format(firstdate.year)
       print('* firstYear',firstYear)
       count = 0;
       v_cumul = 0;
       vTotal = 0
       for z in Z:
            x_plot.append(z[0])            
            v = z[1]
            v_plot.append(v)            
#            r = z[2]
#            r_plot.append(r)  
            numYear = '{:04d}'.format(z[0].year)
            count = count +1
            if (numYear == firstYear):
                vTotal = vTotal+ v
            else:
                vMean = vTotal/float(count)
                vMean_plot.append(vMean)
                year_plot.append(numYear)
                firstYear = numYear
                #print(' numYear ',numYear,'vTotal',vTotal,'vMean',vMean)
                vTotal = v
       tmp = 0
       for vi in reversed(v_plot):
           v_cumul = v_cumul + vi
           #print ('index',tmp,'v_cumul',v_cumul,'vi',vi)
           v_cumul_plot.append(v_cumul)  
           x_cumul_plot.append(tmp) 
           tmp = tmp +1
       newA = zip(year_plot,vMean_plot)
       newB = zip(x_cumul_plot,v_cumul_plot)
       listA = np.append(listA,newA)    # couples (y,vMean) of each curves   
       listB = np.append(listB,newB)       
       #  see fot Twin axys
       #  https://stackoverflow.com/questions/10514315/how-to-add-a-second-x-axis-in-matplotlib
       if (init > 0):
           plt.plot(x_plot, v_plot, color=colors[i])  
       else:
           if (i >= 3):
               plt.plot(x_plot, v_plot, linestyle='--' ,color=colors[i%3]) 
           else:
               plt.plot(x_plot, v_plot, color=colors[i%3])   
       #print('v_plot',v_plot)
       print('len(v_plot)',len(v_plot),'lungh',lungh,'min between 2: ',min(len(v_plot),lungh))
#       for j in range(1,lungh): 
#           dataSaved[i1][j] = lungh-j
#       i1 = i1+1        
#       for j in range(1,min(len(v_plot),lungh)): 
#           dataSaved[i1][j] = v_plot[j]       
#       i1 = i1+1        
       i = i+1        

       
    ax = plt.gca()
    ax2 = ax.twiny()
    ax.xaxis_date()
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b'))
    ax.xaxis.set_minor_formatter(DateFormatter('\n%Y'))
    fig.autofmt_xdate(ha='center', which='minor')
    if (y_ax == 0):
        plt.yticks(np.arange(0, 1, 0.1))
    ax2.set_xlim(-2, len(x_plot))
    ax2.set_xticks(np.arange( -2, len(x_plot), 1))
    ax2.invert_xaxis()
    ax.legend(tipoLegend, loc='upper left')   
    plt.show()
    eol = ''
    label0 = label +  'perMonth' +eol
    fig.savefig('C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result29062019\\'+label0+'.jpg', dpi=300)
    
##################################################
    #                                        #
    #       CHART PER YEAR                   #
    #                                        #
##################################################    
    
    print ('tipo2: ', tipo2)
    fig=plt.gcf()
    i = init
#    x_size = lungh/4
#    print('x_size ',x_size)

    plt.rcParams['figure.figsize'] = (x_size, 5)
    plt.title(title+ '\n PER YEAR \n')
    plt.ylabel(unit+' of Unit x Year')
    plt.xlabel(' years')       

    for A in listA: # listA contains len(listZ) couples of (year array, vMean array) where vMean is the current y-axis var of listZ
        years = np.array([])
        votes = np.array([])
        for a in A:
#            print ('year',a[0])
#            print ('vote x year',a[1])
            years =  np.append(years,a[0]) 
            votes =  np.append(votes,a[1]) 
        print ('years',years)
        print ('vote x year',votes)
        #plt.plot(years, votes, color=colors[i]) # linestyle='--'
        if (init > 0):
           plt.plot(years, votes, color=colors[i])  
        else:
            if (i >= 3):
               plt.plot(years, votes, linestyle='--' ,color=colors[i%3]) 
            else:
               plt.plot(years, votes, color=colors[i%3])        
        i = i+1       
    plt.legend(tipo2, loc='upper left')   
    plt.show()
    eol = ''
    label1 = label + 'perYear' + eol
    fig.savefig('C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result29062019\\'+label1+'.jpg', dpi=300)

##################################################
    #                                        #
    #       CHART PER MONTH CUMULATIVE       #
    #                                        #
##################################################    
    
    fig=plt.gcf()
    i = init
    print ('tipo1: ', tipo1)

    plt.rcParams['figure.figsize'] = (x_size, 5)
    plt.title(title+ '\n CUMULATIVE \n')
    plt.ylabel(unit+' Cumulative Units x Month')       
    plt.xlabel('# months')       

    for B in listB: # listB contains len(listZ) couples of (year array, vMean array) where vMean is the current y-axis var of listZ
       x_plot = np.array([])
       v_cumulplot = np.array([])
       for b in B:
            x_plot = np.append(x_plot,b[0])     
            v_cumulplot = np.append(v_cumulplot,b[1])
            
#       if (typePlot == 0):
#           type = 'v% '
#       if (typePlot == 1):
#           type = 'repu '
#       if (typePlot == 2):
#           type = 'conte v '
#       if (typePlot == 3):
#           type = 'len '
#       print (type+' cumul',v_cumulplot)
       

       #plt.plot(x_plot, v_cumulplot, color=colors[i])   #linestyle=':' 
       if (init > 0):
           plt.plot(x_plot, v_cumulplot, color=colors[i])  
       else:
           if (i >= 3):
               plt.plot(x_plot, v_cumulplot, linestyle='--' ,color=colors[i%3]) 
           else:
               plt.plot(x_plot, v_cumulplot, color=colors[i%3])        
       
       for j in range(1,lungh): 
           dataSaved[i1][j] = j
       i1 = i1+1       
       for j in range(1,min(len(v_cumulplot),lungh)): 
           dataSaved[i1][j] = v_cumulplot[j]
       print('dataSaved[',i1,']',dataSaved[i1])

       i = i+1        
       i1 = i1+1        

#    ax = plt.gca()
#    ax.xaxis_date()
#    ax.xaxis.set_major_locator(MonthLocator())
#    ax.xaxis.set_minor_locator(mdates.YearLocator())
#    ax.xaxis.set_major_formatter(DateFormatter('%b'))
#    ax.xaxis.set_minor_formatter(DateFormatter('\n%Y'))
#    fig.autofmt_xdate(ha='center', which='minor')
       

    plt.xticks(np.arange( 0, len(x_plot), 1))
    plt.legend(tipo1, loc='upper left')   
    plt.show()
    eol = ''
    label2 = label + 'Cumulative' + eol
    fig.savefig('C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result29062019\\'+label2+'.jpg', dpi=300)
    np.savetxt('C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result29062019\\'+label+'.csv', dataSaved, delimiter=";",fmt='%.3f')

    
    
    
def autoscale_based_on(ax, lines):
    ax.dataLim = mtransforms.Bbox.unit()
    for line in lines:
        xy = np.vstack(line.get_data()).T
        ax.dataLim.update_from_data_xy(xy, ignore=False)
    ax.autoscale_view()
    

# Map list positive values x -> [0,1]
def normalization (x):
    # see https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.norm.html for other type of norm
    norm1 = x / np.linalg.norm(x,np.inf)    #np.inf -> [0,1]
    print ('norm\n',norm1)
    return norm1

def normalizationFactor (x, fact):
    # see https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.norm.html for other type of norm
    norm1 = x*fact / np.linalg.norm(x,np.inf)    #np.inf -> [0,1]
    print ('norm\n',norm1)
    return norm1

#def normalization1 (x, lower, upper):
#    l_norm = [lower + (upper - lower) * t for t in x]
#    print ('norm\n',l_norm)
#    return l_norm

    
def normalization1(list, range): # range should be (lower_bound, upper_bound)
  l = np.array(list) 
  a = np.max(l)
  c = np.min(l)
  b = range[1]
  d = range[0]
  m = (b - d) / (a - c)
  pslope = (m * (l - c)) + d
  return pslope

    
#     Return (product, rating) of grup of prod which match with criteria: 
     #  (long or short period) AND (low, medium, high rating)
# 1. take group of prods > start date and (long or short period)  
# 2. CALCULATING RATING Mean of  group aggregated in 1.  
# 3. SPLITTING BY RATING        
    
def list_groupProdsCriteria (Z, IndexProd, IndexRating, IndexTime, presence, rating, dateCol, startDate):    #  Z = zip_longest(X0, y)  where X0 = X + 'prod column'
    
        in1 = []
        in2 = []
        in3 = []
        row = 0
        if (presence == 'long'):    # long from 01/01/2009 and after 01/06/2011  and [30,60]
           numMonth = 60
        else:                       # short from 01/06/2011 to 12/12/2014  and [0,30]
           numMonth = 20            # befor 40
           
        if (rating == 'lowRating'):
            RatingValue = 3.6 
        if (rating == 'mediumRating'):
            RatingValue = 4.0
        if (rating == 'highRating'):
            RatingValue = 4.4
            
        # 1. take group of prods > start date and (long or short period)   
        startDate = datetime.datetime.strptime(startDate, '%d/%m/%Y')
#        print ('IndexProd, IndexRating, IndexTime',IndexProd, IndexRating, IndexTime)
        for z in Z:
             #print ('ratingValue = ', r, 'prod = ', p)            
            currentDate = datetime.datetime.strptime(dateCol[row][0], '%d/%m/%Y')
            row = row +1
#            print ('z[0][IndexTime],numMonth',z[0][IndexTime],numMonth)
            if (currentDate >= startDate and ((numMonth)  >= float(z[0][IndexTime]) >= (numMonth - 20))):               
#                print ('date prod: ',z[0][IndexProd],currentDate,' # months presence :',z[0][IndexTime], 'rating', z[0][IndexRating])
                in1.append(z[0][IndexProd])   # prod
                in2.append(z[0][IndexRating]) # rating
                in3.append(z[1][0]) # vote
        in1 = np.array(in1)
        in2 = np.array(in2)
        in3 = np.array(in3)
        print (in1)
        print (in2)
        print (in3)
        zipped = IT.zip_longest(in1,in2,in3)
        dict_prod_rating = dict(zip(in1,in2))
        
        print ('len dict ',len(dict_prod_rating),'\n',dict_prod_rating.keys())
        print (dict_prod_rating.values())
#        for zi in zipped:
#            print(zi)
        in1A = []
        in2A = []        
        in3A = []        
        #zipped = zip(in1,in2)
        
        # 2. CALCULATING RATING Mean of  group aggregated in 1.       
        for p in dict_prod_rating.keys():
            i = 0
            summa = 0
            v_summa = 0
            for zi in zipped:
                p1 = zi[0]
                r1 = zi[1]
                v1 = zi[2]
                if (p1 == p):
                    summa = summa + float(r1)
                    v_summa = v_summa + float(v1)
                    i = i + 1
                    rat = float(summa)/i
                    vmean = float(v_summa)/i
#                print ('p',p, 'p1',p1)
#            print ('p',p, rat)
            zipped = IT.zip_longest(in1,in2,in3)
            in1A.append(p)
            in2A.append(rat)
            in3A.append(vmean)
        dict_prod_rating = {key:value for key, value in zip(in1A,in2A)}
        dict_prod_vote = {key:value for key, value in zip(in1A,in3A)}
#        print ('\n\ndict_prod_rating',dict_prod_rating)
#        print ('\n\ndict_prod_vote',dict_prod_vote)
        print ('\nLEN dict_prod_rating',len (dict_prod_rating))
        
#        exit
        out1 = []
        out2 = []
        
        # 3. SPLITTING BY RATING        
        #        minRating = 3.6
        #        mediumRating = 4.0
        #        maxumRating = 4.4      
        i = 1
        for p,r in dict_prod_rating.items():
            if (rating == 'lowRating' and (float(r) < RatingValue)):
                    out1.append(p)   # prodsID
                    out2.append(r)   # ratings    
                    print ('product ID, average rating',i,p,r)
            if (rating == 'mediumRating' and ((RatingValue + 0.3) > float(r) > (RatingValue - 0.3))):
                    out1.append(p)  
                    out2.append(r)           
                    print ('product ID, average rating',i,p,r)
            if (rating == 'highRating' and (float(r) > RatingValue)):
                    out1.append(p)  
                    out2.append(r)    
                    print ('product ID, average rating',i,p,r)            
            i = i +1                                     
            
        print ('\n-----------------------------\n')    
        out1 = np.array(out1)    
        for p,v in dict_prod_vote.items():
            for p1 in out1:
                if (p == p1):
                    print ('product ID, average %vote+ ',i,p,v)   
        print ('\n-----------------------------\n')    
            
        statisticSimple (out2, 'rating')
        return out1, out2               # list of prods and rating which satisfy ABOVE CRITERIA


def createUniqueProd(Z, IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature, presence, rating, plotting):
    
    # w1 = (MEAN H VOTES x reviews) x REVIEWER -> column 8, w2 = MEAN H VOTES x CONTEXT (WIN = 4 LEFT) -> column 21
    
        x = []
        y = []
        ya = []
        yb = []
        yc = []
        k = []
        j = []
        w1 = []
        w2 = []
        nrev = []

#        zip(newX, newy, ya, yb, y_pred, dateCol)
#        x = z[0][index_feature]
#        y = v%+ = z[1][0]
#        ya =  1 - %v+  = z[2][0]
#        yb = v+ =  z[3][0]
#        yc (y_pred) =  z[4][0]
#        datecolumn


        row = 0
        print('firstDate',Z[0][5][0])
        firstDate = datetime.datetime.strptime(Z[0][5][0], '%d/%m/%Y')
        print('* firstDate',firstDate)
        firstMonth = '{:02d}'.format(firstDate.month)
        firstYear = '{:04d}'.format(firstDate.year)
        vTotal = 0.
        vaTotal = 0.
        vbTotal = 0.
        vcTotal = 0.
        lTotal = float(0.)
        w1fTotal = float(0.)
        w2fTotal = float(0.)
        rTotal = float(0.)
        numRev = 0
        numRev1 = 0
        # taking mean per month %v , mean rating, mean first feature w1, mean second feature w2 
        # merging all products of list builded in list_groupProds[]
        for z in Z:
            currentDate = datetime.datetime.strptime(z[5][0], '%d/%m/%Y')
            numMonth = '{:02d}'.format(currentDate.month)
            numYear = '{:04d}'.format(currentDate.year)
            row = row +1
            #print('default currentDate',currentDate,'numMonth',numMonth,'numYear',numYear,'firstMonth',firstMonth,'firstYear',firstYear,'row',row)
            if (numMonth == firstMonth and numYear == firstYear):
                r = float(z[0][IndexRating])
                l =  float(z[0][IndexLenFeature])  
                w1f = float(z[0][IndexWinFeature])     
                w2f = float(z[0][Index2WinFeature])     
                v = z[1]
                va = z[2]
                vb = z[3]
                vc = z[4]
#                if (not (v == 0 and va == 0)):        # IF (vtot > 0) DIMITRI SETTING (In this case win always red values)
                vTotal = vTotal+ v      # SUM not for all values, but only for these related to the reviews that valorized v+%
                vaTotal = vaTotal+ va   # SUM not for all values, but only for these related to the reviews that valorized v+%
                vcTotal = vcTotal+ vc
                numRev1 = numRev1 + 1      #  ENDIF DIMITRI SETTING
                vbTotal = vbTotal+ vb
                rTotal = rTotal + r
                lTotal = lTotal + l
                w1fTotal = w1fTotal + w1f
                w2fTotal = w2fTotal + w2f
                numRev = numRev + 1
                #print('currentDate',currentDate,'v',v,'vTotal',vTotal)
            else :
                # calculate mean Month past values 
                vMean = vTotal/float(numRev1)        # Divide not for all numReviews, but only the reviews that valorized v+%
                vaMean = vaTotal/float(numRev1)      # Divide not for all numReviews, but only the reviews that valorized v+% 
                vcMean = vcTotal/float(numRev1)
                vbMean = vbTotal/float(numRev)
                rMean = rTotal/float(numRev)
                lMean =  lTotal/float(numRev) 
                w1fMean = w1fTotal/float(numRev)
                w2fMean = w2fTotal/float(numRev)
                refDate = '01/'+ numMonth +'/' + numYear        # es "01/01/2009"
                x.append(datetime.datetime.strptime(refDate, '%d/%m/%Y'))   # date
                y.append(vMean)     # %v+
                ya.append(vaMean)   # 1 - %v+ 
                yb.append(vbMean)   # v+
                yc.append(vcMean)   # v+
                k.append(rMean)   # r
                j.append(lMean)   # l
                w1.append(w1fMean)   # w1
                w2.append(w2fMean)   # w2      
                nrev.append(numRev1)
                print(' change date refDate',refDate,' calculate vMean',vMean,'Num rev',numRev)                
                # new values
                r = float(z[0][IndexRating]) 
                l = float(z[0][IndexLenFeature])     
                v = z[1]
                va = z[2]
                vb = z[3]
                vc = z[4]
                w1f = float(z[0][IndexWinFeature])    
                w2f = float(z[0][Index2WinFeature])
                firstMonth = numMonth
                firstYear = numYear
                vTotal = v
                vaTotal = va
                vbTotal = vb
                vcTotal = vc
                rTotal = r
                lTotal = l
                w1fTotal = w1f
                w2fTotal = w2f
                numRev = 1    
                numRev1 = 1    
                #print(' (new) currentDate ',currentDate,'v',v,'vTotal',vTotal)

        #exit

#        prova = np.random.rand(100)*10
#        print ('prova random\n',prova)
        print ('w1+\n',w1)
        k = normalization (k)

        df = pd.DataFrame({'date' : x, '%vote+' : y })
        df = df.sort_values(by=['date'])
        print (df)
#        if (plotting):
#            plotting_trends1 (x,y, rating+' '+presence+'Time','b','-', True,rating+' '+presence+'Time mean %vote+')        

        dfa = pd.DataFrame({'date' : x, '1- %vote+' : ya })
        dfa = dfa.sort_values(by=['date'])
#        print (dfa)

        dfb = pd.DataFrame({'date' : x, 'vote+' : yb })
        dfb = dfb.sort_values(by=['date'])
#        print (dfb)
        
        dfc = pd.DataFrame({'date' : x, '%vote+ predict' : yc })
        dfc = dfc.sort_values(by=['date'])
#        print (dfc)
        
        dfnrev = pd.DataFrame({'date' : x, 'nrev in date' : nrev })
        dfnrev = dfnrev.sort_values(by=['date'])
#        print (dfc)              

        kf = pd.DataFrame({'date' : x, 'rating' : k })       
        kf = kf.sort_values(by=['date'])        
        
        
#        print (kf)
#        if (plotting):
#            plotting_trends1 (x,k, rating+' '+presence+'Time','r','-', True,rating+' '+presence+'Time mean Rating')
       
        jf = pd.DataFrame({'date' : x, 'len ' : j })       
        jf = jf.sort_values(by=['date'])
#        print (jf)
        mean1 = sum(j)/float(len(j))
        print ('mean len full period',rating, '{0:.3f}'.format(mean1))
        j = normalization (j)
        jf = pd.DataFrame({'date' : x, 'len Normal' : j })       
        jf = jf.sort_values(by=['date'])
#        print (jf)
#        if (plotting):
#            plotting_trends1 (x,j, rating+' '+presence+'Time','r','-', True,rating+' '+presence+'Time mean Rating')
       
        w1f = pd.DataFrame({'date' : x, ' vote+ x reviewer' : w1 })
        w1f = w1f.sort_values(by=['date'])     
#        print (w1f)
        mean1 = sum(w1)/float(len(w1))
        print ('mean vote+ x reviewer full period',rating,'{0:.3f}'.format(mean1))
        w1 = normalization (w1)
        w1f = pd.DataFrame({'date' : x, 'mean vote+ x reviewer (normal)' : w1 })
        w1f = w1f.sort_values(by=['date'])     
#        print (w1f)
#        if (plotting):
#            plotting_trends1 (x,w1, rating+' '+presence+'Time','g','-', True,rating+' '+presence+'Time MEAN H VOTES x reviews) x REVIEWER')
        
        w2f = pd.DataFrame({'date' : x, 'mean vote+ x context' : w2 })
        w2f = w2f.sort_values(by=['date'])
#        print (w2f)
#        if (plotting):
#            plotting_trends1 (x,w2, rating+' '+presence+'Time','c','-', True,rating+' '+presence+'Time MEAN H VOTES x CONTEXT')
        
        return x, y, ya, yb, yc, k,j,w1,w2, nrev   # date, %v+, 1 - %v+, v+,  %v+ pred, rating (normal), len (normal), first feature (normal), second feature, numRevs x month

        
def createUniqueProdDIMI(Z, IndexRating, IndexWinFeature, Index2WinFeature, IndexLenFeature, presence, rating, plotting):
    
    # w1 = (MEAN H VOTES x reviews) x REVIEWER -> column 8, w2 = MEAN H VOTES x CONTEXT (WIN = 4 LEFT) -> column 21
    
        x = []
        y = []
        ya = []
        yb = []
        yc = []
        k = []
        j = []
        w1 = []
        w2 = []
        nrev = []

#        zip(newX, newy, ya, yb, y_pred, dateCol)
#        x = z[0][index_feature]
#        y = v%+ = z[1][0]
#        ya =  1 - %v+  = z[2][0]
#        yb = v+ =  z[3][0]
#        yc (y_pred) =  z[4][0]
#        datecolumn


        row = 0
        print('firstDate',Z[0][5][0])
        firstDate = datetime.datetime.strptime(Z[0][5][0], '%d/%m/%Y')
        print('* firstDate',firstDate)
        firstMonth = '{:02d}'.format(firstDate.month)
        firstYear = '{:04d}'.format(firstDate.year)
        vTotal = 0.
        vaTotal = 0.
        vbTotal = 0.
        vcTotal = 0.
        lTotal = float(0.)
        w1fTotal = float(0.)
        w2fTotal = float(0.)
        rTotal = float(0.)
        numRev = 0
        numRev1 = 0
        # taking mean per month %v , mean rating, mean first feature w1, mean second feature w2 
        # merging all products of list builded in list_groupProds[]
        for z in Z:
            currentDate = datetime.datetime.strptime(z[5][0], '%d/%m/%Y')
            numMonth = '{:02d}'.format(currentDate.month)
            numYear = '{:04d}'.format(currentDate.year)
            row = row +1
            #print('default currentDate',currentDate,'numMonth',numMonth,'numYear',numYear,'firstMonth',firstMonth,'firstYear',firstYear,'row',row)
            if (numMonth == firstMonth and numYear == firstYear):
                r = float(z[0][IndexRating])
                l =  float(z[0][IndexLenFeature])  
                w1f = float(z[0][IndexWinFeature])     
                w2f = float(z[0][Index2WinFeature])     
                v = z[1]
                va = z[2]
                vb = z[3]
                vc = z[4]
                if (not (v == 0 and va == 0)):        # IF (vtot > 0) DIMITRI SETTING
                    vTotal = vTotal+ v      # SUM not for all values, but only for these related to the reviews that valorized v+%
                    vaTotal = vaTotal+ va   # SUM not for all values, but only for these related to the reviews that valorized v+%
                    vcTotal = vcTotal+ vc
                    numRev1 = numRev1 + 1      #  ENDIF DIMITRI SETTING
                vbTotal = vbTotal+ vb
                rTotal = rTotal + r
                lTotal = lTotal + l
                w1fTotal = w1fTotal + w1f
                w2fTotal = w2fTotal + w2f
                numRev = numRev + 1
                #print('currentDate',currentDate,'v',v,'vTotal',vTotal)
            else :
                # calculate mean Month past values 
                vMean = vTotal/float(numRev1)        # Divide not for all numReviews, but only the reviews that valorized v+%
                vaMean = vaTotal/float(numRev1)      # Divide not for all numReviews, but only the reviews that valorized v+% 
                vcMean = vcTotal/float(numRev1)
                vbMean = vbTotal/float(numRev)
                rMean = rTotal/float(numRev)
                lMean =  lTotal/float(numRev) 
                w1fMean = w1fTotal/float(numRev)
                w2fMean = w2fTotal/float(numRev)
                refDate = '01/'+ numMonth +'/' + numYear        # es "01/01/2009"
                x.append(datetime.datetime.strptime(refDate, '%d/%m/%Y'))   # date
                y.append(vMean)     # %v+
                ya.append(vaMean)   # 1 - %v+ 
                yb.append(vbMean)   # v+
                yc.append(vcMean)   # v+
                k.append(rMean)   # r
                j.append(lMean)   # l
                w1.append(w1fMean)   # w1
                w2.append(w2fMean)   # w2      
                nrev.append(numRev1)
                # print(' change date refDate',refDate,' calculate vMean',vMean,'Num rev',numRev)                
                # new values
                r = float(z[0][IndexRating]) 
                l = float(z[0][IndexLenFeature])     
                v = z[1]
                va = z[2]
                vb = z[3]
                vc = z[4]
                w1f = float(z[0][IndexWinFeature])    
                w2f = float(z[0][Index2WinFeature])
                firstMonth = numMonth
                firstYear = numYear
                vTotal = v
                vaTotal = va
                vbTotal = vb
                vcTotal = vc
                rTotal = r
                lTotal = l
                w1fTotal = w1f
                w2fTotal = w2f
                numRev = 1    
                numRev1 = 1    
                #print(' (new) currentDate ',currentDate,'v',v,'vTotal',vTotal)

        #exit

#        prova = np.random.rand(100)*10
#        print ('prova random\n',prova)
        print ('w1+\n',w1)
        k = normalization (k)

        df = pd.DataFrame({'date' : x, '%vote+' : y })
        df = df.sort_values(by=['date'])
        print (df)
#        if (plotting):
#            plotting_trends1 (x,y, rating+' '+presence+'Time','b','-', True,rating+' '+presence+'Time mean %vote+')        

        dfa = pd.DataFrame({'date' : x, '1- %vote+' : ya })
        dfa = dfa.sort_values(by=['date'])
#        print (dfa)

        dfb = pd.DataFrame({'date' : x, 'vote+' : yb })
        dfb = dfb.sort_values(by=['date'])
#        print (dfb)
        
        dfc = pd.DataFrame({'date' : x, '%vote+ predict' : yc })
        dfc = dfc.sort_values(by=['date'])
#        print (dfc)
        
        dfnrev = pd.DataFrame({'date' : x, 'nrev in date' : nrev })
        dfnrev = dfnrev.sort_values(by=['date'])
#        print (dfc)              

        kf = pd.DataFrame({'date' : x, 'rating' : k })       
        kf = kf.sort_values(by=['date'])        
        
        
#        print (kf)
#        if (plotting):
#            plotting_trends1 (x,k, rating+' '+presence+'Time','r','-', True,rating+' '+presence+'Time mean Rating')
       
        jf = pd.DataFrame({'date' : x, 'len ' : j })       
        jf = jf.sort_values(by=['date'])
#        print (jf)
        mean1 = sum(j)/float(len(j))
        print ('mean len full period',rating, '{0:.3f}'.format(mean1))
        j = normalization (j)
        jf = pd.DataFrame({'date' : x, 'len Normal' : j })       
        jf = jf.sort_values(by=['date'])
#        print (jf)
#        if (plotting):
#            plotting_trends1 (x,j, rating+' '+presence+'Time','r','-', True,rating+' '+presence+'Time mean Rating')
       
        w1f = pd.DataFrame({'date' : x, ' vote+ x reviewer' : w1 })
        w1f = w1f.sort_values(by=['date'])     
#        print (w1f)
        mean1 = sum(w1)/float(len(w1))
        print ('mean vote+ x reviewer full period',rating,'{0:.3f}'.format(mean1))
        w1 = normalization (w1)
        w1f = pd.DataFrame({'date' : x, 'mean vote+ x reviewer (normal)' : w1 })
        w1f = w1f.sort_values(by=['date'])     
#        print (w1f)
#        if (plotting):
#            plotting_trends1 (x,w1, rating+' '+presence+'Time','g','-', True,rating+' '+presence+'Time MEAN H VOTES x reviews) x REVIEWER')
        
        w2f = pd.DataFrame({'date' : x, 'mean vote+ x context' : w2 })
        w2f = w2f.sort_values(by=['date'])
#        print (w2f)
#        if (plotting):
#            plotting_trends1 (x,w2, rating+' '+presence+'Time','c','-', True,rating+' '+presence+'Time MEAN H VOTES x CONTEXT')
        
        return x, y, ya, yb, yc, k,j,w1,w2, nrev   # date, %v+, 1 - %v+, v+,  %v+ pred, rating (normal), len (normal), first feature (normal), second feature, numRevs x month

        
        
        # THE GOAL IS: VERIFY THE CORRELATION BETWEEN VOTES AND MAIN FEATURES IN THE 6 CLASSES OF PROD (IN HIGH PERMANENCE AND lOW RATING MUST BE BETTER!)
        
        # order by date and take all prods of a month
        # create syntethic prod SP such as syntesis of group of prod of current month
        # calculate mean rating, mean %hv, sd %hv of SP


	  	

feature_names_senti = ['RATING','SAMPLE ENT MAX1','RATIO USEFUL / ALL','1/2 RATIO USEFUL / ALL','DENS POS:NVJ','# REVIEWS x PROD','# REVIEWS x REVIEWER', '% VOTES x REVIEWER', 
             '# (MEAN H VOTES x reviews) x REVIEWER', 'CURRENT LIFETIME PROD (month)', 'FULL LIFETIME PROD  (month)', '# H VOTES/# T VOTES x PROD', 'MEAN H VOTES x CONTEXT (WIN = 2)', 
             'COEFF VARIAT H VOTES x CONTEXT (WIN = 2)', 'DIFF H (CURRENT VOTE' 'MEAN VOTES) (WIN = 2)', 'MEAN H VOTES x CONTEXT (WIN = 2 LEFT)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 2 LEFT)', 
             'DIFF (CURRENT VOTE  MEAN VOTES) (WIN = 2 LEFT)', 'MEAN H VOTES x CONTEXT (WIN = 4)', 
             'COEFF VARIAT H VOTES x CONTEXT (WIN = 4)',' DIFF (CURRENT VOTE MEAN VOTES) (WIN = 4)', 'MEAN H VOTES x CONTEXT (WIN = 4 LEFT)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 4 LEFT)', 
             'DIFF (CURRENT VOTE MEAN VOTES) (WIN = 4 LEFT)', 'DENS. COMPOUND WORDS',	'LEN TITLE/LEN TEXT',	'SENT: POSNEGSUM',	
             'SENT: NUMPOS/ALL','SENT: NUMNEG/ALL',	'SENT: NUMCONTINUOUSPOS',	'SENT: NUMCONTINUOUSNEG',	'SENT 1/2: POSNEGSUM' ,	
             'SENT 1/2: NUMPOS/ALL' ,'SENT 1/2: NUMNEG/ALL' ,'LENGTH REVIEW','LENGTH NORMAL REVIEW']

feature_names = ['RATING','SAMPLE ENT MAX1','RATIO USEFUL / ALL','1/2 RATIO USEFUL / ALL','DENS POS:NVJ','# REVIEWS x PROD','# REVIEWS x REVIEWER', '% VOTES x REVIEWER', 
             '# (MEAN H VOTES x reviews) x REVIEWER', 'CURRENT LIFETIME PROD (month)', 'FULL LIFETIME PROD  (month)', '# H VOTES/# T VOTES x PROD', 'MEAN H VOTES x CONTEXT (WIN = 2)', 
             'COEFF VARIAT H VOTES x CONTEXT (WIN = 2)', 'DIFF H (CURRENT VOTE' 'MEAN VOTES) (WIN = 2)', 'MEAN H VOTES x CONTEXT (WIN = 2 LEFT)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 2 LEFT)', 
             'DIFF (CURRENT VOTE MEAN VOTES) (WIN = 2 LEFT)', 'MEAN H VOTES x CONTEXT (WIN = 4)', 
             'COEFF VARIAT H VOTES x CONTEXT (WIN = 4)',' DIFF (CURRENT VOTE MEAN VOTES) (WIN = 4)', 'MEAN H VOTES x CONTEXT (WIN = 4 LEFT)', 'COEFF VARIAT H VOTES x CONTEXT (WIN = 4 LEFT)', 
             'DIFF (CURRENT VOTE  MEAN VOTES) (WIN = 4 LEFT)', 'DENS. COMPOUND WORDS',	'LEN TITLE/LEN TEXT','LENGTH REVIEW','LENGTH NORMAL REVIEW']


        
        
def core(X,y,typeExp,thresh, feature_names_local):
    X = np.array(X)
    y = np.array(y)  
    statistic (y)
    print(typeExp, 'Dim input matrix X', X.shape, 'Dim input matrix y', y.shape)
    X_train, X_test, y_train, y_test = \
        get_data_split(X,y,0.2)  # X_train, X_test, y_train, y_test of CROSS VALIDATION PROCESS (all data extracted from primitive training X,y)
    grid = tune_models_hyperparams(X_train, y_train, X_test, y_test, models, cv=3,  # cross-validation of 
                                   verbose=2, n_jobs=-1)
    print_grid_results(grid,typeExp)                        # print best score of cross validation and best params..
    best_model = get_best_model(typeExp, grid, X_test, y_test)    # best model name with respect to Mean squared error regression loss    
    test_dataset('', grid, X_test, y_test, X_train)       
    if (thresh > -1.):
        #featureExtractionPost (X_train,X_test,y_train,y_test,best_model,thresh, feature_names_local, 1 )
        featureExtractionrfPost (X_train,X_test,y_train,y_test,grid,thresh, feature_names_local)   # <------------- IMPORTANT: Calculate MSE and R^2 only for the features over threshold
        return best_model
    else:
        return best_model


def simpleOLS(X,y,feature_names_local):
    X = np.array(X, dtype='float')
    y = np.array(y, dtype='float')    
    print('Dim input matrix X', X.shape, 'Dim input matrix y', y.shape)
    X_train, X_test, y_train, y_test = \
            get_data_split(X,y,0.2)  # X_train, X_test, y_train, y_test of CROSS VALIDATION PROCESS (all data extracted from primitive training X,y)
    X_train = np.array(X_train, dtype='float')
    y_train = np.array(y_train, dtype='float').ravel()
    print(y_train.shape,X_train.shape)
    modelOLS = sm.OLS(y_train,X_train)
    results = modelOLS.fit()
    print('params=',results.params)
    print(results.summary())
    for feature_list_index in range(len(feature_names_local)):
            print('x',feature_list_index+1,' ',feature_names_local[feature_list_index])

# PROBLEM WITH RESHAPE X_train !!!!!!
#    yp=modelOLS.predict(X_train)
#    resid=y_train-yp
#    rss=np.sum(resid**2)
#    MSE=rss/(results.nobs-2)
#    print('MSE=',MSE)




def featureExtractionrfPost (X_train,X_test,y_train,y_test,grid,thresh, feature_names_local ):
    
        for name, model in grid.items():
            if (name == 'RandomForest'):  
                    print (' chosen = ', name)      
                    best_model = model.best_estimator_.steps[1][1]
            break

        try:
            feature_importance = best_model.feature_importances_ 
        except AttributeError:
            print (' feature_importances_ Not exist for the model ', best_model)
            return
    #    for feature in zip(feature_names, best_model.best_estimator_.feature_importances_):
        #print('\nName Model ', best_model.__class__.__name__,' Features \n')
#        zipped = zip(feature_names_local, feature_importance)
#        for feature in zipped:
#            print(feature)       
        zipSort = sorted(zip(feature_importance,feature_names_local), key=lambda x: x[0], reverse=True)
       # print('\nFeatures ordered: \n',sorted(zip(feature_importance,feature_names_local), key=lambda x: x[0]))            
        print('\nName Model ', best_model.__class__.__name__,' Features Ordered\n')
        for feature in zipSort:
            print(feature)     
        #if (index == 1):
        print(' threshold=',thresh,'\n')
        sfm = SelectFromModel(best_model, threshold=thresh, prefit=True)     # <---------------- PROBLEM HERE (THE model has been already fitted but here not. Must be added prefit=True )
        # Train the selector
        for feature_list_index in sfm.get_support(indices=True):
            print(feature_names_local[feature_list_index],'Relevance ','{0:.3f}'.format(feature_importance[feature_list_index]))
        X_important_train = sfm.transform(X_train)
        X_important_test = sfm.transform(X_test)        
        print(X_important_train.shape)
        print(X_important_test.shape)
        
#            'RandomForest__n_estimators':   [500, 1000],
#            'RandomForest__max_depth':      [15, 18]    
        # IMPORTANT: IT NEEDS TO RE-CALCULATING ALL : FITTING AND PREDICT OF THE OLD MODEL APPLIED TO THE FEATURES SELECTED  
        # bECAUSE SelectFromModel () DOESN'T RETURN predict
        regr = RandomForestRegressor(random_state=0, n_estimators=500, max_depth=15)
        regr.fit(X_important_train,y_train)
        y_pred = regr.predict(X_important_test)
        print ('MSE','{0:.3f}'.format(mean_squared_error(y_test, y_pred)))
        print ('R2', '{0:.3f}'.format(r2_score(y_test, y_pred)))        


# RUN ONLY WITH feature_importances_ ATTRIBUTE
def featureExtractionPost (X_train,X_test,y_train,y_test,best_model,thresh, feature_names_local, index ):
        model = best_model.best_estimator_.steps[index][1]       # the model is positioned in different places with respect to the pipeline used in the Test
        try:
            feature_importance = model.feature_importances_ 
        except AttributeError:
            print (' feature_importances_ Not exist for the model ', model.__class__.__name__)
            return
    #    for feature in zip(feature_names, best_model.best_estimator_.feature_importances_):
        print('\nName Model ', model.__class__.__name__,' Features \n')
        zipped = zip(feature_names_local, feature_importance)
        for feature in zipped:
            print(feature)       
        zipSort = sorted(zip(feature_importance,feature_names_local), key=lambda x: x[0], reverse=True)
       # print('\nFeatures ordered: \n',sorted(zip(feature_importance,feature_names_local), key=lambda x: x[0]))            
        print('\nFeatures Ordered\n')
        for feature in zipSort:
            print(feature)     
        #if (index == 1):
        print(' threshold=',thresh,'\n')
        sfm = SelectFromModel(model, threshold=thresh, prefit=True)     # <---------------- PROBLEM HERE (THE model has been already fitted but here not. Must be added prefit=True )
        # Train the selector
        for feature_list_index in sfm.get_support(indices=True):
            print('**',feature_names_local[feature_list_index],'Importance ',feature_importance[feature_list_index])
        X_important_train = sfm.transform(X_train)
        X_important_test = sfm.transform(X_test)        
        print(X_important_train.shape)
        print(X_important_test.shape)
        
#            'RandomForest__n_estimators':   [500, 1000],
#            'RandomForest__max_depth':      [15, 18]    
        # IMPORTANT: IT NEEDS TO RE-CALCULATING ALL : FITTING AND PREDICT OF THE OLD MODEL APPLIED TO THE FEATURES SELECTED  
        # bECAUSE SelectFromModel () DOESN'T RETURN predict
        regr = RandomForestRegressor(random_state=0, n_estimators=500, max_depth=15)
        regr.fit(X_important_train,y_train)
        y_pred = regr.predict(X_important_test)
        print ('MSE',mean_squared_error(y_test, y_pred)) 
        print ('R2',r2_score(y_test, y_pred)) 



        
##
#   if (typePrefilter == 3):
#            SGDRegressor(penalty="elasticnet"), threshold='median'
#   A pipeline set Feature Selection based on SGDRegressor, concatening the model winning the race of best model 
##                   
def featureExtractionPre (X, y, X_test, y_test, modelName, feature_names_local, **common_grid_kwargs):    
           grids = {}                                                          # empty lists/dicts
           for model in models:
               print (' name model = ', model['title'])
               if (model['title'] == modelName):  
                    print (' chosen = ', model['title'])                   
                    clf1=SelectFromModel(SGDRegressor(penalty="l2"), threshold='0.8*median')
                    clf1.fit(X, y) 
                    print('Dim feature Original matrix X ',X.shape)
                    #print('Feature_importances ',clf1.estimator_.feature_importances_)
                    X_newTrain,X_newTest = featureExtract1_corpus(X,X_test,y,clf1, feature_names_local, "0.8*median", True)                    
                    pipe = Pipeline([                                           # sequence of operations Standar Scale transform your data in noemal distribution will have a mean value 0 and standard deviation of 1. G
                                ('feature_selection', clf1),
                                ("scale", StandardScaler()),
                                (model['name'], model['model'])   ])
                    clf = GridSearchCV(pipe, param_grid=model['param_grid'], **common_grid_kwargs)
                    grids[model['name']] = clf.fit(X, y)     
                    train_score=clf.score(X,y)
                    test_score=clf.score(X_test, y_test)  # <-------- using X_test, Not X_newTest
                    best_params = clf.best_params_              
                    print (' train_score = ',train_score,' test_score = ',test_score, ' best_params= ',best_params)
                    model['train_score'] = train_score
                    model['test_score'] = test_score
                    for key, value in best_params.items():
                        #print (key,value)
                        model['best_params'][key] = value
                    print ('model[best_params]',model['best_params'])
                    joblib.dump(grids[model['name']], './{}.pkl'.format(model['name']))
           return grids, X_newTrain,X_newTest

##
#   if (typePrefilter == 1):
#            SelectKBest( -> Chi2
#   if (typePrefilter == 2):
#            SelectKBest( -> ANOVA
#    No Pipeline: first: Feature Selection SelectKBest, Fit and get the features selected. 
#    After launch best model limited to the features selected
##           
def featureExtractionPre1 (X, y, X_test, y_test, modelName, numExtract, typePrefilter, feature_names_local, **common_grid_kwargs):    
           grids = {}                                                          # empty lists/dicts
           for model in models:
               print (' name model = ', model['title'])
               if (model['title'] == modelName):  
                    print (' chosen = ', model['title'])  
                    if (typePrefilter == 1):      
                       # score_func=chi2 IT'S NOT GOOD BECAUSE it DOESN'T RUN WITH NEGATIVE VALUES
                       # mutual_info_regression(X, y, discrete_features=’auto’, n_neighbors=3, copy=True, random_state=None)                       
                        b1 = SelectKBest(score_func= mutual_info_regression, k=numExtract)    # es k=4
                        X_newTrain,X_newTest = featureExtract1_corpus(X,X_test,y,b1,feature_names_local,'mutual_info_regression', False)
                    if (typePrefilter == 2):
                        b2 = SelectKBest(f_classif, k=numExtract)    # es k=4 ANOVA
                        X_newTrain,X_newTest = featureExtract1_corpus(X,X_test,y,b2,feature_names_local,'ANOVA', False)                        
                    pipe = Pipeline([                                           # sequence of operations Standar Scale transform your data in noemal distribution will have a mean value 0 and standard deviation of 1. G
                                 ("scale", StandardScaler()),
                                 (model['name'], model['model'])   ])
                    clf = GridSearchCV(pipe, param_grid=model['param_grid'], **common_grid_kwargs)
                    grids[model['name']] = clf.fit(X_newTrain, y)     
                    train_score=clf.score(X_newTrain,y)
                    #test_score=clf.score(X_test, y_test)  Using featureExtract1_corpus loop mask to rebuild X_test restricted to only the features selected
                    best_params = clf.best_params_              
                    print (' train_score = ',train_score,' test_score = ',best_params)
                    model['train_score'] = train_score
                    #model['test_score'] = test_score
                    for key, value in best_params.items():
                        #print (key,value)
                        model['best_params'][key] = value
                    print ('model[best_params]',model['best_params'])
                    joblib.dump(grids[model['name']], './{}.pkl'.format(model['name']))
           return grids, X_newTrain, X_newTest


def featureExtract1_corpus(X_train,X_test,y_old,b,feature_names_local, typeSelect, alreadyFit):    
#    print (' y = ',y_old)
#    print (' X_train = ',X_train)
#    print (' X_test = ',X_test)
    if (alreadyFit):
        X_newTrain = b.transform(X_train)
        X_newTest = b.transform(X_test)
    else:    
        y = np.asarray(y_old, dtype="|S6")       
        X_newTrain = b.fit_transform(X_train,y)
        X_newTest = b.transform(X_test)
    X_newTrain =np.array(X_newTrain);
    print (' X_newTrain[0] = ',X_newTrain[0])
    print('Dim feature selection matrix X_newTrain ',X_newTrain.shape)
#    if (alreadyFit):    # SelectKBest has no attribute 'threshold_'
#        print(' threshold= ',b.threshold_)
    mask = b.get_support() #list of booleans
    new_features = [] # The list of your K best features
    origin_features = [] 
    for bool, feature in zip(mask, feature_names_local):
        origin_features.append(feature)
        if bool:
            new_features.append(feature)
    print(' \nSelected features Select ('+typeSelect+')', new_features)
    print(' \nOriginal features ', origin_features)
    return X_newTrain, X_newTest


def coreFeatureExtraction(X,y,typeExp,modelName,typePrefilter, feature_names_local): # typePrefilter == 1 or 2 or 3
    X = np.array(X)
    y = np.array(y)    
    print(typeExp, 'Dim input matrix X', X.shape, 'Dim input matrix y', y.shape)
    X_train, X_test, y_train, y_test = \
        get_data_split(X,y,0.2)  # X_train, X_test, y_train, y_test of CROSS VALIDATION PROCESS (all data extracted from primitive training X,y)    
    if (typePrefilter <= 2):
        grids, X_newTrain, X_newTest = featureExtractionPre1(X_train, y_train, X_test, y_test, modelName, 8, typePrefilter, feature_names_local, cv=3,  # cross-validation of 
                                            verbose=2, n_jobs=-1)
        index = 1
    if (typePrefilter == 3):
        grids, X_newTrain, X_newTest = featureExtractionPre(X_train, y_train, X_test, y_test, modelName, feature_names_local, cv=3,  # cross-validation of 
                                            verbose=2, n_jobs=-1)
        X_newTest = X_test   # <--------------------Not change X_test
        index = 2
    print_grid_results(grids,typeExp)     
    best_model = get_best_model(typeExp, grids, X_newTest, y_test)    # best model name with respect to Mean squared error regression loss    
    test_dataset('', grids, X_newTest, y_test, X_newTrain)           
    #featureExtractionPost (X_train,X_test,y_train,best_model,thresh, feature_names_local )
    featureExtractionPost (X_newTrain,X_newTest,y_train,y_test,best_model,0.3, feature_names_local, index )


# LIENAR POSITIVE/NEGATIVE CORRELATION
def All_pcc(X,y,feature_names_local):       # numFeatures = 28 No Senti, numFeatures = 36 Senti
#        print(' ',X[0])
        numFeatures = len(feature_names_local)
        print ('len',numFeatures)
        X = np.array(X)
        y = np.array(y)    
        thresPositiveCorr = 0.60
        thresNegativeCorr = - 0.60
        X_T = X.transpose()
        y_T = np.array(y.transpose())
        print('X_T ',X_T)        
        print('y_T ',y_T)        
        i = 0;
        print('X_T.shape=',X_T.shape)
        for feature_list_index in list(range(numFeatures)):
#           print(i,') ',feature_names[feature_list_index],'first value ',X[0][feature_list_index])
#           print(X_T[feature_list_index].shape)
           pearson_y = scstat.pearsonr(X_T[feature_list_index].ravel(),y_T.ravel())[0]
           if (pearson_y > (thresPositiveCorr - 0.3) or pearson_y < (thresNegativeCorr + 0.3)):
               print(i,') ','pearson (',feature_names_local[feature_list_index],', %votes =',pearson_y)
           i=i+1
           for feature_list_index1 in list(range(numFeatures)):
               if (feature_list_index != feature_list_index1):
                   pearson = scstat.pearsonr(X_T[feature_list_index].ravel(),X_T[feature_list_index1].ravel())[0]
                   if (pearson > thresPositiveCorr or pearson < thresNegativeCorr):
                       print('pearson (',feature_names_local[feature_list_index],',',feature_names_local[feature_list_index1],')= ',pearson)
                

#    Plotting Only one type of ARIMA models (p,0,1)  red dotted
def ArimaModels (y, splitDays,label):   

#    y = [i - 0.5 for i in y_orig]
    split_point = len(y) - splitDays
    print('split_point=',split_point)
    dataset, validation = y[0:split_point], y[split_point:]
    print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
    differenced = difference(y, 1)
    # fit model
    plt.ylabel(label) 
    plt.plot(y, 'b') 
    plt.show() 
    for p in range(8):
            try:
                print('p=',p)
                model = ARIMA(differenced, order=(p,0,1))
                model_fit = model.fit(transparams=True,disp=0)
                print('p=',p,' OK!')
            # print summary of fit model
                print(model_fit.summary())
    #            fig=plt.gcf()
                #print(model_fit.fittedvalues)
                plt.ylabel('balanced %vote+  p='+str(p)) 
                plt.plot(y, 'b', model_fit.fittedvalues, 'r--') 
                plt.show() 
                #name='ARIMA v+percent p='+str(p+ ' d=0 q=1')
                #fig.savefig('C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result11012019\\'+name+'.jpg', dpi=300)
    
            except:
                pass
            
            
#    Plotting original curve y and ARIMA models (p,d,q)  red dotted
#  SEE https://people.duke.edu/~rnau/411arim.htm
#   https://machinelearningmastery.com/make-sample-forecasts-arima-python/  CAPT 4.  One-Step Out-of-Sample Forecast 
def ArimaModels2 (y_orig, splitDays, theme,col, label):   # label = '%vote+'  or label = 'vote+'

    #y = [i - 0.5 for i in y_orig]
    y = y_orig
    split_point = len(y) - splitDays
    print('split_point=',split_point)
    dataset, validation = y[0:split_point], y[split_point:]
    print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
    differenced = difference(y, 1)
    fig=plt.gcf()
    y_size = len(y)/25
    plt.rcParams['figure.figsize'] = (y_size, 5)
    plt.ylabel(label) 
    plt.plot(y_orig, col)       # plot original curve
    plt.show() 
    name = 'original'
    fig.savefig('C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result11012019\\'+theme+'orig.jpg')

    for p in range(2):
        for d in range(2):
            for q in range(3):
                if (p==0 and q==0 and d<=1):
                    continue
                try:
                    print('p=',p,'q=',q,'d=',d)
                    model = ARIMA(differenced, order=(p,d,q))
                    model_fit = model.fit(transparams=True)     # fit ARIMA (p,d,q)
                    print('p=',p,'q=',q,'d=',d,' OK!')
                    print(model_fit.summary())  # print summary of fit model
                    y_fit = [-i for i in model_fit.fittedvalues]  # curve Arima estimation                  
                    #    Plotting original curve y and ARIMA model red dotted
                    fig=plt.gcf()
                    #print(model_fit.fittedvalues)
                    y_size = len(y)/25
                    plt.rcParams['figure.figsize'] = (y_size, 5)
                    plt.ylabel(' '+label+' ARIMA p='+str(p)+' d='+str(d)+' q='+str(q)) 
                    plt.plot(y, col, y_fit, 'r--')      # plot original curve y and Arima fitting y_fit
                    plt.show() 
                    name=theme+' ARIMA '+label+' p='+str(p)+' d='+str(d)+' q='+str(q)
                    fig.savefig('C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result11012019\\'+name+'.jpg', dpi=300)
    
                except:
                    pass


# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]



def main(args) : 
    
    # PROVE  INPUT: 2 --opt_arg 3 --predict
#    print(args.opt_pos_arg)
#    print(args.opt_arg)
#    print(args.switch)
#    exit;
    
    choice = args.input;
    print("Argument values:", choice )   
    selectionAfter = False;
   
    
    if choice == 'general_test':    # INPUT:  + SENTIMENT
        X, y = DataGenTest (1) 
        simpleOLS(X,y,feature_names_senti)
        
        best_model_ = core(X,y,'general_test',0.03,feature_names_senti)
        modelName = best_model_.best_estimator_.steps[1][1].__class__.__name__
        print('modelName Chosen: '+modelName)     
        coreFeatureExtraction(X, y, 'general_test'+' - Best features',modelName,2,feature_names_senti)      # chose : 1,2 or 3
        All_pcc(X,y,36,feature_names_senti)  
 
        newTrainX, newTrainy = list_sameLen_textReview(IT.zip_longest(X, y), 3, 27, 200, 0)           # TRAINING FILE SAME lEN
        All_pcc(newTrainX, newTrainy,36,feature_names_senti)  

        newTrainX, newTrainy = list_sameRating (IT.zip_longest(X, y), 0, 1)
        All_pcc(newTrainX, newTrainy,36,feature_names_senti)
        best_model = core(newTrainX, newTrainy, 'test rating (1) ',0.03,feature_names_senti)
        modelName = best_model.best_estimator_.steps[1][1].__class__.__name__
        print('modelName Chosen: ',modelName)     
        coreFeatureExtraction(newTrainX, newTrainy,'test rating (1)  '+' - Best features',modelName,2,feature_names_senti)      # chose : 1,2 or 3
        
        newTrainX, newTrainy = list_sameColumn (IT.zip_longest(X, y), 2, 0.6, True, feature_names_senti)
        All_pcc(newTrainX, newTrainy,36,feature_names_senti)
        best_model = core(newTrainX, newTrainy, 'test High Usefull/All (0.6) ',0.03,feature_names_senti)
        modelName = best_model.best_estimator_.steps[1][1].__class__.__name__
        print('modelName Chosen: ',modelName)     
        coreFeatureExtraction(newTrainX, newTrainy,'test High Usefull/All (0.6) '+' - Best features',modelName,2,feature_names_senti)      # chose : 1,2 or 3

        newTrainX, newTrainy = list_sameColumn (IT.zip_longest(X, y), 4, 0.6, True, feature_names_senti)
        All_pcc(newTrainX, newTrainy,36,feature_names_senti)
        best_model = core(newTrainX, newTrainy, 'test DENS POS:NVJ (0.6) ',0.03,feature_names_senti)
        modelName = best_model.best_estimator_.steps[1][1].__class__.__name__
        print('modelName Chosen: ',modelName)     
        coreFeatureExtraction(newTrainX, newTrainy,'test DENS POS:NVJ (0.6)  '+' - Best features',modelName,2,feature_names_senti)      # chose : 1,2 or 3


        X0, X, y_, y__ = DataGenTest (4)  
        # TEST ARIMA %VOTE+
        prodID = 'B003ZSP0WW' 
        newX, newy = list_sameProd (IT.zip_longest(X0, y_),34,prodID,feature_names_senti)
        newy = np.array(newy)    
       # print('newy: ',newy) 
       # print(newy.shape[0]) 
        newy1 = plotting_trends (newy, date, 'prodID=B003ZSP0WW','b','-', False,'%vote+')
        ArimaModels2 (newy1,7,'prodID=B003ZSP0WW','b','%vote+')
                
        # TEST ARIMA %VOTE+
        prodID = 'B0041Q38NU' 
        newX, newy = list_sameProd (IT.zip_longest(X0, y_),34,prodID,feature_names_senti)
        newy = np.array(newy)
        newy1 = plotting_trends (newy, date, 'prodID=B0041Q38NU','b','-', False,'%vote+')
        ArimaModels2 (newy1,7,'prodID=B0041Q38NU','b','%vote+')

        # TEST ARIMA VOTE+
        prodID = 'B0041Q38NU' 
        newX, newy = list_sameProd (IT.zip_longest(X0, y__),34,prodID,feature_names_senti)
        newy = np.array(newy)    
        newy1 = plotting_trends (newy, date, 'prodID=B0041Q38NU','g','-', False,'vote+')
        ArimaModels (newy1,7,'vote+')
        k = range(1,8)
        k = sm.add_constant(k)
        print(k)
        k = np.array(k)
        print(k.shape)
        

        
    elif choice == 'len_test':        # INPUT:  + SENTIMENT                                           # 1 Experiment : training Set with reviews same review length and others...
        X, y, newTestX, newTesty, newTestX2, newTesty2, newTestX3, newTesty3 = DataGenTest (2)        # Training DATA
        newTrainX, newTrainy = list_sameLen_textReview(IT.zip_longest(X, y), 3, 35, 50, 0)           # TRAINING FILE SAME lEN
        best_model_ = core(newTrainX, newTrainy, 'len_test: 50 ',-1,feature_names_senti)
    
    elif choice == 'len_test No Senti':    # INPUT: NO SENTIMENT                                                    # 1 Experiment : training Set with reviews same review length and others...
        X, y = DataGenTest (3)        # Training DATA
        print('\nTrain  ',X[0])
        print('\nLen Normal  ',X[0][27],'\nLen ',X[0][26])
        print(X.shape)   
        All_pcc(X,y,28,feature_names)  
        
        newTrainX, newTrainy = list_sameLen_textReview(IT.zip_longest(X, y), 3, 27, 200, 0)           # TRAINING FILE SAME lEN
        All_pcc(newTrainX, newTrainy,28,feature_names)  
        newTrainX, newTrainy = list_sameRating (IT.zip_longest(X, y), 0, 1)
        All_pcc(newTrainX, newTrainy,28,feature_names)  
        
        best_model_ = core(newTrainX, newTrainy, 'No Senti len_test: 50 Normal Len',-1,feature_names)
        newTrainX, newTrainy = list_sameLen_textReview(IT.zip_longest(X, y), 3, 26, 50, 0)           # TRAINING FILE SAME lEN
        best_model_ = core(newTrainX, newTrainy, 'No Senti len_test: 50 Original Len',-1,feature_names)        

        newTrainX, newTrainy = list_sameLen_textReview(IT.zip_longest(X, y), 3, 27, 50, 0)           # TRAINING FILE SAME lEN
        nameExp = 'No Senti len_test: 50 Normal Len ';
        best_model = core(newTrainX, newTrainy, nameExp, 0.02,feature_names)
        if (selectionAfter):
            print('selectionAfter: '+selectionAfter)     
            modelName = best_model.best_estimator_.steps[1][1].__class__.__name__
            print('modelName Chosen: '+modelName)     
            coreFeatureExtraction(newTrainX, newTrainy, nameExp+' - Best features',modelName,2,feature_names)      # chose : 1,2 or 3

        big_y = [i * 100 for i in y]
        best_model_ = core(X,big_y,'general_test',0.05,feature_names)     # ONLY A TEST!!!! the same test above but the y is multiply for a factor 100
 
        best_model_ = core(X,y,'general_test',0.05,feature_names)         # the last arg is the threshold of features selection
        if (selectionAfter):
            print('selectionAfter: ',selectionAfter)     
            modelName = best_model.best_estimator_.steps[1][1].__class__.__name__
            print('modelName Chosen: ',modelName)     
            coreFeatureExtraction(X,y,'general_test'+' - Best features',modelName,3,feature_names)      # chose : 1,2 or 3
       
    
    
    
    pathfile = "C:\\UniDatiSperimentali\\PRODUCT REVIEW RUGGIERO\\VOTE-RATING-ENTROPY\\OutNew\\Result11012019\\Result"
    predict_save_results (newTestX, newTesty, newTrainX, newTrainy, best_model_, 'len 50', 'Test y %votes', pathfile)
    predict_save_results (newTestX2, newTesty2, newTrainX, newTrainy, best_model_, 'len 50', 'Test y2  %votes', pathfile)
    predict_save_results (newTestX3, newTesty3, newTrainX, newTrainy, best_model_, 'len 50', 'Test y3  %votes', pathfile)
     
    
    

    

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

 
