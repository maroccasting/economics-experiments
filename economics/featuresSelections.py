
# SEE https://machinelearningmastery.com/feature-selection-machine-learning-python/


# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
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

"""
Created on Wed Nov 21 01:55:15 2018

Statistical tests can be used to select those features that have the strongest relationship with the output variable.
The example below uses the chi squared (chi^2) statistical test for non-negative features to select 4 of the best features 

@author: paolo
"""

# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
print(features)


"""

The Recursive Feature Elimination (or RFE) works by recursively removing attributes and building a model on those attributes that remain.
It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute

@author: paolo
"""

from sklearn.svm import SVR
from sklearn.feature_selection import RFE
# feature extraction
model = SVR(kernel='rbf', gamma=1.0/10, C=100.0, cache_size=1000)   # substitute with the best model
rfe = RFE(model, 3)
fit = rfe.fit(X, y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_
# USING https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html TO CONCATENATE THE FEATURES SELECTED AND LAUNCH THEM

"""

Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into a compressed form.

@author: paolo
"""


from sklearn.decomposition import PCA
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)



"""

Bagged decision trees like Random Forest and Extra Trees can be used to estimate the importance of features.

@author: paolo
"""

from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)

