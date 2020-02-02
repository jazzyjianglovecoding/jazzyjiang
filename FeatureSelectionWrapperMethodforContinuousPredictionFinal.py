
# coding: utf-8

# In[1]:

# Some methods to show plot in notebook:
get_ipython().magic(u'matplotlib inline')
# Import libraries to be used}
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
import seaborn as sns
import scipy as scipy


# In[7]:

df= pd.read_csv('featureselectioncontinuous.csv')
# Using panda to read the csv file
X=df.iloc[:,1:127]
y=df.iloc[:,0]


# In[8]:

feature_names = list(X.columns.values)


# In[9]:

X=X.values
y=y.values


# In[11]:

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
lr = linear_model.LinearRegression()

sfs = SFS(lr, 
          k_features=30, 
          forward=True, 
          floating=False, 
          scoring='r2',
          cv=4)

sfs = sfs.fit(X, y)
print('\nSequential Floating Forward Selection (k=30):')
print(sfs.k_feature_idx_)
print('CV Score:')
print(sfs.k_score_)

pd.DataFrame.from_dict(sfs.get_metric_dict()).T

plt.figure(figsize=(19,10))
fig = plot_sfs(sfs.get_metric_dict(), kind=None)
plt.title('Sequential Forward Selection (R Sqaure)')
plt.grid()
plt.show()


# In[12]:

idxs_selected=sfs.k_feature_idx_
featureindex = []
for i in idxs_selected:
    featureindex.append(i)
featuredataframe=df.iloc[:,1:127]
features_dataframe_new = featuredataframe.iloc[:,featureindex]
features_dataframe_new.columns


# In[16]:

import matplotlib
print('\nSequential Floating Forward Selection (k=30):')
print(features_dataframe_new.columns)
print('ROC-AUC Score:')
print(sfs.k_score_)

pd.DataFrame.from_dict(sfs.get_metric_dict()).T
plt.figure(figsize=(19,10))
fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10)
plt.xlabel('Number of Features', fontsize=15)
plt.ylabel('R-Square Score', fontsize=15)
plt.title('Sequential Forward Selection (w. StdDev)',fontsize=18)
plt.grid()
plt.show()


# In[72]:

Xnew=df.iloc[:,1:127]
names=Xnew.columns


# In[77]:

# Feature Extraction with RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge,Lasso
ridge=Ridge(alpha=0.5)

rfe = RFE(ridge,30)
fit = rfe.fit(X, y)
print ("Features sorted by their rank using Logistic Regression:")
print (sorted(zip(map(lambda X: round(X, 4), rfe.ranking_),names)))


# In[100]:

lasso = Lasso(alpha=0.009)
rfe = RFE(lasso,30)
fit = rfe.fit(X, y)
print ("Features sorted by their rank using Logistic Regression:")
print (sorted(zip(map(lambda X: round(X, 4), rfe.ranking_),names)))


# In[101]:

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator = lasso, step = 1, cv=4, scoring='r2')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)


# In[102]:

rfecv.grid_scores_


# In[108]:

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(8,5.5))
plt.title('RFECV Selection(estimator:lasso;scoring:r-square)',fontsize=18)
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)
plt.xlabel("Number of Features Selected",fontsize=15)
plt.ylabel("Cross validation R-Square Score",fontsize=15)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_,'b')
plt.grid()
plt.show()

