
# coding: utf-8

# # Hiscox Project: Prediction of Settle or Dismiss Final
# 
# 

# ## Data Wrangling
# 
# ### Importing Data with Pandas

# In[2]:

# Some methods to show polt in notebook:
get_ipython().magic(u'matplotlib inline')
# Import libraries to be used
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing


# In[225]:

dataframenew= pd.read_csv('Featureselectionbinary.csv')
# Using panda to read the csv file


# Next, we can browse the dataframe by just type in the name of dataset 

# In[226]:

# Now inspect your dataframe
dataframenew


# Above is the summary of dataframe in the form of Pandas dataframe.
# 
# Then, we can get more information about the dataframe using `.info()`

# In[227]:

# Give more information about the dataframe
dataframenew.info()


# We can describe the feature in our dataframe and get some basic statistics using `.describe()`

# In[228]:

# Get descriptive statistics of the dataset 
dataframenew.describe()


# While the following code removes the `NaN` values from those remaining feature using `.dropna()`:
#     df = df.dropna()

# ## Exploratory Data Analysis
# 
# ### Exploring Data through Visualizations 

# Now that we have a basic understanding of what we are trying to predict, let’s predict it.
# 
# ## Baseline Model

# In[229]:

dataframenew.corr()


# In[230]:

names = dataframenew.columns.values
names


# In[231]:

X=dataframenew.iloc[:,1:134]
y=dataframenew.iloc[:,0]


# In[596]:

formula1 = 'DependentVariableSettleDismis~ StNumberofDaystoDimissal+StWithinDaysofIPODatewe+StMCapNoInsiderHoldings+NormalisedNetWorth+StRevenue+StStockPriceClose+StStockPriceOpen+StSP500ReturnDuringClassPeri+StGoogleHits+RestatementofEarnings+ViolationsofGAAP+FalsePositivesFailtoDisclo+AccountingImproprietiesandIna+FilingPeriod0+StockOptionDating+MergerAcquisitionbyCompany+ProductsFailuresDelaysMis+Liquidity+RegulatoryViolationsFDAFTC+_ISCACatego_3+_ISCACatego_6+_ISCACatego_7+_ISCACatego_9+_ISector_44+_ISector_42+_ISector_12+_ISector_37+_ISector_18+_ISector_30+_ISector_7+_IMarketCap_3+_IMarketCap_7+_IMarketCap_9+_ICircuit_4+_ICircuit_5+_ICircuit_8+_ICircuit_10+_IPlantiff_3+_IStockExch_2+_IUSIPOCate_3'


# In[597]:

from sklearn.model_selection import train_test_split
from patsy import dmatrices
dftrain, dftest = train_test_split(dataframenew, test_size = 0.2)
y_train1,x_train1 = dmatrices(formula1, data=dftrain,return_type='dataframe')
y_test1,x_test1 = dmatrices(formula1, data=dftest,return_type='dataframe')
y_train1num=np.squeeze(y_train1)
x_train1num=np.squeeze(x_train1)
y_test1num=np.squeeze(y_test1)
x_test1num=np.squeeze(x_test1)
x_train1.columns


# Use this Train Test Split without changing Features 

# In[598]:

import statsmodels.discrete.discrete_model as sm
model1 = sm.Logit(y_train1,x_train1)
res = model1.fit()
res.summary()


# In[599]:

print(res.summary().as_latex())


# In[600]:

from statsmodels.nonparametric.kde import KDEUnivariate
kde_res1 = KDEUnivariate(res.predict())
kde_res1.fit()
plt.figure(figsize=(9,6))
plt.plot(kde_res1.support,kde_res1.density)
plt.fill_between(kde_res1.support,kde_res1.density, alpha=0.2)
plt.title("Distribution of our Predictions",fontsize=16)


# In[601]:

plt.figure(figsize=(10,3))
plt.scatter(res.predict(),x_train1['FalsePositivesFailtoDisclo'] , alpha=0.2)
plt.grid(b=True, which='major', axis='x')
plt.xlabel("Predicted chance of settle",fontsize=17)
plt.ylabel("Violation of GAAP",fontsize=17)
plt.title("The Change of Settlement Probability by Violation of GAAP",fontsize=17)


# In[602]:

plt.figure(figsize=(10,3))
plt.scatter(res.predict(),x_train1['RestatementofEarnings'] , alpha=0.2)
plt.grid(b=True, which='major', axis='x')
plt.xlabel("Predicted chance of settle",fontsize=17)
plt.ylabel("RestatementofEarnings",fontsize=17)
plt.ylim([-0.07,1.07])
plt.title("The Change of Settlement Probability by RestatementofEarnings",fontsize=17)


# In[603]:

plt.figure(figsize=(10,5))
plt.scatter(res.predict(),x_train1.StNumberofDaystoDimissal, alpha=0.2)
plt.grid(True, linewidth=0.15)
plt.title("The Change of Settlement Probability by Time of Settlement",fontsize=17)
plt.xlabel("Predicted chance of settle",fontsize=17)
plt.ylabel("Time to Settle/Dismiss",fontsize=17)


# In[640]:

from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=0.1)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
y_pred1 = cross_val_predict(logreg,x_train1num,y_train1num,cv=8)
y_pred_flag1 = y_pred1 > 0.5
accuracyscores=cross_val_score(logreg, x_train1num, y_train1num, cv=8)
from sklearn.metrics import classification_report,accuracy_score,log_loss,hamming_loss,matthews_corrcoef,roc_auc_score,precision_recall_fscore_support
print (pd.crosstab(y_train1.DependentVariableSettleDismis,y_pred_flag1,rownames = ['Actual'],colnames = ['Predicted']))
print ('\n \n')
print (classification_report(y_train1,y_pred_flag1))
print("Accuracy : %f"%accuracy_score(y_train1,y_pred_flag1))
print("matthews_corrcoef : %f"%matthews_corrcoef(y_train1,y_pred_flag1)) 
print("hamming_loss : %f"%hamming_loss(y_train1,y_pred_flag1))
print(precision_recall_fscore_support(y_train1,y_pred_flag1,average='weighted'))
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# In[641]:

from sklearn.metrics import auc,roc_curve
fpr, tpr, thresholds = roc_curve(y_train1, y_pred1)
roc_auc1 = auc(fpr, tpr)
print ("Area under the ROC curve : %f" % roc_auc1)
# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc1)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)
plt.title('Receiver Operating Characteristic Curve',fontsize=17)
plt.legend(loc="lower right")
plt.show()


# In[606]:

from sklearn import tree 
clf1 = tree.DecisionTreeClassifier(random_state = 1337,
                     criterion = 'gini',
                     splitter = 'best',
                     max_depth = 5,
                     min_samples_leaf = 1)  
y_pred2 = cross_val_predict(clf1,x_train1num,y_train1num,cv=8)
y_pred_flag2 = y_pred2 > 0.5
scores=cross_val_score(clf1, x_train1num, y_train1num, cv=8)
print (pd.crosstab(y_train1.DependentVariableSettleDismis,y_pred2,rownames = ['Actual'],colnames = ['Predicted']))
print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
print ('\n \n')
print (classification_report(y_train1,y_pred_flag2))
print("Accuracy : %f"%accuracy_score(y_train1,y_pred_flag2))
print("matthews_corrcoef : %f"%matthews_corrcoef(y_train1,y_pred_flag2)) 
print("hamming_loss : %f"%hamming_loss(y_train1,y_pred_flag2)) 
print(precision_recall_fscore_support(y_train1,y_pred_flag2,average='weighted'))
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# In[607]:

fpr, tpr, thresholds = roc_curve(y_train1, y_pred2)
roc_auc2 = auc(fpr, tpr)
print ("Area under the ROC curve : %f" % roc_auc2)
# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[608]:

from sklearn import tree 
clf1 = tree.DecisionTreeClassifier()  
y_pred2 = cross_val_predict(clf1,x_train1num,y_train1num,cv=8)
y_pred_flag2 = y_pred2 > 0.5
scores=cross_val_score(clf1, x_train1num, y_train1num, cv=8)
print (pd.crosstab(y_train1.DependentVariableSettleDismis,y_pred2,rownames = ['Actual'],colnames = ['Predicted']))
print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
print ('\n \n')
print (classification_report(y_train1,y_pred_flag2))
print("Accuracy : %f"%accuracy_score(y_train1,y_pred_flag2))
print("matthews_corrcoef : %f"%matthews_corrcoef(y_train1,y_pred_flag2)) 
print("hamming_loss : %f"%hamming_loss(y_train1,y_pred_flag2)) 
print(precision_recall_fscore_support(y_train1,y_pred_flag2,average='weighted'))
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# In[609]:

fpr, tpr, thresholds = roc_curve(y_train1, y_pred2)
roc_auc2 = auc(fpr, tpr)
print ("Area under the ROC curve : %f" % roc_auc2)
# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[610]:

import sklearn.ensemble as sk
clf2 = sk.RandomForestClassifier()
y_pred3 = cross_val_predict(clf2,x_train1num,y_train1num,cv=8)
y_pred_flag3 = y_pred3 > 0.5
scores=cross_val_score(clf2, x_train1num, y_train1num, cv=8)
print (pd.crosstab(y_train1.DependentVariableSettleDismis,y_pred3,rownames = ['Actual'],colnames = ['Predicted']))
print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
print ('\n \n')
print (classification_report(y_train1,y_pred_flag3))
print("Accuracy : %f"%accuracy_score(y_train1,y_pred_flag3))
print("matthews_corrcoef : %f"%matthews_corrcoef(y_train1,y_pred_flag3))  
print("hamming_loss : %f"%hamming_loss(y_train1,y_pred_flag3))
print(precision_recall_fscore_support(y_train1,y_pred_flag3,average='weighted'))
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# In[611]:

fpr, tpr, thresholds = roc_curve(y_train1, y_pred3)
roc_auc3 = auc(fpr, tpr)
print ("Area under the ROC curve : %f" % roc_auc3)
# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc3)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[612]:

import sklearn.ensemble as sk
clf2 = sk.RandomForestClassifier(n_estimators=108,
    bootstrap=False,
    max_features=10,
    criterion='entropy',
    random_state=1337,
    max_depth=None,
    verbose=3,
    min_samples_split=3,                             
    min_samples_leaf=6)
y_pred3 = cross_val_predict(clf2,x_train1num,y_train1num,cv=8)
y_pred_flag3 = y_pred3 > 0.5
scores=cross_val_score(clf2, x_train1num, y_train1num, cv=8)
print (pd.crosstab(y_train1.DependentVariableSettleDismis,y_pred3,rownames = ['Actual'],colnames = ['Predicted']))
print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
print ('\n \n')
print (classification_report(y_train1,y_pred_flag3))
print("Accuracy : %f"%accuracy_score(y_train1,y_pred_flag3))
print("matthews_corrcoef : %f"%matthews_corrcoef(y_train1,y_pred_flag3))  
print("hamming_loss : %f"%hamming_loss(y_train1,y_pred_flag3))
print(precision_recall_fscore_support(y_train1,y_pred_flag3,average='weighted'))
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# In[613]:

fpr, tpr, thresholds = roc_curve(y_train1, y_pred3)
roc_auc3 = auc(fpr, tpr)
print ("Area under the ROC curve : %f" % roc_auc3)
# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc3)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[614]:

from sklearn import svm
clf3 = svm.SVC()
y_pred4 = cross_val_predict(clf3,x_train1num,y_train1num,cv=8)
y_pred_flag4 = y_pred4 > 0.5
scores=cross_val_score(clf3, x_train1num, y_train1num, cv=8)
print (pd.crosstab(y_train1.DependentVariableSettleDismis,y_pred4,rownames = ['Actual'],colnames = ['Predicted']))
print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
print ('\n \n')
print (classification_report(y_train1,y_pred_flag4))
print("Accuracy : %f"%accuracy_score(y_train1,y_pred_flag4))
print("matthews_corrcoef : %f"%matthews_corrcoef(y_train1,y_pred_flag4))  
print("hamming_loss : %f"%hamming_loss(y_train1,y_pred_flag4))
print(precision_recall_fscore_support(y_train1,y_pred_flag4,average='weighted'))
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# In[615]:

fpr, tpr, thresholds = roc_curve(y_train1, y_pred4)
roc_auc4 = auc(fpr, tpr)
print ("Area under the ROC curve : %f" % roc_auc4)
# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc3)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[616]:

from sklearn.linear_model import SGDClassifier
clf4 = SGDClassifier()
y_pred5 = cross_val_predict(clf4,x_train1num,y_train1num,cv=8)
y_pred_flag5 = y_pred5 > 0.5
scores=cross_val_score(clf4, x_train1num, y_train1num, cv=8)
print (pd.crosstab(y_train1.DependentVariableSettleDismis,y_pred5,rownames = ['Actual'],colnames = ['Predicted']))
print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
print ('\n \n')
print (classification_report(y_train1,y_pred_flag5))
print("Accuracy : %f"%accuracy_score(y_train1,y_pred_flag5))
print("matthews_corrcoef : %f"%matthews_corrcoef(y_train1,y_pred_flag5))  
print("hamming_loss : %f"%hamming_loss(y_train1,y_pred_flag5))
print(precision_recall_fscore_support(y_train1,y_pred_flag5,average='weighted'))
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# In[617]:

fpr, tpr, thresholds = roc_curve(y_train1, y_pred5)
roc_auc5= auc(fpr, tpr)
print ("Area under the ROC curve : %f" % roc_auc5)
# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc3)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[618]:

from sklearn.linear_model import SGDClassifier
clf4 = SGDClassifier(penalty='l2', alpha= 0.01, loss='log', n_jobs=1, verbose=3, fit_intercept= True, shuffle= True)
y_pred5 = cross_val_predict(clf4,x_train1num,y_train1num,cv=8)
y_pred_flag5 = y_pred5 > 0.5
scores=cross_val_score(clf4, x_train1num, y_train1num, cv=8)
print (pd.crosstab(y_train1.DependentVariableSettleDismis,y_pred5,rownames = ['Actual'],colnames = ['Predicted']))
print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
print ('\n \n')
print (classification_report(y_train1,y_pred_flag5))
print("Accuracy : %f"%accuracy_score(y_train1,y_pred_flag5))
print("matthews_corrcoef : %f"%matthews_corrcoef(y_train1,y_pred_flag5))  
print("hamming_loss : %f"%hamming_loss(y_train1,y_pred_flag5))
print(precision_recall_fscore_support(y_train1,y_pred_flag5,average='weighted'))
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# In[619]:

from sklearn.ensemble import GradientBoostingClassifier
clf5= GradientBoostingClassifier()
y_pred6 = cross_val_predict(clf5,x_train1num,y_train1num,cv=8)
y_pred_flag6 = y_pred6 > 0.5
scores=cross_val_score(clf5, x_train1num, y_train1num, cv=8)
print (pd.crosstab(y_train1.DependentVariableSettleDismis,y_pred6,rownames = ['Actual'],colnames = ['Predicted']))
print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
print ('\n \n')
print (classification_report(y_train1,y_pred_flag6))
print("Accuracy : %f"%accuracy_score(y_train1,y_pred_flag6))
print("matthews_corrcoef : %f"%matthews_corrcoef(y_train1,y_pred_flag6))  
print("hamming_loss : %f"%hamming_loss(y_train1,y_pred_flag6))
print(precision_recall_fscore_support(y_train1,y_pred_flag6,average='weighted'))
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# In[620]:

fpr, tpr, thresholds = roc_curve(y_train1, y_pred6)
roc_auc6= auc(fpr, tpr)
print ("Area under the ROC curve : %f" % roc_auc6)
# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc3)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[621]:

from xgboost import XGBClassifier
XGB= XGBClassifier()
y_pred7 = cross_val_predict(XGB,x_train1num,y_train1num,cv=8)
y_pred_flag7 = y_pred7 > 0.5
scores=cross_val_score(XGB, x_train1num, y_train1num, cv=8)
print (pd.crosstab(y_train1.DependentVariableSettleDismis,y_pred7,rownames = ['Actual'],colnames = ['Predicted']))
print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
print ('\n \n')
print (classification_report(y_train1,y_pred_flag7))
print("Accuracy : %f"%accuracy_score(y_train1,y_pred_flag7))
print("matthews_corrcoef : %f"%matthews_corrcoef(y_train1,y_pred_flag7))  
print("hamming_loss : %f"%hamming_loss(y_train1,y_pred_flag7))
print(precision_recall_fscore_support(y_train1,y_pred_flag7,average='weighted'))
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# In[622]:

fpr, tpr, thresholds = roc_curve(y_train1, y_pred7)
roc_auc7= auc(fpr, tpr)
print ("Area under the ROC curve : %f" % roc_auc7)
# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc3)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[623]:

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
y_pred8 = cross_val_predict(mlp,x_train1num,y_train1num,cv=8)
y_pred_flag8 = y_pred8 > 0.5
scores=cross_val_score(XGB, x_train1num, y_train1num, cv=8)
print (pd.crosstab(y_train1.DependentVariableSettleDismis,y_pred8,rownames = ['Actual'],colnames = ['Predicted']))
print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
print ('\n \n')
print (classification_report(y_train1,y_pred_flag8))
print("Accuracy : %f"%accuracy_score(y_train1,y_pred_flag8))
print("matthews_corrcoef : %f"%matthews_corrcoef(y_train1,y_pred_flag8))  
print("hamming_loss : %f"%hamming_loss(y_train1,y_pred_flag8))
print(precision_recall_fscore_support(y_train1,y_pred_flag8,average='weighted'))
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# In[624]:

fpr, tpr, thresholds = roc_curve(y_train1, y_pred8)
roc_auc8= auc(fpr, tpr)
print ("Area under the ROC curve : %f" % roc_auc8)
# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc3)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# ### Tuning Pipeline Hyperparameters(Random Search)

# In[284]:

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV



# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


# specify parameters and distributions to sample from
param_dist = {"criterion": ['gini', 'entropy'],
              "max_features": sp_randint(1, 20),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf1, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(x_train1num,y_train1num)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)




# In[272]:

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV



# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 20),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "verbose": [0,1,2,3]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf2, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(x_train1num,y_train1num)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)



# In[322]:

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV



# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


# specify parameters and distributions to sample from
param_dist = {"loss": ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
              "penalty":  ['none', 'l2', 'l1', 'elasticnet'],
              "alpha": [0.0001, 0.001,0.01,0.1,1],
              "fit_intercept":[True,False],
             "shuffle":[True,False],
             "verbose": [1, 2,3],
             "n_jobs":[1,2,3]}

# run randomized search
n_iter_search = 30
random_search = RandomizedSearchCV(clf4, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(x_train1num,y_train1num)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)





# ### Deploy the Best Model and Apply it on Test Dataset

# In[639]:

logregfinal = linear_model.LogisticRegression(C=0.1)
logfinal=logregfinal.fit(x_train1num,y_train1num)
y_predfinal = logfinal.predict(x_test1num)
y_pred_flagfinal = y_predfinal > 0.5
print (pd.crosstab(y_test1.DependentVariableSettleDismis,y_predfinal,rownames = ['Actual'],colnames = ['Predicted']))
print ('\n \n')
print (classification_report(y_test1,y_pred_flagfinal))
print("Accuracy : %f"%accuracy_score(y_test1,y_pred_flagfinal))
print("matthews_corrcoef : %f"%matthews_corrcoef(y_test1,y_pred_flagfinal))  
print("hamming_loss : %f"%hamming_loss(y_test1,y_pred_flagfinal))
print(precision_recall_fscore_support(y_test1,y_pred_flagfinal,average='weighted'))
import time
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# In[642]:

from sklearn.metrics import auc,roc_curve
fpr, tpr, thresholds = roc_curve(y_test1, y_predfinal)
roc_aucfinal = auc(fpr, tpr)
print ("Area under the ROC curve : %f" % roc_aucfinal)
# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_aucfinal)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)
plt.title('Receiver Operating Characteristic Curve',fontsize=17)
plt.legend(loc="lower right")
plt.show()


# In[ ]:



