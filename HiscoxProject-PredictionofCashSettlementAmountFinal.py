
# coding: utf-8

# # Hiscox Project: Prediction of Cashsettlement Amonunt 
# 

# ## Data Wrangling
# 
# ### Importing Data with Pandas

# In[2]:

# Some methods to show plot in notebook:
get_ipython().magic(u'matplotlib inline')
# Import libraries to be used
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
import seaborn as sns
import scipy as scipy


# In[3]:

df= pd.read_csv('Featureselectioncontinuous.csv')
# Using panda to read the csv file


# Next, we can browse the dataframe by just type in the name of dataset 

# In[4]:

# Now inspect your dataframe
df


# Above is the summary of dataframe in the form of Pandas dataframe.
# 
# Then, we can get more information about the dataframe using `.info()`

# In[5]:

# Give more information about the dataframe
df.info()


# We can describe the feature in our dataframe and get some basic statistics using `.describe()`

# In[6]:

# Get descriptive statistics of the dataset titanicdata
df.describe()


# While the following code removes the `NaN` values from those remaining feature using `.dropna()`:
#     df = df.dropna()

# ## Exploratory Data Analysis
# 
# ### Exploring Data through Visualizations 

# Now that we have a basic understanding of what we are trying to predict, let’s predict it.
# 
# ## Baseline Model

# In[7]:

X=df.iloc[:,1:127]
y=df.iloc[:,0]
names = df.columns.values
names


# In[239]:

formula2 = 'StCashSettlementAmount~StMarketCap+StClassPeriodLengthDays+StAssets+SP500ReturnDuringClassPeri+StMktValueDropNoInsider+StGoogleHits+NormalisedProfits+StStockPrice3weekprior+InsiderTradingSpecificallyAll+FilingPeriod0+ClinicalTrialFailuresDelays+ViolationsofGAAP+StockOptionDating+MisrepresentationDisclosure+_ISector_3+_ISector_10+_ISector_31+_ISector_32+_IStockExch_2+_IStockExch_4+_ICircuit_9+_ICircuit_4+_ISCACatego_2+_ISCACatego_5'


# _ISCACatego_5+_IPlantiff_1+_ICircuit_9+ _ISCACatego_12+
# _IStockExch_2+_IStockExch_4+_ISector_10+_ICircuit_4_ISCACatego_5+
# _IPlantiff_1+_ICircuit_9+ _ISCACatego_12+_ISector_32+_ISector_3+_ISCACatego_2+ _ISector_4_ISCACatego_8+_IStockExch_3+_ISector_20
# 

# In[240]:

from sklearn.model_selection import train_test_split
from patsy import dmatrices
train, test = train_test_split(df, test_size = 0.2)
y_train2,x_train2 = dmatrices(formula2, data=train,return_type='dataframe')
y_test2,x_test2 = dmatrices(formula2, data=test,return_type='dataframe')
y_train2num=np.squeeze(y_train2)
x_train2num=np.squeeze(x_train2)
x_train2.columns


# In[241]:

import statsmodels.discrete.discrete_model as sm
model2 = sm.OLS(y_train2,x_train2)
res2 = model2.fit()
res2.summary()


# In[242]:

print(res2.summary().as_latex())


# In[243]:

import matplotlib as mpl
reg_results = sm.OLS(y_train2,x_train2,formula=formula2,data=df).fit().summary()
sns.set(style="ticks") 
mpl.rc("figure", figsize=(12, 13))
mpl.rc('xtick', labelsize=14) 
mpl.rc('ytick', labelsize=14)
sns.coefplot(formula2,df)
ax = plt.gca()
plt.setp(ax.get_xticklabels(), rotation=90)


# In[269]:

get_ipython().run_cell_magic(u'time', u'', u"from sklearn import linear_model\nregr = linear_model.LinearRegression()\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.model_selection import cross_val_predict\nypred1 = cross_val_predict(regr,x_train2num,y_train2num,cv=4)\nscores_nmse = cross_val_score(regr, x_train2num, y_train2num, scoring = 'neg_mean_squared_error',cv=8)\nscores_nmae1= cross_val_score(regr, x_train2num, y_train2num, scoring = 'neg_mean_absolute_error',cv=8)\nscores_nmae2= cross_val_score(regr, x_train2num, y_train2num, scoring = 'neg_median_absolute_error',cv=8)\nscores_r2= cross_val_score(regr, x_train2num, y_train2num, scoring = 'r2',cv=4)")


# In[270]:

from scipy.stats.stats import pearsonr
print("Mean Squared Error: %f (+/- %f)" % (-scores_nmse.mean(), scores_nmse.std() * 2))
print("Mean Absolute Error: %f (+/- %f)" % (-scores_nmae1.mean(), scores_nmae1.std() * 2))
print("Media Absolute Error: %f (+/- %f)" % (-scores_nmae2.mean(), scores_nmae2.std() * 2))
print("R Square: %f (+/- %f)" % (scores_r2.mean(), scores_r2.std() * 2))
y=y_train2.values[:,0]
corr, p_value=pearsonr(y,ypred1)
print("Pearson coefficient : %f"%corr)
print("P-Value : %f"%p_value)


# In[311]:

fig, ax = plt.subplots(figsize=(9,3))
ax.scatter(y, ypred1)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Actual Values',fontsize=16)
ax.set_ylabel('Predicted Values',fontsize=16)
ax.set_title('Predicted vs Actual',fontsize=18)
plt.xlim((-3,3.7))
plt.show()


# In[312]:

from sklearn import preprocessing
residual= np.subtract(y,ypred1)
stanresidual=preprocessing.scale(residual)
fig, ax = plt.subplots(figsize=(9,3))
ax.scatter(ypred1,residual)
ax.plot([-3, 3], [0, 0], 'k--', lw=4)
ax.set_xlabel('Predicted Values',fontsize=16)
ax.set_ylabel('Residual Values',fontsize=16)
ax.set_title('Predicted vs Residual',fontsize=18)
plt.xlim((-2,3))
plt.show()


# In[401]:

from matplotlib import pyplot
from pandas import DataFrame
from scipy import stats
residualnew=DataFrame(residual)
print(residualnew.describe())
import seaborn as sns
fig, ax = plt.subplots(figsize=(6,5))
plt.xlim((-3,3))
sns.distplot(residual, kde=False, fit=stats.gamma,color='blue')
sns.plt.title('Error Distribution',fontsize=18)


# In[400]:

from statsmodels.graphics.gofplots import qqplot
import pylab
# Q-Q plot
fig, ax = plt.subplots(figsize=(6,5))
stats.probplot(residual, dist="norm", plot=pylab)
ax.set_title("Probability Plot",fontsize=18)
ax.set_xlabel('Theoretical Quantiles',fontsize=16)
ax.set_ylabel('Ordered Values',fontsize=16)
pylab.show()


# In[404]:

sum([1 for x in residual if x > 0])


# In[405]:

np.shape(residual)


# In[372]:

from sklearn.preprocessing import scale
from sklearn import cross_validation
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNetCV
alphas = 10**np.linspace(10,-2,100)*0.5
ridge = Ridge(normalize=True)
coefs = []
for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(x_train2, y_train2)
    coefs.append(ridge.coef_)
coefs=np.squeeze(coefs)
np.shape(coefs)


# In[374]:

np.shape(alphas)
fig, ax = plt.subplots(figsize=(7,6))
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha',fontsize=18)
plt.ylabel('weights',fontsize=18)


# In[375]:

from sklearn import linear_model
from sklearn import datasets
diabetes = datasets.load_diabetes()

xnew=x_train2.values
ynew=y_train2num.values
alphas, _, coefs = linear_model.lars_path(xnew, ynew, method='lasso', verbose=True)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

fig, ax = plt.subplots(figsize=(7,6))
plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|',fontsize=18)
plt.ylabel('Coefficients',fontsize=18)
plt.axis('tight')
plt.legend()
plt.show()


# In[271]:

get_ipython().run_cell_magic(u'time', u'', u"ridgecv = RidgeCV(alphas=alphas, scoring='r2', normalize=True)\nypred3 = cross_val_predict(ridgecv,x_train2num,y_train2num,cv=4)\nscores_nmse = cross_val_score(ridgecv, x_train2num, y_train2num, scoring = 'neg_mean_squared_error',cv=8)\nscores_nmae1= cross_val_score(ridgecv, x_train2num, y_train2num, scoring = 'neg_mean_absolute_error',cv=8)\nscores_nmae2= cross_val_score(ridgecv, x_train2num, y_train2num, scoring = 'neg_median_absolute_error',cv=8)\nscores_r2= cross_val_score(ridgecv, x_train2num, y_train2num, scoring = 'r2',cv=8)")


# In[272]:

print("Mean Squared Error: %f (+/- %f)" % (-scores_nmse.mean(), scores_nmse.std() * 2))
print("Mean Absolute Error: %f (+/- %f)" % (-scores_nmae1.mean(), scores_nmae1.std() * 2))
print("Media Absolute Error: %f (+/- %f)" % (-scores_nmae2.mean(), scores_nmae2.std() * 2))
print("R Square: %f (+/- %f)" % (scores_r2.mean(), scores_r2.std() * 2))
corr, p_value=pearsonr(y,ypred3)
print("Pearson coefficient : %f"%corr)
print("P-Value : %f"%p_value)


# In[410]:

ridgecvmodel=ridgecv.fit(x_train2num,y_train2num)
ridgecvmodel.alpha_


# In[273]:

get_ipython().run_cell_magic(u'time', u'', u"lassocv = LassoCV(max_iter=100000, normalize=True)\nypred4 = cross_val_predict(lassocv,x_train2num,y_train2num,cv=8)\nscores_nmse = cross_val_score(lassocv, x_train2num, y_train2num, scoring = 'neg_mean_squared_error',cv=8)\nscores_nmae1= cross_val_score(lassocv, x_train2num, y_train2num, scoring = 'neg_mean_absolute_error',cv=8)\nscores_nmae2= cross_val_score(lassocv, x_train2num, y_train2num, scoring = 'neg_median_absolute_error',cv=8)\nscores_r2= cross_val_score(lassocv, x_train2num, y_train2num, scoring = 'r2',cv=8)")


# In[274]:

print("Mean Squared Error: %f (+/- %f)" % (-scores_nmse.mean(), scores_nmse.std() * 2))
print("Mean Absolute Error: %f (+/- %f)" % (-scores_nmae1.mean(), scores_nmae1.std() * 2))
print("Media Absolute Error: %f (+/- %f)" % (-scores_nmae2.mean(), scores_nmae2.std() * 2))
print("R Square: %f (+/- %f)" % (scores_r2.mean(), scores_r2.std() * 2))
corr, p_value=pearsonr(y,ypred4)
print("Pearson coefficient : %f"%corr)
print("P-Value : %f"%p_value)


# In[411]:

lassocvmodel=lassocv.fit(x_train2num,y_train2num)
lassocvmodel.alpha_


# In[275]:

get_ipython().run_cell_magic(u'time', u'', u"elastic = ElasticNetCV(cv=8, random_state=0)\nypred34 = cross_val_predict(elastic,x_train2num,y_train2num,cv=8)\nscores_nmse = cross_val_score(elastic, x_train2num, y_train2num, scoring = 'neg_mean_squared_error',cv=8)\nscores_nmae1= cross_val_score(elastic, x_train2num, y_train2num, scoring = 'neg_mean_absolute_error',cv=8)\nscores_nmae2= cross_val_score(elastic, x_train2num, y_train2num, scoring = 'neg_median_absolute_error',cv=8)\nscores_r2= cross_val_score(elastic, x_train2num, y_train2num, scoring = 'r2',cv=8)")


# In[276]:

print("Mean Squared Error: %f (+/- %f)" % (-scores_nmse.mean(), scores_nmse.std() * 2))
print("Mean Absolute Error: %f (+/- %f)" % (-scores_nmae1.mean(), scores_nmae1.std() * 2))
print("Media Absolute Error: %f (+/- %f)" % (-scores_nmae2.mean(), scores_nmae2.std() * 2))
print("R Square: %f (+/- %f)" % (scores_r2.mean(), scores_r2.std() * 2))
corr, p_value=pearsonr(y,ypred34)
print("Pearson coefficient : %f"%corr)
print("P-Value : %f"%p_value)


# In[413]:

elasticcvmodel=elastic.fit(x_train2num,y_train2num)
elasticcvmodel.alpha_


# In[277]:

get_ipython().run_cell_magic(u'time', u'', u"from sklearn.ensemble import GradientBoostingRegressor\nGBR = GradientBoostingRegressor()\nypred7 = cross_val_predict(GBR,x_train2num,y_train2num,cv=8)\nscores_nmse = cross_val_score(GBR, x_train2num, y_train2num, scoring = 'neg_mean_squared_error',cv=8)\nscores_nmae1= cross_val_score(GBR, x_train2num, y_train2num, scoring = 'neg_mean_absolute_error',cv=8)\nscores_nmae2= cross_val_score(GBR, x_train2num, y_train2num, scoring = 'neg_median_absolute_error',cv=8)\nscores_r2= cross_val_score(GBR, x_train2num, y_train2num, scoring = 'r2',cv=8)")


# In[278]:

print("Mean Squared Error: %f (+/- %f)" % (-scores_nmse.mean(), scores_nmse.std() * 2))
print("Mean Absolute Error: %f (+/- %f)" % (-scores_nmae1.mean(), scores_nmae1.std() * 2))
print("Media Absolute Error: %f (+/- %f)" % (-scores_nmae2.mean(), scores_nmae2.std() * 2))
print("R Square: %f (+/- %f)" % (scores_r2.mean(), scores_r2.std() * 2))
corr, p_value=pearsonr(y,ypred7)
print("Pearson coefficient : %f"%corr)
print("P-Value : %f"%p_value)


# In[384]:

get_ipython().run_cell_magic(u'time', u'', u"from sklearn.ensemble import GradientBoostingRegressor\nGBR = GradientBoostingRegressor(\n                    verbose = 1,\n                    min_samples_leaf=8,\n                    n_estimators= 85,\n                    learning_rate =0.0869,\n                    min_samples_split=3,\n                    loss = 'ls',\n                    max_depth = 2)\nypred7 = cross_val_predict(GBR,x_train2num,y_train2num,cv=8)\nscores_nmse = cross_val_score(GBR, x_train2num, y_train2num, scoring = 'neg_mean_squared_error',cv=8)\nscores_nmae1= cross_val_score(GBR, x_train2num, y_train2num, scoring = 'neg_mean_absolute_error',cv=8)\nscores_nmae2= cross_val_score(GBR, x_train2num, y_train2num, scoring = 'neg_median_absolute_error',cv=8)\nscores_r2= cross_val_score(GBR, x_train2num, y_train2num, scoring = 'r2',cv=8)")


# In[385]:

print("Mean Squared Error: %f (+/- %f)" % (-scores_nmse.mean(), scores_nmse.std() * 2))
print("Mean Absolute Error: %f (+/- %f)" % (-scores_nmae1.mean(), scores_nmae1.std() * 2))
print("Media Absolute Error: %f (+/- %f)" % (-scores_nmae2.mean(), scores_nmae2.std() * 2))
print("R Square: %f (+/- %f)" % (scores_r2.mean(), scores_r2.std() * 2))


# In[386]:

get_ipython().run_cell_magic(u'time', u'', u"from xgboost import XGBRegressor \nxgb = XGBRegressor ()\nypred8 = cross_val_predict(xgb,x_train2num,y_train2num,cv=8)\nscores_nmse = cross_val_score(xgb, x_train2num, y_train2num, scoring = 'neg_mean_squared_error',cv=8)\nscores_nmae1= cross_val_score(xgb, x_train2num, y_train2num, scoring = 'neg_mean_absolute_error',cv=8)\nscores_nmae2= cross_val_score(xgb, x_train2num, y_train2num, scoring = 'neg_median_absolute_error',cv=8)\nscores_r2= cross_val_score(xgb, x_train2num, y_train2num, scoring = 'r2',cv=8)")


# In[388]:

print("Mean Squared Error: %f (+/- %f)" % (-scores_nmse.mean(), scores_nmse.std() * 2))
print("Mean Absolute Error: %f (+/- %f)" % (-scores_nmae1.mean(), scores_nmae1.std() * 2))
print("Media Absolute Error: %f (+/- %f)" % (-scores_nmae2.mean(), scores_nmae2.std() * 2))
print("R Square: %f (+/- %f)" % (scores_r2.mean(), scores_r2.std() * 2))


# In[390]:

get_ipython().run_cell_magic(u'time', u'', u"from xgboost import XGBRegressor \nxgbtuned = XGBRegressor (\n                    n_estimators = 108,\n                    learning_rate = 0.07,\n                    max_depth = 3)\nypred8 = cross_val_predict(xgb,x_train2num,y_train2num,cv=8)\nscores_nmse = cross_val_score(xgbtuned, x_train2num, y_train2num, scoring = 'neg_mean_squared_error',cv=8)\nscores_nmae1= cross_val_score(xgbtuned, x_train2num, y_train2num, scoring = 'neg_mean_absolute_error',cv=8)\nscores_nmae2= cross_val_score(xgbtuned, x_train2num, y_train2num, scoring = 'neg_median_absolute_error',cv=8)\nscores_r2= cross_val_score(xgbtuned, x_train2num, y_train2num, scoring = 'r2',cv=8)")


# In[391]:

print("Mean Squared Error: %f (+/- %f)" % (-scores_nmse.mean(), scores_nmse.std() * 2))
print("Mean Absolute Error: %f (+/- %f)" % (-scores_nmae1.mean(), scores_nmae1.std() * 2))
print("Media Absolute Error: %f (+/- %f)" % (-scores_nmae2.mean(), scores_nmae2.std() * 2))
print("R Square: %f (+/- %f)" % (scores_r2.mean(), scores_r2.std() * 2))


# ### Tuning Pipeline Hyperparameters(Random Search)

# In[389]:

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn.grid_search import RandomizedSearchCV
import scipy.stats as st


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

xgbnew = XGBRegressor ()
# specify parameters and distributions to sample from
one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)
param_dist = { "n_estimators": sp_randint(80, 120),
    "max_depth": sp_randint(2, 15),
    "learning_rate": st.uniform(0.05, 0.1),
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
             }

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(xgbnew, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(x_train2num,y_train2num)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)


# In[378]:

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
import scipy.stats as st
from sklearn.grid_search import RandomizedSearchCV



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
param_dist = {"n_estimators": st.randint(80, 120),
              "learning_rate": st.uniform(0.05, 0.3),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "max_depth": sp_randint(1, 14),
              "verbose":[0,1,2],}

# run randomized search
n_iter_search = 30
random_search = RandomizedSearchCV(GBR, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(x_train2num,y_train2num)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)



# ### Deploy the Best Model and Apply it on Test Dataset

# In[427]:

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error,median_absolute_error, r2_score
y_test2num=np.squeeze(y_test2)
x_test2num=np.squeeze(x_test2)
elastic = ElasticNet(alpha=0.0014, l1_ratio=0.5)
elasticfinal=elastic.fit(x_train2num, y_train2num)
y_testpred=elasticfinal.predict(x_test2num)
scores_nmse = mean_squared_error(y_test2num,y_testpred)
scores_nmae1= mean_absolute_error(y_test2num,y_testpred)
scores_nmae2= median_absolute_error(y_test2num,y_testpred)
scores_r2= r2_score(y_test2num,y_testpred)


# In[428]:

print("Mean Squared Error: %f " % (scores_nmse.mean()))
print("Mean Absolute Error: %f " % (scores_nmae1.mean()))
print("Media Absolute Error: %f " % (scores_nmae2.mean()))
print("R Square: %f " % (scores_r2.mean()))
print("Pearson coefficient : %f"%corr)
print("P-Value : %f"%p_value)


# In[433]:

from sklearn import preprocessing
residualout= np.subtract(y_test2num,y_testpred)
stanresidual=preprocessing.scale(residualout)
fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(y_testpred,residualout)
ax.plot([-3, 3], [0, 0], 'k--', lw=4)
ax.set_xlabel('Predicted Values',fontsize=16)
ax.set_ylabel('Residual Values',fontsize=16)
ax.set_title('Predicted vs Residual',fontsize=18)
plt.xlim((-2,3))
plt.show()


# In[434]:

fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(y_test2num,y_testpred)
ax.plot([y_test2num.min(), y_test2num.max()], [y_test2num.min(), y_test2num.max()], 'k--', lw=4)
ax.set_xlabel('Actual Values',fontsize=16)
ax.set_ylabel('Predicted Values',fontsize=16)
ax.set_title('Predicted vs Actual',fontsize=18)
plt.xlim((-3,3.7))
plt.show()


# In[431]:

residualoutframe=DataFrame(residualout)
print(residualoutframe.describe())
import seaborn as sns
fig, ax = plt.subplots(figsize=(6,5))
plt.xlim((-3,3))
sns.distplot(residualoutframe, kde=False, fit=stats.gamma,color='blue')
sns.plt.title('Error Distribution',fontsize=18)


# In[432]:

from statsmodels.graphics.gofplots import qqplot
import pylab
# Q-Q plot
fig, ax = plt.subplots(figsize=(6,5))
stats.probplot(residualout, dist="norm", plot=pylab)
ax.set_title("Probability Plot",fontsize=18)
ax.set_xlabel('Theoretical Quantiles',fontsize=16)
ax.set_ylabel('Ordered Values',fontsize=16)
pylab.show()


# In[ ]:



