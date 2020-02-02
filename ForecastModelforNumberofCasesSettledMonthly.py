
# coding: utf-8

# # Forecasting Model of Monthly Settled Cases

# ##  Overview

# In this tutorial, we will work through a time series forecasting project from end-to-end, from reviewing the dataset and defining the problem to training a final model and making predictions. This project is not exhaustive, but shows how you can get good results quickly by working through a time series forecasting problem systematically.
# The steps of this project that we will through are as follows. 
# 1. Problem Description
# 2. Experimental Setup
# 3. Persistence
# 4. Data Analysis 
# 5. ARIMA Models
# 6. Time series evaluation

# ## 1. Problem Description

# The problem is to predict the Number of Settlement by month of SCA over years.

# In[1]:

# separate out a validation dataset
from pandas import Series
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
# Import statsmodel
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller
# evaluate persistence model on time series
from sklearn.metrics import mean_squared_error
from math import sqrt
get_ipython().magic(u'matplotlib inline')
import pylab
from pandas import DataFrame
from pandas import TimeGrouper
# create and summarize stationary version of time series
from statsmodels.tsa.stattools import adfuller
from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR 
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from math import sqrt
import warnings
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy
from arch import arch_model
from pandas import Series
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from pandas import Series
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import Series
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import boxcox
import numpy
from statsmodels.tsa.arima_model import ARIMAResults
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy
from sklearn.ensemble import RandomForestRegressor
from pandas import DataFrame, concat

series = Series.from_csv('Monthly Count of Settlement.csv',header=0)


# ## 2. Experimental Setup

# We will take the data set and split it into training and test sets for the purposes of the experiment. 

# ## 4. Data Analysis

# Now we have a baseline prediction method and performance; now we can start digging into our data.
# 
# We can use summary statistics and plots of the data to quickly learn more about the structure of the prediction problem. In this section, we will look at the data from five perspectives:
# 
# 1. Summary Statistics. 
# 2. Line Plot.
# 4. Density Plots.
# 5. Box and Whisker Plot.

# ### 4.1 Summary Statistics
# 
# Summary statistics provide a quick look at the limits of observed values. It can help to get a quick idea of what we are working with. The example below calculates and prints summary statistics for the time series.

# In[2]:

# summary statistics of time series
print(series.describe())


# ### 4.2 Line Plot

# The first step before getting bogged down in data analysis and modeling is to establish a baseline of performance. This will provide both a template for evaluating models and a performance measure by which all more elaborate predictive models can be compared. The baseline prediction for time series forecasting is called the naive forecast (somethimes referred to as persistence).
# 
# Here we will use walk-forward validation discussed in class.  

# A line plot of a time series can provide a lot of insight into the problem.

# Running the naive forecast prints the prediction and observation for each iteration of the test dataset. The example ends by printing the RMSE for the model. In this case, we can see that the forecast achieved an RMSE of 0.897. 

# In[3]:

# plot times series data for overview
pyplot.figure(1)
series.plot()
pylab.ylabel('Monthly Settled Cases')
pylab.xlabel('Date')
pyplot.show()


# Some observations from the plot include:
# - There may be an increasing trend of cash settlement amount over time.
# - There might be systematic seasonality to the cash settlement amount for each year.
# - The seasonal signal appears to be growing over time, suggesting a multiplicative relationship (increasing change).
# - There do not appear to be any obvious outliers.
# - The seasonality might suggests that the series is almost certainly non-stationary.
# 
# There may be benefit in explicitly modeling the seasonal component and removing it. You may also explore using differencing with one or two levels in order to make the series stationary. The increasing trend or growth in the seasonal component may suggest the use of a log or other power transform.
# 

# ### 4.4 Density Plot and Transformation 

# Reviewing plots of the density of observations can provide further insight into the structure of the data. The example below creates a histogram and density plot of the observations without any temporal structure.

# In[4]:

pyplot.figure(3)
pyplot.subplot(211)
series.hist()
pyplot.subplot(212) 
series.plot(kind='kde')
pyplot.show(3)


# In[5]:

series=np.sqrt(series)
pyplot.figure(3)
pyplot.subplot(211)
series.hist()
pyplot.subplot(212) 
series.plot(kind='kde')
pyplot.show(3)


# It is clear a unimodel distribution with one clear peak. From the pdf graph and the histogram, we can say that the it is almost symmetric with a small long tail(the distribution have a few number of occurrences far from the central part) on the right.
# For positive skew where tail is on the positive end, we can apply square root transformation,log transformation and inverse/reciprocal transformation. Therefore, if the log transformation is not sufficient, we can also use the next level of transformation. On the other hand, Box Cox runs all transformations automatically so that you can choose the best one.

# ### 4.5 Box and Whisker Plot 
# 
# We can group the monthly data by year and get an idea of the spread of observations for each year and how this may be changing. We do expect to see some trend (increasing mean or median), but it may be interesting to see how the rest of the distribution may be changing. 
# 
# The example code below groups the observations by year and creates one box and whisker plot for each year of observations. The last year (2017) only contains 3 months and may not be a useful comparison with the 12 months of observations for other years. Therefore, only data between 1988 and 2016 was plotted.

# In[6]:

groups = series['1989':'2005'].groupby(TimeGrouper('A')) 
years = DataFrame()
for name, group in groups:
  years[name.year] = group.values
years.boxplot(return_type ='dict',figsize=(12,6))
pyplot.show(4)


# In[7]:

split_point = int(round(len(series)*0.8))
train, holdontest = series[0:split_point], series[split_point:]
print('Train %d, holdonTest %d' % (len(train), len(holdontest)))
train.to_csv('trainSet.csv')
holdontest.to_csv('holdontestSet.csv')


# In[8]:

series=Series.from_csv('trainSet.csv',header=0)


# The plot above has 9 box and whisker plots side-by-side, one for each of the 9 years of selected data. 
# 
# Some observations from reviewing the plots include:
#     
# - The median values for each year (red line) may show an increasing trend.
# - There are outliers half of the time

# ## 5.  Time Series Models

# ### 5.1 Checking stationarity 

# Analysis of the time series data assumes that we are working with a stationary time series. As we have seen in previous sections time series is almost certainly non-stationary. 
# 
# We can make it stationary by first differencing the series and using a statistical test to confirm that the result is stationary.
# 
# For stationarity, the code below creates a deseasonalized version of the series and saves it to file `stationary.csv`.

# The plot does not show any obvious seasonality or trend, suggesting the seasonally differenced dataset is a good starting point for modeling. 
# 
# We can use the the augmented Dickey-Fuller statistical significance test to check that the output series is stationary. 
# 

# In[9]:

X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.80)
train, test = X[0:train_size], X[train_size:]
# check if stationary
result = adfuller(train)
print('ADF Statistic: %f' % result[0]) 
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
 print('\t%s: %.3f' % (key, value)) 


# The results show that the test statistic value -4.099 is smaller than the critical value at 1% of -3.47. This suggests that we can reject the null hypothesis with a significance level of less than 1% (i.e. a low probability that the result is a statistical fluke). 
# 
# Rejecting the null hypothesis means that the time series is stationary or does not have time-dependent structure.
# 

# In[10]:

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) 


# ## 3. Naive forecast

# In[11]:

import matplotlib.axes as ax
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.80)
train, test = X[0:train_size], X[train_size:]

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# predict
	yhat = history[-1]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
#	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
    
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
mape= mean_absolute_percentage_error(test, predictions)
print('MAPE: %.3f' % mape)
# plot
plt.figure(figsize=(10,6))
plt.plot(test,label='test')
plt.plot(predictions, color='green',linestyle='--',label='predictions')
plt.legend(fontsize=14)
plt.show()


# In[12]:

X = series.values
train_size = int(len(X) * 0.80)
train, test = X[0:train_size], X[train_size:]
# train autoregression
model = AR(train)
model_fit = model.fit()
window = model_fit.k_ar
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)
#	print('predicted=%f, expected=%f' % (yhat, obs))
print('Lag: %s' % model_fit.k_ar)
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
mape= mean_absolute_percentage_error(test, predictions)
print('MAPE: %.3f' % mape)
# plot
plt.figure(figsize=(10,6))
plt.plot(test,label='test')
plt.plot(predictions, color='green',linestyle='--',label='predictions')
plt.legend(fontsize=14)
plt.show()


# In[13]:

from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic
X = series.values
train_size = int(len(X) * 0.80)
train, test = X[0:train_size], X[train_size:]
res = sm.tsa.arma_order_select_ic(train, max_ar=5, max_ma=5, ic=['aic', 'bic'], trend='nc')
res.aic_min_order
res.bic_min_order


# In[14]:

model = ARMA(train, order=(1, 1)) #5,7 is decided using arma_order_select_ic method
results = model.fit(trend='nc', method='css-mle')
print(results.summary2())


# ### 5.2 Manually Configure the ARIMA
# 
# We will use the difference adjusted dataset as an input to the ARIMA model. 
# 
# The `ARIMA(p,d,q)` model requires three parameters and is traditionally configured manually.
# 
# The first step is to select the lag values for the Autoregression (`AR`) and Moving Average (`MA`) parameters, `p` and `q` respectively. 
# 
# We can do this by reviewing Autocorrelation Function (`ACF`) and Partial Autocorrelation Function (`PACF`) plots. Note, we are now using the seasonally differenced `stationary.csv` as our dataset. This is because the manual seasonal differencing performed is different from the lag=1 differencing performed by the ARIMA model with the `d` parameter. (It also suggests that no further differencing may be required, and that the `d` parameter may be set to 0.) The example below creates ACF and PACF plots for the series.

# In this section, we will develop an Autoregressive Integrated Moving Average, or ARIMA, model for the problem. 
# 
# An ARIMA model can be considered as a special type of regression model--in which the dependent variable has been stationarized and the independent variables are all lags of the dependent variable and/or lags of the errors. Alternatively, you can think of a hybrid ARIMA/regression model as a regression model which includes a correction for autocorrelated errors. 
# 
# We will approach modeling by both manual and automatic configuration of the ARIMA model. This will be followed by investigating the residual errors of the chosen model. As such, this section is broken down into the following steps:
# 1. Checking stationarity
# 2. Manually Configure the ARIMA
# 3. Running the ARIMA model
# 4. Review Residual Errors
# 5. Finalize model 
# 6. Making predictions

# We can make the following observations from the above plots.
# 
# - The ACF shows a significant lag for 1 month.with perhaps some significant lag at 13 months.
# - The PACF shows a significant lag for 1 month, with perhaps some significant lag at 13 months.
#   
# Both the ACF and PACF show a drop-off at the same point, perhaps suggesting a mix of AR and MA.
# 
# This quick analysis suggests an `ARIMA(1,0,1)` on the stationary data may be a good starting point. The historic observations will be seasonally differenced prior to the fitting of each ARIMA model. The differencing will be inverted for all predictions made to make them directly comparable to the expected observation in the original sale count units. Experimentation shows that this configuration of ARIMA does not converge and results in errors by the underlying
# 
# 

# ### 5.31 Running the ARIMA model

# The example below demonstrates the performance of the selected ARIMA model using the experiemental setup.

# In[19]:

# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.80)
train, test = X[0:train_size], X[train_size:]

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	model = ARIMA(history, order=(1,0,1)) #using manually selected paratments
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
#	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))  
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
mape= mean_absolute_percentage_error(test, predictions)
print('MAPE: %.3f' % mape)
# plot
plt.figure(figsize=(10,6))
plt.plot(test,label='test')
plt.plot(predictions, color='green',linestyle='--',label='predictions')
plt.legend(fontsize=14)
plt.show()


# In[20]:

print(model_fit.summary().as_latex())


# Running the ARIMA(0,1,1) model results in an RMSE of 0.844, which is better than the persistence RMSE of 0..
# 
# This is a great start, but we may be able to get improved results with a better configured ARIMA model.

# In[25]:

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	X = X.astype('float32')
	train_size = int(len(X) * 0.80)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(trend='nc', disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)
	return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# load dataset
series = Series.from_csv('trainSet.csv')
# evaluate parameters
p_values = range(1, 5)
d_values = range(0, 2)
q_values = range(1, 5)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)


# In[26]:

# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.80)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	model = ARIMA(history, order=(4,1,1)) #using manually selected paratments
	model_fit = model.fit()
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
#	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
print(model_fit.summary())   
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
mape= mean_absolute_percentage_error(test, predictions)
print('MAPE: %.3f' % mape)
# plot
plt.figure(figsize=(10,6))
plt.plot(test,label='test')
plt.plot(predictions, color='green',linestyle='--',label='predictions')
plt.legend(fontsize=14)
plt.show()


# In[33]:

# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.80)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	model = ARIMA(history, order=(4,1,1)) #using manually selected paratments
	model_fit = model.fit(disp=False, trend='c',transparams=False)
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
#	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
print(model_fit.summary())   
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
mape= mean_absolute_percentage_error(test, predictions)
print('MAPE: %.3f' % mape)
# plot
plt.figure(figsize=(10,6))
plt.plot(test,label='test')
plt.plot(predictions, color='green',linestyle='--',label='predictions')
plt.legend(fontsize=14)
plt.show()


# In[34]:

print(model_fit.summary().as_latex())


# ### 5.4 Review residual errors

# A good final check of a model is to review residual forecast errors. Ideally, the distribution of residual errors should be a Gaussian with a zero mean. We can check this by using summary statistics and plots to investigate the residual errors from the ARIMA(0,1,1) model. The example below calculates and summarizes the residual forecast errors.

# In[36]:

# summarize ARIMA forecast residuals
# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.80)
train, test = X[0:train_size], X[train_size:]

# walk-forward validation
history = [x for x in train]
predictions = list()

for i in range(len(test)):
	# difference data
	# predict
	model = ARIMA(history, order=(4,1,1))
	model_fit = model.fit(disp=False, trend='c',transparams=False)
	yhat = model_fit.forecast()[0]
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)

# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())


# We can see that the distribution has a right shift and that the mean is non-zero. This is perhaps a sign that the predictions are biased.
# 
# We can examine this further by plotting the distribution of redidual errors. 

# In[67]:

print(residuals.describe())
plt.figure(figsize=(8,5))
sns.distplot(residuals, kde=False, fit=stats.gamma,color='green')
sns.plt.title('Residual Error Distribution',fontsize=14)


# In[69]:

# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.80)
train, test = X[0:train_size], X[train_size:]
# walk-forward validation
history = [x for x in train]
predictions = list()
bias = -0.079681
for i in range(len(test)):
	# predict
	model = ARIMA(history, order=(4,1,1))
	model_fit = model.fit(disp=False, trend='c',transparams=False)
	yhat = model_fit.forecast()[0]
	yhat = yhat+bias
	predictions.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
mape= mean_absolute_percentage_error(test, predictions)
print('MAPE: %.3f' % mape)
# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()


# In[74]:

pyplot.figure(figsize=(11,7))
pyplot.subplot(211)
plot_acf(residuals, ax=pyplot.gca())
plt.title('Autocorrelation',fontsize=14)
pyplot.subplot(212)
plot_pacf(residuals, ax=pyplot.gca())
plt.title('Partial Autocorrelation',fontsize=14)
pyplot.show()


# ## Model Validation

# ### 5.5 Finalize model  

# Having manually selected the ARIMA(2,0,2) model we are in a position to finalizing the model by fitting our selected ARIMA model on the dataset, in this case on a transformed version of the entire dataset. Once fit, the model can be saved to file for later use. 

# In[89]:

X = series.values
X = X.astype('float32')
# difference data
def __getnewargs__(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
 
ARIMA.__getnewargs__ = __getnewargs__

model = ARIMA(X, order=(4,1,1))
model_fit = model.fit(trend='nc', disp=0)
# bias constant, could be calculated from in-sample mean residual
bias = -0.079681
# save model
model_fit.save('model.pkl')
numpy.save('model_bias.npy', [bias])


# Running the code above creates two local files:
# 
# - `model.pkl` This is the `ARIMAResult` object from the call to `ARIMA.fit()`. This includes the coefficients and all other internal data returned when fitting the model.
# - `model_bias.npy` This is the bias value stored as a one-row, one-column NumPy array.

# ### 5.6 Making predictions

# A natural case may be to load the model and make a single forecast. This is relatively straightforward and involves restoring the saved model and the bias and calling the `forecast()` function. To invert the seasonal differencing, the historical data must also be loaded. The example below loads the model, makes a prediction for the next time step, and prints the prediction.

# In[90]:

from statsmodels.tsa.arima_model import ARIMAResults

series = Series.from_csv('trainSet.csv')
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
yhat = float(model_fit.forecast()[0])
yhat = bias + yhat
print('Predicted: %.3f' % yhat)


# This prediction gives the sort of result we would expect in the first instance. 
# 
# Now let us explore the model properly and use it in a simulated operational manner. In the Experimental setup section, we split the original dataset into test and training. We can load the train.csv file now and use it see how well our model really is on unseen data. There are two ways we might proceed:
# 
# - Load the model and use it to forecast foreward in time over many months. The forecast beyond the first one or two months will quickly start to degrade in performanace as we get further away from the known data. 
# 
# - Load the model and use it in a rolling-forecast manner, updating the transform and model for each time step. This is the preferred method as it is how one would use this model in practice as it would achieve the best performance.
# 
# As with model evaluation in previous sections, we will make predictions in a rolling-forecast manner. This means that we will step over lead times in the validation dataset and take the observations as an update to the history.

# In[94]:

# load and prepare datasets
dataset = Series.from_csv('trainSet.csv')
X = dataset.values.astype('float32')
history = [x for x in X]
validation = Series.from_csv('holdontestSet.csv')
y = validation.values.astype('float32')
# load model
model_fit = ARIMAResults.load('model.pkl')
bias = numpy.load('model_bias.npy')
# make first prediction
predictions = list()
yhat = float(model_fit.forecast()[0])
yhat = bias + yhat
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))
# rolling forecasts
for i in range(1, len(y)):
	# predict
	model = ARIMA(history, order=(4,1,1))
	model_fit = model.fit(trend='nc', disp=0, start_ar_lags=6)
	yhat = model_fit.forecast()[0]
	yhat = bias + yhat
	predictions.append(yhat)
	# observation
	obs = y[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
    
rmse = sqrt(mean_squared_error(y, predictions))
print('RMSE: %.3f' % rmse)
mape= mean_absolute_percentage_error(y, predictions)
print('MAPE: %.3f' % mape)
# plot
plt.figure(figsize=(10,6))
plt.plot(y,label='holdontest')
plt.plot(predictions, color='green',linestyle='--',label='predictions')
plt.legend(fontsize=14)
plt.show()


# From the plot above we can see that the `ARIMA` model is working in the manner expected, but that our predictions are not always inline with the test data. 
# Before we look at the errors or failure modes in more detail let us now appraoch this learning problem using a diffrent set of techniques. 
#     

# ## 6. Supervised Learning setup

# In previous sections we have trained an ARIMA model on time series data.
# 
# Here we examin how time series forecasting can be framed as a supervised learning problem. This re-framing of your time series data allows you access to the suite of standard linear and nonlinear machine learning algorithms on your problem (for example eg. Boosted Trees). 

# ### 6.1 Sliding Window (univerate) 
# 
# Time series data can be reformulated as supervised learning. Given a sequence of numbers for a time series dataset, we can restructure the data to look like a supervised learning problem. We can do this by using previous time steps as input variables and use the next time step as the output variable. Let’s make this concrete with an example. Imagine we have a time series below and are trying to learn the mapping function from input to output
# 
# $Y = f(X)$
# 
# for input variables ($X$) and ouptut variables ($Y$). Starting with the following data:
# 
# <table width="250">
#   <tr>
#     <th>time</th>
#     <th>measure</th> 
#   </tr>
#   <tr>
#     <td>1</td>
#     <td>100</td> 
#   </tr>
#   <tr>
#     <td>2</td>
#     <td>110</td> 
#   </tr>
#     <tr>
#     <td>3</td>
#     <td>108</td> 
#   </tr>
#     <tr>
#     <td>4</td>
#     <td>115</td> 
#   </tr>
#     <tr>
#     <td>5</td>
#     <td>120</td> 
#   </tr>
# </table>
# 
# <br>
# 
# <p> We can restructure this time series dataset as a supervised learning problem by using the value at the previous time step to predict the value at the next time-step. Re-organizing the time series dataset this way, the data would look as follows:<p>
# 
# <table width="250">
#   <tr>
#     <th>$X$</th>
#     <th>$y$</th> 
#   </tr>
#   <tr>
#     <td>?</td>
#     <td>100</td> 
#   </tr>
#   <tr>
#     <td>100</td>
#     <td>110</td> 
#   </tr>
#     <tr>
#     <td>110</td>
#     <td>108</td> 
#   </tr>
#     <tr>
#     <td>108</td>
#     <td>115</td> 
#   </tr>
#     <tr>
#     <td>115</td>
#     <td>120</td> 
#   </tr>
#     <tr>
#     <td>120</td>
#     <td>?</td> 
#   </tr>
# </table>
#     

# Take a look at the above transformed dataset and compare it to the original time series. Here are some observations:
# 
# - We can see that the previous time step is the input ($X$) and the next time step is the output ($y$) in our supervised learning problem.
# - We can see that the order between the observations is preserved, and must continue to be preserved when using this dataset to train a supervised model.
# - We can see that we have no previous value that we can use to predict the first value in the sequence. We will delete this row as we cannot use it.
# - We can also see that we do not have a known next value to predict for the last value in the sequence. We may want to delete this value while training our supervised model also.
# 
# The use of prior time steps to predict the next time step is called the sliding window method. For short, it may be called the window method in some literature. In statistics and time series analysis, this is called a lag or lag method. The number of previous time steps is called the window width or size of the lag. This sliding window is the basis for how we can turn any time series dataset into a supervised learning problem. From this simple example, we can notice a few things:
# 
# - We can see how this can work to turn a time series into either a regression or a classification supervised learning problem for real-valued or labeled time series values.
# - We can see how once a time series dataset is prepared this way that any of the standard linear and nonlinear machine learning algorithms may be applied, as long as the order of the rows is preserved.
# - We can see how the width sliding window can be increased to include more previous time steps.
# - We can see how the sliding window approach can be used on a time series that has more than one value, or so-called multivariate time series.
# 

# Let us frame the Champagne sales forecast problem as a supervised learning one (where the outputs are the differenced series values) so we can apply well-known ML regressors from `scikit`. We start with extracting features.

# In[104]:

series = Series.from_csv('trainSet.csv', header=0)
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.80)
train, test = X[0:train_size], X[train_size:]

# creating lag features with pandas
diff = DataFrame(difference(series, months_in_year))
dataframe = concat([diff.shift(3), diff.shift(2), diff.shift(1), diff], axis=1)
dataframe.columns = ['t-2', 't-1', 't', 't+1']
print(len(dataframe))
dataframe.head()


# Two remarks:
# 
# - There are 12 less rows in diff (hence in dataframe) than in series, since we are dealing with differenced series
# - The first rows contain nans because of the lag features. Let's discard them and get the values in a form we can use within scikit.

# In[133]:

XX = dataframe.values[3:,0:-1]
yy = dataframe.values[3:,-1]


# We define the training and test sets in a way such that the 1st element in the supervised learning test set corresponds to the 1st element in the previous time series test set.

# In[134]:

train_size = int(len(series) * 0.80) - 3 - 12 # because of the lag and of the difference
XX_train = XX[0:train_size]
XX_test = XX[train_size:]
yy_train = yy[0:train_size]
yy_test = yy[train_size:]


# We can compare the sizes of train and test sets with what we had before, and check that the size of the test set is the same.

# In[135]:

print(train_size)
print(len(XX)-train_size)


# ## 7. Random Forest Models

# Let's train the regressor we want to work with:

# In[136]:

model = RandomForestRegressor()
model.fit(XX_train, yy_train)


# We reuse the previous evaluation code but this time we make predictions with this regressor

# In[137]:

# walk-forward validation
history = [x for x in train]
prediction_sl = list()
for i in range(len(test)):
	yhat = model.predict(XX_test[i,:])[0]
	yhat = inverse_difference(history, yhat, months_in_year)
	prediction_sl.append(yhat)
	# observation
	obs = test[i]
	history.append(obs)
	prediction_sl[i]=yhat
#	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))


# In[138]:

rmse = sqrt(mean_squared_error(test, prediction_sl))
print('RMSE: %.3f' % rmse)


# In[139]:

pyplot.plot(test, color='green')
pyplot.plot(prediction_sl, color='orange')
pyplot.show()


# In[ ]:



