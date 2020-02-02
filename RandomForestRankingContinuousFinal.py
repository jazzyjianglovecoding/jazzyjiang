
# coding: utf-8

# # Hiscox Project: Prediction of Settle or Dismiss(start with stepwise)
# 
# 

# ## Data Wrangling
# 
# ### Importing Data with Pandas

# In[35]:

# Some methods to show polt in notebook:
get_ipython().magic(u'matplotlib inline')
# Import libraries to be used
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing


# In[36]:

dataframenew= pd.read_csv('Featureselectioncontinuous.csv')
# Using panda to read the csv file


# Next, we can browse the dataframe by just type in the name of dataset 

# In[37]:

X=dataframenew.iloc[:,1:127]
y=dataframenew.iloc[:,0:1]


# Firstly we need to import the `RandonForestClassifier` and `preprocessing` modules from the Scikit-learn library

# In[38]:

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


# We need to seed the random seed and initialize the label encoder in order to preprocess some of our features

# In[39]:

# Set the random seed
np.random.seed(12)
# Initialize label encoder
label_encoder = preprocessing.LabelEncoder()


# We need to initialize the Random Forest model with the following parameters: number of estimators as `1000`, max features as `2` and oob_score as `True`, see the Sci-kit learn documentation for [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and [User Guide](http://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees) for usage.

# In[40]:

# Initialize the Random Forest model
from sklearn.ensemble import RandomForestRegressor
randomforest = RandomForestRegressor(max_features=2,n_estimators=500,oob_score=True) 
# Use OOB scoring and put estimators number to 500


# In[41]:

# Define our features
features = [ u'FilingPeriod0', u'BankruptcyBinary', u'MisrepresentationDisclosure',
       u'FalsePositivesFailtoDisclo', u'ClinicalTrialFailuresDelays',
       u'ExecutiveCompensationIssues', u'InsiderTradingSpecificallyAll',
       u'StockPriceManipulationUporD', u'StockOptionDating',
       u'RestatementofEarnings', u'MissingRelatedtoEarningsProj',
       u'AccountingImproprietiesandIna', u'ViolationsofGAAP',
       u'ProductsFailuresDelaysMis', u'TenderOffertoBuyBackShares',
       u'MergerAcquisitionbyCompany', u'SarbanesOxley',
       u'MergerAcquisitionbyOthersA', u'SpinoffDivesture',
       u'RegulatoryViolationsFDAFTC', u'MisappropriationofIntellectual',
       u'ImproperSettlementofLawsuit', u'CriminalIllegalDeceptiveA',
       u'BreachofFiduciaryDuties', u'BadManagementGeneralAllegat',
       u'SelfDealingInclRelatedParty', u'Subprime', u'FCPA', u'Liquidity',
       u'StGoogleHits', u'SP500ReturnDuringClassPeri', u'StIPOPrice',
       u'StClassPeriodLengthDays', u'StFilingPeriodDays',
       u'StNumberofDaystoSettle', u'StWithinDaysofIPODate', u'StMarketCap',
       u'StStockPriceOpen', u'StStockPrice3weekprior',
       u'StStockPriceClose',  u'StMktValueDropNoInsider', u'StRevenue',
       u'StAssets', u'NormalisedProfits', u'NormalisedNetWorth',
       u'NormalisedPERatio',u'NormalisedMCaptoNetWorth', u'NormalisedProfittoAsset',
       u'NormalisedProfittoNetWorth', u'_ISector_2', u'_ISector_3',
       u'_ISector_4', u'_ISector_5', u'_ISector_6', u'_ISector_7',
       u'_ISector_8', u'_ISector_9', u'_ISector_10', u'_ISector_11',
       u'_ISector_12', u'_ISector_13', u'_ISector_14', u'_ISector_15',
       u'_ISector_16', u'_ISector_17', u'_ISector_18', u'_ISector_19',
       u'_ISector_20', u'_ISector_21', u'_ISector_22', u'_ISector_23',
       u'_ISector_24', u'_ISector_25', u'_ISector_26', u'_ISector_27',
       u'_ISector_28', u'_ISector_29', u'_ISector_30', u'_ISector_31',
       u'_ISector_32', u'_ISector_33', u'_ISector_34', u'_ISector_35',
       u'_ISector_36', u'_ISector_37', u'_ISector_38', u'_ISector_39',
       u'_ISector_40', u'_ISector_41', u'_ISector_42', u'_ISector_43',
       u'_ISector_44', u'_IUSIPOCate_2', u'_IUSIPOCate_3', u'_ISCACatego_2',
       u'_ISCACatego_3', u'_ISCACatego_4', u'_ISCACatego_5', u'_ISCACatego_6',
       u'_ISCACatego_7', u'_ISCACatego_8', u'_ISCACatego_9', u'_ISCACatego_10',
       u'_ISCACatego_11', u'_ISCACatego_12', u'_ICircuit_2', u'_ICircuit_3',
       u'_ICircuit_4', u'_ICircuit_5', u'_ICircuit_6', u'_ICircuit_7',
       u'_ICircuit_8', u'_ICircuit_9', u'_ICircuit_10', u'_ICircuit_11',
       u'_IStockExch_2', u'_IStockExch_3', u'_IStockExch_4', u'_IPlantiff_1',
       u'_IPlantiff_2', u'_IPlantiff_3', u'_ISettlingD_2', u'_ISettlingD_3']
# Train the model
randomforest.fit(X=X[features],y=y['StCashSettlementAmount'])
# Print OOB accuracy
print("OOB accuracy : %f"%randomforest.oob_score_)
# Print feature importances
for feature, importance in zip(features, randomforest.feature_importances_):
    print (feature, ":", importance)


# In[42]:

randomforest.feature_importances_


# Define our features to be used and train the Random Forest model using the `.fit()` function and passing `X=df[features]` and `y=df["Survived"])` as arguments.

# Finally we print the out-of-bag accuracy using `rf_model.oob_score_` followed by the feature importances.

# In[45]:

import matplotlib
from pylab import barh,plot,yticks,show,grid,xlabel,figure
wscores = list(zip(features,randomforest.feature_importances_))
wchi2 = sorted(wscores,key=lambda x:x[1]) 
topchi2 = list(zip(*wchi2[-35:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.figure(figsize=(13,9))
barh(x,topchi2[1],align='center',alpha=.2,color='c')
plot(topchi2[1],x,'-o',markersize=2,alpha=.8,color='c')
yticks(x,labels,fontsize=13)
matplotlib.rc('xtick', labelsize=16) 
xlabel('Random Forest Importance',fontsize=18)
show()

