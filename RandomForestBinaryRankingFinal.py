
# coding: utf-8

# ## Data Wrangling
# 
# ### Importing Data with Pandas

# In[18]:

# Some methods to show polt in notebook:
get_ipython().magic(u'matplotlib inline')
# Import libraries to be used
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing


# In[19]:

dataframenew= pd.read_csv('Featureselectionbinary.csv')
# Using panda to read the csv file


# Next, we can browse the dataframe by just type in the name of dataset 

# In[24]:

X=dataframenew.iloc[:,1:134]
y=dataframenew.iloc[:,0:1]


# Firstly we need to import the `RandonForestClassifier` and `preprocessing` modules from the Scikit-learn library

# In[21]:

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


# We need to seed the random seed and initialize the label encoder in order to preprocess some of our features

# In[22]:

# Set the random seed
np.random.seed(12)
# Initialize label encoder
label_encoder = preprocessing.LabelEncoder()


# We need to initialize the Random Forest model with the following parameters: number of estimators as `1000`, max features as `2` and oob_score as `True`, see the Sci-kit learn documentation for [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and [User Guide](http://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees) for usage.

# In[23]:

# Initialize the Random Forest model
randomforest = RandomForestClassifier(max_features=2,n_estimators=500,oob_score=True) 
# Use OOB scoring and put estimators number to 500


# In[25]:

# Define our features
features = [ 'FilingPeriod0',
       'BankruptcyBinary', 'MisrepresentationDisclosure',
       'FalsePositivesFailtoDisclo', 'ClinicalTrialFailuresDelays',
       'ExecutiveCompensationIssues', 'InsiderTradingSpecificallyAll',
       'StockPriceManipulationUporD', 'StockOptionDating',
       'RestatementofEarnings', 'MissingRelatedtoEarningsProj',
       'AccountingImproprietiesandIna', 'ViolationsofGAAP',
       'ProductsFailuresDelaysMis', 'TenderOffertoBuyBackShares',
       'MergerAcquisitionbyCompany', 'SarbanesOxley',
       'MergerAcquisitionbyOthersA', 'SpinoffDivesture',
       'RegulatoryViolationsFDAFTC', 'MisappropriationofIntellectual',
       'ImproperSettlementofLawsuit', 'CriminalIllegalDeceptiveA',
       'BreachofFiduciaryDuties', 'BadManagementGeneralAllegat',
       'SelfDealingInclRelatedParty', 'Subprime', 'FCPA', 'Liquidity',
       'StPercentageDecreaseStockPrice', 'StPercentageDecreaseNew',
       'StSP500ReturnDuringClassPeri', 'StGoogleHits',
       'StClassPeriodLengthdays', 'StFilingPerioddays',
       'StNumberofDaystoDimissal', 'StWithinDaysofIPODatewe',
        'StStockPriceOpen',
       'StStockPrice3weekprior', 'StStockPriceClose',
       'StMCapNoInsiderHoldings',
       'StMktValueDropNoInsider', 'StRevenue', 'StAssets', 'StIPOPrice',
       'NormalisedProfits', 'NormalisedNetWorth', 'NormalisedPERatio',
       'NormalisedMCaptoNetWorth', 'NormalisedProfittoRevenue',
       'NormalisedProfittoAsset', 'NormalisedProfittoNetWorth',
       '_ISector_5', '_ISector_6', '_ISector_7', '_ISector_8',
       '_ISector_9', '_ISector_10', '_ISector_11', '_ISector_12',
       '_ISector_13', '_ISector_14', '_ISector_15', '_ISector_16',
       '_ISector_17', '_ISector_18', '_ISector_19', '_ISector_20',
       '_ISector_21', '_ISector_22', '_ISector_23', '_ISector_24',
       '_ISector_25', '_ISector_26', '_ISector_27', '_ISector_28',
       '_ISector_29', '_ISector_30', '_ISector_31', '_ISector_32',
       '_ISector_33', '_ISector_34', '_ISector_35', '_ISector_36',
       '_ISector_37', '_ISector_38', '_ISector_39', '_ISector_40',
       '_ISector_41', '_ISector_42', '_ISector_43', '_ISector_44',
       '_IUSIPOCate_2', '_IUSIPOCate_3', '_ISCACatego_2', '_ISCACatego_3',
       '_ISCACatego_4', '_ISCACatego_5', '_ISCACatego_6', '_ISCACatego_7',
       '_ISCACatego_8', '_ISCACatego_9', '_ISCACatego_10',
       '_ISCACatego_11', '_ISCACatego_12', '_ICircuit_2', '_ICircuit_3',
       '_ICircuit_4', '_ICircuit_5', '_ICircuit_6', '_ICircuit_7',
       '_ICircuit_8', '_ICircuit_9', '_ICircuit_10', '_ICircuit_11',
       '_IStockExch_2', '_IStockExch_3', '_IStockExch_4', '_IStockExch_5',
       '_IPlantiff_1', '_IPlantiff_2']
# Train the model
randomforest.fit(X=X[features],y=y['DependentVariableSettleDismis'])
# Print OOB accuracy
print("OOB accuracy : %f"%randomforest.oob_score_)
# Print feature importances
for feature, importance in zip(features, randomforest.feature_importances_):
    print (feature, ":", importance)


# In[30]:

randomforest.feature_importances_


# Define our features to be used and train the Random Forest model using the `.fit()` function and passing `X=df[features]` and `y=df["Survived"])` as arguments.

# Finally we print the out-of-bag accuracy using `rf_model.oob_score_` followed by the feature importances.

# In[52]:

import matplotlib
from pylab import barh,plot,yticks,show,grid,xlabel,figure
wscores = list(zip(features,randomforest.feature_importances_))
wchi2 = sorted(wscores,key=lambda x:x[1]) 
topchi2 = list(zip(*wchi2[-38:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.figure(figsize=(13,11))
barh(x,topchi2[1],align='center',alpha=.2,color='b')
plot(topchi2[1],x,'-o',markersize=2,alpha=.8,color='b')
yticks(x,labels,fontsize=13)
matplotlib.rc('xtick', labelsize=16) 
xlabel('Random Forest Importance',fontsize=18)
show()


# In[ ]:



