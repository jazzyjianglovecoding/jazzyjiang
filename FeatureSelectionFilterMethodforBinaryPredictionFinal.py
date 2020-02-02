
# coding: utf-8

# In[2]:

# Some methods to show plot in notebook:
get_ipython().magic(u'matplotlib inline')
# Import libraries to be used}
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as math
import seaborn as sns
import scipy as scipy


# In[3]:

df= pd.read_csv('ChiTestSelection.csv')
# Using panda to read the csv file
X=df.iloc[:,1:111]
y=df.iloc[:,0:1]


# In[4]:

from sklearn.feature_selection import chi2
import seaborn as sns
import matplotlib.pyplot as plt
# compute chi2 for each feature
chi2score = chi2(X,y)[0]
chi2score


# In[5]:

from pylab import barh,plot,yticks,show,grid,xlabel,figure
figure(figsize=(11,8))
features=['FilingPeriod0', 'BankruptcyBinary', 'MisrepresentationDisclosure',
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
       '_ISCACatego_11', '_ISCACatego_12', '_IMarketCap_2',
       '_IMarketCap_3', '_IMarketCap_4', '_IMarketCap_5', '_IMarketCap_6',
       '_IMarketCap_7', '_IMarketCap_8', '_IMarketCap_9', '_IMarketCap_10',
       '_IMarketCap_11', '_IMarketCap_12', '_ICircuit_2', '_ICircuit_3',
       '_ICircuit_4', '_ICircuit_5', '_ICircuit_6', '_ICircuit_7',
       '_ICircuit_8', '_ICircuit_9', '_ICircuit_10', '_ICircuit_11',
       '_IStockExch_2', '_IStockExch_3', '_IStockExch_4', '_IStockExch_5',
       '_IPlantiff_1', '_IPlantiff_2', '_IPlantiff_3']
wscores = list(zip(features,chi2score))
wchi2 = sorted(wscores,key=lambda x:x[1]) 
topchi2 = list(zip(*wchi2[-30:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
barh(x,topchi2[1],align='center',alpha=.2,color='g')
plot(topchi2[1],x,'-o',markersize=2,alpha=.8,color='g')
yticks(x,labels,fontsize=12)
xlabel('$\chi^2$',fontsize=15)
show()


# In[ ]:




# In[ ]:



