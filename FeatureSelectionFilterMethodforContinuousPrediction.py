
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


# In[2]:

df= pd.read_csv('CorrelationCheckContinuous.csv')
# Using panda to read the csv file


# In[7]:

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white")
# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:



