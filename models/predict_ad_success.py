#!/usr/bin/env python
# coding: utf-8

# ## Predict ad success
# This is a binary classification problem where you need to predict whether an ad buy will lead to a netgain.

# In[1]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# There are 3 csv files in the current version of the dataset:
# 

# In[2]:


# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# The next hidden code cells define functions for plotting data. Click on the "Code" button in the published kernel to reveal the hidden code.

# In[3]:


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# In[4]:


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# In[5]:


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# Now you're ready to read in the data and use the plotting functions to visualize the data.

# ### Let's check 1st file: /kaggle/input/Dataset/sample_submission.csv

# In[6]:

dir = os.path.abspath(os.curdir)

nRowsRead = 1000 # specify 'None' if want to read whole file
# sample_submission.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv(dir+'\\datasets\\predict_ad_success_sample_submission.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = dir+'\\datasets\\predict_ad_success_sample_submission.csv'


# nRow, nCol = df1.shape
# print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[7]:

def predict_ad_success_getTop1(n):
    return df1.head(n)


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[8]:

def predict_ad_success_plotPerColumnDistribution1():
    return plotPerColumnDistribution(df1, 10, 5)


# ### Let's check 2nd file: /kaggle/input/Dataset/Train.csv

# In[9]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# Train.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv(dir+'\\datasets\\predict_ad_success_train.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = dir+'\\datasets\\predict_ad_success_train.csv'
nRow, nCol = df2.shape

# print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[10]:


def predict_ad_success_getTop2(n):
    return df2.head(n)


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[11]:


def predict_ad_success_plotPerColumnDistribution2():
    return plotPerColumnDistribution(df2, 10, 5)


# Correlation matrix:

# In[12]:


def predict_ad_success_plotCorrelationMatrix2():
    return plotCorrelationMatrix(df2, 8)


# Scatter and density plots:

# In[13]:


def predict_ad_success_plotScatterMatrix2():
    return plotScatterMatrix(df2, 9, 10)


# ### Let's check 3rd file: /kaggle/input/Dataset/Test.csv

# In[14]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# Test.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df3 = pd.read_csv(dir+'/datasets/predict_ad_success_test.csv', delimiter=',', nrows = nRowsRead)
df3.dataframeName = dir+'/datasets/predict_ad_success_test.csv'
nRow, nCol = df3.shape

# print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[15]:


def predict_ad_success_getTop3(n):
    return df3.head(n)


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[17]:


def predict_ad_success_plotPerColumnDistribution3():
    return plotPerColumnDistribution(df3, 10, 5)


# Correlation matrix:

# In[18]:


def predict_ad_success_plotCorrelationMatrix3():
    return plotCorrelationMatrix(df3, 8)


# Scatter and density plots:

# In[19]:


def predict_ad_success_plotScatterMatrix3():
    return plotScatterMatrix(df3, 9, 10)


# ## Conclusion
# This concludes your starter analysis! To go forward from here, click the blue "Fork Notebook" button at the top of this kernel. This will create a copy of the code and environment for you to edit. Delete, modify, and add code as you please. Happy Kaggling!
