#!/usr/bin/env python
# coding: utf-8

# # Advertising Logistic Regresson

# ### Predict whether or not a user will click on an advertisement

# ## Imports 

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


dir = os.path.abspath(os.curdir)

# In[10]:

# import model
ad_data = pd.read_csv(dir+'\\datasets\\user_behavior_advertising.csv')
# print(ad_data)


# ## Exploratory Data Analysis

# In[11]:

def getTop(n):
    return ad_data.head(n)


# In[12]:

def getInfo():
    return ad_data.info()


# In[13]:


def getDescription():
    return ad_data.describe()


# In[14]:

def all():
  return ad_data.to_json()

def getAge():
    return ad_data['Age'].to_json() 

def getDailyInternetUsage():
    return ad_data['Daily Internet Usage'].to_json()

def getAreaIncome():
    return ad_data['Area Income'] 

def getDailyTimeSpentonSite():
    return ad_data['Daily Time Spent on Site'] 

def getMale():
    return ad_data['Male']    

def plotAgeFrequency():
    return ad_data['Age'].plot.hist(bins=30)


# In[15]:

def getAgeAreaIncome():
    return ad_data.loc[:, ['Age','Area Income']].to_json()

def plotAgeAreaIncome():
    return sns.jointplot(x='Age',y='Area Income',data=ad_data)


# In[16]:

def getAgeDailyTimeSpentonSite():
    
    return ad_data.loc[:, ['Age','Daily Time Spent on Site']].to_json()


def plotAgeDailyTimeSpentonSite():
    return sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde',color='red')


# In[17]:

def getDailyTimeSpentonSiteDailyInternetUsage():
    return ad_data['Daily Time Spent on Site'],ad_data['Daily Internet Usage'], 

def plotDailyTimeSpentonSiteDailyInternetUsage():
    return sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data)


# In[18]:


sns.pairplot(ad_data,hue='Clicked on Ad')


# ## Logistic Regression Model

# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

# In[20]:

X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


      


# ### Train and fit a logistic regression model on the training set

# In[21]:

def getPredictions():
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)


# ## Predictions and Evaluations

# In[23]:
    predictions = logmodel.predict(X_test)
    return predictions


# In[24]:

from sklearn.metrics import classification_report,confusion_matrix
def precision():
    predictions = getPredictions()
    classificationReport = classification_report(y_test,predictions)
    confusionMatrix = confusion_matrix(y_test,predictions)
    # confusionMatrix
    return classificationReport
