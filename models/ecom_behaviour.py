#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. .read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
from scipy import stats

        
pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 100)
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[4]:
import os
dir = os.path.abspath(os.curdir)

visData = pd.read_csv(dir+"\\datasets\\ecom_behavior_visitorLogsData.csv")
userData = pd.read_csv(dir+"\\datasets\\ecom_behavior_userTable.csv")


# Taking only User Visited sites

# In[5]:


visData.drop(['City','Country'],axis=1,inplace=True)


# In[6]:


# visData[(visData['UserID'].isnull()==True) & (visData['ProductID'].isnull()==False)].groupby('webClientID').count()
bb = visData.groupby('webClientID').agg({'UserID':['size','count']}).reset_index()
bb.columns=['webclient','usersize','usercount']
bb['diff'] = bb['usersize']-bb['usercount']

def getTop(n):
    return bb.head(n)


# In[8]:


bb[(bb['usercount']!=0) & (bb['diff']!=0)]


# No such user exist in which user id is not halfway null for any webclientID

# In[12]:


visUsrData = visData[visData['UserID'].isnull()==False].copy()

def getShape():
    return print(visUsrData.shape)


# Correcting Dates

# In[13]:



from datetime import datetime
from tqdm import tqdm_notebook,tqdm

from scipy import stats
tqdm.pandas(desc="Running Date Time conversion")

def getOutput():

    def correctTimeStamp(x):
        if x==x:
            try:
                x = pd.to_datetime(x,format='%Y-%m-%d %H:%M:%S')
            except ValueError:
                x = int(x)
                x=x/ 10**9
                x = datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S.%f')
            return x
        else:
            return np.nan

        
    visUsrData['VisitDateTime'] = visUsrData['VisitDateTime'].progress_apply(correctTimeStamp)


    # ### Handling Missing Data & removing duplicates

    # In[14]:


    # visUsrData['UserID'].sample(20)
    visUsrData = visUsrData[visUsrData.duplicated()==False].copy()


    # ### Feature Engineering and Imputing missing values

    # In[15]:


    visUsrData['VisitDateTime']=visUsrData.groupby(['UserID','webClientID'])['VisitDateTime'].transform(lambda x: x.fillna(x.min()))
    visUsrData['Activity'] = visUsrData.sort_values(['UserID','VisitDateTime']).groupby('UserID')['Activity'].transform(lambda x:x.fillna(method = 'bfill'))
    visUsrData['Activity'].fillna('pageload',inplace=True)


    # In[16]:


    visUsrData['OS'] = visUsrData['OS'].apply(lambda x: x.lower())
    visUsrData['ProductID'] = visUsrData['ProductID'].apply(lambda x: x if x!=x else str(x).lower())

    visUsrData['SevenDays']=0
    visUsrData.loc[visUsrData['VisitDateTime']>='2018-05-21','SevenDays']=1
    visUsrData['FifteenDays']=0
    visUsrData.loc[visUsrData['VisitDateTime']>='2018-05-13','FifteenDays']=1
    visUsrData['isActive'] = visUsrData['Activity'].apply(lambda x: x==x)

    visUsrData['is7Active']=0
    visUsrData.loc[(visUsrData['isActive']==1) &(visUsrData['SevenDays']==1),'is7Active']=1

    visUsrData['VisitDate'] = visUsrData['VisitDateTime'].dt.date
    visUsrData['Activity'] = visUsrData['Activity'].apply(lambda x: x if x!=x else str(x).lower())

    visUsrData['Pageloads_last_7_days']=0
    visUsrData.loc[(visUsrData['Activity']=='pageload') & (visUsrData['SevenDays']==1),'Pageloads_last_7_days']=1

    visUsrData['Clicks_last_7_days']=0
    visUsrData.loc[(visUsrData['Activity']=='click') & (visUsrData['SevenDays']==1),'Clicks_last_7_days']=1


    visUsrData['FifteenDaysActive']=0
    visUsrData.loc[((visUsrData['FifteenDays']==1) & (visUsrData['isActive']==True)),'FifteenDaysActive']=1
    visUsrData['pageloads_actvity']=0
    visUsrData.loc[(visUsrData['Activity']=='pageload'),'pageloads_actvity']=1

    visUsrData['ProductID'] = visUsrData.sort_values(['UserID','VisitDateTime']).groupby('UserID')['ProductID'].transform(lambda x:x.fillna(method = 'bfill'))


    # ### Creating Input Features

    # ## No_of_days_Visited_7_Days

    # In[17]:


    df = pd.DataFrame(visUsrData.groupby(['UserID'])['webClientID'].count().reset_index().drop('webClientID',axis=1))
    df


    # In[18]:



    df = df.merge(visUsrData.groupby(['UserID','VisitDate','is7Active']).count().reset_index().groupby('UserID')['is7Active'].sum().reset_index(),on='UserID',how='left')
    df['is7Active'].fillna(0,inplace=True)


    # In[19]:


    df['is7Active'].value_counts()


    # Using User Data

    # In[20]:


    visUsrData = visUsrData.merge(userData,on='UserID',how='inner')
    visUsrData['Signup Date']=pd.to_datetime(visUsrData['Signup Date'],format="%Y-%m-%d %H:%M:%S")
    visUsrData['Signup Date'] = visUsrData['Signup Date'].dt.tz_localize(None)
    visUsrData['User_Vintage'] = (visUsrData['VisitDateTime'].max() - visUsrData['Signup Date']).dt.days


    # ## User_Vintage

    # In[21]:


    df = df.merge(visUsrData.groupby(['UserID'])['User_Vintage'].max().reset_index(),on='UserID',how='left')
    df['User_Vintage']=df['User_Vintage']+1


    # ## Most_Active_OS

    # In[22]:



    df['Most_Active_OS'] = visUsrData.groupby(['UserID'])['OS'].agg(lambda x: stats.mode(x)[0][0]).values


    # In[23]:


    df['Most_Active_OS'].value_counts()


    # ## Pageloads_last_7_days & Clicks_last_7_days

    # In[24]:


    df ['Pageloads_last_7_days'] = visUsrData.groupby(['UserID'])['Pageloads_last_7_days'].sum().values
    df ['Clicks_last_7_days'] = visUsrData.groupby(['UserID'])['Clicks_last_7_days'].sum().values


    # In[25]:


    df['Pageloads_last_7_days'].head(5)


    # ## Recently_Viewed_Product

    # In[26]:


    mask = visUsrData['pageloads_actvity']==1
    df = df.merge(visUsrData.sort_values(['UserID','VisitDateTime'])[mask].groupby(['UserID'])
                .agg({'ProductID':'last'}).reset_index().rename(columns={'ProductID':'Recently_Viewed_Product'}),on='UserID',how='left')


    # ## Most_Viewed_product_15_Days

    # In[27]:


    mask2 = (visUsrData['pageloads_actvity']==1)  & (visUsrData['FifteenDays']==1)
    df = df.merge(visUsrData.sort_values(['UserID','VisitDateTime'],ascending=[True, False])[mask2].groupby(['UserID'])
                .agg({'ProductID':lambda x: stats.mode(x)[0][0]}).reset_index().rename(columns={'ProductID':'Most_Viewed_product_15_Days'}),on='UserID',how='left')


    # ## No_Of_Products_Viewed_15_Days

    # In[28]:


    visUsrData['ProductID'] = visUsrData.groupby(['UserID'])['ProductID'].transform(lambda x: x.fillna(stats.mode(x)[0][0]))
    print(visUsrData['ProductID'].isnull().sum())
    visUsrData['ProductID'].fillna('Product101',inplace=True)
    print(visUsrData['ProductID'].isnull().sum())


    # In[29]:


    mask3 = visUsrData['FifteenDays']==1

    df = df.merge(visUsrData[mask3].groupby(['UserID']).agg({'ProductID':'nunique'}).reset_index().rename(columns={'ProductID':'No_Of_Products_Viewed_15_Days'}),on='UserID',how='left')


    # Fill in missing values

    # In[30]:


    # df['Most_Viewed_product_15_Days']='P12345'
    # df['Recently_Viewed_Product']='P12345'
    # df['Pageloads_last_7_days']=1
    # df['Clicks_last_7_days']=1

    df.reset_index(inplace=True)
    df.rename(columns={'is7Active':'No_of_days_Visited_7_Days'},inplace=True)

    df['Recently_Viewed_Product'].fillna('Product101',inplace=True)
    df['Most_Viewed_product_15_Days'].fillna('Product101',inplace=True)
    df['No_Of_Products_Viewed_15_Days'].fillna(0,inplace=True)

    df = df.reindex(['UserID','No_of_days_Visited_7_Days','No_Of_Products_Viewed_15_Days','User_Vintage','Most_Viewed_product_15_Days',
        'Most_Active_OS','Recently_Viewed_Product','Pageloads_last_7_days','Clicks_last_7_days'], axis=1)


    # Re Index

    # In[31]:


    df.isnull().sum()


    # In[33]:


    df.to_csv(dir+'\\datasets\\output\\ecom_behavior_input_feats_v27.csv',index=False)


# In[ ]:




