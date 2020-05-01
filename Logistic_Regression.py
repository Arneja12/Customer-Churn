#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


df=pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv")


# In[39]:


df.head()


# In[40]:


df_churn=df[['tenure','age','income','ed','employ','callcard','wireless','churn']]
df_churn['churn']=df_churn['churn'].astype('int')
df_churn.head()


# In[41]:


X=np.asarray(df_churn[['tenure','age','income','ed','employ','callcard','wireless']])
X[0:5]


# In[42]:


y=np.asarray(df_churn['churn'])
y[0:5]


# In[43]:


from sklearn import preprocessing
X=preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[44]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test=train_test_split(X,y,test_size=0.2,random_state=4)
print('Train set:',X_train.shape , y_train.shape)
print('Test set:', X_test.shape, y_test.shape)


# In[100]:


from sklearn.linear_model import LogisticRegression
LR= LogisticRegression(C=0.001,solver='liblinear').fit(X_train,y_train)


# In[101]:


yhat=LR.predict(X_test)
yhat


# In[102]:


yhat_prob= LR.predict_proba(X_test)
yhat_prob


# In[103]:


from sklearn.metrics import jaccard_similarity_score , classification_report
jaccard_similarity_score(y_test,yhat)


# In[104]:


print(classification_report(y_test,yhat))

