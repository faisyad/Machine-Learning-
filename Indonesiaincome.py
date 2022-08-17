#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[3]:


df = pd.read_csv("D:\Machine Learning\Project+Exercise Single Linear Regression\Indonesiacapita.csv")
df


# In[5]:


df.columns.values


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Years')
plt.ylabel('income per capita')
plt.scatter(df.years, df.percapitaincome, color = 'red', marker = '+')


# In[8]:


reg = linear_model.LinearRegression()
reg.fit(df[['years']].values, df.percapitaincome)


# In[9]:


reg.get_params()


# In[10]:


reg.predict([[2022]])


# In[11]:


reg.coef_


# In[12]:


reg.intercept_


# In[13]:


72693.90909091*2022+-142759713.1060606


# In[14]:


d=pd.read_csv("D:\Machine Learning\Project+Exercise Single Linear Regression\Indonesiacapita.csv")
d


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Years')
plt.ylabel('Income per Capita')
plt.scatter(d.years, d.percapitaincome, color='red', marker ='.')
plt.plot(d.years, reg.predict(d[['years']]), color='red')

