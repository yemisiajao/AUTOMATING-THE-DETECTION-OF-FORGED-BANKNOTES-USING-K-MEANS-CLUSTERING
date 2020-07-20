#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the needed python libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 


# In[2]:


#read the dataset from the file's directory 
data = pd.read_csv('Banknote_authentication.csv')
data


# In[22]:


#extract values needed
x = data['V1']
y = data['V2']


# In[3]:


#compute the statistical analysis of the dataset
data.describe()


# In[4]:


#plot a graph to check the linear correlation of the data features
sns.pairplot(data)


# In[24]:


#performing normaliztion on the features
mean_x=x.mean()
mean_y=y.mean()
max_x=x.max()
max_y=y.max()
min_x=x.min()
min_y=y.min()
for i in range(0,x.size):
    x[i] = (x[i] - mean_x) / (max_x - min_x)
for i in range(0,y.size):
    y[i] = (y[i] - mean_y) / (max_y - min_y)


# In[25]:


#performing clustering
fake_real = np.column_stack((x, y))
km_res = KMeans(n_clusters=2).fit(fake_real)

km_res.cluster_centers_


# In[26]:


km_res = KMeans(n_clusters=2).fit(fake_real)
km_res.cluster_centers_


# In[14]:


km_res = KMeans(n_clusters=2).fit(fake_real)
km_res.cluster_centers_


# In[28]:


#group the dataset
xval_0 = []
yval_0 = []
xval_1 = []
yval_1 = []

for i in range (0, x.size):
    if(km_res.labels_[i] ==0):
        xval_0.append(x[i])
        yval_0.append(y[i])
    else:
        xval_1.append(x[i])
        yval_1.append(y[i])
        


# In[29]:


#plotting the values and the clusters
clusters = km_res.cluster_centers_

plt.scatter(xval_0, yval_0, c='red')
plt.scatter(xval_1, yval_1, c='green')
plt.scatter(clusters[:,0], clusters[:,1], s=200, c='blue')

plt.xlabel('V1')
plt.ylabel('V2')
plt.savefig('fig1')


# In[ ]:




