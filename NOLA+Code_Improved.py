
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

import scipy
from scipy import stats

from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt
import seaborn as sb
from pylab import rcParams

import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[2]:


address = '//prc-cs-f9dkb42/ecozzolino$/Desktop/code/nolabblistings.csv'
nola = pd.read_csv(address)


# In[3]:


print(nola.head(10))
print(nola.describe())


# In[4]:


rcParams['figure.figsize'] = 5, 4 #this is the size of the plot
sb.set_style('whitegrid') #this is the style: white grid


# In[5]:


nola.plot(kind='scatter', x='reviews', y='price', c=['darkgray'], s=150)
plt.xlabel('Number of Reviews')
plt.ylabel('Price')
plt.title('Number of Reviews by Price')
plt.show()


# In[6]:


nola.plot(kind='scatter', x='overall_satisfaction', y='price', c=['darkgray'], s=150)
plt.xlabel('Overall Satisfaction (1-5')
plt.ylabel('Price')
plt.title('Overall Satisfaction by Price')
plt.show()


# In[7]:


nola.plot(kind='scatter', x='overall_satisfaction', y='reviews', c=['darkgray'], s=150)
plt.xlabel('Number of Reviews')
plt.ylabel('Overall Satisfaction 1-5')
plt.title('Number of Reviews by Overall Satisfaction')
plt.show()


# In[8]:


nola.plot(kind='scatter', x='bedrooms', y='price', c=['darkgray'], s=150)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
plt.title('Price by Number of Bedrooms')
plt.show()


# In[9]:


nola.plot(kind='scatter', x='bedrooms', y='accommodates', c=['darkgray'], s=150)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Accommodation')
plt.title('Accommodation by Number of Bedrooms')
plt.show()


# In[10]:


nola_trim = nola[['reviews', 'price', 'overall_satisfaction', 'bedrooms', 'accommodates', 'minstay', 'room_type']]
#create dataframe for only focal vars


# In[11]:


print(nola_trim.head(10))
print(nola_trim.describe())
#look at new dataset


# In[ ]:





# In[12]:


nola_trim.isnull().any()
#satisfaction, bedrooms, minstay have missing


# In[13]:


miss = nola_trim.isnull()
miss.head(10)


# In[14]:


def reject_outliers(nola_trim, m =2.):
    d = np.abs(nola_trim - np.median(nola_trim))
    mdev = np.median(d)
    s = d/mdev if mdev else 0
    return nola_trim[s>m]
nola_trim.head(10)
#this doesn't work.


# In[15]:


nola_nomiss = nola_trim.dropna()
print(nola_nomiss.head(10))
#drop missing values
nola_nomiss.head(10)


# In[16]:


X = nola_nomiss[['reviews', 'price', 'overall_satisfaction', 'bedrooms', 'accommodates', 'minstay', 'room_type']]
sb.pairplot(X)
plt.show()


# In[17]:


corr = X.corr()
corr
sb.heatmap(corr,xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.show()


# In[18]:


pearsonr_coefficient, p_value = pearsonr(nola_nomiss.reviews, nola_nomiss.price)
print('PearsonR Correlation Coefficient - Reviews, Price %0.3f' % (pearsonr_coefficient))

pearsonr_coefficient, p_value = pearsonr(nola_nomiss.accommodates, nola_nomiss.price)
print('PearsonR Correlation Coefficient - Accommodates, Price %0.3f' % (pearsonr_coefficient))

table = pd.crosstab(nola_nomiss.room_type, nola_nomiss.overall_satisfaction)
chi2, p, dof, expected = chi2_contingency(table.values)
print('Chi-square Statistic - Room Type, Overall Satisfaction %0.3f p_value %0.3f' % (chi2, p))

table = pd.crosstab(nola_nomiss.room_type, nola_nomiss.price)
chi2, p, dof, expected = chi2_contingency(table.values)
print('Chi-square Statistic - Room Type, Price %0.3f p_value %0.3f' % (chi2, p))


# In[19]:


nola_bin = pd.get_dummies(nola_nomiss['room_type'])
nolaBinNomiss = pd.concat([nola_nomiss, nola_bin], axis=1)
nolaBinNomiss.head(10)


# In[20]:


y = nolaBinNomiss.price
#X = nola_nomiss.drop('price', axis=1)
X = nolaBinNomiss[['reviews', 'overall_satisfaction', 'bedrooms', 'accommodates', 'minstay', 'Private room', 'Shared room']]
#create DV and focal IV's for analysis


# In[21]:


#y.groupby(X.room_type).mean()
#look at means by room type to see if there's natural order to them


# In[22]:


#nola_nomiss['roomtypecat'] = pd.Categorical(nola_nomiss.room_type).labels
#this spits out a weird error - Entry point for launching an IPython kernal ?


# In[23]:


np.asarray(nolaBinNomiss)


# In[24]:


results = sm.OLS(y, X).fit()
results.summary()


# In[25]:


X
y


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=1/3,
                                                        random_state=123)


# In[27]:


scaler = preprocessing.StandardScaler().fit(X_train)


# In[28]:


X_train_scaled = scaler.transform(X_train)
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))


# In[29]:


pipeline=make_pipeline(preprocessing.StandardScaler(),
                                        RandomForestRegressor(n_estimators=100))
#print(pipeline.get_params())
#create pipeline for evaluating models


# In[30]:


hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                                    'randomforestregressor__max_depth': [None, 5, 3, 1]}


# In[31]:


clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)


# In[32]:


print(clf.best_params_)
#discover which parameters are best for these data
print(clf.refit)
#refit on the training set


# In[33]:


y_pred = clf.predict(X_test)
#predict a new set of data
print(r2_score(y_test, y_pred))
# r2: 0.5676
#much better model once accounting for bedrooms, accommedations
print(mean_squared_error(y_test, y_pred))
# mse: 13730.57


# In[34]:


#save the model for reuse later
joblib.dump(clf, 'rf_regressor.pkl')
clf2 =joblib.load('rf_regressor.pkl')
clf2.predict(X_test)

