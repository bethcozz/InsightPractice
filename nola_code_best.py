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

address = '~/Desktop/code/nolabblistings.csv'
nola = pd.read_csv(address)
#read in dataset

print(nola.head(10))
print(nola.describe())
#look at the data

room_type = nola['room_type']
neighborhood = nola['neighborhood']
reviews = nola['reviews']
price = nola['price']
overall_satisfaction = nola['overall_satisfaction']
accommodates = nola['accommodates']
bedrooms = nola['bedrooms']
minstay = nola['minstay']
#identify variables of interest and create as shorthand objects

rcParams['figure.figsize'] = 5, 4 #this is the size of the plot
sb.set_style('whitegrid') #this is the style: white grid


nola.plot(kind='scatter', x='reviews', y='price', c=['darkgray'], s=150)
plt.xlabel('Number of Reviews')
plt.ylabel('Price') 
plt.title('Number of Reviews by Price')
plt.show()


nola.plot(kind='scatter', x='overall_satisfaction', y='price', c=['darkgray'], s=150)
plt.xlabel('Overall Satisfaction (1-5')
plt.ylabel('Price') 
plt.title('Overall Satisfaction by Price')
plt.show()

nola.plot(kind='scatter', x='overall_satisfaction', y='reviews', c=['darkgray'], s=150)
plt.xlabel('Number of Reviews')
plt.ylabel('Overall Satisfaction 1-5') 
plt.title('Number of Reviews by Overall Satisfaction')
plt.show()

nola.plot(kind='scatter', x='bedrooms', y='price', c=['darkgray'], s=150)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price') 
plt.title('Price by Number of Bedrooms')
plt.show()

nola.plot(kind='scatter', x='bedrooms', y='accommodates', c=['darkgray'], s=150)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Accommodation') 
plt.title('Accommodation by Number of Bedrooms')
plt.show()

nola_trim = nola[['reviews', 'price', 'overall_satisfaction', 'bedrooms', 'accommodates', 'minstay']]
#create dataframe for only focal vars

print(nola_trim.reviews)
print(nola_trim.price)
print(nola_trim.overall_satisfaction)
print(nola_trim.bedrooms)
print(nola_trim.accommodates)
print(nola_trim.minstay)
#transform variable classes from string into float

print(nola_trim.head(10))
print(nola_trim.describe())
#look at new dataset

def reject_outliers(nola_trim, m =2.):
    d = np.abs(nola_trim - np.median(nola_trim))
    mdev = np.median(d)
    s = d/mdev if mdev else 0
    return nola_trim[s>m]
#eliminate outliers outside 2 standard deviations

nola_nomiss = nola_trim.dropna()
print(nola_nomiss.head(10))
#drop missing values

X = nola_nomiss[['reviews', 'price', 'overall_satisfaction', 'bedrooms', 'accommodates', 'minstay']]
sb.pairplot(X)
plt.show()

corr = X.corr()
corr
sb.heatmap(corr,xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.show()

pearsonr_coefficient, p_value = pearsonr(reviews, price)
print 'PearsonR Correlation Coefficient %0.3f' % (pearsonr_coefficient)

pearsonr_coefficient, p_value = pearsonr(accommodates, price)
print 'PearsonR Correlation Coefficient %0.3f' % (pearsonr_coefficient)

table = pd.crosstab(room_type, overall_satisfaction)
chi2, p, dof, expected = chi2_contingency(table.values)
print 'Chi-square Statistic %0.3f p_value %0.3f' % (chi2, p)

table = pd.crosstab(room_type, price)
chi2, p, dof, expected = chi2_contingency(table.values)
print 'Chi-square Statistic %0.3f p_value %0.3f' % (chi2, p)


y = nola_nomiss.price
X = nola_nomiss.drop('price', axis=1)
#create DV and focal IV's for analysis

print y
print X

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                                test_size=0.2,
                                                                                random_state=123)
#create test and train groups for both IVs and DV
#works if I do not stratify by y
scaler = preprocessing.StandardScaler().fit(X_train)
#ValueError: could not convert string to float: Private room
#had to eliminate room_type
X_train_scaled = scaler.transform(X_train)
print X_train_scaled.mean(axis=0)
print X_train_scaled.std(axis=0)
#standardize variables

pipeline=make_pipeline(preprocessing.StandardScaler(),
                                        RandomForestRegressor(n_estimators=100))
print pipeline.get_params()
#create pipeline for evaluating models

hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                                    'randomforestregressor__max_depth': [None, 5, 3, 1]}
#specify hyperparameters

clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)
#fit and tune model

print clf.best_params_
#discover which parameters are best for these data
print clf.refit
#refit on the training set

y_pred = clf.predict(X_test)
#predict a new set of data
print r2_score(y_test, y_pred)
# r2: 0.5676
#much better model once accounting for bedrooms, accommedations
print mean_squared_error(y_test, y_pred)
# mse: 13730.57

#save the model for reuse later
joblib.dump(clf, 'rf_regressor.pkl')
clf2 =joblib.load('rf_regressor.pkl')
clf2.predict(X_test)

