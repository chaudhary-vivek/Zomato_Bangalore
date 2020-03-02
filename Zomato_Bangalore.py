# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:38:36 2020

@author: Chaudharyji
"""

# Zomato Bangalore data

##########################
# Part 1: Preprocessing###
##########################

# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Importing the dataset
df = pd.read_csv('zomato.csv')

# Deleting unnnecessary columns
df = df.drop(['url', 'address','phone', 'dish_liked', 'reviews_list', 'menu_item', 'cuisines'], axis = 1)

# Renaming certain colummns
df = df.rename(columns = {'approx_cost(for two people)' : 'cost'})
df = df.rename(columns = {'listed_in(type)' : 'listing'})
df = df.rename(columns = {'listed_in(city)' : 'area'})
df = df.rename(columns = {'rest_type' : 'type'})
df = df.rename(columns = {'book_table' : 'booking'})
df = df.rename(columns = {'online_order' : 'order'})

# Order - Feature engineering

# Checking for null values
df[df['order'].isnull()]
# No null values
# Seeing the value counts
df['order'].value_counts()

# Online - Feature engineering

# Checking for null values
df[df['booking'].isnull()]
# No null values
df['booking'].value_counts()

# Rate - Feature engineering

# Dropping the /5
df['rate'] = df['rate'].str.split('/').str[0]
# Seeing the number of values
#rate_values = df['rate'].value_counts()
# Replacing the - and NEW values with NaN
df['rate'] = df['rate'].replace('-', np.NaN)
df['rate'] = df['rate'].replace('NEW', np.NaN)
# Converting the datatype to float
df['rate'] = df['rate'].astype(float)
# Seeing the distribution
sns.distplot(df['rate'])
# Dropping null values
df = df.dropna(subset = ['rate'])

# Vote - Feature engineering

# Checking for null values
df[df['votes'].isnull()]
# No null values
# Seeing the value counts
df['votes'].value_counts()
# There are 2000 instances of 0 votes
df['votes'] = df['votes'].replace(0, np.NaN)
#Replacing those values with the mean
df = df.dropna(subset = ['votes'])


# Location - Feature engineering

# Checking for null values
df[df['location'].isnull()]
# there are only 21 null values
df['location'].value_counts()
# there are 93 values
# We will compare it with area
# Area is a better feature, so we will delete this feature
df = df.drop(['location'], axis = 1)

# Area - value engineering

# Checking for null values
df[df['area'].isnull()]
# No null values
# Checking the number of values
df['area'].value_counts()
# Area has a more even distribution, hence we can delete location

# Cost - Value engineering

# Checking for null values
df[df['cost'].isnull()]
# There are 342 null values
# Removing commas
df['cost'] = df['cost'].str.replace(',', '')
# Changing the datatype to float
df['cost'] = df['cost'].astype(float)
# Dropping the null values
df = df.dropna(subset = ['cost'])

# Listing - feature engineering

# Checking for null values
df[df['listing'].isnull()]
# No null values
# Checking the number of values
df['listing'].value_counts()

# Type - Feature engineering

# Checking for null values
df[df['type'].isnull()]
# There are 204 null values
# We will drop them
df = df.dropna(subset = ['type'])
# Checking for the longest value
df.loc[df['type'].map(len).argmax(), 'type']
# So the longest entry in this feature has only two comma separated values
# Separating the datatype by using str.split
df['type1'] = df['type'].str.split(', ').str[0]
df['type2'] = df['type'].str.split(', ').str[1]
# Deleting the original type column
df = df.drop(['type'], axis =1)
# There are several nan values in the type 2 column, replacing them with 'none'
df['type2'] = df['type2'].fillna('none')

############################
# Part 2 : Vizualization####
############################

# Vizualising for order (online orders)

# Pie
x = df['order'].value_counts()
plt.pie(x, labels = x.index, autopct='%1.1f%%')
plt.legend(x.index, loc = 'best')
plt.title('Restaurants accepting online orders in Bengaluru')
# Horizontal bar
x = df['order'].value_counts()
plt.barh(x.index, x)
plt.title('Restaurants accepting online orders in Bengaluru')

# Vizualising for booking (table booking)

# Pie
x = df['booking'].value_counts()
plt.pie(x, labels = x.index, autopct = '%1.1f%%')
plt.legend(x.index, loc = 'best')
# Horizontal bar
x = df['booking'].value_counts()
plt.barh(x.index, x)
plt.legend(x.index, loc = 'best')

# Vizualising the area

# Horizontal bar
rest_area = df['area'].value_counts()[:20]
sns.barplot(rest_area, rest_area.index)


# Vizualising the type of restautrants

listing_types = df['listing'].value_counts()
sns.barplot( listing_types, listing_types.index)
plt.pie(listing_types,labels = listing_types.index, autopct='%1.1f%%')
plt.legend(listing_types.index, loc = 'best')




############################
# Part 3 : Making the model#
############################

# Some columns are not necessary for the modelling, so we may remove it
df = df.drop(['name','type1', 'type2'], axis = 1)

# Orders - encoding
# Only two values yes and no
# But the datatype is object, so we will get dummy variables
df['order'] = pd.get_dummies(df['order'], drop_first = True)

# Booking - encoding
# Only two values, yes and no
# So, we can get dummy variables for the feature
df['booking'] = pd.get_dummies(df['booking'], drop_first = True)

# Area - encoding
# We will carry out label encoding in this feature
# Initiating the encoder object
encoder = LabelEncoder()
# Fitting the feature in the label encoder
df['area'] = encoder.fit_transform(df['area'])

# Listing - Encoding
# carrying out label encoding
encoder1 = LabelEncoder()
# Fitting the column onto the label encoder
df['listing'] = encoder1.fit_transform(df['listing'])

# Listing - Encoding
# carrying out label encoding
encoder1 = LabelEncoder()
# Fitting the column onto the label encoder
df['listing'] = encoder1.fit_transform(df['listing'])


# Making a heatmp of correlation

corr = df.corr(method = 'kendall')
sns.heatmap(corr, annot = True)
# Looks dicey :/

y = df['rate']
x = df.drop(['rate'], axis = 1)
# Doing train test plit on the data
from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 353)


# Feature selection
#
## Selecting important features using lasso
#from sklearn.linear_model import Lasso
#from sklearn.feature_selection import SelectFromModel
#model = SelectFromModel(Lasso(alpha = 0.05, random_state = 0))
#model.fit(x_train, y_train)
#model.get_support()
## Columns order, booking, listing, area are not important
#x_train = x_train.drop(['order', 'booking', 'listing', 'area'], axis = 1)
#x_test = x_test.drop(['order', 'booking', 'listing', 'area'], axis = 1)

# Hyper parameter tuning

# Importing randomizedsearchcv
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(100, 1200, 12)]
# Max levels in every tree
max_depth = [int(x) for x in np.linspace(5, 30, 6)]
# Number of features in each split
max_features = ['auto', 'sqrt']
# Minimum samples in a node
min_samples_split = [2, 5, 10, 15, 100]
#  Minimum samples in each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Creating a dictionary using above above parameterss to feed into the regressor

random_grid = {'n_estimators' : n_estimators,
               'max_features' : max_features,
               'max_depth' : max_depth,
               'min_samples_split' : min_samples_split,
               'min_samples_leaf' : min_samples_leaf
               }

#
## Random_forest_model
#from sklearn.ensemble import RandomForestRegressor
#rf = RandomForestRegressor(n_estimators = 100, random_state = 329, min_samples_leaf = 0.0001)
#rf.fit(x_train, y_train)

# Random forest model using grid search cv
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring = 'neg_mean_squared_error', n_iter = 20, verbose = 2, random_state =42, n_jobs =1 )
rf_random.fit(x_train, y_train)

###########################
# Part 4: Model Evaluation#
###########################

## Making prediction on the test data
#y_pred = rf.predict(x_test)

# Making prediction using the tuned model
y_pred = rf_random.predict(x_test)

# Calculating the r2 score
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
# R2 score is 0.652
# R2 score after dropping the features as per lasso is 0.631, hence we keep the features
# After hyper parameter tuning, the R2 score is 0.671 :/

# Seeing the residual graph
sns.distplot(y_test - y_pred)
# Residuals are normally distributed

# Seeing the scatter plot
plt.scatter(y_test, y_pred)















