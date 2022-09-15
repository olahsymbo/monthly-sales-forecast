#!/usr/bin/env python3
# -*- coding: utf-8 -*-  
import os
import sys
import inspect

app_path = inspect.getfile(inspect.currentframe())
file_dir = os.path.realpath(os.path.dirname(app_path))

from datetime import datetime, timedelta,date
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import load_data 
import warnings
warnings.filterwarnings("ignore")  
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 

#%% 
 
data = []

root_path = file_dir + '/data' 
 
datafolder = load_data.load_data(root_path)

data = pd.concat(datafolder, axis=0, ignore_index=True)
data.head()

data.describe()

print(data.columns)

dn= data.groupby('Item Id')['Items Ordered'].sum().reset_index()

#create a new dataframe to model the difference
df_diff = dn.copy()
#%%
#add previous sales to the next row
df_diff['prev_sales'] = df_diff['Items Ordered'].shift(1)
df_diff = df_diff.dropna()

df_diff['diff'] = (df_diff['Items Ordered'] - df_diff['prev_sales'])
df_supervised = df_diff.drop(['prev_sales'],axis=1)
#%%
#adding lags
n_lags = 13
for inc in range(1,n_lags):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc) 
    
#drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)
 
#%% split the main data into training and testing samples.
df_supervised1 = df_supervised.copy() 

df_model_train_set, df_model_test_set = train_test_split(df_supervised1, test_size=0.3, random_state=10) 

# remove Items Ordered and Item Id from the train and test sets
df_model_train = df_model_train_set.drop(['Items Ordered','Item Id'],axis=1)
df_model_test = df_model_test_set.drop(['Items Ordered','Item Id'],axis=1)
 
#%%  
#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(df_model_train.values) 
train_set_scaled = scaler.transform(df_model_train.values) 
test_set_scaled = scaler.transform(df_model_test.values)

#%% organize the train and test data with train labels and test labels
X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1] 
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1] 

#%% Train Gradient Boosting Model
clf = GradientBoostingRegressor(random_state=10)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

pred_test_set = [] 
yred_reshape = y_pred.reshape(y_pred.shape[0], 1)
pred_test_set = np.concatenate([yred_reshape,X_test],axis=1) 

#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)
#%%
#create dataframe that shows the predicted sales
result_list = []
sales_dates = list(df_model_test_set['Item Id'])
act_sales = list(df_model_test_set['Items Ordered'])
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['Item Id'] = sales_dates[index]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)
 
#%%
# we need to select the rows of predicted samples from the main data samples and merge the predicted result (df_result)
df_sales_pred = data.loc[data['Item Id'].isin(df_result['Item Id'])]  
df_sales_pred_names = pd.merge(df_sales_pred,df_result,on='Item Id',how='left')

# sort the sales in descending order
df_sales_pred_sorted = df_sales_pred_names.sort_values('pred_value', ascending = False)

# remove duplicates
df_new = df_sales_pred_sorted.drop_duplicates() 

plt.figure()
colors = (0,0,0)
plt.scatter(df_new['pred_value'], df_new['Items Ordered'],c=colors, alpha=0.3)
plt.title('Scatter of sales')
plt.ylabel('Items Ordered')
plt.xlabel('pred_value')
plt.show()
 