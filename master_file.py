# coding: utf-8

# # Brazilian E-Commerce Public Dataset by Olist

# # Problem statement

# Most customers do not post a review rating or any comment after purchasing a product which is a challenge for any ecommerce platform to perform If a company predicts whether a customer liked/disliked a product so that they can recommend more similar and related products as well as they can decide whether or not a product should be sold at their end.
# This is crucial for ecommerce based company because they need to keep track of each product of each seller , so that none of products discourage their customers to come shop with them again. Moreover, if a specific product has very few rating and that too negetive, a company must not drop the product straight away, may be many customers who found the product to be useful haven't actually rated it.
#
# Some reasons could possibly be comparing your product review with those of your competitors beforehand,gaining lots of insight about the product and saving a lot of manual data pre-processin,maintain good customer relationship with company,lend gifts, offers and deals if the company feels the customer is going to break the relation.
#
# Objective of this case study is centered around predicting customer satisfaction with a product which can be deduced after predicting the product rating a user would rate after he makes a purchase.

# # Constraints

# High Accuracy
#
# Low latency (Rating should be known within the completion of the order)
#
# Prone to outliers
#

# ## Table of Contents
# 1.[Loading all packages and dataset](#first-bullet)<br>
# 2.[EDA for all the Tables one by one](#second-bullet)<br>
# 3.[EDA for Customers Tables](#third-bullet)<br>
# 4.[Exploring Orders Table](#fourth-bullet)<br>
# 3.[Descriptive statistics of Customer]()<br>
# 4.[Descriptive statistics of Seller]()<br>
# 5.[Customer purchase prediction]()<br>
# 6.[

# ## Loading packages and dataset
# <a class="anchor" id="first-bullet"></a>

# Imprting the Datasets

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "data/" directory.

import time, warnings
import datetime as dt
import re
import datetime
from datetime import datetime as dt
from sklearn.metrics.pairwise import cosine_similarity
from prettytable import PrettyTable
import logging
logging.basicConfig(filename='file.log',filemode='w',format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
# visualizations
import matplotlib.pyplot as plt

# get_ipython().magic(u'matplotlib inline')
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
import os
from mpl_toolkits.basemap import Basemap
import cufflinks as cf
#
py.offline.init_notebook_mode(connected=True)
cf.go_offline()

# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans

# Importing libraries for building the neural network
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb
from eli5.sklearn import PermutationImportance
import eli5

# Reading the Files from csv

##reading and checking dataset
df_cust = pd.read_csv('./Original_Data/olist_customers_dataset.csv')
df_loc = pd.read_csv('./Original_Data/olist_geolocation_dataset.csv')
df_items = pd.read_csv('./Original_Data/olist_order_items_dataset.csv')
df_pmt = pd.read_csv('./Original_Data/olist_order_payments_dataset.csv')
df_rvw = pd.read_csv('./Original_Data/olist_order_reviews_dataset.csv')
df_products = pd.read_csv('./Original_Data/olist_products_dataset.csv')
df_orders = pd.read_csv('./Original_Data/olist_orders_dataset.csv')
df_sellers = pd.read_csv('./Original_Data/olist_sellers_dataset.csv')
df_cat_name = pd.read_csv('./Original_Data/product_category_name_translation.csv')

# ## Customer Segmentation
#
# Customers who shop on Olist have different needs and they have their own different profile. We should adapt our actions depending on that.
#
# <a class="anchor" id="second-bullet"></a>

# # Exploring Tables one by one
# <a class="anchor" id="second-bullet"></a>

df_cust.head()
df_loc.head()
df_items.head()
df_pmt.head()
df_rvw.head()
df_products.head()
df_orders.head()
df_sellers.head()
df_cat_name.head().T
df_orders.describe().T

print('Printing Customers Table')
df_cust.head()
df_cust.isnull().sum()
df_cust.customer_state.value_counts().plot(kind='pie', figsize=(8, 10), autopct='%.1f%%', radius=2)
plt.legend()
plt.show()
# Top 10 cities with their value counts
df_cust.customer_city.value_counts().sort_values(ascending=False)[:10]
df_cust.info()
print(
    'Total Nos of Customers: {} \n Total generated IDs: {}'.format(len(df_cust['customer_id'].unique()), len(df_cust)))
print('Total Nos of Customers: {} \n Total generated unique IDs: {}'.format(len(df_cust['customer_unique_id'].unique()),
                                                                            len(df_cust)))
df_cust['customer_unique_id'].duplicated().sum()

# dropping ALL duplicte values

df_cust.sort_values('customer_unique_id', inplace=True)
df_cust.drop_duplicates(subset='customer_unique_id', keep=False, inplace=True)

# displaying data
df_cust['customer_unique_id'].duplicated().sum()

# Now check if any Duplicates is still in place:

print(
    'Total Nos of Customers: {} \n Total generated IDs: {}'.format(len(df_cust['customer_id'].unique()), len(df_cust)))
print('Total Nos of Customers: {} \n Total generated unique IDs: {}'.format(len(df_cust['customer_unique_id'].unique()),
                                                                            len(df_cust)))

# In[23]:


df_cust.shape, df_cust.info()

# In[24]:

#
# df_cust['customer_state'].value_counts().iplot()

# # Exploring Orders Table

# In[25]:


df_orders.head(3)

# In[26]:


df_orders['order_status'].value_counts()

# In[27]:


df_orders.isnull().sum().plot(kind='pie', radius=1)

# In[28]:


# After checking the Data, found that Filling the Null Values correspondence to Order Approved at
# imputation of Date columms to make the data for time series analysis
df_orders.fillna(method='ffill', axis=1, inplace=True)

# In[29]:


# Checking the Null Value After filling
df_orders.isnull().sum()

# In[30]:


df_orders.info()


# In[31]:


def convert_to_date(date_time, cols):
    for i in cols:
        date_time[i] = pd.to_datetime(date_time[i], format='%Y-%m-%d')


# In[32]:


convert_to_date(df_orders, ['order_approved_at', 'order_delivered_carrier_date'])

# In[33]:


convert_to_date(df_orders,
                ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date'])
df_orders.sort_values(by='order_purchase_timestamp', inplace=True)

# In[34]:


df_orders['shipping_time_delta'] = df_orders['order_estimated_delivery_date'] - df_orders[
    'order_delivered_customer_date']
df_orders['shipping_duration'] = df_orders['order_delivered_customer_date'] - df_orders['order_purchase_timestamp']
df_orders['estimated_duration'] = df_orders['order_estimated_delivery_date'] - df_orders['order_purchase_timestamp']

# In[35]:


df_orders[df_orders['shipping_duration'] > df_orders['estimated_duration']]
# Clearly can see that 6548 Records out of 99441 got delayed in Delivery as per Estimated Duration


# In[36]:


# Checking in %age for delay in shipment its only 7.885 Percent
a = df_orders[df_orders['shipping_duration'] > df_orders['estimated_duration']]
b = a.count() / len(df_orders) * 100
b

# In[37]:


## Visualize the data for Delivery
lab = df_orders[list(df_orders.columns)[2]].value_counts().keys().tolist()
# values
val = df_orders[list(df_orders.columns)[2]].value_counts().values.tolist()
#
trace = go.Pie(labels=lab,
               values=val,
               marker=dict(colors=['royalblue', 'lime'], line=dict(color="white", width=1.3)),
               rotation=90,
               hoverinfo="label+value+text",
               hole=.4)
layout = go.Layout(dict(title=list(df_orders.columns)[2] + " Delivery Status", plot_bgcolor="rgb(243,243,243)",
                        paper_bgcolor="rgb(243,243,243)", ))

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
#
# **Observations**
#
# 1. Almost 97.77% of orders are marked as delivered, some values are canceled, approved e.t.c. Note that such circumstances are very rare which shows in the data too, so this feature is of no use. We can drop it.
#
# 2. There are some orders which have missing order_delivered_carrier_date, order_delivered_customer_date and very few have order_approved_at missing.
#

# In[38]:


# All the Dates columns has been converted into Datetime format
df_orders.info()

# In[39]:


print('Total orders: {} \n Total records: {}'.format(len(df_orders['order_id'].unique()), len(df_orders)))

# In[40]:


df_orders.duplicated().sum()

# # Exploring Orders Item Tables

# In[41]:


df_items.info()
# there is no any NULL values


# In[42]:


print('Number of duplcated records: {} \n Number of duplcated order lines: {}'
      .format(df_items.duplicated().sum(),
              df_items[['order_id', 'product_id']].duplicated().sum()))

# In[43]:


df_items["shipping_limit_date"] = df_items["shipping_limit_date"].apply(lambda d: (dt.strptime(d, '%Y-%m-%d %H:%M:%S')))

# In[44]:


df_items["year"] = df_items["shipping_limit_date"].dt.year
df_items["month"] = df_items["shipping_limit_date"].dt.month
df_items["day"] = df_items["shipping_limit_date"].dt.day
df_items['time'] = df_items["shipping_limit_date"].dt.time
df_items['hour'] = df_items["shipping_limit_date"].dt.hour


# In[45]:


def time(X):
    if int(X) >= 0 and int(X) < 6:
        return 'Mid_Night'
    elif int(X) >= 6 and int(X) < 12:
        return 'Morning'
    elif int(X) >= 12 and int(X) < 18:
        return 'Afternoon'
    elif int(X) >= 18 and int(X) < 24:
        return 'Evening'


# In[46]:


df_items['timing'] = df_items['hour'].apply(time)

# In[47]:


import calendar

df_items['Month_Cat'] = df_items['month'].apply(lambda x: calendar.month_abbr[x])

# In[48]:


df_items


# In[49]:

try:
    def Seasonal(X):
        if int(X) >= 3 and int(X) <= 4:
            return 'Spring'
        elif int(X) >= 4 and int(X) <= 6:
            return 'Summer'
        elif int(X) >= 6 and int(X) <= 8:
            return 'Mansoon'
        elif int(X) >= 9 and int(X) <= 10:
            return 'Outumn'
        elif int(X) >= 11 and int(X) <= 3:
            return 'Winter'

except Exception as e:
    logging.error("Exception occurred", exc_info=True)
# In[50]:


df_items['Seasons'] = df_items['month'].apply(Seasonal)

# In[51]:


max(df_items.freight_value)


# In[52]:


def freight(X):
    if int(X) >= 0 and int(X) < 50:
        return 'L1'
    elif int(X) >= 6 and int(X) < 100:
        return 'L2'
    elif int(X) >= 12 and int(X) < 200:
        return 'L3'
    elif int(X) >= 18 and int(X) < 300:
        return 'L4'
    elif int(X) >= 300:
        return 'L5'


# In[53]:


df_items['freight_value_levels'] = df_items['freight_value'].apply(freight)

# In[54]:


df_items.head(2)

# In[55]:


# According to Days and Time visualizations and plotting
plt.figure(figsize=(10, 5))
plt.title('Year Wise Details')
sns.countplot(x='year', data=df_items)
plt.xlabel('Years')

plt.figure(figsize=(10, 5))
plt.title("Month wise")
sns.countplot(x="Month_Cat", data=df_items)
plt.xlabel("Months")

plt.figure(figsize=(10, 5))
plt.title('Time Wise Information')
sns.countplot(x='timing', data=df_items)

plt.figure(figsize=(10, 5))
plt.title('Freight Wise Information')
sns.scatterplot(x='freight_value', y='price', data=df_items)

plt.figure(figsize=(10, 5))
plt.title('Seasonal Sales Information')
sns.countplot(x='Seasons', data=df_items)

# In[56]:


df_items.describe().T

# In[57]:


print('Number of sellers :', df_items.seller_id.unique().shape[0])
print('Number of unique products are : ', df_items.product_id.unique().shape[0])

# In[58]:


print('Total Unique Orders ids : {} \n out of total orders i.e.  : {}'.format(len(df_items['order_id'].unique()),
                                                                              len(df_items)))

# In[59]:


print('Number of duplcated records: {} \nNumber of duplcated order lines: {}'
      .format(df_items.duplicated().sum(),
              df_items[['order_id', 'product_id']].duplicated().sum()))

# In[60]:


df_items.head(3)

# In[61]:


df_items_consolidated = df_items.groupby(by=['product_id', 'order_id']).agg({
    'order_item_id': 'count', 'seller_id': 'first',
    'shipping_limit_date': 'first',
    'price': 'first',
    'freight_value': 'first'
}).reset_index()

# In[62]:


# Renaming new quantity column
df_items_consolidated.rename(columns={'order_item_id': 'qty'}, inplace=True)
df_items_consolidated.head(3)

# In[63]:


# Merging or Combining the Orders and Items dataset. New Name " df_order_items_final"
df_order_items_final = df_orders.merge(df_items_consolidated, on='order_id')
df_order_items_final.head()

# In[64]:


df_order_items_final.describe()

# In[65]:


df_order_items_final.isnull().sum()

# In[66]:


# Again checking the Unique Orders and the Dulicated Orders from the New combined Tables
print('Total Unique orders in old table : {}'.format(len(df_items_consolidated['order_id'].unique())))
print('Total Unique orders after combining the Order and Items Table : {}'.format(
    len(df_order_items_final['order_id'].unique())))
print('Total Unique orders in actual Table : {}'.format(len(df_orders['order_id'].unique())))

print('Checking Duplicates Orders if any : {}'.format(df_order_items_final.duplicated().sum()))

# In[67]:


df_order_items_final.info()

# In[68]:


# final dataset after Preprocessing or Items and Orders Table : df_order_items_final

df_order_items_final.to_csv('prep_items_orders.csv')

# # Working with Products

# In[69]:


df_products.head()

# In[70]:


df_products.columns

# In[71]:


df_products.info()

# In[72]:


df_products['product_id'].equals(df_order_items_final['product_id'])

# In[73]:


df_order_items_final = df_order_items_final.merge(df_products, on='product_id')


# In[74]:


df_order_items_final['product_weight_g'].fillna('mean', inplace=True)
df_order_items_final['product_length_cm'].fillna('mean', inplace=True)
df_order_items_final['product_height_cm'].fillna('mean', inplace=True)
df_order_items_final['product_width_cm'].fillna('mean', inplace=True)


# In[75]:


def subst_mean(dat, cols):
    '''Function takes in name of a data frame and list of columns to substitute, nan cells will be filled with mean'''
    for col in cols:
        dat[col] = dat[col].fillna(dat[col].mean())


# In[76]:


subst_mean(df_order_items_final, ['product_length_cm',
                                  'product_weight_g',
                                  'product_height_cm',
                                  'product_width_cm'
                                  ])

# In[77]:


df_order_items_final.info()

# In[78]:


df_order_items_final.isnull().sum()

# In[79]:


df_order_items_final.describe().T

# In[80]:


## Calculate Shipping Volume - As we have Length, Height and Weidth, so easily can calculate the Volume to calculate


# In[81]:


df_order_items_final.info()

# In[82]:


df_order_items_final['order_shipping_volume'] = df_order_items_final['product_height_cm'] * df_order_items_final[
    'product_length_cm'] * df_order_items_final['product_width_cm']

# In[83]:


# Daily Orders Informations
df_order_items_final['date_by_ordinal'] = df_order_items_final['order_purchase_timestamp'].apply(
    lambda date: date.toordinal())


# In[84]:


# Converting again in a proper format i.e timedelta:
def int_to_datetime(dat, cols):
    '''Function takes in name of a data frame and list of columns to convert cells will be converted from int to datetime'''
    for col in cols:
        dat[col] = pd.to_timedelta(dat[col], 'ns').dt.days


# In[85]:


int_to_datetime(df_order_items_final, ['shipping_duration', 'shipping_time_delta', 'estimated_duration'])

# In[86]:


df_SKUs_consolidated = df_order_items_final.groupby(['product_id', 'order_purchase_timestamp']).agg(
    {'customer_id': 'count'}).reset_index()

# In[87]:


df_orders_daily = df_order_items_final.groupby('order_purchase_timestamp').agg({
    'date_by_ordinal': 'first',
    'qty': 'sum',
    'order_id': 'count',
    'order_shipping_volume': 'sum',
    'freight_value': 'sum',
    'price': 'sum',
    'shipping_duration': 'mean',
    'shipping_time_delta': 'mean',
    'estimated_duration': 'mean'
}).reset_index()

# In[88]:


df_orders_daily = df_orders_daily.merge(df_order_items_final, on='order_purchase_timestamp')
df_orders_daily = df_orders_daily.merge(df_SKUs_consolidated, on='order_purchase_timestamp')

# In[89]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.set_title('Total range')
ax2.set_title('Up to 500')

ax2.set_xlim([0, 500])

sns.distplot(df_order_items_final['price'],
             bins=100,
             ax=ax1,
             axlabel='Unit price',
             kde=False)
sns.distplot(df_order_items_final['price'],
             bins=1000,
             ax=ax2,
             axlabel='Unit price',
             kde=False);

# In[90]:


plt.figure(figsize=(15, 15))
sns.heatmap(df_order_items_final.corr(),
            cmap='seismic_r',
            vmax=0.3, center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            annot=True);


# In[91]:


## Function takes in name of a data frame and list of columns to convert cells will be converted from datetime to int64'''

def datetime_to_int(dat, cols):
    for col in cols:
        dat[col] = dat[col].values.astype(np.int64)


# In[92]:


# Trying linear regression
X = df_order_items_final[['shipping_duration', 'estimated_duration']]
y = df_order_items_final['freight_value']

# In[93]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# In[94]:


# applying Linear Models
lm_model = LinearRegression()

# In[95]:


lm_model.fit(X_train, y_train)

# In[96]:


# Trying to Generate predictions
y_preditions_train = lm_model.predict(X_train)
y_preditions_test = lm_model.predict(X_test)

# In[97]:


# Validating the precision in the model using the r2-score

print(r2_score(y_train, y_preditions_train))
print(r2_score(y_test, y_preditions_test))

# In[98]:


# Binning of product prices
df_order_items_final['price_round'] = round(df_order_items_final['price'] / 10) * 10

# In[99]:


df_ship_price_test = df_order_items_final.groupby('price_round').agg({
    'shipping_duration': ['max', 'min'],
    'estimated_duration': ['max', 'min']
}).reset_index().sort_values('price_round')

# In[100]:


df_ship_price_test['true_ship_range'] = df_ship_price_test.shipping_duration['max'] - \
                                        df_ship_price_test.shipping_duration['min']
df_ship_price_test['estimated_ship_range'] = df_ship_price_test.estimated_duration['max'] - \
                                             df_ship_price_test.estimated_duration['min']

# In[101]:


df_ship_price_test

# In[102]:


plt.figure(figsize=(15, 5))
sns.scatterplot(data=df_ship_price_test,
                y='true_ship_range',
                x='price_round')
sns.scatterplot(data=df_ship_price_test,
                y='estimated_ship_range',
                x='price_round')
plt.legend([
    'Range of shipping durations',
    'Range of estimated shipping durations'
]);

# In[103]:


plt.figure(figsize=(15, 5))
sns.scatterplot(data=df_order_items_final, x='price', y='shipping_duration')
plt.xlabel('Product Price')
plt.ylabel('Shipping duration')

# In[104]:


plt.figure(figsize=(15, 5))
sns.scatterplot(data=df_order_items_final, x='price', y='estimated_duration')
plt.xlabel('Product Price')
plt.ylabel('Estimated duration')

# In[105]:


plt.figure(figsize=(15, 5))
sns.scatterplot(data=df_order_items_final, x='price', y='shipping_time_delta')
plt.xlabel('Product Price')
plt.ylabel('Planned vs. true delivery duration')

# In[106]:


# Most Selling Products List

plt.xlabel('Product category')
plt.ylabel('Units sold')
df_order_items_final['product_category_name'].value_counts().plot(kind="bar", fontsize=15, figsize=(15, 5))
plt.xlim(0, 10);

# In[107]:


# Product Category  wise list
df_order_items_final['product_category_name'].value_counts().head(50)

# In[108]:


# Category of Products with Higher Turnovers
df_order_items_final.groupby('product_category_name').agg({'price': 'sum'}).sort_values(by='price',
                                                                                        ascending=False).plot(
    kind="bar", fontsize=15, figsize=(15, 5))
plt.xlabel('Product category')
plt.ylabel('Total turnover')
plt.xlim(0, 10)

# In[109]:


df_order_items_final.groupby('product_category_name').agg({'price': 'sum'}).sort_values(by='price',
                                                                                        ascending=False).head(20)

# In[110]:


df_products.groupby('product_category_name').agg({'product_id': 'count'}).sort_values(by='product_id',
                                                                                      ascending=False).plot(kind="bar",
                                                                                                            fontsize=15,
                                                                                                            figsize=(
                                                                                                            15, 5))
plt.xlabel('Product category')
plt.ylabel('Active SKUs')
plt.xlim(0, 10)

# In[111]:


df_products['product_category_name'].value_counts().head(20)

# # Working with Geographical Datasets

# # Olist E-commerce Data is used to trace customer location based on zip code using latitude and longitude
# #(1) What kinds of products are sold frequently? and which words are appeared mainly in review comments
# #(2) Showing Delivery Dates Time Histogram: Estimated Dates vs Actual Dates
# #(3) Geospatial scatterplot using latitudes and longitudes data, linking Zip code for customers
# #(4) Interesting Correlation between features & Freight payments by locations
# #(5) Clustering can seperate customers?
# #(6) To be improved

# In[112]:


df_loc.head(3)

# In[113]:


df_sellers.head(3)

# In[114]:


df_cust.head(3)

# In[115]:


df_sellers['seller_state'].nunique()

# In[116]:


geo_data = pd.concat([df_loc, df_sellers, df_cust], axis=1)

# In[117]:


geo_data.head()

# In[118]:


# project is finding the Top 10 products and sellers, analysis of orders
# by their geolocation and obtain information about Brazilian's online e-commerce profiles.
# The below displayed MAP shows the location of purchased product orders and according to the distribution
# of populations, it also visualised and explains the highest shopping rate

lat = geo_data['geolocation_lat']
lon = geo_data['geolocation_lng']
# states=df_loc['geolocation_state']
plt.figure(figsize=(10, 10))

m = Basemap(llcrnrlat=-55.401805, llcrnrlon=-92.269176, urcrnrlat=13.884615, urcrnrlon=-27.581676)
m.bluemarble()
m.drawmapboundary(fill_color='#46bcec')  # Make your map into any style you like
m.fillcontinents(color='#f2f2f2', lake_color='#46bcec')  # Make your map into any style you like
# m.drawcoastlines()
m.drawstates()
m.scatter(lon, lat, zorder=10, alpha=0.5, color='purple')

# In[119]:


geo_data.head(3)

# In[120]:


geo_data.info()

# In[121]:


df_orders['delivered_time'] = pd.to_datetime(df_orders['order_delivered_customer_date'],
                                             format='%Y-%m-%d').dt.date
df_orders['estimate_time'] = pd.to_datetime(df_orders['order_estimated_delivery_date'],
                                            format='%Y-%m-%d').dt.date

# In[122]:


df_orders['weekly'] = pd.to_datetime(df_orders['order_delivered_customer_date'],
                                     format='%Y-%m-%d').dt.week

# In[123]:


df_orders['yearly'] = pd.to_datetime(df_orders['order_delivered_customer_date']).dt.to_period('M')

df_orders['yearly'] = df_orders['yearly'].astype(str)

# In[124]:


df_orders['diff_days'] = df_orders['delivered_time'] - df_orders['estimate_time']
df_orders['diff_days'] = df_orders['diff_days'].dt.days

# In[125]:


plt.figure(figsize=(15, 5))
sns.lineplot(x='weekly', y='diff_days', data=df_orders, color="green", linewidth=5,
             markers=True, dashes=False, estimator='mean')

plt.xlabel("Weeks", size=14)
plt.ylabel("Difference Days", size=14)
plt.title("Average Difference Days per Week", size=15, weight='bold')

# In[126]:


# Extracting attributes for purchase date - Year and Month
df_orders['order_purchase_year'] = df_orders['order_purchase_timestamp'].apply(lambda x: x.year)

# In[127]:


# # Locations vy ZIP
lats = list(df_items.query('order_purchase_year == 2018')['geolocation_lat'].dropna().values)[:30000]
longs = list(df_items.query('order_purchase_year == 2018')['geolocation_lng'].dropna().values)[:30000]
locations = list(zip(lats, longs))

# # Creating a map using folium
map1 = folium.Map(location=[-15, -50], zoom_start=4.0)
#
FastMarkerCluster(data=locations).add_to(map1)

# map1


# In[128]:


plt.figure(figsize=(10, 10))
sns.countplot(x='geolocation_state', data=df_loc,
              order=df_loc['geolocation_state'].value_counts().sort_values().index,
              palette='icefire_r')

# The below graph shows the number of product orders purchased
# based on the states. According to that, "SP" has the highest rate
# and there is a huge gap between the rest of the states.


# In[129]:


df2 = pd.read_csv("../input/geolocation_olist_public_dataset.csv").sample(n=50000)
#
mapbox_access_token = 'pk.eyJ1IjoibGVlZG9oeXVuIiwiYSI6ImNqbjl1Y2hmcTB6dTQzcnBiNDZ2cXcwbGEifQ.hcPVtUhnyzXDXZbQQH0nMw'
data = [go.Scattermapbox(
    lon = df2['lng'],
    lat = df2['lat'],
    marker = dict(
        size = 3,

#     ))]

layout = dict(
        title = 'Geo Locations based on Zip code',
        mapbox = dict(
            accesstoken = mapbox_access_token,
            center= dict(lat=-20,lon=-60),
            bearing=5,
            pitch=5,
            zoom=2.3,
        )
    )
fig = dict( data=data, layout=layout )
iplot( fig, validate=False)


# # Identifying the Top Best Customers, Sellers, Hot Selling Prodcts

# # Finding the Best Customers based on Orders, Product and Items
#
# ## Table to be Use:
# ## Items
# ## Products
# ## Customer
# ## Orders

# In[130]:


# Merge data
total_orders = pd.merge(df_orders, df_items)
product_orders = pd.merge(total_orders, df_products, on="product_id")
product_orders.info()

# In[131]:


print(product_orders['product_id'].nunique(), product_orders['product_id'].str[-8:].nunique())

# In[132]:


# Top 15 Products Details

plt.figure(figsize=(30, 10))
sns.countplot(x='product_id', data=product_orders,
              order=product_orders['product_id'].value_counts()[:15].sort_values().index).set_title("Top 15 Products",
                                                                                                    fontsize=15,
                                                                                                    weight='bold')

# In[133]:


product_orders.columns

# In[134]:


# Grouping Top 20 Products Information

product_orders.groupby(["product_category_name"])["product_id"].count().sort_values(ascending=False).head(20)

group_category = product_orders.groupby(['product_id', 'product_category_name', ])['product_id'].count().sort_values(
    ascending=False).head(20)
group_category.plot.bar()
group_category

# In[135]:


# Top 20 Seller
# Merging of seller`` dataset with with the ``product orders`` data.

seller_products = pd.merge(product_orders, df_sellers, on="seller_id")
seller_products.info()

# In[136]:


seller_products['seller_state'].value_counts().plot.bar()

# In[137]:


# Visualization Chart shows the Top 20 Seller & 20 Best products.

plt.figure(figsize=(10, 10))
seller_products['seller_id'].value_counts()[:20].plot.pie(autopct='%1.1f%%',
                                                          shadow=True, startangle=90, cmap='tab20')
plt.title("Top 20 Seller", size=14, weight='bold')

plt.figure(figsize=(10, 10))
seller_products['product_category_name'].value_counts()[:20].plot.pie(autopct='%1.1f%%',
                                                                      shadow=True, startangle=90, cmap='tab20')
plt.title("Top 20 Products", size=14, weight='bold')

# In[138]:


# Assuming for the orders' product category of these sellers, we can use 'product category' values. Below table shows the Top 10 sellers category, and since they can sell multiple product types, garden tools are the most selling product of the best seller.
#
seller_category = seller_products.groupby(['seller_id', 'product_category_name'])['seller_id'].count().sort_values(ascending=False).head(20)
seller_category.iplot()

# In[139]:


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15))
group_category.plot.barh(ax=ax1, cmap='summer')
seller_category.plot.barh(ax=ax2, cmap='autumn')

ax1.set_title('Top20 Product', fontweight='bold')
ax2.set_title('Top20 Seller', fontweight='bold')

ax1.set_xlabel('Count', fontsize=15)
ax1.set_ylabel('Product Name', fontsize=15)
ax1.xaxis.set_tick_params(labelsize=12)
ax1.yaxis.set_tick_params(labelsize=15)

ax2.set_xlabel('Count', fontsize=15)
ax2.set_ylabel('Product Name', fontsize=15)
ax2.xaxis.set_tick_params(labelsize=12)
ax2.yaxis.set_tick_params(labelsize=15)

# In[140]:


# Weekly Analysis for Popular Items
product_orders['order_week'] = pd.to_datetime(product_orders['order_purchase_timestamp'],
                                              format='%Y-%m-%d').dt.week

# In[141]:


# Weekly popular items

items_weekly = product_orders.groupby(['order_week', 'product_category_name'])[
    'product_category_name'].count().sort_values(ascending=False)
most_products = items_weekly.reset_index(name='count')

# Find the max value of row
max_selling_products = most_products[most_products['count']
                                     == most_products.groupby(['order_week'])['count'].transform(max)]
max_selling_products.head(10)

# In[142]:


max_selling_products['product_category_name'].value_counts().plot.bar()

# # We have analysis the Order, Items, Products, Customers and Seller, So now need to validate the Payments and Revenue

# In[143]:


df_pmt.head(2)

# In[144]:


payments = pd.merge(seller_products, df_pmt, on="order_id")
payments.info()

# In[145]:


# Drop all irrelevant columns that to make more sense to work
payments = payments.drop(columns=['product_name_lenght', 'product_description_lenght',
                                  'product_photos_qty', 'product_weight_g', 'product_length_cm',
                                  'product_height_cm', 'product_width_cm'])

# In[146]:


# Most used Payment method for orders

payments['payment_type'].groupby(payments['payment_type']).count().plot(kind='pie', radius=3,
                                                                        labels=payments.payment_type.unique(),
                                                                        autopct='%.1f%%')
plt.legend()
plt.show()

# In[147]:


payments['delivered_time'] = pd.to_datetime(payments['order_delivered_customer_date'],
                                            format='%Y-%m-%d').dt.date
payments['estimate_time'] = pd.to_datetime(payments['order_estimated_delivery_date'],
                                           format='%Y-%m-%d').dt.date

# In[148]:


# Yearly & Weekly feature created based on order delivered customer date and analysing the difference as well for Delivery

payments['weekly'] = pd.to_datetime(payments['order_delivered_customer_date'],
                                    format='%Y-%m-%d').dt.week

payments['yearly'] = pd.to_datetime(payments['order_delivered_customer_date']).dt.to_period('M')
payments['yearly'] = payments['yearly'].astype(str)

payments['diff_days'] = payments['delivered_time'] - payments['estimate_time']
payments['diff_days'] = payments['diff_days'].dt.days

# In[149]:


price_details = payments.groupby(['order_id', 'price', 'product_category_name',
                                  'yearly', 'weekly'])[['freight_value', 'payment_value']].sum().reset_index()

# In[150]:


# Formula to Calculate : *** Total order value can be calculated by sum of `price` and `freight value`.***

price_details['total_order_value'] = price_details['price'] + price_details['freight_value']

# Also need to Calculate `Gross Profit` and `Profit Margin` by `payment value` and `total order value`

price_details['gross_profit'] = price_details['payment_value'] - price_details['total_order_value']
price_details['profit_margin'] = price_details['gross_profit'] / price_details['payment_value']
price_details['profit_margin'] = price_details['profit_margin'].astype('int64')

price_details.sort_values('gross_profit', ascending=False).head(10)

# In[151]:


# ## 3.3. Payments
plt.figure(figsize=(25, 15))

sns.lineplot(x='yearly', y='gross_profit',
             data=price_details[price_details['product_category_name'] \
                                == 'cama_mesa_banho'], label='bed_bath_table', color="green")
sns.lineplot(x='yearly', y='gross_profit',
             data=price_details[price_details['product_category_name'] \
                                == 'beleza_saude'], label='beauty_health', color="blue")
sns.lineplot(x='yearly', y='gross_profit',
             data=price_details[price_details['product_category_name'] \
                                == 'esporte_lazer'], label='sports_leisure', color="red")
sns.lineplot(x='yearly', y='gross_profit',
             data=price_details[price_details['product_category_name'] \
                                == 'moveis_decoracao'], label='home_decoration', color="orange")
sns.lineplot(x='yearly', y='gross_profit',
             data=price_details[price_details['product_category_name'] \
                                == 'informatica_acessorios'], label='Informatic_accessories', color="purple")
plt.title("Gross Profit of Top 5 Products (2016-2018)", fontweight='bold')

# In[152]:


price_details.columns

# #Order_reviews data

# In[153]:


df_rvw.head(3)

# In[154]:


# pie chart for review_score
df_rvw.review_score.value_counts().plot(kind='pie', radius=2, autopct='%.1f%%')
plt.legend()
plt.show()

# In[155]:


# how many null/missing entries are present
df_rvw.isnull().sum()

# **Observations**
#
# * Review_score by maximum customers is 5 star(57%) and 4star(19.2%)
#
# *  review_comment_title and review_comment_message have lots of entires as blank or null, which is a problem. This is however xpected because most customers don't prefer to write reviews.
#
# *  As the percentage of null/blank value is over 30% (here it is about 80%) , so we drop these two features.

# # ********************************************WIP in below codes**********************

# In[156]:


# calculate Revenue for each row and create a new dataframe with YearMonth - Revenue columns
revenue = price_details.groupby(['yearly'])['payment_value'].sum().reset_index()
revenue.head()

# In[157]:


# calculating for Yearly revenue growth rate
# using pct_change() function to see monthly percentage change
revenue['YearlyGrowth'] = revenue['payment_value'].pct_change()

revenue.head()

# # Average Revenue per Customer Purchase

# In[158]:


payments.columns

# In[159]:


# creating monthly active customers dataframe by counting unique Customer IDs
plt.figure(figsize=(10, 5))
df_monthly_active = payments.groupby('weekly')['customer_id'].nunique().reset_index().plot.bar()

# In[160]:


# create a new dataframe for average revenue by taking the mean of it
df_monthly_order_avg = payments.groupby('weekly')['payment_value'].mean().reset_index().plot.bar()

# In[161]:


# Order Reviews Exploratory


# In[162]:


# pie chart for review_score
df_rvw.review_score.value_counts().plot(kind='pie', radius=3, autopct='%.1f%%')
plt.legend()
plt.show()

# In[163]:


df_rvw.isnull().sum().plot.bar()

# Review_score by maximum customers is 5 star(57%) and 4star(19.2%)
#
# review_comment_title and review_comment_message have lots of entires as blank or null, which is a problem. This is however xpected because most customers don't prefer to write reviews.
#
# As the percentage of null/blank value is over 30% (here it is about 80%) , so we drop these two features.

# # *Creating the Models based on EDA*

# In[164]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, Normalizer, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import itertools
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# In[165]:


data = pd.read_csv('preprocessed_data.csv')
data.columns

# In[166]:


data.shape

# In[167]:


# spliting data to train and test data
X = data.drop('Score', axis=1)
Y = data.Score.values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.33, stratify=Y, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# # Normalising all the numerical features

# In[168]:


std_scaler = Normalizer()
min_max = MinMaxScaler()

# payment_sequential feature
payment_sequential_train = std_scaler.fit_transform(X_train.payment_sequential.values.reshape(-1, 1))
payment_sequential_test = std_scaler.transform(X_test.payment_sequential.values.reshape(-1, 1))

# payment_installments feature
payment_installments_train = std_scaler.fit_transform(X_train.payment_installments.values.reshape(-1, 1))
payment_installments_test = std_scaler.transform(X_test.payment_installments.values.reshape(-1, 1))

# Payment value feature
payment_value_train = std_scaler.fit_transform(X_train.payment_value.values.reshape(-1, 1))
payment_value_test = std_scaler.transform(X_test.payment_value.values.reshape(-1, 1))

# price
price_train = std_scaler.fit_transform(X_train.price.values.reshape(-1, 1))
price_test = std_scaler.transform(X_test.price.values.reshape(-1, 1))

# freight_value
freight_value_train = std_scaler.fit_transform(X_train.freight_value.values.reshape(-1, 1))
freight_value_test = std_scaler.transform(X_test.freight_value.values.reshape(-1, 1))

# product_name_length
product_name_length_train = std_scaler.fit_transform(X_train.product_name_length.values.reshape(-1, 1))
product_name_length_test = std_scaler.transform(X_test.product_name_length.values.reshape(-1, 1))

# product_description_length
product_description_length_train = std_scaler.fit_transform(X_train.product_description_length.values.reshape(-1, 1))
product_description_length_test = std_scaler.transform(X_test.product_description_length.values.reshape(-1, 1))

# product_photos_qty
product_photos_qty_train = std_scaler.fit_transform(X_train.product_photos_qty.values.reshape(-1, 1))
product_photos_qty_test = std_scaler.transform(X_test.product_photos_qty.values.reshape(-1, 1))

# delivery_days
delivery_days_train = std_scaler.fit_transform(X_train.delivery_days.values.reshape(-1, 1))
delivery_days_test = std_scaler.transform(X_test.delivery_days.values.reshape(-1, 1))

# estimated_days
estimated_days_train = std_scaler.fit_transform(X_train.estimated_days.values.reshape(-1, 1))
estimated_days_test = std_scaler.transform(X_test.estimated_days.values.reshape(-1, 1))

# ships_in
ships_in_train = std_scaler.fit_transform(X_train.ships_in.values.reshape(-1, 1))
ships_in_test = std_scaler.transform(X_test.ships_in.values.reshape(-1, 1))

# seller_popularity
seller_popularity_train = min_max.fit_transform(X_train.seller_popularity.values.reshape(-1, 1))
seller_popularity_test = min_max.transform(X_test.seller_popularity.values.reshape(-1, 1))

# # Normalising Categorical features

# In[169]:


# initialising oneHotEncoder

onehot = CountVectorizer()
cat = OneHotEncoder()
# payment_type
payment_type_train = onehot.fit_transform(X_train.payment_type.values)
payment_type_test = onehot.transform(X_test.payment_type.values)

# customer_state
customer_state_train = onehot.fit_transform(X_train.customer_state.values)
customer_state_test = onehot.transform(X_test.customer_state.values)

# seller_state
seller_state_train = onehot.fit_transform(X_train.seller_state.values)
seller_state_test = onehot.transform(X_test.seller_state.values)

# product_category_name
product_category_name_train = onehot.fit_transform(X_train.product_category_name.values)
product_category_name_test = onehot.transform(X_test.product_category_name.values)

# arrival_time
arrival_time_train = onehot.fit_transform(X_train.arrival_time.values)
arrival_time_test = onehot.transform(X_test.arrival_time.values)

# delivery_impression
delivery_impression_train = onehot.fit_transform(X_train.delivery_impression.values)
delivery_impression_test = onehot.transform(X_test.delivery_impression.values)

# estimated_del_impression
estimated_del_impression_train = onehot.fit_transform(X_train.estimated_del_impression.values)
estimated_del_impression_test = onehot.transform(X_test.estimated_del_impression.values)

# ship_impression
ship_impression_train = onehot.fit_transform(X_train.ship_impression.values)
ship_impression_test = onehot.transform(X_test.ship_impression.values)

# existing_cust
existing_cust_train = cat.fit_transform(X_train.existing_cust.values.reshape(-1, 1))
existing_cust_test = cat.transform(X_test.existing_cust.values.reshape(-1, 1))

# **Stacking the data**

# In[170]:


# stacking up all the encoded features
X_train_vec = hstack((payment_sequential_train, payment_installments_train, payment_value_train, price_train,
                      freight_value_train, product_name_length_train, product_description_length_train,
                      product_photos_qty_train, delivery_days_train, estimated_days_train, ships_in_train,
                      payment_type_train, customer_state_train, seller_state_train, product_category_name_train,
                      arrival_time_train, delivery_impression_train, estimated_del_impression_train,
                      ship_impression_train, seller_popularity_train))

X_test_vec = hstack((payment_sequential_test, payment_installments_test, payment_value_test, price_test,
                     freight_value_test, product_name_length_test, product_description_length_test,
                     product_photos_qty_test, delivery_days_test, estimated_days_test, ships_in_test,
                     payment_type_test, customer_state_test, seller_state_test, product_category_name_test,
                     arrival_time_test, delivery_impression_test, estimated_del_impression_test,
                     ship_impression_test, seller_popularity_test))

print(X_train_vec.shape, X_test_vec.shape)

# # Naive Bayes

# # Hyper parameter Tuning

# In[171]:


naive = MultinomialNB(class_prior=[0.5, 0.5])

param = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# for the bow based model
NB = GridSearchCV(naive, param, cv=3, refit=False, return_train_score=True, scoring='roc_auc')
NB.fit(X_train_vec, y_train)

# In[172]:


NB.best_params_

# # Fitting the Model

# In[173]:


clf = MultinomialNB(alpha=0.0001, class_prior=[0.5, 0.5])
clf.fit(X_train_vec, y_train)

# predicted value of y probabilities
y_pred_train = clf.predict_proba(X_train_vec)
y_pred_test = clf.predict_proba(X_test_vec)

# predicted values of Y labels
pred_label_train = clf.predict(X_train_vec)
pred_label_test = clf.predict(X_test_vec)

# Confusion Matrix
cf_matrix_train = confusion_matrix(y_train, pred_label_train)
cf_matrix_test = confusion_matrix(y_test, pred_label_test)

fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_pred_train[:, 1])
fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_pred_test[:, 1])

train_auc = round(auc(fpr_train, tpr_train), 3)
test_auc = round(auc(fpr_test, tpr_test), 3)

plt.plot(fpr_train, tpr_train, color='red', label='train-auc = ' + str(train_auc))
plt.plot(fpr_test, tpr_test, color='blue', label='test-auc = ' + str(test_auc))
plt.plot(np.array([0, 1]), np.array([0, 1]), color='black', label='random model auc = ' + str(0.5))
plt.xlabel('False Positive Rate(FPR)')
plt.ylabel('True Positive Rate(TPR)')
plt.title('ROC curve')
plt.legend()
plt.show()
print('Best AUC for the model is {} '.format(test_auc))

# In[174]:


# plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cf_matrix_test / np.sum(cf_matrix_test), annot=True, fmt='.2%', cmap='Greens')
plt.show()

# In[175]:


# f1 score
print('Train F1_score for this model is : ', round(f1_score(y_train, pred_label_train), 4))
print('Test F1_score for this model is : ', round(f1_score(y_test, pred_label_test), 4))

# In[176]:


print('Train Accuracy score for this model : ', round(accuracy_score(y_train, pred_label_train), 4))
print('Test Accuracy score for this model : ', round(accuracy_score(y_test, pred_label_test), 4))

# # Observations
#
# 1. Naive bayes performed pretty decent in terms of minimal overfitting in train and test performances.
# 2. Both train and test f1 score was 0.86 and accuracy 77%.
# 3. But the confusion matrix says it has misclassified many points as False Positives.
# 4. AUC score for test data was 0.694.

# # Logistic Regression

# # Hyper parameter Tuning

# In[177]:


# we have used max_iter 1000 as it was causing exception while fitting
Logi = LogisticRegression(max_iter=1000, solver='lbfgs')

param = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 30]}

# for the bow based model
LR = GridSearchCV(Logi, param, cv=3, refit=False, return_train_score=True, scoring='roc_auc')
LR.fit(X_train_vec, y_train)

# In[178]:


LR.best_params_

# **NOTE**
#
# * For performance measurement we will not use accuracy as a metric as the data set is highly imbalanced.
# * We will use AUC score and f1 score as performance metric.

# In[179]:


# model
clf = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs')
clf.fit(X_train_vec, y_train)

# In[180]:


# predicted value of y probabilities
y_pred_train = clf.predict_proba(X_train_vec)
y_pred_test = clf.predict_proba(X_test_vec)

# predicted values of Y labels
pred_label_train = clf.predict(X_train_vec)
pred_label_test = clf.predict(X_test_vec)

# Confusion Matrix
cf_matrix_train = confusion_matrix(y_train, pred_label_train)
cf_matrix_test = confusion_matrix(y_test, pred_label_test)

fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_pred_train[:, 1])
fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_pred_test[:, 1])

train_auc = round(auc(fpr_train, tpr_train), 3)
test_auc = round(auc(fpr_test, tpr_test), 3)

plt.plot(fpr_train, tpr_train, color='red', label='train-auc = ' + str(train_auc))
plt.plot(fpr_test, tpr_test, color='blue', label='test-auc = ' + str(test_auc))
plt.plot(np.array([0, 1]), np.array([0, 1]), color='black', label='random model auc = ' + str(0.5))
plt.xlabel('False Positive Rate(FPR)')
plt.ylabel('True Positive Rate(TPR)')
plt.title('ROC curve')
plt.legend()
plt.show()
print('Best AUC for the model is {} '.format(test_auc))

# In[181]:


# plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cf_matrix_test / np.sum(cf_matrix_test), annot=True, fmt='.2%', cmap='Greens')
plt.show()

# In[182]:


# f1 score
print('Train F1_score for this model is : ', round(f1_score(y_train, pred_label_train), 4))
print('Test F1_score for this model is : ', round(f1_score(y_test, pred_label_test), 4))

# In[183]:


print('Train Accuracy score for this model : ', round(accuracy_score(y_train, pred_label_train), 4))
print('Test Accuracy score for this model : ', round(accuracy_score(y_test, pred_label_test), 4))

# # Observations
#
# 1. Logistic regression performs considerably better than Naive bayes in terms of f1 score, however AUC score being almost the same.
# 2. Misclassification of False positives reduced which resulted in the increase of f1 score of 92%.
# 3. Accuracy was 86% for both train and test which shows the model doesn't overfit at all.

# # Decision Tree

# # HyperParmater tuning

# In[184]:


# model initialize
DT = DecisionTreeClassifier(class_weight='balanced')

# hyper parameters
param = {'max_depth': [1, 5, 10, 15, 20], 'min_samples_split': [5, 10, 100, 300, 500, 1000]}

# Grid search CV
DT = GridSearchCV(DT, param, cv=3, refit=False, return_train_score=True, scoring='roc_auc')
DT.fit(X_train_vec, y_train)

# In[185]:


# best params
DT.best_params_

# In[186]:


# model
clf = DecisionTreeClassifier(class_weight='balanced', max_depth=20, min_samples_split=300)
clf.fit(X_train_vec, y_train)

# predicted value of y probabilities
y_pred_train = clf.predict_proba(X_train_vec)
y_pred_test = clf.predict_proba(X_test_vec)

# predicted values of Y labels
pred_label_train = clf.predict(X_train_vec)
pred_label_test = clf.predict(X_test_vec)

# Confusion Matrix
cf_matrix_train = confusion_matrix(y_train, pred_label_train)
cf_matrix_test = confusion_matrix(y_test, pred_label_test)

# taking the probabilit scores instead of the predicted label
# predict_proba returns probabilty scores which is in the 2nd column thus taking the second column
fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_pred_train[:, 1])
fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_pred_test[:, 1])

train_auc = round(auc(fpr_train, tpr_train), 3)
test_auc = round(auc(fpr_test, tpr_test), 3)

plt.plot(fpr_train, tpr_train, color='red', label='train-auc = ' + str(train_auc))
plt.plot(fpr_test, tpr_test, color='blue', label='test-auc = ' + str(test_auc))
plt.plot(np.array([0, 1]), np.array([0, 1]), color='black', label='random model auc = ' + str(0.5))
plt.xlabel('False Positive Rate(FPR)')
plt.ylabel('True Positive Rate(TPR)')
plt.title('ROC curve')
plt.legend()
plt.show()
print('Best AUC for the model is {} '.format(test_auc))

# In[187]:


# plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cf_matrix_test / np.sum(cf_matrix_test), annot=True, fmt='.2%', cmap='Greens')
plt.show()

# In[188]:


# f1 score
print('Train F1_score for this model is : ', round(f1_score(y_train, pred_label_train), 4))
print('Test F1_score for this model is : ', round(f1_score(y_test, pred_label_test), 4))

# In[189]:


print('Train Accuracy score for this model : ', round(accuracy_score(y_train, pred_label_train), 4))
print('Test Accuracy score for this model : ', round(accuracy_score(y_test, pred_label_test), 4))

# # Observations
#
# 1. Decision Tree does nothing better interms of both f1 score , auc score and accuracy comes out to be 0.708 and 70%.
# 2. It misclassfied False Positives to a lot.
# 3. Model doesn't overfit but doesn't perform better either.

# # Random Forest

# # Hyperparameter Tuning

# In[190]:


# param grid
# we have limit max_depth to 10 so that the model doesn't overfit
param = {'min_samples_split': [5, 10, 30, 50, 100], 'max_depth': [5, 7, 10]}

# Random forest classifier
RFclf = RandomForestClassifier(class_weight='balanced')

# using grid search cv to tune parameters
RF = GridSearchCV(RFclf, param, cv=5, refit=False, n_jobs=-1, verbose=1, return_train_score=True, scoring='roc_auc')
RF.fit(X_train_vec, y_train)

# In[191]:


RF.best_params_

# In[192]:


# model
clf = RandomForestClassifier(class_weight='balanced', max_depth=10, min_samples_split=5)
clf.fit(X_train_vec, y_train)

# predicted value of y probabilities
y_pred_train = clf.predict_proba(X_train_vec)
y_pred_test = clf.predict_proba(X_test_vec)

# predicted values of Y labels
pred_label_train = clf.predict(X_train_vec)
pred_label_test = clf.predict(X_test_vec)

# Confusion Matrix
cf_matrix_train = confusion_matrix(y_train, pred_label_train)
cf_matrix_test = confusion_matrix(y_test, pred_label_test)

# taking the probabilit scores instead of the predicted label
# predict_proba returns probabilty scores which is in the 2nd column thus taking the second column
fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_pred_train[:, 1])
fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_pred_test[:, 1])

train_auc = round(auc(fpr_train, tpr_train), 3)
test_auc = round(auc(fpr_test, tpr_test), 3)

plt.plot(fpr_train, tpr_train, color='red', label='train-auc = ' + str(train_auc))
plt.plot(fpr_test, tpr_test, color='blue', label='test-auc = ' + str(test_auc))
plt.plot(np.array([0, 1]), np.array([0, 1]), color='black', label='random model auc = ' + str(0.5))
plt.xlabel('False Positive Rate(FPR)')
plt.ylabel('True Positive Rate(TPR)')
plt.title('ROC curve')
plt.legend()
plt.show()
print('Best AUC for the model is {} '.format(test_auc))

# In[193]:


# plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cf_matrix_test / np.sum(cf_matrix_test), annot=True, fmt='.2%', cmap='Greens')
plt.show()

# In[194]:


# f1 score
print('Train F1_score for this model is : ', round(f1_score(y_train, pred_label_train), 4))
print('Test F1_score for this model is : ', round(f1_score(y_test, pred_label_test), 4))

# In[195]:


print('Train Accuracy score for this model : ', round(accuracy_score(y_train, pred_label_train), 4))
print('Test Accuracy score for this model : ', round(accuracy_score(y_test, pred_label_test), 4))

# # Observations
#
# 1. Random forest performs better than logistic regression in terms of f1 score and accuracy.
# 2. It gives an f1 score of 90.13% and doesn't seem to overfit.
# 3. Misclassification rate is still not that great.
# 4. AUC is score is 0.718
# 5. Accuracy score is 83%.

# # GBDT

# # Hyper parameter tuning

# In[196]:


# param grid
# we have limit max_depth to 8 so that the model doesn't overfit
param = {'min_samples_split': [5, 10, 30, 50], 'max_depth': [3, 5, 7, 8]}

GBDTclf = GradientBoostingClassifier()

clf = GridSearchCV(RFclf, param, cv=5, refit=False, return_train_score=True, scoring='roc_auc')
clf.fit(X_train_vec, y_train)

# In[197]:


# best parameters
clf.best_params_

# In[198]:


import pickle

# In[199]:


# Model
clf = GradientBoostingClassifier(max_depth=8, min_samples_split=5)
clf.fit(X_train_vec, y_train)

# save the model to disk
Pkl_Filename = "final_model.pkl"
with open(Pkl_Filename, 'wb') as file:
    pickle.dump(clf, file)

# predicted value of y probabilities
y_pred_train = clf.predict_proba(X_train_vec)
y_pred_test = clf.predict_proba(X_test_vec)

# predicted values of Y labels
pred_label_train = clf.predict(X_train_vec)
pred_label_test = clf.predict(X_test_vec)

# Confusion Matrix
cf_matrix_train = confusion_matrix(y_train, pred_label_train)
cf_matrix_test = confusion_matrix(y_test, pred_label_test)

# taking the probabilit scores instead of the predicted label
# predict_proba returns probabilty scores which is in the 2nd column thus taking the second column
fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_pred_train[:, 1])
fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_pred_test[:, 1])

train_auc = round(auc(fpr_train, tpr_train), 3)
test_auc = round(auc(fpr_test, tpr_test), 3)

plt.plot(fpr_train, tpr_train, color='red', label='train-auc = ' + str(train_auc))
plt.plot(fpr_test, tpr_test, color='blue', label='test-auc = ' + str(test_auc))
plt.plot(np.array([0, 1]), np.array([0, 1]), color='black', label='random model auc = ' + str(0.5))
plt.xlabel('False Positive Rate(FPR)')
plt.ylabel('True Positive Rate(TPR)')
plt.title('ROC curve')
plt.legend()
plt.show()
print('Best AUC for the model is {} '.format(test_auc))

# In[200]:


# plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cf_matrix_test / np.sum(cf_matrix_test), annot=True, fmt='.2%', cmap='Greens')
plt.show()

# In[201]:


# f1 score
print('Train F1_score for this model is : ', round(f1_score(y_train, pred_label_train), 4))
print('Test F1_score for this model is : ', round(f1_score(y_test, pred_label_test), 4))

# In[202]:


print('Train Accuracy score for this model : ', round(accuracy_score(y_train, pred_label_train), 4))
print('Test Accuracy score for this model : ', round(accuracy_score(y_test, pred_label_test), 4))

# # Observations
#
# 1. Gradient Boosted classifier results the best f1 score of 0.9243 and auc score of 0.745.
# 2. Misclassification of False Positives and True negetives is also reduced to 11% also true positive rate is 83%.
# 3. Accuracy score is 86% for test and 87% for train data.
# 4. Model does overfit a slight comapred to rest of the models.

# # Observations
#
# 1. We created a standard deep Neural network model and trained it for 20 epochs this resulted f1 score very similar to our best ML model yet which is GBDT.
# 2. Kindly note that this neural network was very little hyper-parameter tuning done,and still results in a very decent performance.
# 3. However the auc score of GBDT is still better than the NN model.
# 4. Important thing to note that NN based models can be much better than conventional ML models for such problems.

# # Results

# In[203]:


from prettytable import PrettyTable

table = PrettyTable()
table.field_names = ["Model", "F1_score", " AUC_score ", " Accuracy "]

table.add_row(["Naive Bayes", '0.8575', '0.694', '0.7689'])
table.add_row(["Logistic Regression", '0.9217', '0.699', '0.8605'])
table.add_row(["Decision Tree", '0.8031', '0.713', '0.7021'])
table.add_row(["Random Forest", '0.9013', '0.718', '0.8315', ])
table.add_row(["GBDT**(BEST)", '0.9243', '0.745', '0.8651'])
# table.add_row(["Deep NN",'0.9233','0.710','0.8629'])

print(table)

# # Summary
#
# 1. GBDT performs better in comparision to rest of the model in terms of all the performance metric.
# 2. Logistic regression performs fairly similar to GBDT, but GBDT is more robust to outliers.
# 3. Rating prediction is not fairly dependent directly on most of the features, so the performance is not at its peak.
# 4. We have used f1 score as our primary performance metric.
# 5. We have taken care of overfitting in each model.
# 6. Each model performed better after we removed neutral review score from the data.

# # Applying Association Rules

# There are many data analysis tools available to the python analyst and it can be challenging to know which ones to use in a particular situation. A useful (but somewhat overlooked) technique is called association analysis which attempts to find common patterns of items in large data sets. One specific application is often called market basket analysis.

# # Why Association Analysis?
# In today’s world, there are many complex ways to analyze data (clustering, regression, Neural Networks, Random Forests, SVM, etc.). The challenge with many of these approaches is that they can be difficult to tune, challenging to interpret and require quite a bit of data prep and feature engineering to get good results. In other words, they can be very powerful but require a lot of knowledge to implement properly.
#
# Association analysis is relatively light on the math concepts and easy to explain to non-technical people. In addition, it is an unsupervised learning tool that looks for hidden patterns so there is limited need for data prep and feature engineering. It is a good start for certain cases of data exploration and can point the way for a deeper dive into the data using other approaches.
#
# As an added bonus, the python implementation in MLxtend should be very familiar to anyone that has exposure to scikit-learn and pandas. For all these reasons, I think it is a useful tool to be familiar with and can help you with your data analysis problems.
#
# One quick note - technically, market basket analysis is just one application of association analysis. In this post though, I will use association analysis and market basket analysis interchangeably.

# Support is the relative frequency that the rules show up. In many instances, you may want to look for high support in order to make sure it is a useful relationship. However, there may be instances where a low support is useful if you are trying to find “hidden” relationships.
#
# Confidence is a measure of the reliability of the rule. A confidence of .5 in the above example would mean that in 50% of the cases where Diaper and Gum were purchased, the purchase also included Beer and Chips. For product recommendation, a 50% confidence may be perfectly acceptable but in a medical situation, this level may not be high enough.
#
# Lift is the ratio of the observed support to that expected if the two rules were independent (see wikipedia). The basic rule of thumb is that a lift value close to 1 means the rules were completely independent. Lift values > 1 are generally more “interesting” and could be indicative of a useful rule pattern.
#
# One final note, related to the data. This analysis requires that all the data for a transaction be included in 1 row and the items should be 1-hot encoded.

# In[204]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[205]:
#
#
customer = pd.read_csv('./olist_customers_dataset.csv')
order = pd.read_csv('./olist_orders_dataset.csv')
item = pd.read_csv('./olist_order_items_dataset.csv')
product = pd.read_csv('./olist_products_dataset.csv')
product_category = pd.read_csv('./product_category_name_translation.csv')
payment = pd.read_csv('./olist_order_payments_dataset.csv')
review = pd.read_csv('./olist_order_reviews_dataset.csv')

# # In[206]:
#
#
# # merge the dataset
df = customer.merge(order, how='inner', on='customer_id')
df = df.merge(item, how='inner', on='order_id')
df = df.merge(product, how='inner', on='product_id')
df = df.merge(product_category, how='left', on='product_category_name')
df = df.merge(payment, how='inner', on='order_id')
df = df.merge(review, how='inner', on='order_id')
#
df.shape
df
#
# # In[207]:
#
#
df.to_csv('Combine_data_collaborative_Filtering.csv')
#
# # In[208]:
#
#
# # Load data
df = pd.read_csv("Combine_data_collaborative_Filtering.csv")
#
# # Only keep the orders that only contains 1 order
order_product = df.groupby('order_id').product_id.count().sort_values(ascending=False)
df = df[df.order_id.isin((order_product[order_product == 1]).index)]
#
# # Select columns for analysis
item_profile = df[['customer_unique_id', 'product_id', 'review_score']]
item_profile.head()
#
# # In[209]:
#
#
# # Calculate average rating of different product
Ratings_mean = item_profile.groupby('product_id')['review_score'].mean().reset_index().rename(
    columns={'review_score': 'mean_rating'})
item_profile = pd.merge(item_profile, Ratings_mean, how='inner', on=['product_id'])
item_profile.head()
#
# # In[210]:
#
#
print(len(item_profile.customer_unique_id.drop_duplicates().tolist()))
print(len(item_profile.product_id.drop_duplicates().tolist()))
#
# # In[211]:
#
#
# # Create a pivot table
new_df = item_profile.head(10000)
pivot = pd.pivot_table(new_df, index='product_id', columns='customer_unique_id', values='review_score')
pivot.head()
#
# # In[212]:
#
#
pivot.shape
#
# # In[213]:
#
#
Center the mean around 0 (centered cosine/pearson)
pivot_norm = pivot.apply(lambda x: x - np.nanmean(x), axis=1)
pivot_norm.head()
#
#
# # ## Item Based CF
#
# # In[214]:
#
#
# # Fill NaN with 0
pivot.fillna(0, inplace=True)
pivot.head()
#
# # ### Calculate Similar Items
#
# # In[223]:
#
#
# # Convert into dataframe
item_sim_df = pd.DataFrame(cosine_similarity(pivot, pivot),
#                            index=pivot.index,
#                            columns=pivot.index)
item_sim_df
#
#
# # In[224]:
#
#
def get_similar_product(product_id):
    if product_id not in pivot.index:
        return None, None
    else:
        sim_product = item_sim_df.sort_values(by=product_id, ascending=False).index[1:]
        sim_score = item_sim_df.sort_values(by=product_id, ascending=False).loc[:, product_id].tolist()[1:]
        return sim_product, sim_score

#
# # In[219]:
#

product, score = get_similar_product("a9516a079e37a9c9c36b9b78b10169e8")
for x, y in zip(product[:10], score[:10]):
    print("{} with similarity of {}".format(x, y))
#
#
# # In[220]:
#
#
# # Predict the rating of product x by user y
def predict_rating(customer_unique_id, product_id, max_neighbor=10):
    product, scores = get_similar_product(product_id)
    product_arr = np.array([x for x in product])
    sim_arr = np.array([x for x in scores])

    # Select only the product that has already rated by user x
    filtering = pivot[customer_unique_id].loc[product_arr] != 0

    # Calculate the predicted score
    s = np.dot(sim_arr[filtering][:max_neighbor],
               pivot[customer_unique_id].loc[product_arr[filtering][:max_neighbor]]) / np.sum(
        sim_arr[filtering][:max_neighbor])

    return s
#
#
# # In[221]:
#
#
predict_rating("00115fc7123b5310cf6d3a3aa932699e", "a9516a079e37a9c9c36b9b78b10169e8")
#
#
# # ## Get recommendation
#
# # In[ ]:
#
#
# # Recommend top n_product for customer x
def get_recommendation(customer_unique_id, n_product=1):
    predicted_rating = np.array([])

    for _product in pivot.index:
        predicted_rating = np.append(predicted_rating, predict_rating(customer_unique_id, _product))

    # Don't recommend sth that the customer has already rated
    temp = pd.DataFrame({'predicted': predicted_rating, 'name': pivot.index})
    filtering = (pivot[customer_unique_id] == 0.0)
    temp = temp.loc[filtering.values].sort_values(by='predicted', ascending=False)

    # Recommend n_product product
    return product.loc[product_index.loc[temp.name[:n_product]]]


# # In[ ]:
#
#
# # Recommendation for a particular customer
get_recommendation("0010a452c6d13139e50b57f19f52e04e")
#
# # ### Summary
# # Since most customers (70%+) only rate 1 product, it's hard to recommend similar products at leaset for the subset (i.e. first 10000 rows). Possible solutions:
# # - Use the whole dataset: same concern of the limited ratings
# # - Group the items based on the categories
#
# # In[ ]:
#
#
# # merge the dataset
df = customer.merge(order, how='inner', on='customer_id')
df = df.merge(item, how='inner', on='order_id')
df = df.merge(product, how='inner', on='product_id')
df = df.merge(product_category, how='left', on='product_category_name')
df = df.merge(payment, how='inner', on='order_id')
df = df.merge(review, how='inner', on='order_id')

df.shape
df
#
df.to_csv('Combine_data_collaborative_Filtering.csv')
#
# # Load data
df = pd.read_csv("Combine_data_collaborative_Filtering.csv")
#
# # Only keep the orders that only contains 1 order
order_product = df.groupby('order_id').product_id.count().sort_values(ascending=False)
df = df[df.order_id.isin((order_product[order_product == 1]).index)]
#
# # Select columns for analysis
item_profile = df[['customer_unique_id', 'product_id', 'review_score']]
item_profile.head()

# # Calculate average rating of different product
Ratings_mean = item_profile.groupby('product_id')['review_score'].mean().reset_index().rename(
    columns={'review_score': 'mean_rating'})
item_profile = pd.merge(item_profile, Ratings_mean, how='inner', on=['product_id'])
item_profile.head()
#
print(len(item_profile.customer_unique_id.drop_duplicates().tolist()))
print(len(item_profile.product_id.drop_duplicates().tolist()))
#
# # Create a pivot table
new_df = item_profile.head(10000)
pivot = pd.pivot_table(new_df, index='product_id', columns='customer_unique_id', values='review_score')
pivot.head()

pivot.shape
#
# # Center the mean around 0 (centered cosine/pearson)
pivot_norm = pivot.apply(lambda x: x - np.nanmean(x), axis=1)
pivot_norm.head()
#
# ## Item Based CF
#
# # Fill NaN with 0
pivot.fillna(0, inplace=True)
pivot.head()
#
# ### Calculate Similar Items
#
# # Convert into dataframe
item_sim_df = pd.DataFrame(cosine_similarity(pivot, pivot),
                           index=pivot.index,
                           columns=pivot.index)
item_sim_df
#
#
def get_similar_product(product_id):
    if product_id not in pivot.index:
        return None, None
    else:
        sim_product = item_sim_df.sort_values(by=product_id, ascending=False).index[1:]
        sim_score = item_sim_df.sort_values(by=product_id, ascending=False).loc[:, product_id].tolist()[1:]
        return sim_product, sim_score

#
product, score = get_similar_product("a9516a079e37a9c9c36b9b78b10169e8")
for x, y in zip(product[:10], score[:10]):
    print("{} with similarity of {}".format(x, y))

#
# # Predict the rating of product x by user y
def predict_rating(customer_unique_id, product_id, max_neighbor=10):
    product, scores = get_similar_product(product_id)
    product_arr = np.array([x for x in product])
    sim_arr = np.array([x for x in scores])

    # Select only the product that has already rated by user x
    filtering = pivot[customer_unique_id].loc[product_arr] != 0

    # Calculate the predicted score
    s = np.dot(sim_arr[filtering][:max_neighbor],
               pivot[customer_unique_id].loc[product_arr[filtering][:max_neighbor]]) / np.sum(
        sim_arr[filtering][:max_neighbor])

    return s
#
#
predict_rating("00115fc7123b5310cf6d3a3aa932699e", "a9516a079e37a9c9c36b9b78b10169e8")
#
#
# ## Get recommendation
#
# # Recommend top n_product for customer x
def get_recommendation(customer_unique_id, n_product=1):
    predicted_rating = np.array([])

    for _product in pivot.index:
        predicted_rating = np.append(predicted_rating, predict_rating(customer_unique_id, _product))

    # Don't recommend sth that the customer has already rated
    temp = pd.DataFrame({'predicted': predicted_rating, 'name': pivot.index})
    filtering = (pivot[customer_unique_id] == 0.0)
    temp = temp.loc[filtering.values].sort_values(by='predicted', ascending=False)

#     # Recommend n_product product
    return product.loc[product_index.loc[temp.name[:n_product]]]
#
#
# # Recommendation for a particular customer
get_recommendation("0010a452c6d13139e50b57f19f52e04e")
#
# ### Summary
# '''
# Since
# most
# customers(70 % +)
# only
# rate
# 1
# product, it
# 's hard to recommend similar products at leaset for the subset (i.e. first 10000 rows). Possible solutions:
# - Use
# the
# whole
# dataset: same
# concern
# of
# the
# limited
# ratings
# - Group
# the
# items
# based
# on
# the
# categories  #########################################################################################################################
#
# # # *In progress Work ---- Pipelining*
#
# # In[ ]:
# '''
#
# # imports
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pickle
#
#
# # # Function 1
#
# # In[ ]:
#
#
# # Takes test data as input and results Test labels as ouput
def function1(X_test):
    # loads best model
    # Load the Model back from file
    with open('final_model.pkl', 'rb') as file:
        clf = pickle.load(file)

    # predict labels
    pred_label = clf.predict(X_test)

    # predicted value of y probabilities
    pred_proba = clf.predict_proba(X_test)

    return [pred_label, pred_proba]
#
#
# # # Function 2
#
# # In[ ]:
#
#
# # takes true label and predicted label and shows all necessary performance plots and metrics
def function2(y_true, y_pred, pred_proba):
    # Confusion Matrix
    cf_matrix_test = confusion_matrix(y_true, y_pred)

    # taking the probabilit scores instead of the predicted label
    # predict_proba returns probabilty scores which is in the 2nd column thus taking the second column
    fpr_test, tpr_test, threshold_test = roc_curve(y_true, pred_proba[:, 1])
    test_auc = round(auc(fpr_test, tpr_test), 3)
#
#     # ROC_AUC plot and score
    print('ROC_AUC curve plot :\n\n')
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_test, tpr_test, color='blue', label='test-auc = ' + str(test_auc))
    plt.plot(np.array([0, 1]), np.array([0, 1]), color='black', label='random model auc = ' + str(0.5))
    plt.xlabel('False Positive Rate(FPR)')
    plt.ylabel('True Positive Rate(TPR)')
    plt.title('ROC curve')
    plt.legend()
    plt.show()
    print('\n\t*********Best AUC for the model is {}  **********'.format(test_auc))
#
#     # plot confusion matrix
    print('\n\nConfusion Matrix : \n\n')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cf_matrix_test / np.sum(cf_matrix_test), annot=True, fmt='.2%', cmap='Greens')
    plt.show()

    # f1 score
    print('\n\nTest F1_score for this model is : ----->', round(f1_score(y_true, y_pred), 4))

    # accuracy score
    print('\n\nTest Accuracy score for this model : ----->', round(accuracy_score(y_true, y_pred), 4))
#
# # In[ ]:
#
#
# # # Function Call
# # #Function 1 to get the predicted values
X_test = sp.sparse.load_npz('X_test_vec.npz')
pred_label,pred_proba = function1(X_test)
#
# # #Function 2 to get the performance metrics and plots
y_test = pd.read_csv('test_labels.csv').values.ravel()
function2(y_test,pred_label,pred_proba)
#

