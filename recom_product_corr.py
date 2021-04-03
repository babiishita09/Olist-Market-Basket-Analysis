
# coding: utf-8

# # Creating a Product based Recommender System with the help of Correlation.

import pandas as pd
import numpy as np

def recom_product_corr():

    combined=pd.read_csv("Combined.csv")
    print(combined.head())
    print(combined.shape)

    df1=pd.read_csv("Combined.csv",usecols=['product_id','customer_unique_id','product_category_name_english','review_score'])
    print(df1)

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('white')
    # get_ipython().magic(u'matplotlib inline')
    df1.groupby('product_category_name_english')['review_score'].mean().sort_values(ascending=False).head()

    df1.groupby('product_category_name_english')['review_score'].count().sort_values(ascending=False).head()
    reviews = pd.DataFrame(df1.groupby('product_category_name_english')['review_score'].mean())
    print(reviews.head())

    reviews['num of reviews'] = pd.DataFrame(df1.groupby('product_category_name_english')['review_score'].count())
    reviews.head()

    # Let's explore the data a bit and get a look at some of the best review product.

    plt.figure(figsize=(10,4))
    reviews['num of reviews'].hist(bins=70)

    plt.figure(figsize=(10,4))
    reviews['review_score'].hist(bins=70)


    # Now that we have a general idea of what the data looks like,

    # # Recommending Similar Product

    # Create a matrix that has the customer unique id on one access and the product_category_name english on another axis. Each cell will then consist of the review the customer gave to that product there will be a lot of NaN values, because most people have not review most of the product

    df2 = df1.pivot_table(index='customer_unique_id',columns='product_category_name_english',values='review_score')
    df2.head()


    # Most reviewed product:

    reviews.sort_values('num of reviews',ascending=False).head(10)

    bedbath_user_reviews = df2['bed_bath_table']
    telephony_user_reviews = df2['telephony']
    bedbath_user_reviews.head()


    # ## We can then use corrwith() method to get correlations between two pandas series:

    similar_to_bedbath = df2.corrwith(bedbath_user_reviews)
    similar_to_telephony = df2.corrwith(telephony_user_reviews)

    # clean this by removing NaN values and using a DataFrame instead of a series
    corr_bedbath = pd.DataFrame(similar_to_bedbath,columns=['Correlation'])
    corr_bedbath.dropna(inplace=True)
    print(corr_bedbath.head())

    # we can sort the dataframe by correlation, we should get the most similar products, however we get some results that don't really make sense. This is because there are a lot of products only reviewed once by customer who also reviewed bed bath table (it was the most popular product).

    corr_bedbath.sort_values('Correlation',ascending=False).head(10)

    corr_bedbath = corr_bedbath.join(reviews['num of reviews'])
    corr_bedbath.head()

    # # Recommendation based on product correlated with bed bath table product

    # filtering out products that have less than 100 reviews (this value was chosen based off the histogram from earlier).


    corr_bedbath[corr_bedbath['num of reviews']>100].sort_values('Correlation',ascending=False).head()


    # # Recommendation based on product correlated with Telephony product

    # In[24]:


    corr_telephony = pd.DataFrame(similar_to_telephony,columns=['Correlation'])
    corr_telephony.dropna(inplace=True)
    corr_telephony = corr_telephony.join(reviews['num of reviews'])
    corr_telephony[corr_telephony['num of reviews']>100].sort_values('Correlation',ascending=False).head()


    # # Methodology:
    # Users are separated into repeat customers and first time customers and the recommendation system works as follows.
    #
    # Repeat Customers
    # Collaborative filtering recommendation
    # Hot Products
    # Popular in your area
    # New Customers
    # Hot products
    # Popular in your area

