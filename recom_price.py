# coding: utf-8

# # Nearest Neighbor price based Collaborative Filtering

# In[1]:


import pandas as pd
import numpy as np

# In[5]:

def recom_price():
    combined = pd.read_csv("Combined.csv")
    print(combined.head())

    df1 = pd.read_csv("Combined.csv",
                      usecols=['product_id', 'customer_unique_id', 'product_category_name_english', 'review_score',
                               'price'])
    print(df1.head())

    df1.groupby('product_category_name_english')['price'].mean().sort_values(ascending=False).head()

    price = pd.DataFrame(df1.groupby('product_category_name_english')['price'].mean())
    print(price.head())

    print('create a Pivot matrix')

    df2 = df1.pivot_table(index='product_category_name_english', columns='customer_unique_id', values='price').fillna(0)
    print(df2.head())

    from scipy.sparse import csr_matrix
    price_df_matrix = csr_matrix(df2.values)
    from sklearn.neighbors import NearestNeighbors
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(price_df_matrix)

    print('Checking the Shape', price_df_matrix.shape)

    print('choose a random product')

    query_index = np.random.choice(df2.shape[0])
    print(query_index)
    distances, indices = model_knn.kneighbors(df2.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)

    print('Based on that product what we have choosen 5 nearest distance product get recommended based on prices')

    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(df2.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, df2.index[indices.flatten()[i]], distances.flatten()[i]))

    print('Calculate the accuracy')

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    import matplotlib.pyplot as plt

    df3 = pd.read_csv("Combined.csv", usecols=['product_category_name_english', 'price'])
    print(df3)

    df4 = pd.get_dummies(df3)
    print(df4)

    X = df4.drop('price', axis=1)
    y = df4['price']

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_log_error

    lreg = LinearRegression()
    lreg.fit(X_train, y_train)

    pred_train = lreg.predict(X_train)
    train_score = np.sqrt(mean_squared_log_error(y_train, pred_train))

    pred_test = lreg.predict(X_test)
    test_score = np.sqrt(mean_squared_log_error(y_test, pred_test))

    print('Training score:', train_score)
    print('Test score:', test_score)
