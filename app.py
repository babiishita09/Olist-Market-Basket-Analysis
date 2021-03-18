import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#set page layout
st.set_page_config(page_title="Olist E-commerce",page_icon="*",initial_sidebar_state='expanded')
st.markdown("<h1 style='text-align:center;color:orange;'>Masker_Basket_Analysis</h1>",unsafe_allow_html=True)

from PIL import Image
image = Image.open(r"C:\Users\ishita\Downloads\Market_Basket-ver0.1\Market_Basket\market_basket_img.png")
st.image(image, use_column_width=True)
import time


def main():
    activities=["EDA","Visualization","Model","About Us"]
    option=st.sidebar.selectbox("selection Option",activities)


    #Dealing withnthe EDA part
    if option=='EDA':
        st.markdown("# Exploratory Data Analysis")

        data = st.file_uploader("Upload dataset:", type=['csv','xlsx','txt','json'])


        if data is not None:
            if st.checkbox("read data"):
                df=pd.read_csv(data)
                st.dataframe(df.head(50))

                st.markdown("### Select Option")
                selectbox = st.selectbox("",('Display Shape', 'Display Columns','Select multiple columns','Display Summary','Display Null Values','Display the data types','Display correlation of various columns'))
                if selectbox=='Display Shape':
                    st.write(df.shape)
                if selectbox=='Display Columns':
                    st.write(df.columns)
                if selectbox=="Select multiple columns":
                    selected_columns=st.multiselect('Select preferred columns:', df.columns)
                    df1 = df[selected_columns]
                    st.dataframe(df1)
                if selectbox=="Display Summary":
                    st.write(df.describe().T)
                if selectbox=="Display Null Values":
                    st.write(df.isnull().sum())
                if selectbox=="Display the data types":
                    st.write(df.dtypes)
                if selectbox=="Display correlation of various columns":
                    st.write(df.corr())

    # Dealing with visualization part
    elif option=="Visualization":
        st.markdown("# Visualization")

        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])

        if data is not None:

            if st.checkbox("read data"):
                df = pd.read_csv(data)
                st.dataframe(df.head(50))




                if st.checkbox("Display Heatmap"):
                    st.write(sns.heatmap(df.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
                    st.pyplot()

                if st.checkbox("Display Pairplot"):
                    st.write(sns.pairplot(df,diag_kind='kde'))
                    st.pyplot()

                if st.checkbox("Display Pie Chart"):
                    all_columns=df.columns.to_list()
                    pie_columns=st.selectbox("Select column to display",all_columns)
                    piechart =df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
                    st.write(piechart)
                    st.pyplot()
                if st.checkbox("Display barchart"):
                    all_columns = df.columns.to_list()
                    bar_columns = st.selectbox("Select column to display", all_columns)
                    barchart = df[bar_columns].value_counts().plot(kind='bar')
                    # plt.xlim(0,20)
                    st.write(barchart)
                    st.pyplot()

                if st.checkbox("Display linechart"):
                    all_columns = df.columns.to_list()
                    line_columns = st.selectbox("Select column to display", all_columns)
                    df1=df[line_columns].value_counts().plot(kind='line')
                    st.write(df1)
                    st.pyplot()
                    # Geolocation data
                if st.checkbox("map(Only for Geolocation data)"):
                    df2 = pd.DataFrame({'lat': df['geolocation_lat'][:50000], 'lon': df['geolocation_lng'][:50000]})
                    st.map(df2)

    #Dealing with Model part
    elif option=="Model":
        st.markdown("# Model Building")

        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox("Select Multiple Columns"):
                new_data = st.multiselect("Select your preferred columns", df.columns)
                df1= df[new_data]
                st.dataframe(df1)

        #Dividing data into X and Y variables
                X=df1.iloc[:0:-1]
                Y=df1.iloc[:,-1]

            seed=st.sidebar.slider('Seed',1,200)
            classifier_name=st.sidebar.selectbox("Select your preferred classifier:",('KNN','SVM','LR','naive_bayes','Decision Tree','Gradient Boosting Classifier','Random Forest'))

            def add_parameter(name_of_classifier):
               params=dict()
               if name_of_classifier=="SVM":
                   C = st.sidebar.slider("C",0.01,15.0)
                   params["C"]=C
               elif name_of_classifier=='KNN':
                   K=st.sidebar.slider("K",1,15)
                   params["K"]=K
               elif name_of_classifier=="Decision Tree":
                   max_depth=st.sidebar.slider("max_depth",1,20)
                   min_samples_split=st.sidebar.slider("min_samples_split",5,1000)
                   params["max_depth"]=max_depth
                   params['min_samples_split']=min_samples_split
               elif name_of_classifier=="Random Forest":
                   max_depth = st.sidebar.slider("max_depth", 1, 20)
                   min_samples_split = st.sidebar.slider("min_samples_split", 5, 1000)
                   params["max_depth"] = max_depth
                   params['min_samples_split'] = min_samples_split
               elif name_of_classifier=="Gradient Boosting Classifier":
                   max_depth = st.sidebar.slider("max_depth", 1, 10)
                   min_samples_split = st.sidebar.slider("min_samples_split", 5, 100)
                   params["max_depth"] = max_depth
                   params['min_samples_split'] = min_samples_split
               return params
        #calling the function
            params=add_parameter(classifier_name)

        #defing a function for our classifier
            def get_classifier(name_of_classifier,params):
               clf = None
               if name_of_classifier=='SVM':
                  clf=SVC(C=params['C'])
               elif name_of_classifier=="KNN":
                  clf=KNeighborsClassifier(n_neighbors=params['K'])
               elif name_of_classifier=="LR":
                  clf=LogisticRegression()
               elif name_of_classifier=="naive_bayes":
                  clf=MultinomialNB()
               elif name_of_classifier=="Decision Tree":
                  clf=DecisionTreeClassifier(max_depth=params["max_depth"],min_samples_split=params['min_samples_split'])
               elif name_of_classifier=="Ramdom Forest":
                  clf=RandomForestClassifier(max_depth=params["max_depth"],min_samples_split=params['min_samples_split'])
               elif name_of_classifier=="Gradient Boosting Classifier":
                  clf=GradientBoostingClassifier(max_depth=params["max_depth"],min_samples_split=params['min_samples_split'])
               else:
                  st.warning("Select your choice of algorithm")
               return clf

            clf=get_classifier(classifier_name,params)
            
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=seed)
            st.write("X_train_shape:",X_train.shape)
            st.write("Y_train_shape:",Y_train.shape)
            clf.fit(X_train,Y_train)
            Y_pred=clf.predict(X_test)
            st.write("Predictions:",Y_pred)
            accuracy=accuracy_score(Y_test,Y_pred)
            st.write("Name of classifier:",classifier_name)
            st.write("Accuracy:",accuracy)


if __name__ == "__main__" :
    main()