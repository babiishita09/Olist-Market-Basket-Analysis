import pandas as pd
import numpy as np
from db_conn import conn_db_postgres

def data_loading():
    # try:
    datatype = input("Choose to Load the Datasets -- 1 - csv\n 2 - json (*not Implemented)\n 3 - pdf(* not Implemented)\n 4 - txt(* not Implemented)\n 5 - Load from Database (*not Yet Provided)\n\n")
    try:
        if '1' in datatype:
        ##reading and checking dataset
            df_cust = pd.read_csv('./Original_Data//olist_customers_dataset.csv')
            df_loc = pd.read_csv('./Original_Data/olist_geolocation_dataset.csv')
            df_items = pd.read_csv('./Original_Data/olist_order_items_dataset.csv')
            df_pmt = pd.read_csv('./Original_Data/olist_order_payments_dataset.csv')
            df_rvw = pd.read_csv('./Original_Data/olist_order_reviews_dataset.csv')
            df_products = pd.read_csv('./Original_Data/olist_products_dataset.csv')
            df_orders = pd.read_csv('./Original_Data/olist_orders_dataset.csv')
            df_sellers = pd.read_csv('./Original_Data/olist_sellers_dataset.csv')
            df_cat_name = pd.read_csv('./Original_Data/product_category_name_translation.csv')
            df_model_analysis = pd.read_csv('./Original_Data/model_analysis.csv')
            print('Successfully Loaded Data from CSV files')
        else:
            print('There is some problem in Fetching the records..Kindly choose the correct Choice')

    except:
         print('Database root not found ... Kindly check and restart....Quitting the program ....')