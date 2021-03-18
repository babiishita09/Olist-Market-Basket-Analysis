# This is a sample Python script.
# from eda_cust_file import reading_files
from visualize import visualize_eda
from nlp_review import nlp_review_func
from model_building import model_build
from data_loading_understanding import data_loading
from db_conn import conn_db_postgres
from recom_price import recom_price
from recom_product_corr import recom_product_corr
# from customer_segmentation import cust_seg_rmf
from reports import show_report
# import logging
# logging.basicConfig(filename='msg.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',level=logging.INFO)
# logging.basicConfig(filename='msg2.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',level=logging.error)


def module_start():
    try:
        options = input("Choose the Options from below to run the Module : \n '1' - * Data Loading & Selections \n '2' - * EDA \n '3' - * Recommendation Engine_Price \n '4' - * Recommendation Engine_Product \n '5' - * Reviews \n '6' - * Customer Segmentation\n '7' - * Model Building \n '8' - * Quit/Exit\n\nChoise is :")
    # selection = input("Enter the Selection : \n '1' - EDA \n '2' Visualize the Data" )
        if '1' in options:
            data_loading()
        elif '2' in options:
            reading_files()
        elif '3' in options:
            recom_price()
        elif '4' in options:
            recom_product_corr()
        elif '5' in options:
            nlp_review_func()
        elif '6' in options:
            cust_seg_rmf()
        elif '7' in options:
            model_build()
        elif '8' in options:
            show_report()
        else:
            print('Thank You for visiting the Page.. Hope you enjoyed the Project!!!')
    except Exception as e:
        print(e)

module_start()


