import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output
import plotly.express as px
import dash_table_experiments as dt
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
import dash_table
import numpy as np
import base64
# from eda_cust_file import reading_files
# from visualize import visualize_eda
# from nlp_review import nlp_review_func
# from model_building import model_build
# from data_loading_understanding import data_loading
# from db_conn import conn_db_postgres
# from recom_price import recom_price
# from recom_product_corr import recom_product_corr
# # from customer_segmentation import cust_seg_rmf
# from reports import show_report
from jupyter_dash import JupyterDash

app=dash.Dash(__name__)
df_orders = pd.read_csv('./Original_Data/olist_orders_dataset.csv')
df_rvw = pd.read_csv('./Original_Data/olist_order_reviews_dataset.csv')
df_model_analysis= pd.read_csv('./Original_Data/model_analysis.csv')
df_products = pd.read_csv('./Original_Data/olist_products_dataset.csv')


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )


# Layout section: Bootstrap (https://hackerthemes.com/bootstrap-cheatsheet/)
# ************************************************************************
app.layout=dbc.Container(style={'backgroundColor':'#4DE4FC'},
    children=[
     #1st row

     dbc.Row([
         dbc.Col(html.H1("MARKET BASKET ANALYSIS DASHBOARD"),style={'width':50,'textAlign':'center','color':'black','backgroundColor':'#0716F2'})
     ]),
     #2nd row
     dbc.Row([
         #1st col
              dbc.Col([
                       dcc.Graph(id='pie1',style={'width':'300px','height':'300px'},
		               figure=go.Figure(data=[go.Pie(labels=df_rvw['review_score'].value_counts().keys().tolist(),values=df_rvw['review_score'].value_counts().values.tolist(),rotation=90, hoverinfo='label+value+text', hole=.5)],
	                    layout=go.Layout(title="<b>Review score</b>",paper_bgcolor="#B3B1B1"))
                       )
                ]),
         #2nd col
              dbc.Col([
                       dcc.Graph(id='pie2',style={'width':'300px','height':'300px'},
		               figure=go.Figure(data=[go.Pie(labels=df_orders['order_status'].value_counts().keys().tolist(),values=df_rvw['review_score'].value_counts().values.tolist(),rotation=90, hoverinfo='label+value+text', hole=.5)],
	                    layout=go.Layout(title="<b>Order Status</b>",paper_bgcolor="#F39A9A"))
                       )
                 ]),

         #3rd col
               dbc.Col([
            html.Header('Model Performance',style={'font-size':30,'textAlign':'center', 'font-weight': 'bold'}),
            dash_table.DataTable(
             id='table',
            style_data={
            'color':'black',
             'font-weight': 'bold'
               },
           columns=[{"name": i, "id": i} for i in df_model_analysis.columns],
            data=df_model_analysis.to_dict('records'),
            style_table={
                 'color':'#131DA4',
            'font-weight': 'bold'},
            style_cell={
                'fontFamily': 'Open Sans',
                'textAlign': 'center',
                'overflow': 'hidden',

            })

      ])],style={'width':'100%'}),
    #3rd row
     dbc.Row([
         #1st col
          dbc.Col(
            dbc.Card([dbc.CardImg(src="./assets/nlp_img2.png",top=True),dbc.CardBody(html.H4('Customer review',className='card-text'))],style={"width":'20rem'})

            ),
         #2nd col
           dbc.Col(

             dbc.Card([dbc.CardImg(src="./assets/customer_segmentation_image.png",top=True),dbc.CardBody(html.H6('Customer segmentation',className='card-text'))],style={"width":'15rem','align':'cebter'})
         ),
        #3rd col
          dbc.Col([
           dcc.Graph(
               id='pie3', style={'width': '400px', 'height': '300px'},
               figure=go.Figure(data=[go.Pie(labels=df_products['product_category_name'].value_counts()[:10].keys().tolist(),
                                             values=df_products['product_category_name'].value_counts()[:10].values.tolist(), rotation=90,
                                             hoverinfo='label+value+text', hole=.5)],
                                layout=go.Layout(title="<b>Top 10 Products</b>", paper_bgcolor="#B3B1B1"))
           )])
       ],style={'width':'100%'}),


        #5th row
    dbc.Row([

            dbc.Col([

            ])

        ])

     ])























if __name__=='__main__':
    app.run_server(debug=True)


