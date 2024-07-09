import streamlit as st
from app_pages.multipage import MultiPage
from src.data_management import log10_transform, power_transform

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_saleprice_study import sale_price_study_body
from app_pages.price_prediction import price_prediction_body
# from app_pages.page_project_hypothesis import page_project_hypothesis_body
# from app_pages.page_predict_saleprice import page_predict_saleprice_body
# from app_pages.page_cluster import page_cluster_body
# from app_pages.page_

app = MultiPage(app_name= 'Housing Market') # Craete an instance of the app

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Sale Price Study", sale_price_study_body)
app.add_page("Price Prediction", price_prediction_body)
# app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
# app.add_page("ML: Housing Sale Price", page_predict_saleprice_body)
# app.add_page("ML: Cluster Analysis", page_cluster_body)
# app.add_page()


app.run() # Runs the app
