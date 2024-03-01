import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_saleprice_study import page_saleprice_study_body
from app_pages.page_prospect import page_prospect_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_predict_saleprice import page_predict_saleprice_body
from app_pages.page_cluster import page_cluster_body
# from app_pages.page_

app = MultiPage(app_name= 'Housing Market') # Craete an instance of the app

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Sale Price Study", page_saleprice_study_body)
app.add_page("Prospect Marketer", page_prospect_body)
app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
app.add_page("ML: Prospect Sale Price", page_predict_saleprice_body)
app.add_page("ML: Cluster Analysis", page_cluster_body)
# app.add_page()

app.run() # Runs the app
