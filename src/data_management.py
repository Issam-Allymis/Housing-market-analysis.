import streamlit as st
import pandas as pd
import numpy as np
import joblib

"""@st.cache()
def load_telco_data():
    df = pd.read_csv("outputs/datasets/collection/house-price-2021.csv") 
    return df"""

@st.cache_data
def load_house_prices_data(suppress_st_warning=True, allow_output_mutation=True):
    df = pd.read_csv("outputs/datasets/collection/house-price-2021.csv") 
    return df


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)

