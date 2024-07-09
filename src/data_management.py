import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from feature_engine.selection import SmartCorrelatedSelection
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

# Create required tranformations for numerical variables
def log10_transform(x):
    return np.log10(x)

def power_transform(x, power=1.5):
    return np.power(x, power)


@st.cache_data
def load_house_prices_data(suppress_st_warning=True, allow_output_mutation=True):
    # st.write(os.getcwd()) 
    # Construct the file path
    file_path = os.path.join('inputs', 'housing-prices-data', 'house_prices_records.csv')
    df = pd.read_csv(file_path) 
    
    return df


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)
