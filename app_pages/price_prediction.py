import os
import streamlit as st
import sklearn
import pandas as pd
from src.data_management import load_house_prices_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import (
    predict_saleprice,
    predict_cluster)

current_dir = os.getcwd()


def price_prediction_body():

    # load predict saleprice files
    version = 'v1'
    saleprice_pipe_model = load_pkl_file(
        f"{current_dir}/outputs/ml_pipeline/predict_saleprice/{version}/pipeline_regression.pkl")

    saleprice_pipe_dc_fe = load_pkl_file(
        f'{current_dir}/outputs/ml_pipeline/predict_saleprice/{version}/pipeline_dcfe.pkl')
    
    saleprice_features = (pd.read_csv(f"{current_dir}/outputs/ml_pipeline/predict_saleprice/{version}/X_train.csv")
                      .columns
                      .to_list()
                      )


    
    
    st.write("---")

 # Predict prices of inherited houses
    if st.button("Predict inherited houses prices"):
        inherited_houses = pd.read_csv(os.path.join(current_dir, 'inputs', 'inherited_houses.csv'))
        # inherited_houses = pd.read_csv(f"{current_dir}\inputs\inherited_houses.csv")
        inherited_houses_sorted = inherited_houses.sort_values(by='YearBuilt')
        inherited_houses_prices = saleprice_pipe_model.predict(inherited_houses_sorted) 
        for index in range(len(inherited_houses_prices)):
            price = inherited_houses_prices[index]
            YearBuilt = inherited_houses_sorted.at[index, 'YearBuilt']
            st.write(f"The price of the house built in {YearBuilt} is around {price}.")
            

    # Generate Live Data
    # check_variables_for_UI(saleprice_features, cluster_features)
    X_live = DrawInputsWidgets()


    # predict on live data
    if st.button("Run Predictive Analysis"):
        saleprice_prediction = saleprice_pipe_model.predict(X_live)
        st.write(f"The price of this house is around {saleprice_prediction}")
        

        # predict_cluster(X_live, cluster_features,
        #                 cluster_pipe, cluster_profile)


# def check_variables_for_UI(saleprice_features, cluster_features):
#     import itertools

#     # The widgets inputs are the features used in all pipelines (saleprice, cluster)
#     # We combine them only with unique values
#     combined_features = set(
#         list(
#             itertools.chain(saleprice_features, cluster_features)
#         )
#     )
#     st.write(
#         f"* There are {len(combined_features)} features for the UI: \n\n {combined_features}")



def DrawInputsWidgets():

    st.write(sklearn.__version__)

    # load dataset
    df = load_house_prices_data()
    percentageMin, percentageMax = 0.4, 2.0

# we create input widgets only for 6 features
#    col1, col2, col3, col4 = st.columns(4)
#    col5, col6, col7, col8 = st.columns(4)
# Create 21 columns
    
    # st.write(X_live)
    X_live = pd.DataFrame([], index=[0])

    # Create 21 columns
    columns = st.columns(21)

# Dictionary mapping features to columns
    feature_column_mapping = {
        '1stFlrSF': columns[0],
        '2ndFlrSF': columns[1],
        'BedroomAbvGr': columns[2],
        'BsmtExposure': columns[3],
        'BsmtFinSF1': columns[4],
        'BsmtFinType1': columns[5],
        'BsmtUnfSF': columns[6],
        'GarageArea': columns[7],
        'GarageFinish': columns[8],
        'GarageYrBlt': columns[9],
        'GrLivArea': columns[10],
        'KitchenQual': columns[11],
        'LotArea': columns[12],
        'LotFrontage': columns[13],
        'MasVnrArea': columns[14],
        'OpenPorchSF': columns[15],
        'OverallCond': columns[16],
        'OverallQual': columns[17],
        'TotalBsmtSF': columns[18],
        'YearBuilt': columns[19],
        'YearRemodAdd': columns[20],
    }

   # Create a vertical layout for each feature
    features = ['BsmtFinSF1', 'YearBuilt', 'BsmtUnfSF', 
        'GarageArea', 'GrLivArea', 'BsmtExposure', 
        'LotFrontage', 'MasVnrArea', 'OpenPorchSF', 
        'KitchenQual', 'OverallQual', 'OverallCond', 
        'GarageYrBlt', 'BedroomAbvGr', '1stFlrSF', '2ndFlrSF']

    for feature in features:
        # st.write(f"feature_type_for_{feature} is {df[feature].dtype}")
        options = sorted(df[feature].astype(str).unique())
        if df[feature].dtype in ['object', 'category']:
            st_widget = st.selectbox(
                label=feature,
                # options=sorted(df[feature].unique())
                options=options
            )
        elif df[feature].dtype in ['int64', 'float64']:
            if len(df[feature].unique().tolist()) < 20 :  # Arbitrary threshold for using selectbox
                st_widget = st.selectbox(
                    label=feature,
                    options=sorted(df[feature].unique())
                )
            else:
                st_widget = st.number_input(
                    label=feature,
                    min_value=float(df[feature].min() * percentageMin),
                    max_value=float(df[feature].max() * percentageMax),
                    value=float(df[feature].median())
                )
        X_live[feature] = st_widget

    return X_live

