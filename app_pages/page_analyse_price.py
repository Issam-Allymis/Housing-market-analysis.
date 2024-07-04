import os
import streamlit as st
import pandas as pd
from src.data_management import load_house_prices_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import (
    predict_saleprice,
    predict_cluster)


def page_analyse_price():

    # load predict saleprice files
    version = 'v1'
    saleprice_pipe_model = load_pkl_file(
        f"C:/Users/issam/Housing-market-analysis.1/outputs/ml_pipeline/predict_saleprice/{version}/clf_pipeline_predict.pkl")

    saleprice_pipe_dc_fe = load_pkl_file(
        f'C:/Users/issam/Housing-market-analysis.1/outputs/ml_pipeline/predict_saleprice/{version}/clf_pipeline_model.pkl')
    
    saleprice_features = (pd.read_csv(f"C:/Users/issam/Housing-market-analysis.1/outputs/ml_pipeline/predict_saleprice/{version}/X_train.csv")
                      .columns
                      .to_list()
                      )


    # load cluster analysis files
    version = 'v1'
    cluster_pipe = load_pkl_file(
        f"C:/Users/issam/Housing-market-analysis.1/outputs/ml_pipeline/cluster_analysis/{version}/cluster_pipeline.pkl")
    cluster_features = (pd.read_csv(f"C:/Users/issam/Housing-market-analysis.1/outputs/ml_pipeline/cluster_analysis/{version}/TrainSet.csv")
                        .columns
                        .to_list()
                        )
    cluster_profile = pd.read_csv(
        f"C:/Users/issam/Housing-market-analysis.1/outputs/ml_pipeline/cluster_analysis/{version}/clusters_profile.csv")

    st.write("### Prospect Sale Price Interface")
    st.info(
    f"* The client is interested in understanding the factors that influence property sale prices. "
    f"They want to determine which features are most strongly associated with higher or lower sale prices. "
    f"Additionally, the client wants to identify clusters of properties with similar characteristics. "
    f"Based on these insights, the client aims to highlight the key factors that could help increase "
    f"the sale price of a property or identify properties with the potential for price appreciation."
    )
    st.write("---")

    st.info(
        f"* The cluster profile interpretation allowed us to label the cluster in the following fashion:\n"
        f"* Cluster 0 has houses built between 1977 and 2004, with Type 1 finished square feet ranges\n "
        f"from 646 to 1070, 38% of garage finished and 27% unfinished, Buyers are likely moderate spenders.\n"
        f"* Cluster 1 features houses built between 1992 and 2005, with no Type 1 finished square footage, "
        f"basement unfinished square footage ranging from 855 to 1386, and garage finishes distributed as "
        f"44% RFn, 29% Fin, and 26% Unf. Buyers in this cluster are likely to be high spenders.\n"
        f"* Cluster 2 comprises houses constructed between 1922 and 1958, with no Type 1 finished square "
        f"footage, basement unfinished square footage ranging from 264 to 697, and garage finishes distributed "
        f"as 75% Unf, 10% None, and 7% Fin. Buyers in this cluster are likely to be mid-level spenders."
    )

    # Generate Live Data
    # check_variables_for_UI(saleprice_features, cluster_features)
    X_live = DrawInputsWidgets()

    # predict on live data
    if st.button("Run Predictive Analysis"):
        saleprice_prediction = predict_saleprice(
            X_live, saleprice_features, saleprice_pipe_model, saleprice_pipe_dc_fe)

        predict_cluster(X_live, cluster_features,
                        cluster_pipe, cluster_profile)


def check_variables_for_UI(saleprice_features, cluster_features):
    import itertools

    # The widgets inputs are the features used in all pipelines (saleprice, cluster)
    # We combine them only with unique values
    combined_features = set(
        list(
            itertools.chain(saleprice_features, cluster_features)
        )
    )
    st.write(
        f"* There are {len(combined_features)} features for the UI: \n\n {combined_features}")


def DrawInputsWidgets():

    # load dataset
    df = load_house_prices_data()
    percentageMin, percentageMax = 0.4, 2.0

# we create input widgets only for 6 features
#    col1, col2, col3, col4 = st.columns(4)
#    col5, col6, col7, col8 = st.columns(4)
# Create 21 columns
    """
    columns = st.columns(21)
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18, col19, col20, col21 = columns

    # We are using these features to feed the ML pipeline - values copied from check_variables_for_UI() result
    # 'YearBuilt', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageFinish', 'OverallQual'

    # create an empty DataFrame, which will be the live data
    X_live = pd.DataFrame([], index=[0])

    # from here on we draw the widget based on the variable type (numerical or categorical)
    # and set initial values
    with col1:
        feature = "YearBuilt"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget

    with col2:
        feature = "BsmtFinSF1"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget

    with col3:
        feature = "BsmtUnfSF"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
    X_live[feature] = st_widget

    with col4:
        feature = "TotalBsmtSF"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget

    with col5:
        feature = "GarageFinish"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with col6:
        feature = "OverallQual"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget

    with col7:
        feature = "1stFlrSF"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget
    with col8:
        feature = "2ndFlrSF"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget
    with col9:
        feature = "BedroomAbvGr"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget
    with col10:
        feature = "BsmtExposure"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget
    with col11:
        feature = "BsmtFinType1"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget
    with col12:
        feature = "GarageArea"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget
    with col13:
        feature = "GarageYrBlt"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget
    with col14:
        feature = "GrLivArea"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget
    with col15:
        feature = "KitchenQual"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget
    with col16:
        feature = "LotArea"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget
    with col17:
        feature = "LotFrontage"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget
    with col18:
        feature = "MasVnrArea"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget
    with col19:
        feature = "OpenPorchSF"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget
    with col20:
        feature = "YearRemodAdd"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget
    with col21:
        feature = "OverallCond"
        st_widget = st.selectbox(
            label=feature,
            options=sorted(df[feature].unique())
        )
    X_live[feature] = st_widget

    """    # st.write(X_live)
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
    features = [
    '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtExposure', 'BsmtFinSF1',
    'BsmtFinType1', 'BsmtUnfSF', 'GarageArea', 'GarageFinish', 'GarageYrBlt',
    'GrLivArea', 'KitchenQual', 'LotArea', 'LotFrontage', 'MasVnrArea',
    'OpenPorchSF', 'OverallCond', 'OverallQual', 'TotalBsmtSF', 'YearBuilt',
    'YearRemodAdd'
    ]

    for feature in features:
        if df[feature].dtype == 'object':
            st_widget = st.selectbox(
                label=feature,
                options=sorted(df[feature].unique())
            )
        elif df[feature].dtype in ['int64', 'float64']:
            if len(df[feature].unique()) < 20:  # Arbitrary threshold for using selectbox
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

