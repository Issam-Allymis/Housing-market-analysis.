import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_house_prices_data, load_pkl_file
from src.machine_learning.evaluate_clf import reg_performance
import os


current_dir = os.getcwd()

def price_prediction_body():

    version = 'v1'
    # load needed files
    saleprice_pipe_dc_fe = load_pkl_file(
       f'{current_dir}/Housing-market-analysis.1/outputs/ml_pipeline/predict_saleprice/{version}/clf_pipeline_model.pkl')
    saleprice_pipe_model = load_pkl_file(
       f"{current_dir}/Housing-market-analysis.1/outputs/ml_pipeline/predict_saleprice/{version}/clf_pipeline_model.pkl")
    saleprice_feat_importance = plt.imread(
       f"{current_dir}/Housing-market-analysis.1/outputs/ml_pipeline/predict_saleprice/{version}/features_importance.png")
    X_train = pd.read_csv(
       f"{current_dir}/Housing-market-analysis.1/outputs/ml_pipeline/predict_saleprice/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"{current_dir}/Housing-market-analysis.1/outputs/ml_pipeline/predict_saleprice/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"{current_dir}/Housing-market-analysis.1/outputs/ml_pipeline/predict_saleprice/{version}/y_train.csv").values
    y_test = pd.read_csv(
        f"{current_dir}/Housing-market-analysis.1/outputs/ml_pipeline/predict_saleprice/{version}/y_test.csv").values 

    st.write("### ML Pipeline: Predict Prospect SalePrice")
    # display pipeline training summary conclusions
    st.info(
        f"Since we are interested in this project in detecting housing market prices. \n"
        f"* The pipeline performance on train and test set is 0.57 and -0.44, respectively."
    )

    # show pipelines
    st.write("---")
    st.write("#### There are 2 ML Pipelines arranged in series.")

    st.write(" * The first is responsible for data cleaning and feature engineering.")
    st.write(saleprice_pipe_dc_fe)

    st.write("* The second is for feature scaling and modelling.")
    st.write(saleprice_pipe_model)

    # show feature importance plot
    st.write("---")
    st.write("* The features the model was trained and their importance.")
    st.write(X_train.columns.to_list())
    st.image(saleprice_feat_importance)

    # We don't need to apply dc_fe pipeline, since X_train and X_test
    # were already transformed in the jupyter notebook (Predict Customer Churn.ipynb)

    # evaluate performance on train and test set
    st.write("---")
    st.write("### Pipeline Performance")
    reg_performance(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    pipeline=saleprice_pipe_model
                    )
                    #label_map=["Low Price", "High Price"])
                    