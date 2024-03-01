import streamlit as st
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# code copied from "Modeling and Evaluation" notebooks

def reg_performance(X_train, y_train, X_test, y_test, pipeline):
    st.write("#### Train Set ####")
    y_train_pred = pipeline.predict(X_train)
    st.write("Mean Squared Error (Train):", mean_squared_error(y_train, y_train_pred))
    st.write("R2 Score (Train):", r2_score(y_train, y_train_pred))

    st.write("\n#### Test Set ####")
    y_test_pred = pipeline.predict(X_test)
    st.write("Mean Squared Error (Test):", mean_squared_error(y_test, y_test_pred))
    st.write("R2 Score (Test):", r2_score(y_test, y_test_pred))

