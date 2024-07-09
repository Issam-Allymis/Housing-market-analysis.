import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_house_prices_data, log10_transform
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
version = 'v1'

def sale_price_study_body(): 

    # load data
    df = load_house_prices_data()

    # vars_to_study selected from analysis in notebook 2
    vars_to_study = ['YearBuilt', 'GrLivArea' 
    'LotFrontage', 'OverallQual', '1stFlrSF'] 

    '''['BsmtFinSF1', 'YearBuilt', 'BsmtUnfSF', 'GarageArea', 
    'GrLivArea', 'BsmtExposure', 'LotFrontage', 'MasVnrArea', 
    'OpenPorchSF', 'KitchenQual', 'OverallQual', 'OverallCond', 
    'GarageYrBlt', 'BedroomAbvGr', '1stFlrSF', '2ndFlrSF']
   
    ['YearBuilt' 'GrLivArea' 'LotFrontage' 'OverallQual' '1stFlrSF'] 
    '''

    st.write("### SalePrice Study")
    st.info(
        f"* The client is interested in understanding the patterns within the housing market data "
        f"to identify the most relevant variables "
        f"correlated with high sale prices."
        f" The client is interested in understanding the factors that influence property sale prices. "
        f"They want to determine which features are most strongly associated with higher or lower sale prices. "
        f"Based on these insights, the client aims to highlight the key factors that could help increase "
        f"the sale price of a property or identify properties with the potential for price appreciation."
    )

    st.write("### Hypothesis validation")

    

    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted to better understand how  "
        f"the variables are correlated with Sale Price levels.\n"
        f"The most correlated variable are: **{vars_to_study}**"
    )

    st.info(
    
    f"* The correlation indications and plots below interpretation converge. It is indicated that:\n\n"
    f"* A higher saleprice is typically associated with the First Floor square feet having a larger surface area **['1stFlrSF']**.\n"
    f"* A higher saleprice is typically associated with a larger Size of LotFrontage in square feet **['LotFrontage']**.\n"
    f"* A higher saleprice is typically associated with a larger Above grade (ground) living area in square feet **['GrLivArea']**.\n"
    f"* A higher saleprice is typically associated with a higher Rate of the overall material/quality and finish of the house **['OverallQual']**.\n"
    f"* A higher saleprice is typically associated with the more recent construction dates **['YearBuilt']**.\n"
    )


    if st.button("Feature Importance chart"):    
        # Load the image of feature importance plot
        image_path = os.path.join('outputs', 'ml_pipeline', 'predict_saleprice', 'v1', 'features_importance.png')
        # image_path = f'outputs/ml_pipeline/predict_saleprice/{version}/features_importance.png'
        image = Image.open(image_path)
        image.show()


    # Code copied from "02 - PredictingSalePrice" notebook code - "EDA on selected variables" section
    df_eda = df.filter(vars_to_study + ['SalePrice'])

    # Individual plots per variable
    
    if st.button("SalePrice Levels per Variable"):
        st.write(
            f"* Evidently, properties which span over greater foot print "
            f"and that are kept well tend to cost more. The year "
            f"during which the houses were built in does matter but the " 
            f"OverallQual feature seems to have a higher precedence over other features. "
        )
        saleprice_level_per_variable(df_eda)

    # Parallel plot
    if st.button("Parallel Plot"):
        st.write(
            f"* Information in light blue (almost white) indicates the profile of properties with higher \n"
            f"Sale Price levels, while midnight blue/navy represents properties \n"
            f"with lower Sale Price levels.")
        parallel_plot_saleprice(df_eda)
        plot_yearbuilt(df_eda)

        # inspect data
    if st.button("Inspect Customer Base"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))


# function created using "02 - PredictingSalePrice" notebook code - "Variables Distribution by SalePrice" section
def saleprice_level_per_variable(df_eda):
    target_var = 'SalePrice'

    for col in df_eda.drop([target_var], axis=1).columns.to_list():
        if df_eda[col].dtype == 'object':
            plot_categorical(df_eda, col, target_var)
        else:
            plot_numerical(df_eda, col, target_var)


'''
def plot_categorical(df, col, target_var, max_labels=5):
    fig, axes = plt.subplots(figsize=(12, 5))
    # Select the top max_labels categories based on their counts
    top_categories = df[col].value_counts().head(max_labels).index
    sns.countplot(data=df[df[col].isin(top_categories)], x=col, hue=target_var,
                  order=top_categories)
    plt.xticks(rotation=90)
    plt.title(f"{col}", fontsize=12, y=1.05)
     # Limit the number of legend labels
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(handles[:max_labels], labels[:max_labels], title=target_var)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()
'''


# code copied from "02 - PredictingSalePrice" notebook - "Variables distribution by SalePrice" section
def plot_numerical(df, col, target_var, max_labels=5):
    fig, axes = plt.subplots(figsize=(12, 5))
    # Select the top max_labels categories based on their counts
    top_categories = df[col].value_counts().head(max_labels).index
    data_to_plot = df[df[col].isin(top_categories)]
    sns.scatterplot(data=df, x=col, y=target_var, hue=target_var)
    #sns.histplot(data=df, x=col, hue=target_var, kde=True, element="step")

    legend_labels = data_to_plot[target_var].unique()[:max_labels]
    plt.legend(labels=legend_labels)

    plt.title(f"{col}", fontsize=20, y=1.10)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


# function created using ""02 - PredictingSalePrice" notebook code - Parallel Plot section
def parallel_plot_saleprice(df_eda):

    # Define OverallQual binning and transformation
    OverallQual_map = [-np.inf, 5.0, 6.0, 7.0, np.inf]
    disc = ArbitraryDiscretiser(binning_dict={'OverallQual': OverallQual_map})
    df_parallel = disc.fit_transform(df_eda)

    n_classes = len(OverallQual_map) - 1
    classes_ranges = disc.binner_dict_['OverallQual'][1:-1]
    LabelsMap = {}
    for n in range(n_classes):
        if n == 0:
            LabelsMap[n] = f"{classes_ranges[n-1]} to {classes_ranges[n]}"
        elif n == n_classes - 1:
            LabelsMap[n] = f"+{classes_ranges[-1]}"
        else:
            LabelsMap[n] = f"<{classes_ranges[0]}"

    df_parallel['OverallQual'] = df_parallel['OverallQual'].replace(LabelsMap)
    fig = px.parallel_categories(
        df_parallel, dimensions=['OverallQual'], color="SalePrice", width=850, height=500)
    st.plotly_chart(fig)

def plot_yearbuilt(df_eda):
    YearBuilt_map = [-np.inf, 1900, 2000, np.inf]
    disc = ArbitraryDiscretiser(binning_dict={'YearBuilt': YearBuilt_map})
    df_parallel = disc.fit_transform(df_eda)
    sns.set_style("whitegrid")

    n_classes = len(YearBuilt_map) - 1
    classes_ranges = disc.binner_dict_['YearBuilt'][1:-1]
    LabelsMap = {}
    for n in range(n_classes):
        if n == 0:
            LabelsMap[n] = f"{classes_ranges[n-1]} to {classes_ranges[n]}"
        elif n == n_classes - 1:
            LabelsMap[n] = f"+{classes_ranges[-1]}"
        else:
            LabelsMap[n] = f"<{classes_ranges[0]}"

    

    df_parallel['YearBuilt'] = df_parallel['YearBuilt'].replace(LabelsMap)
    fig = px.parallel_categories(df_parallel, dimensions=['YearBuilt'], color="SalePrice", width=850, height=500)
    st.plotly_chart(fig)


def page_sale_price_study_general_conclusion():

    st.write("### General conclusion on price behaviour")

    # conclusions taken from "02 - " notebook
    st.success(
        f"* "
        f"The correlation study at Sale Price Study supports that; \n\n"

        f"* A property pricing survey showed that value was influenced by several key factors.\n "
        f" Among these factors, the over all quality emerged as the most significant determinant\n"
        f" of property value in Ames, Iowa. Additionally, the size of the land plot proved to be a crucial factor, with properties\n"
        f" sprawling over larger land masses commanding higher prices."
        f" A High-quality construction and larger living spaces are correlated with higher property prices in the housing market."
        f" This insight will be used by the survey team for further discussions and investigations."
    )

        