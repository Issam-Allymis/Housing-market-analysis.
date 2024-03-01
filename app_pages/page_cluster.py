import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.data_management import load_telco_data, load_pkl_file


def page_cluster_body():

    # load cluster analysis files and pipeline
    version = 'v1'
    cluster_pipe = load_pkl_file(
        f"/workspace/Housing-market-analysis./outputs/ml_pipeline/cluster_analysis/v1/cluster_pipeline.pkl")
    cluster_silhouette = plt.imread(
        f"/workspace/Housing-market-analysis./outputs/ml_pipeline/cluster_analysis/v1/clusters_silhouette.png")
    features_to_cluster = plt.imread(
        f"/workspace/Housing-market-analysis./outputs/ml_pipeline/cluster_analysis/v1/features_define_cluster.png")
    heatmapCorr = plt.imread(
        f"/workspace/Housing-market-analysis./images/heatmapCorr.png"
        )
    scatterImg = plt.imread(
        f"/workspace/Housing-market-analysis./images/numerical_plot.png"
    )
    cluster_profile = pd.read_csv(
        f"outputs/ml_pipeline/cluster_analysis/v1/clusters_profile.csv")
    cluster_features = (pd.read_csv(f"outputs/ml_pipeline/cluster_analysis/v1/TrainSet.csv")
                        .columns
                        .to_list()
                        )

    # dataframe for cluster_distribution_per_variable()
    df_saleprice_vs_clusters = load_telco_data().filter(['SalePrice'], axis=1)
    df_saleprice_vs_clusters['Clusters'] = cluster_pipe['model'].labels_

    st.write("### ML Pipeline: Cluster Analysis")
    # display pipeline training summary conclusions
    st.info(
        f"* We refitted the cluster pipeline using fewer variables, and it delivered equivalent "
        f"performance to the pipeline fitted using all variables.\n"
        f"* The pipeline average silhouette score is 0.30"
    )
    st.write("---")

    st.write("#### Cluster ML Pipeline steps")
    st.write(cluster_pipe)

    st.write("#### The features the model was trained with")
    st.write(cluster_features)

    st.write("#### Clusters Silhouette Plot")
    st.image(cluster_silhouette)

    st.write( "#### Heatmap Correlation Plot")
    st.image(heatmapCorr)

    st.write("#### Scatter Plot")
    st.image(scatterImg)

    cluster_distribution_per_variable(df=df_saleprice_vs_clusters, target='SalePrice')

    st.write("#### Most important features to define a cluster")
    st.image(features_to_cluster)

    # text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
    st.write("#### Cluster Profile")
    statement = (
       f"* Among these clusters, properties in Cluster 0 were typically built between 1977 and 2004, " 
       f"with basement finished square footage ranging from 646 to 1070 square feet, unfinished square " 
       f"footage from 130 to 424 square feet, total basement square footage from 894 to 1473 square feet, " 
       f"and garage finishes distributed as 38% finished, 35% ready to finish, and 27% unfinished. The sale " 
       f"prices in this cluster fall within the range of €157,000 to €256,000. \n"

       f"* Properties in Cluster 1 were predominantly built between 1992 and 2005, with no basement finished square " 
       f"footage, unfinished square footage ranging from 855 to 1386 square feet, total basement square footage from " 
       f"870 to 1512 square feet, and garage finishes distributed as 44% ready to finish, 29% finished, and 26% unfinished. " 
       f"The sale prices in this cluster range from €172,009 -- €218,834.\n"

       f"* Cluster 2 properties were mainly constructed between 1922 and 1958, with no basement finished square footage, unfinished " 
       f"square footage ranging from 264 to 697 square feet, total basement square footage from 660 to 979 square feet, and garage " 
       f"finishes distributed as 75% unfinished, 10% none, and 7% finished. The sale prices in this cluster vary between €107,500 -- €153,900."
    )
    st.info(statement)

    # text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
    statement = (
        f"* The cluster profile interpretation allowed us to label the cluster in the following fashion:\n"
        f"* Cluster 0 has houses built between 1977 and 2004, with Type 1 finished square feet ranges"
        f"from 646 to 1070, 38% of garage finished and 27% unfinished, Buyers are likely moderate spenders.\n"
        f"* Cluster 1 features houses built between 1992 and 2005, with no Type 1 finished square footage, "
        f"basement unfinished square footage ranging from 855 to 1386, and garage finishes distributed as "
        f"44% RFn, 29% Fin, and 26% Unf. Buyers in this cluster are likely to be high spenders.\n"
        f"* Cluster 2 comprises houses constructed between 1922 and 1958, with no Type 1 finished square "
        f"footage, basement unfinished square footage ranging from 264 to 697, and garage finishes distributed "
        f"as 75% Unf, 10% None, and 7% Fin. Buyers in this cluster are likely to be mid-level spenders."
    )
    st.success(statement)

    # hack to not display the index in st.table() or st.write()
    cluster_profile.index = [" "] * len(cluster_profile)
    st.table(cluster_profile)


# code coped from "07 - Modeling and Evaluation - Cluster Sklearn" notebook - under "Cluster Analysis" section
def cluster_distribution_per_variable(df, target):

    df_bar_plot = df.value_counts(["Clusters", target]).reset_index()
    df_bar_plot.columns = ['Clusters', target, 'Count']
    df_bar_plot[target] = df_bar_plot[target] #.astype('object')

    st.write(f"#### Clusters distribution across {target} levels")
    fig = px.bar(df_bar_plot, x='Clusters', y='Count',
                 color=target, width=800, height=350)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    # we replaced fig.show() for a streamlit command to render the plot
    st.plotly_chart(fig)

    df_relative = (df
                   .groupby(["Clusters", target])
                   .size()
                   .groupby(level=0)
                   .apply(lambda x:  100*x / x.sum())
                   .reset_index()
                   .sort_values(by=['Clusters'])
                   )
    df_relative.columns = ['Clusters', target, 'Relative Percentage (%)']

    st.write(f"#### Relative Percentage (%) of {target} in each cluster")
    fig = px.line(df_relative, x='Clusters', y='Relative Percentage (%)',
                  color=target, width=800, height=350)
    fig.update_layout(xaxis=dict(tickmode='array',
                      tickvals=df['Clusters'].unique()))
    fig.update_traces(mode='markers+lines')
    # we replaced fig.show() for a streamlit command to render the plot
    st.plotly_chart(fig)