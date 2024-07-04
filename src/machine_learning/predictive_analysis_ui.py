import streamlit as st
from feature_engine.encoding import OrdinalEncoder

def FeatEngineering_CategoricalEncoder(df_feat_eng, columns):
  list_methods_worked = []
  print(df_feat_eng)
  print(columns)
  #for col in df_feat_eng.select_dtypes(include='category').columns:
  #  df_feat_eng[col] = df_feat_eng[col].astype('object')

  try: 
    """ 
    encoder= OrdinalEncoder(encoding_method='arbitrary', variables = [f"{column}_ordinal_encoder"])
  
    df_feat_eng = encoder.fit_transform(df_feat_eng)
    list_methods_worked.append(f"{column}_ordinal_encoder")
    """
    encoder = OrdinalEncoder(encoding_method='arbitrary', variables=columns)
    df_feat_eng[columns] = encoder.fit_transform(df_feat_eng[columns])
        
    for column in columns:
            list_methods_worked.append(f"{column}_ordinal_encoder")
  except Exception as e:
        print(f"Error: {e}")
        for column in columns:
            if f"{column}_ordinal_encoder" in df_feat_eng.columns:
                df_feat_eng.drop([f"{column}_ordinal_encoder"], axis=1, inplace=True)

  print("list_methods_worked:",list_methods_worked)  
  return df_feat_eng


def predict_saleprice(X_live, saleprice_features, saleprice_pipeline_model, saleprice_dc_fe):

    # from live data, subset features related to this pipeline
    X_live_saleprice = X_live.filter(saleprice_features)
    

    # apply data cleaning / feat engine pipeline to live data

    # we need to change this string values to number
    features_engineering = [ 'BsmtExposure', 'BsmtFinType1', 'GarageFinish', 'KitchenQual']
    X_live_saleprice=FeatEngineering_CategoricalEncoder(X_live, features_engineering)
    X_live_saleprice_dc_fe = saleprice_dc_fe.transform(X_live_saleprice)

    # predict
  #  saleprice_prediction = saleprice_model.predict(X_live_saleprice_dc_fe)
  #  saleprice_prediction_proba = saleprice_model.predict_proba(
   #     X_live_saleprice_dc_fe)
    print(saleprice_pipeline_model)
    saleprice_prediction = saleprice_pipeline_model.predict(X_live_saleprice_dc_fe)
    print(saleprice_prediction)
   # saleprice_prediction_proba = saleprice_pipeline_model.predict_proba(
   #     X_live_saleprice_dc_fe)
  
    # st.write(churn_prediction_proba)

    # Create a logic to display the results
    #saleprice_prob = saleprice_prediction_proba[0, saleprice_prediction]*100#[0]
    #if saleprice_prediction == 1:
    #    saleprice_result = 'will'
    #else:
    #    saleprice_result = 'will not'

    statement = (
     #   f'### There is {saleprice_prob.round(1)}% probability '
        f'that this prospect **{saleprice_prediction} Sale Price**.')

    st.write(statement)

    return saleprice_prediction


def predict_tenure(X_live, tenure_features, tenure_pipeline, tenure_labels_map):

    # from live data, subset features related to this pipeline
    X_live_tenure = X_live.filter(tenure_features)

    # predict
    tenure_prediction = tenure_pipeline.predict(X_live_tenure)
    tenure_prediction_proba = tenure_pipeline.predict_proba(X_live_tenure)
    # st.write(tenure_prediction_proba)

    # create a logic to display the results
    proba = tenure_prediction_proba[0, tenure_prediction][0]*100
    tenure_levels = tenure_labels_map[tenure_prediction[0]]

    if tenure_prediction != 1:
        statement = (
            f"* In addition, there is a {proba.round(2)}% probability the prospect "
            f"will stay **{tenure_levels} months**. "
        )
    else:
        statement = (
            f"* The model has predicted the prospect would stay **{tenure_levels} months**, "
            f"however we acknowledge that the recall and precision levels for {tenure_levels} is not "
            f"strong. The AI tends to identify potential churners, but for this prospect the AI is not "
            f"confident enough on how long the prospect would stay."
        )

    st.write(statement)


def predict_cluster(X_live, cluster_features, cluster_pipeline, cluster_profile):

    # from live data, subset features related to this pipeline
    # X_live_cluster = X_live.filter(cluster_features)

    # from live data, subset features related to this pipeline
    print(X_live)
    print(cluster_features)
    X_live_cluster = X_live.filter(cluster_features)
    
    print(X_live_cluster)
    import sklearn
    print(sklearn.__version__)

    # predict
    cluster_prediction = cluster_pipeline.predict(X_live_cluster)

    statement = (
        f"### The prospect is expected to belong to **cluster {cluster_prediction[0]}**")
    st.write("---")
    st.write(statement)

  	# text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
    #statement = (
    #    f"* Evidentally, properties which span over  greater land mass "
    #    f"and that are kept well tend to cost more, the year "
    #    f"during which the houses were built in does matter but the " 
    #    f"OverallQual feature seems to have a higher precedence over other features. "
    #)
    #st.info(statement)

  	# text based on "07 - Modeling and Evaluation - Cluster Sklearn" notebook conclusions
    #statement = (
    #    f"* The cluster profile interpretation allowed us to label the cluster in the following fashion:\n"
    #    f"* Cluster 0 has houses built between 1977 and 2004, with Type 1 finished square feet ranges\n "
    #    f"from 646 to 1070, 38% of garage finished and 27% unfinished, Buyers are likely moderate spenders.\n"
    #    f"* Cluster 1 features houses built between 1992 and 2005, with no Type 1 finished square footage, "
    #    f"basement unfinished square footage ranging from 855 to 1386, and garage finishes distributed as "
    #    f"44% RFn, 29% Fin, and 26% Unf. Buyers in this cluster are likely to be high spenders.\n"
    #    f"* Cluster 2 comprises houses constructed between 1922 and 1958, with no Type 1 finished square "
    #    f"footage, basement unfinished square footage ranging from 264 to 697, and garage finishes distributed "
    #    f"as 75% Unf, 10% None, and 7% Fin. Buyers in this cluster are likely to be mid-level spenders."
    #)
    #st.success(statement)

    # hack to not display index in st.table() or st.write()
    cluster_profile.index = [" "] * len(cluster_profile)
    # display cluster profile in a table - it is better than in st.write()
    st.table(cluster_profile)
