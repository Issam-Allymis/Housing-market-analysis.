import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    # conclusions taken from "02 - Churned Customer Study" notebook
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