import streamlit as st

def page_summary_body():

    st.write("### Quick Project Summary")

    # copied from README file - "Business Requirements" section
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in discovering how the house attributes correlate with the sale price. "
        f"Therefore, the client expects data visualisations of the correlated variables against the sale price "
        f"to show that. This objective is delivered in page Sales Price Study.\n"
        f"* 2 - The client is interested in predicting the house sale price from her four inherited houses and "
        f"any other house in Ames, Iowa. This objective is delivered in page Price Prediction."
        )

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* A homeowner is an individual who owns a residential property.\n"
        f"* A homebuyer is someone who is interested in purchasing a property and is actively searching for one.\n"
        f"* Closing costs are fees associated with finalizing a real estate transaction, including legal fees, "
        f"appraisal fees, and title insurance.\n "
        f"* The Multiple Listing Service (MLS) is a database used by real estate agents to share information about "
        f"properties for sale. " 
        f"has used our product/service.\n\n"
        f"**Project Dataset**\n"
        f"* The dataset represents a **housing records from Ames, Iowa,** "
        f"indicating house profile (Floor Area, Basement, Garage, Kitchen, "
        f"Lot, Porch, Wood Deck, Year Built)and its respective sale price , "
        f"for houses built between 1872 and 2010. "
        f"and profile (like gender, partner, dependents).")

    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/Issam-Allymis/Housing-market-analysis.).")
    

