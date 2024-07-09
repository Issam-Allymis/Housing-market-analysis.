|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|



## Business Requirements
As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.


## Hypothesis and how to validate?

### **Hypotheses:**

- **Hypothesis 1:** The size of the lot (LotArea) positively correlates with the sale price (SalePrice) of a house.

- **Hypothesis 2:** Houses with a larger above-ground living area (GrLivArea) tend to have a higher sale price.
- **Hypothesis 3:** The quality of the kitchen (KitchenQual) positively influences the sale price of a house.
- **Hypothesis 4:** The presence of a masonry veneer (MasVnrArea) increases the sale price of a house.
- **Hypothesis 5:** The year of construction (YearBuilt) affects the sale price of a house, with newer houses commanding higher prices.

### **Validating Hypotheses:**

- **Data Exploration and Visualization:** Explore the relationships between the selected variables and the sale price using visualizations such as scatter plots, histograms, and heatmaps.

- **Machine Learning Models:** Train regression models to predict the sale price based on the identified features. Evaluate the models' performance and analyze feature importance.
**Business Requirements:**

- **Requirement 1:** Identify key features that significantly impact house prices to inform pricing strategies.

- **Requirement 2:** Understand the preferences of homebuyers by analyzing the importance of various features in determining the sale price.
- **Requirement 3:** Develop predictive models to estimate house prices accurately, facilitating decision-making for buyers, sellers, and real estate investors.
- **Requirement 4:** Visualize the distribution of house prices and important features to gain insights into market trends and demand.

### **Mapping to Data Visualizations and ML Tasks:**

- **Data Visualizations:**
    - Scatter plots: to visualize the relationships between numerical features and sale price.
    - Histograms: to explore the distributions of numerical variables.
    - Heatmaps: to display correlations between features and identify patterns.
- **ML Tasks:**
    - Regression modeling: to predict sale prices based on relevant features.
    - Feature importance analysis: to determine the most influential factors driving house prices.
These hypotheses and corresponding analyses aim to provide valuable insights into the elements leading house prices, allowing stakeholders to make calculated decisions which is cruical in the real estate market.


## The rationale to map the business requirements to the Data Visualisations and ML tasks

1. **Predictive Modeling for Property Prices:**

- **Requirement:** Develop machine learning models to predict property prices based on various features such as location, size, amenities, etc.

- **Rationale:** Predicting property prices accurately is crucial for buyers, sellers, and real estate agents to make informed decisions. Machine learning models can utilize historical sales data and property features to provide accurate price estimates.

2. **Identifying Key Factors Affecting Property Prices:**

**Requirement:** Analyze the impact of different factors (e.g., location, square footage, over all condition) on property prices.
**Rationale:** Understanding the key drivers of property prices helps stakeholders identify lucrative investment opportunities and make data-driven decisions. When it comes to visualizations, the likes of scatter plots, heatmaps, and correlation matrices can highlight the relationships between variables.

## ML Business Case
* Frame the business case using the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, which is a commonly used approach for data mining projects.
 Here's how I applied it to the business requirement of predicting property prices:

### Business Case Understanding:

- **Business Objective:** Improve the accuracy of predicting housing prices to assist real estate agents and homebuyers in making informed decisions.

- **Key Stakeholders:** Real estate agents, homebuyers, data scientists, business analysts.

### Data Collection:
- **Data Sources:** Obtain housing market data from public datasets, Kaggle.
- **Data Description:** Collect information on various features such as square footage, size of the garage, Above grade ground living area, garage finish, year built, sale price, etc.
- **Data Permissions:** Ensure compliance with data privacy regulations and obtain necessary permissions to use the data.
- **Data Quality Check:** Assess the quality of the collected data, check for missing values, duplicates, and inconsistencies.

### Data Visualization:
- **Exploratory Data Analysis (EDA):** Visualize the distribution of housing prices, explore relationships between features, and identify patterns or anomalies.
- **Feature Importance:** Visualize the importance of different features in predicting housing prices using techniques such as correlation analysis or feature importance plots.

### Model Training:
- **Model Selection:** Choose appropriate regression models such as Linear Regression, Random Forest, or Gradient Boosting Regression based on the problem complexity and data characteristics.
- **Data Preprocessing:** Handle missing values, encode categorical variables, and scale numerical features as required.
- **Model Evaluation:** Split the data into training and testing sets, train the selected models on the training data, and evaluate their performance using metrics like Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE).

### Model Visualization:
- **Prediction Visualization:** Visualize the predicted housing prices against the actual prices to assess the model's performance visually.
- **Residual Analysis:** Plot the residuals (the difference between predicted and actual prices) to check for heteroscedasticity and ensure model assumptions are met.

### Model Optimization:
- **Hyperparameter Tuning:** Use techniques like Grid Search or Random Search to find the optimal hyperparameters for the selected models.
- **Cross-Validation:** Implement k-fold cross-validation to assess model generalization and reduce overfitting.
- **Ensemble Methods:** Explore ensemble techniques such as Bagging, Boosting, or Stacking to improve model performance further.

### Model Generation:
- **Final Model Selection:** Choose the best-performing model based on evaluation metrics and cross-validation results.
- **Model Persistence:** Train the final model on the entire dataset and save it for future use or deployment.

### Model Deployment:
- **Integration with Business Systems:** Deploy the trained model into production environments.
**API Development:** Develop a RESTful API to allow real-time predictions of housing prices based on user input.

### Business Output:
- **Real-time Price Prediction:** Provide real-time housing price predictions to real estate agents and homebuyers through web or mobile applications.
- **Informed Decision-Making:** Enable stakeholders to make informed decisions based on accurate predictions, leading to improved property valuation and better investment opportunities.

## Dashboard Design
1. **Overview Page:**

- Summary statistics of the housing market dataset.
- Distribution plots for key features like sale price, over all quality like the finishing touches of the property, the condition of the house, etc.
- Map visualization showing housing prices by over all quality.

2. **Feature Analysis Page:**

- Correlation heatmap to visualize relationships between different features.
- Scatter plots to explore the relationship between sale price and numerical features like year remodel date.

3. **Model Evaluation Page:**

- Performance metrics table displaying evaluation metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for different regression models.

4. **Model Interpretation Page:**

- Feature importance plot showing the relative importance of different features in predicting housing prices.

5. Prediction Page:

Input form to enter details of a new property (e.g., square footage, number of bedrooms, location, etc.).
Button to trigger model prediction and display the predicted sale price.
Visualization of the predicted price along with confidence intervals if available.

6. **Dashboard Settings Page:**
- Toggle buttons or checkboxes to show/hide specific plots or widgets based on user preferences.

## Unfixed Bugs

* My model had a certain amount of overfitting, i attempted to modify the preprocessor transformers and the matters improved in the final model before deployed to production.

* Many of the graphics that could've been displayed in the dashboard haven't been included but are included in the notebooks.


## Deployment
### Heroku

* The App live link is: https://housingmarket-51bd98038f4c.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries
* **examples of how libraries were used:**

01. scikit-learn is a Python library for machine learning that provides a wide range of tools and algorithms for tasks such as classification, regression, clustering, and dimensionality reduction.
 
02. Importing and using pandas to create and manipulate dataframes.
03. Importing and using numpy to work with arrays and numerical operations.
04. Importing and using matplotlib to create visualizations, such as line plots.
05. Importing and using seaborn to create statistical visualizations, such as bar plots.
06. Using ProfileReport to display the report in a Jupyter Notebook using profile.to_notebook_iframe().

* **Usage examples**
01. **Scikit-learn:** Used it to train a linear regression model to predict house prices based on features like number of bedrooms and square footage.
02. **Pandas:** Used pandas to read in a CSV file containing housing market data and created a dataframe. I then manipulated the dataframe to clean the data, perform exploratory data analysis, and prepared it for machine learning tasks.
03. **Numpy:** Used numpy to create arrays and perform numerical operations. For example, Used numpy to calculate the mean and standard deviation of housing prices in a dataset.
04. **Matplotlib:** Used matplotlib to create visualizations such as line plots to show the trend of housing prices over time or scatter plots to visualize the relationship between feature variables and the target sale price variable.
05. **Seaborn:** Used for visualizing datasets to understand their distributions, relationships between variables, and potential patterns. Additionally, scatterplots can be created using Seaborn.
06. **ProfileReport:** Used ProfileReport from the pandas_profiling library to generate a comprehensive report on the housing market dataset, including summaries, statistics, and visualizations, all in a Jupyter Notebook environment.
## Credits 

* [NeuralNine](https://www.youtube.com/watch?v=Wqmtf9SA_kk); This source helped me grasp the very basics of machine learning.
* [Code Institute](https://learn.codeinstitute.net/ci_program/diplomainsoftwaredevelopmentpredictiveanalytics); The Data Analysis & Machine Learning Toolkit unit was instrumental in helping me grasp the fundamentals of data analysis and machine learning. However, I acknowledge that I still have a long journey ahead to truly master these concepts.
* [Ken Jee](https://www.youtube.com/watch?v=NQQ3DRdXAXE); This source opened my eyes on how to approach certain problems in data analytics.
### Content 

- I followed a good portion of Instructions from the Develop and Deploy an A.I. System [Code Institute](https://learn.codeinstitute.net/courses/course-v1:CodeInstitute+DDA101+2021_T4/courseware/bba260bd5cc14e998b0d7e9b305d50ec/c83c55ea9f6c4e11969591e1b99c6c35/).



## Acknowledgements (optional)
* I'd like to express my heartfelt gratitude to my family and friends for their unwavering support throughout this incredibly challenging project. Additionally, I extend a profound thanks to the [Code Institute](https://codeinstitute.net/ie/full-stack-software-development-diploma/?hsa_acc=8983321581&hsa_cam=14304747355&hsa_grp=1152289077037598&hsa_ad&hsa_src=o&hsa_tgt=kwd-72018717753321%3Aloc-92&hsa_kw=code%20institute&hsa_mt=e&hsa_net=adwords&hsa_ver=3&msclkid=8f56a94a3de213de270fcc99560f89cd&utm_source=bing&utm_medium=cpc&utm_campaign=CI%20-%20IRL%20-%20Search%20-%20Brand&utm_term=code%20institute&utm_content=CI%20-%20IRL%20-%20Search%20-%20Brand%20-%20Exact) for providing invaluable insights and resources that will shape my future development. This journey has truly opened my eyes to a new realm of coding, altering my perspective from when I started to now. Thanks to this course, I've gained clarity on my career path and have made the decision to pursue further education to deepen my understanding. Project 5, focusing on predictive analytics, presented significant challenges, especially considering my lack of prior experience in this domain and the demands of a full-time job. Despite the hurdles, I embraced the opportunity to learn and grow, and I am eager to continue expanding my knowledge in the field.

