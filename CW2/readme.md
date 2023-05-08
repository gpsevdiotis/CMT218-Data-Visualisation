# Installation Instructions

1. Open cmd and cd to your current dir.

2. Create a virtual environment:
   python -m venv venv

3. Activate it:
   venv\Scripts\activate

4. Install the required libraries:
   python -m pip install -r requirements.txt
   or
   pip install dash pandas plotly statsmodels

5. Run the script:
   py app.py
   
6. Open your browser and direct to your localhost: http://127.0.0.1:8050/

# The app uses the following Python libraries:

dash for creating the web application
<br>
dash_table for creating the table displaying the dataset
<br>
pandas for data manipulation
<br>
plotly for creating the scatter plot and histogram
<br>
statsmodels for calculating the trendline for the scatter plot
<br><br>
# Scatter Plot Section
The scatter plot displays the relationship between BMI and Age by Diabetes Outcome. The plot includes a trendline calculated using Ordinary Least Squares (OLS) regression from the statsmodels library.

The user can choose to display only non-diabetic or diabetic outcomes or both using the radio buttons.

# Histogram Section
The histogram displays the distribution of the selected numerical variables. The user can select the variables to display using the checklist.

# Dataset Section
The table displays the filtered dataset used in this app. It includes the columns Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, and Outcome. The table is paginated, and the user can change the page size using the dropdown at the bottom of the table. The diabetes dataset used in the application can be found on Kaggle - https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset?resource=download . The app provides a link to download the dataset.

# Footer Section
This project was created by Georgios K. Psevdiotis (C1841824).
