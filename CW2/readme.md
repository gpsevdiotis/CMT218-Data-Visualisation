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
   
6. Open your default browser and direct to your localhost: http://127.0.0.1:8050/

The app uses the following Python libraries:

dash for creating the web application
dash_table for creating the table displaying the dataset
pandas for data manipulation
plotly for creating the scatter plot and histogram
statsmodels for calculating the trendline for the scatter plot
Data Preparation
The diabetes.csv file contains the dataset used in this app. The code reads the file into a Pandas DataFrame and sets the column names to a list of strings. It then filters the rows based on the condition that excludes rows where all column values in exclude_cols are equal to 0 and selects rows where at least one column.

Scatter Plot Section
The scatter plot displays the relationship between BMI and Age by Diabetes Outcome. The plot includes a trendline calculated using Ordinary Least Squares (OLS) regression from the statsmodels library.

The user can choose to display only non-diabetic or diabetic outcomes or both using the radio buttons.

Histogram Section
The histogram displays the distribution of the selected numerical variables. The user can select the variables to display using the checklist.

Dataset Section
The table displays the filtered dataset used in this app. It includes the columns Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, and Outcome. The table is paginated, and the user can change the page size using the dropdown at the bottom of the table.

Footer Section
This project was created by Georgios K. Psevdiotis (C1841824).
