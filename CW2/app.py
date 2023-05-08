# Import necessary libraries
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# Read the CSV File 
df = pd.read_csv('diabetes.csv')


# Set the column names of a DataFrame to a list of strings
df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Create a list of column names containing numerical data
numerical_values = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Filter rows of DataFrame based on the condition
# Exclude rows where all column values in exclude_cols are equal to 0
# Select rows where at least one column
exclude_cols = ["Pregnancies", "Outcome"]
df = df.loc[~(df.drop(exclude_cols, axis=1) == 0).any(axis=1)]

# Create the Dash app
app = Dash(__name__)
app.title = 'C1841824'

fig = px.scatter(df, x='Age', y='BMI', color='Outcome', trendline='ols',trendline_scope="overall", trendline_color_override="black")

# Define the layout
app.layout = html.Div([
    html.H1('Data Analysis and Visualisation Creation', style={'text-align': 'center', 'font-family': 'Helvetica Neue, sans-serif'}),
    html.Hr(),
    html.H2('BMI and Age relationship by Diabetes Outcome', style={'text-align': 'center', 'font-weight': 'normal', 'font-family': 'Helvetica Neue, sans-serif'}),
    html.Br(),
    dcc.RadioItems(
        id='output-selector',
        options=[
            {'label': 'Only Non-Diabetic', 'value': 0},
            {'label': 'Only Diabetic', 'value': 1},
            {'label': 'Both', 'value': 'both'}
        ],
        value='both',
        inline=True,
        style={'text-align': 'center', 'font-family': 'Helvetica Neue, sans-serif'}
    ),
    dcc.Graph(id='scatter-plot', figure=fig),
    html.Br(),
    html.Hr(),
    html.Br(),
    html.H2('Histogram of Numerical Variables', style={'text-align': 'center', 'font-weight': 'normal', 'font-family': 'Helvetica Neue, sans-serif'}),
    html.Br(),
    dcc.Checklist(
        id='variable-selector',
        options=[
            {'label': 'Pregnancies', 'value': 'Pregnancies'},
            {'label': 'Glucose', 'value': 'Glucose'},
            {'label': 'BloodPressure', 'value': 'BloodPressure'},
            {'label': 'SkinThickness', 'value': 'SkinThickness'},
            {'label': 'Insulin', 'value': 'Insulin'},
            {'label': 'BMI', 'value': 'BMI'},
            {'label': 'DiabetesPedigreeFunction', 'value': 'DiabetesPedigreeFunction'},
            {'label': 'Age', 'value': 'Age'},
            
        ],
        value=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        labelStyle={'display': 'inline-block'},
        style={'text-align': 'center', 'font-family': 'Helvetica Neue, sans-serif'}),
    html.Br(),
    dcc.Graph(id='histogram-plot'),
    html.Br(),
    html.Hr(),
    html.Br(),
    html.H2('Dataset', style={'text-align': 'center', 'font-weight': 'bold', 'font-style': 'italic', 'text-decoration': 'underline', 'font-family': 'Helvetica Neue, sans-serif'}),
    html.A('diabetes.csv', href='https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset?resource=download', style={'text-align': 'center', 'font-family': 'Helvetica Neue, sans-serif'}),
    dash_table.DataTable(data=df.to_dict('records'), page_size=10, style_cell={'font-family': 'Helvetica Neue, sans-serif'}),
    html.Footer('Georgios K. Psevdiotis', style={'color':'#888888','text-align': 'center', 'font-style': 'italic', 'font-family': 'Helvetica Neue, sans-serif', 'padding-top': '10px'}),
    html.Footer('C1841824', style={'color':'#888888', 'text-align': 'center', 'font-style': 'italic', 'font-family': 'Helvetica Neue, sans-serif'})    
    ])

@app.callback(
    [Output('scatter-plot', 'figure'), Output('histogram-plot', 'figure')],
    [Input('output-selector', 'value'), Input('variable-selector', 'value')]
)

def update_figures(selected_output, selected_variable):
    """
    Updates scatter and histogram plots based on user's selected output and variable.

    Args:
        selected_output (str): The user-selected output.
        selected_variable (str): The user-selected variable.

    Returns:
        tuple: A tuple containing the updated scatter plot and histogram plot figures.
    """
    #######################################
    ########SCATTER PLOT CALLBACK##########
    #######################################
     
    # Check if the selected output is 'both'
    if selected_output == 'both':
        # If it is, use the entire data frame
        filtered_df = df
    else:
        # Otherwise, filter the data frame using the selected output
        filtered_df = df[df['Outcome'] == selected_output]

    # Create a scatter plot with the filtered data frame
    scatter_fig = px.scatter(filtered_df, x='Age', y='BMI', color='Outcome', trendline='ols',trendline_scope="overall", trendline_color_override="black")

    # Create trace for non-diabetic patients
    trace1 = go.Scatter(
        x=filtered_df[filtered_df['Outcome'] == 0]['Age'],
        y=filtered_df[filtered_df['Outcome'] == 0]['BMI'],
        mode='markers',
        marker=dict(color='blue'),
        name='Non-Diabetic',
        hovertemplate='<br>'.join([
            'Age: %{x}',
            'BMI: %{y}',
            'Outcome: Non-Diabetic'
        ]),
        legendgroup='Non-Diabetic',
        legendrank=1
    )

    # Create trace for diabetic patients
    trace2 = go.Scatter(
        x=filtered_df[filtered_df['Outcome'] == 1]['Age'],
        y=filtered_df[filtered_df['Outcome'] == 1]['BMI'],
        mode='markers',
        marker=dict(color='red'),
        name='Diabetic',
        hovertemplate='<br>'.join([
            'Age: %{x}',
            'BMI: %{y}',
            'Outcome: Diabetic'
        ]),
        legendgroup='Diabetic',
        legendrank=2
    )

     # Compute ols trendline
    X = sm.add_constant(filtered_df['Age'])
    model = sm.OLS(filtered_df['BMI'], X).fit()
    trendline_y = model.predict(X)
    trendline_trace = go.Scatter(
    x=filtered_df['Age'], 
    y=trendline_y, 
    mode='lines', 
    line=dict(color='black', width=2), 
    name='OLS Trendline',
    hovertemplate='<br>'.join([
        'Age: %{x}',
        'BMI: %{y}',
        'Outcome: Trendline'
    ])
    )

    # Update the figure with the filtered data and ols trendline
    scatter_fig = go.Figure(data=[trace1, trace2, trendline_trace], layout={
        'xaxis_title': 'Age',
        'yaxis_title': 'BMI',
        'hovermode': 'closest',
        'legend': {'orientation': 'v', 'xanchor': 'right', 'yanchor': 'top', 'y': 1, 'x': 1}
    })

    #######################################
    #######HISTOGRAM PLOT CALLBACK#########
    #######################################
    
    # Check if a variable is selected
    if not selected_variable:
        # If not, use the default variables
        selected_variable = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    # Initialize an empty list to store histogram data
    data = []

    # Initialize the maximum selected value to 0
    max_selected = 0

    # Loop through each selected variable
    for var in selected_variable:   
        # Create a histogram with the filtered data for the current variable
        hist = go.Histogram(
            x=filtered_df[var],
            name=var,
            opacity=0.4,
            hoverinfo='text',
            hovertemplate='<b>%{y}</b><br>' + var.capitalize() + ': %{x:.2f}',
        )

        # Append the histogram to the data list
        data.append(hist)

        # Check if the maximum value for the current variable is greater than the maximum selected value
        max_value = filtered_df[var].max()
        if max_value > max_selected:
            max_selected = max_value

    # Set the maximum range for the histogram
    max_range = min(max_selected, 200)

    # Set the layout for the histogram plot
    layout = go.Layout(
        barmode='overlay',
        xaxis=dict(title='Value', range=[0, max_range]),
        yaxis=dict(title='Count'),
        hovermode='closest'
    )

    # Create a new figure with the histogram data and layout
    histogram_fig = go.Figure(data=data, layout=layout)

    # Update the layout for both the scatter and histogram plots
    scatter_fig.update_layout(transition_duration=400)
    histogram_fig.update_layout(transition_duration=400)

    # Return a tuple containing the scatter and histogram figures
    return scatter_fig, histogram_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)