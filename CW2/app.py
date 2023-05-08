# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
# Incorporate data
df = pd.read_csv('diabetes.csv')

# Create the Dash app
app = Dash(__name__)


# Create the Plotly figure
fig = px.scatter(df, x='Age', y='BMI', color='Outcome', trendline='ols',trendline_scope="overall", trendline_color_override="black")

# Define the layout
app.layout = html.Div([
    html.H1('test'),
    html.Br(),
    dcc.RadioItems(
        id='output-selector',
        options=[
            {'label': 'Only Diabetic', 'value': 1},
            {'label': 'Only Non-Diabetic', 'value': 0},
            {'label': 'Both', 'value': 'both'}
        ],
        value='both',
        inline=True
    ),
    dcc.Graph(id='scatter-plot', figure=fig),
    html.Br(),
])

# Define the callback function
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('output-selector', 'value')
)
def update_figure(selected_output):
    if selected_output == 'both':
        filtered_df = df
    else:
        filtered_df = df[df['Outcome'] == selected_output]

    # Create trace for non-diabetic patients
    trace0 = go.Scatter(
        x=filtered_df[filtered_df['Outcome'] == 0]['Age'],
        y=filtered_df[filtered_df['Outcome'] == 0]['BMI'],
        mode='markers',
        marker=dict(color='blue'),
        name='Non-Diabetic (output=0)',
        hovertemplate='<br>'.join([
            'Age: %{x}',
            'BMI: %{y}',
            'Outcome: Non-Diabetic (output=0)'
        ]),
        legendgroup='Non-Diabetic (output=0)',
        legendrank=1
    )

    # Create trace for diabetic patients
    trace1 = go.Scatter(
        x=filtered_df[filtered_df['Outcome'] == 1]['Age'],
        y=filtered_df[filtered_df['Outcome'] == 1]['BMI'],
        mode='markers',
        marker=dict(color='red'),
        name='Diabetic (output=1)',
        hovertemplate='<br>'.join([
            'Age: %{x}',
            'BMI: %{y}',
            'Outcome: Diabetic (output=1)'
        ]),
        legendgroup='Diabetic (output=1)',
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
    fig = go.Figure(data=[trace0, trace1, trendline_trace], layout={
        'xaxis_title': 'Age',
        'yaxis_title': 'BMI',
        'hovermode': 'closest',
        'legend': {'orientation': 'v', 'xanchor': 'right', 'yanchor': 'top', 'y': 1, 'x': 1}
    })

    #add update transition
    fig.update_layout(transition_duration=400)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)





'''
# Define the layout
app.layout = html.Div([
    
    dcc.RadioItems(
        id='output-selector',
        options=[
            {'label': 'Only Diabetic (output=1)', 'value': 1},
            {'label': 'Only Non-Diabetic (output=0)', 'value': 0},
            {'label': 'Both (output=0 and output=1)', 'value': 'both'}
        ],
        value='both'
    ),
    dcc.Graph(id='scatter-plot', figure=fig),
])

# Define the callback function
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('output-selector', 'value')
)
def update_figure(selected_output):
    if selected_output == 'both':
        filtered_df = df
    else:
        filtered_df = df[df['Outcome'] == selected_output]
    
    # Update the figure with the filtered data
    fig = px.scatter(filtered_df, x='Age', y='BMI', color='Outcome', trendline='ols', color_discrete_sequence=['blue', 'red'])
    fig.update_layout(transition_duration=500)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
'''