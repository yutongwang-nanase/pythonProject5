import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

# read the data from the 'megnet_kvrh.csv' and 'megnet_dielectric.csv' files
df_C = pd.read_csv('megnet_kvrh.csv')
df_D = pd.read_csv('megnet_dielectric.csv')

# define the app
app = dash.Dash(__name__)

# define the layout of the app
app.layout = html.Div(children=[
    html.H1(children='Box Plot'),
    dcc.Dropdown(
        id='data-dropdown',
        options=[
            {'label': 'KVRH Data', 'value': 'C'},
            {'label': 'Dielectric Data', 'value': 'D'}
        ],
        value='C'
    ),
    dcc.Graph(
        id='box-plot',
        figure={
            'data': [
                {'y': df_C[col], 'type': 'box', 'name': col} for col in ['mae', 'mape', 'rmse']
            ],
            'layout': {
                'title': 'Box Plot of mae, mape, and rmse values',
                'yaxis': {'title': 'Values'}
            }
        }
    ),
    html.Div(
        id='data-description',
        children=[
            html.P(id='kvrh-description', children='KVRH is a measure of a material\'s resistance to indentation'),
            html.P(id='dielectric-description', children='Dielectric refers to the ability of a material to store electrical energy')
        ]
    )
])

# define the callback function for the dropdown menu
@app.callback(
    dash.dependencies.Output('box-plot', 'figure'),
    [dash.dependencies.Input('data-dropdown', 'value')]
)
def update_figure(selected_data):
    if selected_data == 'C':
        data = df_C
    else:
        data = df_D
    return {
        'data': [
            {'y': data[col], 'type': 'box', 'name': col} for col in ['mae', 'mape', 'rmse']
        ],
        'layout': {
            'title': 'Box Plot of mae, mape, and rmse values for ' + selected_data + ' Data',
            'yaxis': {'title': 'Values'}
        }
    }

# define the callback function for the data descriptions
@app.callback(
    dash.dependencies.Output('kvrh-description', 'style'),
    dash.dependencies.Output('dielectric-description', 'style'),
    [dash.dependencies.Input('data-dropdown', 'value')]
)
def update_data_description(selected_data):
    if selected_data == 'C':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

# run the app
if __name__ == '__main__':
    app.run_server(debug=True)
