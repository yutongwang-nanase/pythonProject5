import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd

# 读取数据集
df_A = pd.read_csv('cgcnn_log_kvrh.csv')
df_B = pd.read_csv('cgcnn_log_kvrh.csv')
df_C = pd.read_csv('test_results.csv')
df_D = pd.read_csv('test_results.csv')

# 创建Dash应用程序
app = dash.Dash(__name__)

# 创建布局
app.layout = html.Div(children=[
    html.H1(children='我的 Dash 应用程序'),

    # 创建折线图和柱状图的容器
    html.Div([
        dcc.Dropdown(
            id='dropdown-A',
            options=[
                {'label': '数据集A', 'value': 'A'},
                {'label': '数据集B', 'value': 'B'}
            ],
            value='A'
        ),
        dcc.Graph(id='line-chart')
    ]),

    html.Div([
        dcc.Dropdown(
            id='dropdown-C',
            options=[
                {'label': '数据集C', 'value': 'C'},
                {'label': '数据集D', 'value': 'D'}
            ],
            value='C'
        ),
        dcc.Graph(id='bar-chart')
    ])
])

# 创建回调函数，根据下拉列表选项更新图形
@app.callback(
    dash.dependencies.Output('line-chart', 'figure'),
    [dash.dependencies.Input('dropdown-A', 'value')]
)
def update_line_chart(value):
    if value == 'A':
        df = df_A
    else:
        df = df_B

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines', name='折线图'))
    fig.update_layout(title='折线图')

    return fig

@app.callback(
    dash.dependencies.Output('bar-chart', 'figure'),
    [dash.dependencies.Input('dropdown-C', 'value')]
)
def update_bar_chart(value):
    if value == 'C':
        df = df_C
    else:
        df = df_D

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['x'], y=df['y'], name='柱状图'))
    fig.update_layout(title='柱状图')

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
