import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
# 数据
data = {
    'Kvrh': {
        'CGCNN': 0.048,
        'DimeNet': 0.06458061,
        'MEGNet': 0.06817374082073885
    },
    'jdft2d': {
        'CGCNN': 30.101,
        'DimeNet': 57.779957,
        'MEGNet': 53.224194433218166
    }
}
df_A = pd.read_csv('cgcnn_log_kvrh.csv')
df_B = pd.read_csv('test_results.csv')
# 创建Dash应用程序
app = dash.Dash(__name__)

# 定义布局
app.layout = html.Div(children=[

    html.H1('比赛成绩'),
    dcc.Graph(
        id='score-graph',
        figure=go.Figure(
            data=[
                go.Bar(
                    x=list(data.keys()),
                    y=[data[k]['CGCNN'] for k in data.keys()],
                    text=['CGCNN', 'DimeNet', 'MEGNet'],
                    name='CGCNN'
                ),
                go.Bar(
                    x=list(data.keys()),
                    y=[data[k]['DimeNet'] for k in data.keys()],
                    text=['CGCNN', 'DimeNet', 'MEGNet'],
                    name='DimeNet'
                ),
                go.Bar(
                    x=list(data.keys()),
                    y=[data[k]['MEGNet'] for k in data.keys()],
                    text=['CGCNN', 'DimeNet', 'MEGNet'],
                    name='MEGNet'
                )
            ],
            layout=go.Layout(
                barmode='group',
                title='成绩对比',
                xaxis=dict(title='比赛项目'),
                yaxis=dict(title='MAE'),
                height=600,  # 设置高度
                width=300  # 设置宽度
            )
        )
    ),
    html.Div([
        dcc.RadioItems(
            id='select-game',
            options=[
                {'label': 'Kvrh', 'value': 'Kvrh'},
                {'label': 'jdft2d', 'value': 'jdft2d'}
            ],
            value='Kvrh'
        ),
        dcc.Checklist(
            id='select-player',
            options=[
                {'label': 'CGCNN', 'value': 'CGCNN'},
                {'label': 'DimeNet', 'value': 'DimeNet'},
                {'label': 'MEGNet', 'value': 'MEGNet'}
            ],
            value=['CGCNN', 'DimeNet', 'MEGNet']
        )
    ]),
    html.Div([
        html.H1("CGCNN 预测值和真实值的曲线"),
        # 选择项目的 Dropdown 组件
        dcc.Dropdown(
            id='project-dropdown',
            options=[
                {'label': 'kvrh', 'value': 'A'},
                {'label': 'jdft2d', 'value': 'B'}
            ],
            value='A'  # 默认选择项目A
        ),
        # 折线图
        dcc.Graph(id='line-chart')
    ])

])


# 回调函数，用于更新图表
@app.callback(
    Output('score-graph', 'figure'),
    Input('select-game', 'value'),
    Input('select-player', 'value'),
)
def update_graph(selected_game, selected_players):
    fig = go.Figure()
    for player in selected_players:
        fig.add_trace(go.Bar(
            x=[selected_game],
            y=[data[selected_game][player]],
            text=[player],
            name=player
        ))
    fig.update_layout(
        barmode='group',
        title=f"{selected_game}的成绩",
        xaxis=dict(title='比赛项目'),
        yaxis=dict(title='得分')
    )
    return fig

@app.callback(
    Output('line-chart', 'figure'),
    Input('project-dropdown', 'value')
)
def update_chart(selected_project):
    if selected_project == 'A':
        fig = px.line(df_A, x=df_A.index, y=['TRUE', 'predict'], title='kvrh')
    else:
        fig = px.line(df_B, x=df_B.index, y=['TRUE', 'predict'], title='jdft2d')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
