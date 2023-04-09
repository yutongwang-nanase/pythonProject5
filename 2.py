import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output

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

df_C = pd.read_csv('megnet_kvrh.csv')
df_D = pd.read_csv('megnet_dielectric.csv')


markdown_text = '''
## MAPE和RMSE

MAPE指的是平均绝对百分比误差（Mean Absolute Percentage Error），它是预测值与实际值之间的百分比误差的平均值。对于样本数据集D，MAPE可以表示为：

MAPE(D) = (1/n) * Σ | (预测值(i) - 实际值(i)) / 实际值(i) | * 100%

其中n是样本数据集D的大小。

MAPE可以反映模型预测误差相对于实际值的大小，具有良好的可解释性。但是，MAPE存在一些问题，例如在实际值接近于0时，MAPE可能会出现异常值。

相对地，RMSE指的是均方根误差（Root Mean Squared Error），它是预测值与实际值之间的平方误差的平均值的平方根。对于样本数据集D，RMSE可以表示为：

RMSE(D) = sqrt( (1/n) * Σ (预测值(i) - 实际值(i))^2 )

其中n是样本数据集D的大小。

RMSE可以反映模型预测误差的大小，并且能够更好地处理异常值。通常来说，RMSE越小，说明模型的预测越准确。
'''

# 创建Dash应用程序
app = dash.Dash(__name__)

# 定义布局
app.layout = html.Div(children=[

    html.Div([
        html.H1('导航'),
        html.A('跳转到图表1', href='#score'),
        html.Br(),
        html.A('跳转到图表2', href='#line'),
        html.Br(),
        html.A('跳转到图表3', href='#box-plot'),
        html.Br(),
    ], style={'position': 'fixed', 'width': '20%'}),

    # dbc.Nav([
    #     dbc.NavItem(dbc.NavLink("比赛成绩", href="score-graph")),
    #     dbc.NavItem(dbc.NavLink("CGCNN 预测值和真实值的曲线", href="line-chart")),
    #     dbc.NavItem(dbc.NavLink("Box Plot", href="box-plot")),
    #     # dbc.NavItem(dbc.NavLink("KVRH Data", href="#kvrh-description")),
    #     # dbc.NavItem(dbc.NavLink("Dielectric Data", href="#dielectric-description")),
    # ], vertical=True, pills=True),

    # html.Nav(className="sidebar",
    #          children=[
    #              html.H2("目录"),
    #              html.Ul(className="sidebar-items", children=[
    #                  html.Li(html.A("比赛成绩", href="#score-section")),
    #                  html.Li(html.A("CGCNN 预测值和真实值的曲线", href="#line-chart-section")),
    #                  html.Li(html.A("Box Plot", href="#box-plot-section"))
    #              ])
    #          ]),

    # html.Div(className="main-content", children=[
    html.Div([
        html.H1('比赛成绩', id='score'),
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
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.H1("CGCNN 预测值和真实值的曲线", id='line'),
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
        ]),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H1(children='Box Plot', id='box'),
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
                dcc.Markdown(
                    id='kvrh-description',
                    children='''**KVRH** is a measure of a material's resistance to indentation. It is an important material property for applications such as protective coatings, where the ability to withstand damage from impacts is important.'''
                ),
                dcc.Markdown(
                    id='dielectric-description',
                    children='''**Dielectric** refers to the ability of a material to store electrical energy. It is an important property for applications such as capacitors, where the ability to store electrical charge is important.'''
                )
            ]
        ),
        dcc.Markdown(children=markdown_text),
    ], style={'width': '75%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '20%'}),
    html.P('这是底部区域', style={'height': '300px','background-color': '#f2f2f2'}, )
])


# ])


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
        title=f"{selected_game}的MAE成绩对比",
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

        fig = px.line(df_A, x=df_A.index, y=['TRUE', 'predict', 'AE'], title='kvrh')
        fig.update_layout(
            barmode='group',
            title=f"MAE = {df_A.MAE[0]}",
        )

    else:
        fig = px.line(df_B, x=df_B.index, y=['TRUE', 'predict', 'AE'], title='jdft2d')
    return fig


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
            'title': 'Box Plot of mae, mape, and rmse values ',
            # for ' + selected_data + ' Data'
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


# 为 Nav 组件添加样式
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/dZVMbK.css'
})

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8082)
