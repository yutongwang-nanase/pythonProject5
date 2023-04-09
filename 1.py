import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)

# 创建图表数据
x = [1, 2, 3, 4, 5]
y = [1, 3, 2, 4, 3]

# 创建页面布局
app.layout = html.Div([
    # 左侧导航栏
    html.Div([
        html.H3('导航'),
        html.Ul([
            html.Li(html.A('图表1', href='#chart1')),
            html.Li(html.A('图表2', href='#chart2')),
            html.Li(html.A('图表3', href='#chart3'))
        ])
    ], style={'position': 'fixed', 'width': '20%'}),

    # 右侧容器
    html.Div([
        dcc.Graph(id='chart1', figure={'data': [{'x': x, 'y': y, 'type': 'line'}]}),
        dcc.Graph(id='chart2', figure={'data': [{'x': x, 'y': y, 'type': 'line'}]}),
        dcc.Graph(id='chart3', figure={'data': [{'x': x, 'y': y, 'type': 'line'}]})
    ], style={'width': '75%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '20%'})
])

# 运行应用程序
if __name__ == '__main__':
    app.run_server(debug=True)
