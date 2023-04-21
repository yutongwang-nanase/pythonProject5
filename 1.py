import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)

# 创建图表数据
x = [1, 2, 3, 4, 5]
y = [1, 3, 2, 4, 3]

# 创建页面布局
app.layout = html.Div([
    # 页头
    html.Div([
        # 在页头左侧添加动态图标
        html.Iframe(
            src="https://giphy.com/embed/uljidatzCuEs7yo1KV",
            style={"height": "150px", "width": "150px", "border": "none", "pointer-events": "none",
                   "margin-left": "10%"}
        ),

        # 在页头中间添加标题
        html.H1(
            '我的页面',
            style={'color': 'white', 'background-color': '#0074D9', 'border-radius': '10px', 'padding': '20px'}
        ),
        # 在页头右侧添加静态图标，点击跳转到 GitHub 页面
        html.A(
            html.Img(
                src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
                style={"height": "50px", "width": "50px", "margin-left": "20px"}
            ),
            href="https://github.com/",
            target="_blank"
        )
    ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'padding-top': '50px'}),

    # 左侧导航栏
    html.Div([
        html.H3('导航'),
        html.Ul([
            html.Li(html.A('图表1', href='#chart1')),
            html.Li(html.A('图表2', href='#chart2')),
            html.Li(html.A('图表3', href='#chart3'))
        ])
    ], style={'position': 'fixed', 'width': '20%', 'padding-top': '100px'}),

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
