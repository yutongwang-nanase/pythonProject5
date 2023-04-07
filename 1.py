import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# 读取数据
df_A = pd.read_csv('cgcnn_log_kvrh.csv')
df_B = pd.read_csv('test_results.csv')

# 创建 Dash 应用
app = dash.Dash(__name__)

# 布局
app.layout = html.Div([
    html.H1("项目 A 和项目 B 的曲线"),
    # 选择项目的 Dropdown 组件
    dcc.Dropdown(
        id='project-dropdown',
        options=[
            {'label': '项目A', 'value': 'A'},
            {'label': '项目B', 'value': 'B'}
        ],
        value='A'  # 默认选择项目A
    ),
    # 折线图
    dcc.Graph(id='line-chart')
])


# 回调函数：根据选择的项目绘制折线图
@app.callback(
    Output('line-chart', 'figure'),
    Input('project-dropdown', 'value')
)
def update_chart(selected_project):
    if selected_project == 'A':
        fig = px.line(df_A, x=df_A.index, y=['TRUE', 'predict'], title='项目A')
    else:
        fig = px.line(df_B, x=df_B.index, y=['TRUE', 'predict'], title='项目B')
    return fig


# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8082)
