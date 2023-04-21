import dash
import plotly.express as px
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# 读取数据
df = pd.read_csv('picture2/cgcnn_e_form.csv')

# 创建Dash应用
app = dash.Dash(__name__)

# 构建UI界面
app.layout = html.Div([
    html.H1("CGCNN 真实值与预测值对比图"),
    dcc.Dropdown(
        id='project-dropdown',
        options=[
            {'label': 'e_form', 'value': 'e_form'},
            {'label': 'gap', 'value': 'gap'},
            {'label': 'gvrh', 'value': 'gvrh'},
        ],
        value='e_form'
    ),
    dcc.Graph(id='scatter-plot')
])


# 定义回调函数
@app.callback(Output('scatter-plot', 'figure'),
              Input('project-dropdown', 'value'))
def update_scatter_plot(selected_project):
    # 根据下拉列表选择更新数据
    if selected_project == 'e_form':
        df = pd.read_csv('picture2/cgcnn_e_form.csv')
    elif selected_project == 'gap':
        df = pd.read_csv('picture2/cgcnn_gap.csv')
    elif selected_project == 'gvrh':
        df = pd.read_csv('picture2/cgcnn_gvrh.csv')

    # 创建散点图
    fig = px.scatter(df, x="TRUE", y="predict", color="AE", hover_data=["id"])

    # 添加布局信息
    fig.update_layout(title=f"{selected_project.upper()}真实值与预测值对比图",
                      xaxis_title="真实值",
                      yaxis_title="预测值",
                      coloraxis_colorbar=dict(title="AE"))

    return fig


# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True)
