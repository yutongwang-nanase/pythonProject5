import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 创建Dash应用
app = dash.Dash(__name__)

# 生成数据
np.random.seed(42)
x = np.random.normal(0, 1, size=1000)
y = np.random.normal(0, 1, size=1000)

# 绘制图表
fig = px.scatter(x=x, y=y)

# 创建背景图片
bg_color = (255, 255, 255)
img_size = (600, 400)
img = Image.new("RGB", img_size, bg_color)

# 在背景图片上绘制文字
draw = ImageDraw.Draw(img)
font_size = 36
font = ImageFont.truetype("Arial.ttf", font_size)
text = "化学晶体预测"
text_size = draw.textsize(text, font)
text_pos = ((img_size[0] - text_size[0]) // 2, (img_size[1] - text_size[1]) // 2)
draw.text(text_pos, text, font=font, fill=(0, 0, 0))

# 将背景图片转换为Dash支持的格式
bg_image = img.tobytes()

# 定义页面布局
app.layout = html.Div(style={'background-image': 'url(data:image/png;base64,' + bg_image.decode() + ')',
                             'background-size': 'cover',
                             'height': '100vh'},
                      children=[
                          html.H1('动态页面示例'),
                          dcc.Graph(id='my-graph', figure=fig),
                          dcc.Interval(id='update-interval', interval=1000, n_intervals=0)
                      ])

# 定义更新回调函数
@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    dash.dependencies.Input('update-interval', 'n_intervals')
)
def update_graph(n):
    # 更新数据
    x_new = np.random.normal(0, 1, size=1000)
    y_new = np.random.normal(0, 1, size=1000)
    fig_new = px.scatter(x=x_new, y=y_new)
    return fig_new

if __name__ == '__main__':
    app.run_server(debug=True)
