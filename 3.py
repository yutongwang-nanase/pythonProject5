import dash
import dash_html_components as html

app = dash.Dash(__name__)

app.layout = html.Iframe(
    src='assets/index.html',
    style={'width': '1200px', 'height': '1200px', 'border': 'none'}
)

if __name__ == '__main__':
    app.run_server(debug=True)
