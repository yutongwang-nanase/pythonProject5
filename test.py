import plotly.graph_objs as go
import plotly.subplots as sp
import dash
import dash_core_components as dcc
import dash_html_components as html

num_labels = ['<1k', '1k-10k', '10k-100k', '>=100k']
num_values = [2, 5, 3, 3]
num_colors = ['#7B68EE', '#6A5ACD', '#483D8B', '#2c2656']

app_colors = ['#FFA07A', '#FF7F50', '#FF6347', '#FF4500', '#FF8C00']
app_labels = ['stability', 'electronic', 'mechanical', 'optical', 'thermal']
app_values = [4, 4, 3, 1, 1]

type_colors = ['#4169E1', '#00FFFF']
type_labels = ['regression', 'classifacation']
type_values = [3, 10]

data_colors = ['#8B4513', '#20B2AA']
data_labels = ['DTF', 'experiment']
data_values = [9, 4]

# Create figure
fig = sp.make_subplots(rows=2, cols=2,
                       specs=[[{'type': 'domain'}, {'type': 'domain'}],
                              [{'type': 'domain'}, {'type': 'domain'}]])

# Add subplots
fig.add_trace(go.Pie(labels=num_labels, values=num_values, marker=dict(colors=num_colors)),
              row=1, col=1)
fig.add_trace(go.Pie(labels=app_labels, values=app_values, marker=dict(colors=app_colors)),
              row=1, col=2)
fig.add_trace(go.Pie(labels=type_labels, values=type_values, marker=dict(colors=type_colors)),
              row=2, col=1)
fig.add_trace(go.Pie(labels=data_labels, values=data_values, marker=dict(colors=data_colors)),
              row=2, col=2)

# Set subplot titles
fig.update_layout(title='Summary of MatBench Datasets')

# Set subplot sizes and spacing
fig.update_layout(height=800, width=800,
                  margin=dict(l=20, r=20, t=50, b=20),
                  grid=dict(rows=2, columns=2, pattern='independent'),
                  showlegend=False,
                  font=dict(size=16),
                  annotations=[dict(font=dict(size=20), showarrow=False,
                                    text='Data Size'),
                               dict(font=dict(size=20), showarrow=False,
                                    text='Task Categories'),
                               dict(font=dict(size=20), showarrow=False,
                                    text='Task Types'),
                               dict(font=dict(size=20), showarrow=False,
                                    text='Data Sources')])

fig.update_traces(hole=.4, hoverinfo="label+percent+name")

# Run the app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
