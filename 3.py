import dash
import dash_html_components as html
import dash_table
import pandas as pd

# Create a sample dataframe
df = pd.DataFrame({
    'Task name': ['matbench_steels', 'matbench_jdft2d', 'matbench_phonons', 'matbench_expt_gap', 'matbench_dielectric',
                  'matbench_expt_is_metal', 'matbench_glass', 'matbench_log_gvrh', 'matbench_log_kvrh',
                  'matbench_perovskites', 'matbench_mp_gap', 'matbench_mp_is_metal', 'mat_mp_e_form'],
    'Task type/input': ['regression/composition', 'regression/structure', 'regression/structure',
                        'regression/composition', 'regression/structure', 'classification/composition',
                        'classification/composition', 'regression/structure', 'regression/structure',
                        'regression/structure', 'regression/structure', 'classification/structure',
                        'regression/structure'],
    'Target column (unit)': ['yield strength (MPa)', 'exfoliation_en (meV/atom)', 'last phdos peak (cm^-1)',
                             'gap expt (eV)', 'n (unitless)', 'is_metal', 'gfa', 'log10(G_VRH (log10(GPa))',
                             'log10(K_VRH) (log10(GPa))', 'e_form (eV/unit cell)', 'gap pbe (eV)', 'is_metal',
                             'e_form (eV/atom)'],
    'Samples': [312, 636, 126, 4604, 4764, 4921, 5680, 10987, 10987, 18928, 106113, 106113, 132752],
    'MAD (regression) or Fraction True (classification)': [229.3743, 67.202, 323.7870, 1.1432, 0.8085, 0.4981, 0.7104,
                                                           0.2931, 0.2897, 0.5660, 1.3271, 0.4349, 1.0059],
    '': ['download, interactive', 'download, interactive', 'download, interactive', 'download, interactive',
         'download, interactive', 'download, interactive', 'download, interactive', 'download, interactive',
         'download, interactive', 'download, interactive', 'download, interactive', 'download, interactive',
         'download, interactive'],
    'Submissions': [9, 14, 14, 11, 14, 6, 6, 14, 14, 14, 14, 11, 16],
    'Task description': ['Predict the yield strength of steel alloys based on their composition',
                         'Predict the exfoliation energy of 2D materials based on their structure',
                         'Predict the frequency of the last peak in the phonon density of states of materials based on their structure',
                         'Predict the experimental band gap of inorganic compounds based on their composition',
                         'Predict the refractive index of materials based on their structure',
                         'Classify inorganic compounds as metals or non-metals based on their composition',
                         'Classify inorganic compounds as glass formers or glass modifiers based on their composition',
                         'Predict the shear modulus of materials based on their structure',
                         'Predict the bulk modulus of materials based on their structure',
                         'Predict the formation energy of perovskite materials based on their structure',
                         'Predict the band gap of inorganic compounds based on their structure',
                         'Classify inorganic compounds as metals or non-metals based on their structure',
                         'Predict the formation energy per atom of inorganic compounds based on their structure'],
    'Task category': ['Materials science', 'Materials science', 'Materials science', 'Materials science',
                      'Materials science', 'Materials science', 'Materials science', 'Materials science',
                      'Materials science', 'Materials science', 'Materials science', 'Materials science',
                      'Materials science'],
    'Task difficulty': ['Intermediate', 'Advanced', 'Advanced', 'Intermediate', 'Intermediate', 'Beginner', 'Beginner',
                        'Advanced', 'Advanced', 'Advanced', 'Advanced', 'Beginner', 'Advanced']
})

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        dash_table.DataTable(
            id='table1',
            columns=[
                {'name': 'Task name', 'id': 'Task name'},
                {'name': 'Task type/input', 'id': 'Task type/input'},
                {'name': 'Target column (unit)', 'id': 'Target column (unit)'},
                {'name': 'Samples', 'id': 'Samples'}
            ],
            data=df.to_dict('records'),
            style_header={
                'backgroundColor': '#2c3e50',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'border': '1px solid white'
            },
            style_cell={
                'backgroundColor': '#34495e',
                'color': 'white',
                'textAlign': 'center',
                'border': '1px solid white'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#2c3e50'
                },
                {
                    'if': {'column_id': 'Task difficulty'},
                    'backgroundColor': '#16a085',
                    'color': 'white'
                },
                {
                    'if': {'column_id': 'Task category'},
                    'backgroundColor': '#8e44ad',
                    'color': 'white'
                }
            ]
        )
    ]),
    html.Br(),
    html.Div([
        dash_table.DataTable(
            id='table2',
            columns=[
                {'name': 'MAD (regression) or Fraction True (classification)',
                 'id': 'MAD (regression) or Fraction True (classification)'},
                {'name': 'Submissions', 'id': 'Submissions'},
                {'name': 'Task description', 'id': 'Task description'},
                {'name': 'Task category', 'id': 'Task category'},
                {'name': 'Task difficulty', 'id': 'Task difficulty'}
            ],
            data=df.to_dict('records'),
            style_header={
                'backgroundColor': '#2c3e50',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'border': '1px solid white'
            },
            style_cell={
                'backgroundColor': '#34495e',
                'color': 'white',
                'textAlign': 'center',
                'border': '1px solid white'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#2c3e50'
                },
                {
                    'if': {'column_id': 'Task difficulty'},
                    'backgroundColor': '#16a085',
                    'color': 'white'
                },
                {
                    'if': {'column_id': 'Task category'},
                    'backgroundColor': '#8e44ad',
                    'color': 'white'
                },
                {
                    'if': {'state': 'active'},
                    'backgroundColor': 'inherit !important',
                    'border': 'inherit !important'
                }
            ]
        )
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
