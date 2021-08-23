import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime

import json
from distutils.util import strtobool

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output

from get_potential_savings import get_day_results
from extract_peaks import get_peak_days, get_peak_statistics
from risk_profile import calculate_predictability

with open('data/project_params.json') as json_file:
    config = json.load(json_file)
project_params = config['project_params']
analysis_params = config['analysis_params']
analysis_params['weekday_only'] = strtobool(analysis_params['weekday_only'])
analysis_params['peak_hours_only'] = strtobool(analysis_params['peak_hours_only'])
analysis_params['peak_days'] = strtobool(analysis_params['peak_days'])
load_data = pd.read_csv(f"data/{project_params['data_file']}",
                        usecols=['datetime', 'net_load_after_pv', 'battery_power'],
                        parse_dates=['datetime'], index_col=['datetime'])
peak_days = get_peak_days(load_data=load_data,
                          peak_level=analysis_params['peak_level'],
                          weekday_only=analysis_params['weekday_only'],
                          peak_hours_only=analysis_params['peak_hours_only'],
                          peak_hours=(analysis_params['peak_hour_start'],
                                      analysis_params['peak_hour_end'])
                          )
with open(f"data/{project_params['tariff_name']}") as json_file:
    tariff_dict = json.load(json_file)

peak_distribution, fig_peak_dist = get_peak_statistics(load_data=load_data,
                                                       peak_level=analysis_params['peak_level'],
                                                       peak_days=analysis_params['peak_days'],
                                                       weekday_only=analysis_params['weekday_only'],
                                                       peak_hours_only=analysis_params['peak_hours_only'],
                                                       peak_hours=(analysis_params['peak_hour_start'],
                                                                   analysis_params['peak_hour_end']))

peak_days = get_peak_days(load_data=load_data,
                          peak_level=analysis_params['peak_level'],
                          weekday_only=analysis_params['weekday_only'],
                          peak_hours_only=analysis_params['peak_hours_only'],
                          peak_hours=(analysis_params['peak_hour_start'],
                                      analysis_params['peak_hour_end'])
                          )
peak_data = load_data.loc[np.isin(load_data.index.date, peak_days)]
cor_results, fig_hist, fig_peak = \
    calculate_predictability(peak_data=peak_data, start_date=None, end_date=None,
                             weekday_only=analysis_params['weekday_only'])

flex_df_2cd_opt = []
flex_df_2cd_opt_figs = []
flex_df_optimization = []
flex_df_optimization_figs = []
for day in peak_days:
    flex_results_2cd_opt, fig_2cd_opt, fig_optimization, flex_results_optimization =\
        get_day_results(load_data=load_data, 
                        day=day,
                        project_params=project_params,
                        tariff_dict=tariff_dict)
    flex_df_2cd_opt.append(flex_results_2cd_opt)
    flex_df_2cd_opt_figs.append(fig_2cd_opt)
    flex_df_optimization.append(flex_results_optimization)
    flex_df_optimization_figs.append(fig_optimization)

flex_df_optimization = pd.DataFrame.from_dict(flex_df_optimization, orient='columns')
flex_df_2cd_opt = pd.DataFrame.from_dict(flex_df_2cd_opt, orient='columns')

flex_df_2cd_opt_figs_df = pd.DataFrame(index=range(len(flex_df_2cd_opt_figs)))
flex_df_2cd_opt_figs_df.loc[:, 'fig'] = 'a'
for i in range(len(flex_df_2cd_opt_figs)):
    flex_df_2cd_opt_figs_df.loc[i, 'fig'] = flex_df_2cd_opt_figs[i]
flex_df_2cd_opt_figs_df.index = pd.to_datetime(flex_df_2cd_opt['date'])

flex_df_optimization_figs_df = pd.DataFrame(index=range(len(flex_df_optimization_figs)))
flex_df_optimization_figs_df.loc[:, 'fig'] = 'a'
for i in range(len(flex_df_optimization_figs)):
    flex_df_optimization_figs_df.loc[i, 'fig'] = flex_df_optimization_figs[i]
flex_df_optimization_figs_df.index =  pd.to_datetime(flex_df_optimization['date'])

flex_df_optimization.to_csv(f"results/{project_params['project_name']}/optimization_{project_params['project_name']}_flexibility.csv")
flex_df_2cd_opt.to_csv(f"results/{project_params['project_name']}/2cd_opt_{project_params['project_name']}_flexibility.csv")



app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([

    html.H1('EI Pre Deployment Analysis', className="app-header"),

    html.Div([
        html.H2('Peak Event Profile - predictability peak', className="app-header")
    ]),
    dcc.Graph(
        id='peak-event-profile',
        figure=fig_peak
    ),

    html.Div([
        html.H2('Peak Hour Distribution', className="app-header")
    ]),

    dcc.Graph(
        id='peak-hour-dist',
        figure=fig_peak_dist
    ),

    html.Div([
        html.Div([
            html.H3('Select a peak day:')
        ],
        ),

        dcc.Dropdown(
            className="day-dropdown",
            id='day-select',
            options=[{'label': day, 'value': day} for day in peak_days],
            value=peak_days[0],
            clearable=False,
        )], className='dropdown'),

    html.Div([
        html.H2('2cd Opt controller results', className="app-header")
    ]),

    dcc.Graph(
        id='2cd-opt',
    ),

    html.Div([
        html.H2('Optimization results', className="app-header")
    ]),

    dcc.Graph(
        id='optimization',
    ),
    html.Div([
        html.H2('Summary of Flexibility Scores for all days for 2cd OPT', className="app-header")
    ]),
    dt.DataTable(
        id='2cd-opt-table',
        columns=[
            {'name': 'First_charge_risk', 'id': 'first_charge_risk'},
            {'name': 'First_discharge_risk', 'id': 'first_discharge_risk'},
            {'name': 'First_load_risk', 'id': 'first_load_risk'},
            {'name': 'Second_charge_risk', 'id': 'second_charge_risk'},
            {'name': 'Second_discharge_risk', 'id': 'second_discharge_risk'},
            {'name': 'Second_load_risk', 'id': 'second_load_risk'},
            {'name': 'Date', 'id': 'date'},
        ],
        data=flex_df_2cd_opt.to_dict(orient='records')
    ),
    html.Div([
        html.H2('Summary of Flexibility Scores for all days for Optimization', className="app-header")
    ]),
    dt.DataTable(
        id='optimization-table',
        columns=[
            {'name': 'First_charge_risk', 'id': 'first_charge_risk'},
            {'name': 'First_discharge_risk', 'id': 'first_discharge_risk'},
            {'name': 'First_load_risk', 'id': 'first_load_risk'},
            {'name': 'Second_charge_risk', 'id': 'second_charge_risk'},
            {'name': 'Second_discharge_risk', 'id': 'second_discharge_risk'},
            {'name': 'Second_load_risk', 'id': 'second_load_risk'},
            {'name': 'Date', 'id': 'date'},
        ],
        data=flex_df_optimization.to_dict(orient='records')
    ),


    html.Div([
        html.H2('Predictability Histogram', className="app-header")
    ]),
    dcc.Graph(
        id='predictability-profile',
        figure=fig_hist
    ),

])


# @app.callback([
#     Output('2cd-opt-table', 'data'),
#     Output('2cd-opt-table', 'columns'),
#     Output('2cd-opt', 'figure'),
#     Output('optimization', 'figure'),
#     Output('optimization-table', 'data'),
#     Output('optimization-table', 'columns')
#     ],
#     [Input('day-select', 'value')]
# )
# def update_potential(day):
#     date_obj = datetime.strptime(day, '%Y-%m-%d')
#     flex_results_2cd_opt, fig_2cd_opt, fig_optimization, flex_results_optimization = \
#         get_day_results(load_data=load_data, day=date_obj.date(), project_params=project_params,
#                         tariff_dict=tariff_dict)
    
#     flex_results_2cd_opt = pd.DataFrame([flex_results_2cd_opt])
#     columns_2cd_opt = [{"name": i.capitalize(), "id": i} for i in flex_results_2cd_opt.keys()]
#     flex_results_optimization = pd.DataFrame([flex_results_optimization])
#     columns_optimization = [{"name": i.capitalize(), "id": i} for i in flex_results_optimization.keys()]

#     return flex_results_2cd_opt.to_dict(orient='records'), columns_2cd_opt, fig_2cd_opt, fig_optimization, flex_results_optimization.to_dict(orient='records'), columns_optimization

@app.callback([
    Output('2cd-opt', 'figure'),
    Output('optimization', 'figure'),
    ],
    [Input('day-select', 'value')]
)
def update_potential(day):
    date_obj = datetime.strptime(day, '%Y-%m-%d')
    fig_2cd_opt = flex_df_2cd_opt_figs_df.loc[flex_df_2cd_opt_figs_df.index == date_obj, 'fig'][0]
    fig_optimization = flex_df_optimization_figs_df.loc[flex_df_optimization_figs_df.index == date_obj, 'fig'][0]
    return fig_2cd_opt, fig_optimization

if __name__ == '__main__':
    app.run_server(debug=True)


