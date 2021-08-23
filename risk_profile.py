import pandas as pd
import numpy as np
import datetime
from plotly import graph_objs as go
from sklearn.metrics import pairwise_distances


def calculate_flexibility(daily_load, date, demand_target, storage_size_kWh):
    # Make sure data are in the right format
    input_resolution = int((daily_load.index[1] - daily_load.index[0]).total_seconds() / 60)
    daily_load["power_delta"] = demand_target - daily_load["net_load_after_storage"]
    daily_load["power_delta"] = daily_load["power_delta"].clip(lower=0)

    # Select subset of daily load
    daily_load = daily_load[daily_load.index.date == date]
    hourly_energy_delta = daily_load.groupby(daily_load.index.hour)["power_delta"].sum() * input_resolution / 60
    hourly_load_delta = daily_load.groupby(daily_load.index.hour)["net_load_after_pv"].sum() * input_resolution / 60

    # Calculate risk profiles
    first_charge_risk = round(hourly_energy_delta[hourly_energy_delta.index <= 7].sum() / storage_size_kWh, 3)
    first_discharge_risk = round(hourly_energy_delta[(hourly_energy_delta.index >= 8) &
                                                     (hourly_energy_delta.index <= 11)].sum() / storage_size_kWh, 3)
    first_load_risk = round(hourly_load_delta[(hourly_load_delta.index >= 8) &
                                              (hourly_load_delta.index <= 11)].sum() / storage_size_kWh, 3)
    second_charge_risk = round(hourly_energy_delta[(hourly_energy_delta.index >= 12) &
                                                   (hourly_energy_delta.index <= 16)].sum() / storage_size_kWh, 3)
    second_discharge_risk = round(hourly_energy_delta[hourly_energy_delta.index >= 17].sum() / storage_size_kWh, 3)
    second_load_risk = round(hourly_load_delta[hourly_load_delta.index >= 17].sum() / storage_size_kWh, 3)

    flex_results = {
        "first_charge_risk": first_charge_risk,
        "first_discharge_risk": first_discharge_risk,
        "first_load_risk": first_load_risk,
        "second_charge_risk": second_charge_risk,
        "second_discharge_risk": second_discharge_risk,
        "second_load_risk": second_load_risk
    }

    fig = go.Figure()
    fig.add_vrect(x0=datetime.datetime.combine(date, datetime.time(0, 0)),
                  x1=datetime.datetime.combine(date, datetime.time(8, 0)),
                  line_width=0, fillcolor="red", opacity=0.1,
                  annotation_text="1st充电后多余空间: {}".format(first_charge_risk), annotation=dict(font_size=20),
                  annotation_position="top left")
    fig.add_vrect(x0=datetime.datetime.combine(date, datetime.time(8, 0)),
                  x1=datetime.datetime.combine(date, datetime.time(17, 0)),
                  line_width=0, fillcolor="red", opacity=0,
                  annotation_text="1st放电后: {}".format(first_discharge_risk), annotation=dict(font_size=20),
                  annotation_position="top left")
    fig.add_vrect(x0=datetime.datetime.combine(date, datetime.time(8, 0)),
                  x1=datetime.datetime.combine(date, datetime.time(17, 0)),
                  line_width=0, fillcolor="red", opacity=0,
                  annotation_text="1st自消纳剩余: {}".format(first_load_risk), annotation=dict(font_size=20),
                  annotation_position="bottom left")
    fig.add_vrect(x0=datetime.datetime.combine(date, datetime.time(12, 0)),
                  x1=datetime.datetime.combine(date, datetime.time(17, 0)),
                  line_width=0, fillcolor="red", opacity=0.1,
                  annotation_text="2nd充电后多余空间: {}".format(second_charge_risk), annotation=dict(font_size=20),
                  annotation_position="top left")
    fig.add_vrect(x0=datetime.datetime.combine(date, datetime.time(17, 0)),
                  x1=datetime.datetime.combine(date, datetime.time(21, 0)),
                  line_width=0, fillcolor="red", opacity=0,
                  annotation_text="2nd放电后: {}".format(second_discharge_risk), annotation=dict(font_size=20),
                  annotation_position="top left")
    fig.add_vrect(x0=datetime.datetime.combine(date, datetime.time(17, 0)),
                  x1=datetime.datetime.combine(date, datetime.time(21, 0)),
                  line_width=0, fillcolor="red", opacity=0,
                  annotation_text="2nd自消纳剩余: {}".format(second_load_risk), annotation=dict(font_size=20),
                  annotation_position="bottom left")
    fig.add_trace(go.Scatter(x=daily_load.index, y=daily_load['net_load_after_pv'], mode='lines',
                             name='net_load_after_pv'))
    fig.add_trace(go.Scatter(x=daily_load.index, y=daily_load["battery_power"], mode='lines',
                             name='battery_power'))
    fig.add_trace(go.Scatter(x=daily_load.index, y=daily_load["battery_energy"], mode='lines',
                             name='battery_energy'))
    fig.add_trace(go.Scatter(x=daily_load.index, y=daily_load["net_load_after_storage"], mode='lines',
                             name='net_load_after_storage'))
    fig.add_trace(go.Scatter(x=daily_load.index, y=np.array([demand_target] * len(daily_load["battery_power"])),
                             mode='lines', fill='tonexty', showlegend=False, fillcolor='rgba(184, 247, 212, 0.5)'))
    fig.update_layout(title="% of battery size for demand violation & for battery power self-digestion")

    return flex_results, fig


def calculate_predictability(peak_data, start_date=None, end_date=None, weekday_only=True, peak_hours_only=False, peak_hours=(7,18)):
    # Make sure data are in the right format
    data = peak_data.copy(deep=True)

    # Only consider workdays
    if weekday_only:
        data = data[data.index.weekday <= 4]
    # Only consider 8-5pm?
    
    if peak_hours_only:
        data = data[(data.index.hour <= peak_hours[1]) &\
                    (data.index.hour >= peak_hours[0])]
    
    # Only consider data within a date range
    if start_date != None and end_date != None:
        data = data[(data.index.date <= end_date) & (data.index.date >= start_date)]
    unique_dates = set(data.index.date)

    values_list = []
    for date in unique_dates:
        values = list(data[data.index.date == date]["net_load_after_pv"])
        values_list.append(values)

    # Obtain correlation results
    # distance_matrix = pairwise_distances(values_list, metric='correlation')
    cor_matrix = np.corrcoef(values_list)
    cor_matrix_update = cor_matrix[~np.eye(cor_matrix.shape[0], dtype=bool)].reshape(cor_matrix.shape[0], -1)
    cor_results = {
        "mean": cor_matrix_update.mean(),
        "median": np.median(cor_matrix_update),
        "max": cor_matrix_update.max(),
        "min": cor_matrix_update.min()
    }

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=np.reshape(cor_matrix_update, -1),
                                    xbins=dict(start=-1.0, end=1.0, size=0.05)))
    fig_hist.update_xaxes(range=[-1, 1])
    fig_hist.add_vrect(x0=-1, x1=0.7, line_width=0, fillcolor="red", opacity=0.1,
                       annotation_text="Poor correlation", annotation=dict(font_size=20),
                       annotation_position="top left")
    fig_hist.update_layout(title=str(cor_results))

    fig_peak = go.Figure()
    for date in unique_dates:
        values = data[data.index.date == date]
        fig_peak.add_trace(go.Scatter(x=values.index.time, y=values['net_load_after_pv'], mode='lines',
                                      name=str(date)))

    return cor_results, fig_hist, fig_peak
