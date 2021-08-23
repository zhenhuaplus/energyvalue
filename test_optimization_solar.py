import os
import pandas as pd
import numpy as np
from plotly import graph_objs as go
import json
import datetime

from optimization_solar import run_optimization

# working_directory = os.path.dirname(os.path.abspath(__file__))

# Specify inputs
pred_start = datetime.datetime(2021, 1, 15)
pred_end = pred_start + datetime.timedelta(days=1)
project_name = 'hs'

# Read data
data = pd.read_csv("data/solar_test.csv")
data['datetime'] = pd.to_datetime(data['datetime'])
with open("data/jiangsu_202101_1_10kV.json") as json_file:
    tariff_dict = json.load(json_file)

# Run controller
battery_charge, battery_discharge, battery_energy, net_load_after_pv, net_load_after_storage = run_optimization(
    daily_load=data,
    battery_size_kWh=200,
    battery_power_kW=50,
    min_soc=0.00,
    max_soc=1.00,
    current_soc=0.00,
    one_way_efficiency=1.00,
    tariff_dict=tariff_dict,
    solar_to_battery_allowed=True)
data["battery_charge"] = battery_charge
data["battery_discharge"] = battery_discharge
data["battery_energy"] = battery_energy
data["net_load_after_pv"] = net_load_after_pv
data["net_load_after_storage"] = net_load_after_storage
data.to_csv("/Users/zhenhua/Desktop/solar_result.csv")

# Output plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['datetime'],
                         y=data['pv'], mode='lines', name='pv'))
fig.add_trace(go.Scatter(
    x=data['datetime'], y=data['net_load_before_pv'], mode='lines', name='net_load_before_pv'))
fig.add_trace(go.Scatter(
    x=data['datetime'], y=net_load_after_pv, mode='lines', name='net_load_after_pv'))
fig.add_trace(go.Scatter(
    x=data['datetime'], y=battery_charge, mode='lines', name='battery_charge'))
fig.add_trace(go.Scatter(
    x=data['datetime'], y=battery_discharge, mode='lines', name='battery_discharge'))
fig.add_trace(go.Scatter(
    x=data['datetime'], y=battery_energy, mode='lines', name='battery_energy'))
fig.add_trace(go.Scatter(
    x=data['datetime'], y=net_load_after_storage, mode='lines', name='net_load_after_storage'))
fig.show()
