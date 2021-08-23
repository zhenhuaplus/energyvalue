import pandas as pd
import datetime
from plotly import graph_objs as go

from generate_fake_daily_load import generate_fake_data

data = pd.read_csv("data/production_inputs/cz_1day_15min_no_interp.csv")
data["datetime"] = pd.to_datetime(data["datetime"])
data = data.set_index("datetime")
data["battery_energy"] = 0
data["net_load_after_storage"] = data["net_load_after_pv"]

fake_data = generate_fake_data(daily_load=data, peak_hours=(7, 18))
fig = go.Figure()
fig.add_trace(go.Scatter(x=fake_data.index, y=fake_data["net_load_after_pv"], name="original load"))
fig.add_trace(go.Scatter(x=fake_data.index, y=fake_data["net_load_after_pv_fake"], name="fake load"))
fig.show()
