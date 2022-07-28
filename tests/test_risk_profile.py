import pandas as pd
import datetime

from risk_profile import calculate_predictability, calculate_flexibility

data = pd.read_csv("data/xd_0801_0324_1min.csv")
data["datetime"] = pd.to_datetime(data["datetime"])
data = data.set_index("datetime")
data["battery_energy"] = 0
data["net_load_after_storage"] = data["net_load_after_pv"]

flex_results, fig = \
    calculate_flexibility(daily_load=data, date=datetime.date(2021, 1, 15),
                          demand_target=450, storage_size_kWh=200)
fig.show()

cor_results, fig_hist, fig_peak = \
    calculate_predictability(peak_data=data, start_date=datetime.date(2021, 3, 17), end_date=datetime.date(2021, 3, 24),
                             weekday_only=True)
fig_hist.show()
fig_peak.show()
