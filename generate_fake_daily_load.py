import pandas as pd
import numpy as np
import datetime
from plotly import graph_objs as go


def generate_fake_data(daily_load, peak_hours=(7, 18)):
    # Calculate standard deviation
    non_peak_morning = daily_load[daily_load.index.hour < peak_hours[0]]
    non_peak_night = daily_load[daily_load.index.hour > peak_hours[1]]
    peak = daily_load[(daily_load.index.hour <= peak_hours[1]) &
                      (daily_load.index.hour >= peak_hours[0])]
    non_peak_morning_std = np.std(non_peak_morning["net_load_after_pv"])
    non_peak_night_std = np.std(non_peak_night["net_load_after_pv"])
    peak_std = np.std(peak["net_load_after_pv"])

    # Apply standard deviation to peak hours and non-peak hours
    fake_daily_load = daily_load.copy(deep=True)
    error_morning = np.random.normal(0, non_peak_morning_std, non_peak_morning.shape[0])
    error_night = np.random.normal(0, non_peak_night_std, non_peak_night.shape[0])
    error_peak = np.random.normal(0, peak_std, peak.shape[0])
    fake_daily_load["error"] = np.concatenate([error_morning, error_peak, error_night])
    fake_daily_load["net_load_after_pv_fake"] = fake_daily_load["error"] + fake_daily_load["net_load_after_pv"]

    return fake_daily_load
