import pandas as pd
import numpy as np
import datetime
from plotly import graph_objs as go
import plotly.express as px

def get_peak_days(load_data,
                    peak_level=2,
                    weekday_only=False,
                    peak_hours_only=False,
                    peak_hours=(7,18)):
    """
    load_data: input data to be analyzed,
    peak_level: amount of peaks to be considered for each month - values between 1 to 10,
    weekday_only: consider only weekdays,
    peak_hours_only: consider only peak hours,
    peak_hours: tuple with start and end of peak hours
    """
    if weekday_only:
        load_data = load_data[load_data.index.weekday <= 4]
    if peak_hours_only:
        load_data = load_data[(load_data.index.hour <= peak_hours[1]) &\
                                (load_data.index.hour >= peak_hours[0])]
    monthly_peak_days = load_data.groupby([load_data.index.month,
                                            load_data.index.year])['net_load_after_pv'].nlargest(int(peak_level*10))
    monthly_peak_days.index = monthly_peak_days.index.droplevel(0).droplevel(0)
    monthly_peak_days = monthly_peak_days.groupby([monthly_peak_days.index.year,
                                                    monthly_peak_days.index.month,
                                                    monthly_peak_days.index.date]).max()
    return list(monthly_peak_days.index.levels[2])

def get_peak_statistics(load_data,
                        peak_days=False,
                        peak_level=3,
                        weekday_only=False,
                        peak_hours_only=False,
                        peak_hours=(7,18)):
    """
    load_data: input data to be analyzed,
    peak_days: consider only peak days identified from get_peak_days()
    peak_level: amount of peaks to be considered for each month - values between 1 to 10,
    weekday_only: consider only weekdays,
    peak_hours_only: consider only peak hours,
    peak_hours: tuple with start and end of peak hours
    """
    title = "all days"
    if weekday_only:
        load_data = load_data[load_data.index.weekday <= 4]
    if peak_hours_only:
        load_data = load_data[(load_data.index.hour <= peak_hours[1]) &\
                                (load_data.index.hour >= peak_hours[0])]
    if peak_days:
        title = "selected peak days"
        load_data = load_data.loc[np.isin(load_data.index.date,
                                            get_peak_days(load_data=load_data, peak_level=peak_level,
                                                           weekday_only=weekday_only,
                                                           peak_hours_only=peak_hours_only,
                                                           peak_hours=peak_hours))]
    peak_time = load_data['net_load_after_pv'].groupby(load_data.index.date).idxmax()
    peak_hours = peak_time.dt.hour
    peak_hours = pd.DataFrame(peak_hours.values, index = peak_hours.index)
    peak_hours['peak'] = 1
    peak_hours.columns = ['hour', 'peak']
    peak_distribution = peak_hours.groupby('hour').sum()
    fig = px.bar(peak_distribution, x=peak_distribution.index, y='peak')
    fig.update_xaxes(range=[0, 24])
    fig.update_layout(title=f"Peak Distribution for {title}")
    return peak_distribution, fig