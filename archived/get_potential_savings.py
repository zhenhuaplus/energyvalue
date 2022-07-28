import pandas as pd
import numpy as np
import json
from distutils.util import strtobool
import os

pd.set_option('mode.chained_assignment', None)

from archived.extract_peaks import get_peak_days, get_peak_statistics
from energyimpact.controller.rule_based_dt_2cd_opt import run_rule_based_dt_2cd_opt
from energyimpact.controller.optimization import run_optimization

from risk_profile import calculate_predictability, calculate_flexibility

def get_day_results(load_data, day, project_params, tariff_dict):
    if not os.path.exists(f"results/{project_params['project_name']}"):
        os.makedirs(f"results/{project_params['project_name']}")
    daily_load = load_data.loc[load_data.index.date == day]
    initial_demand_target = max(daily_load['net_load_after_pv'])
    results = run_rule_based_dt_2cd_opt(project_name=project_params['project_name'],
                                        daily_load=daily_load.reset_index(),
                                        tariff_dict=tariff_dict,
                                        battery_size_kWh=project_params['battery_size_kWh'],
                                        battery_power_kW=project_params['battery_power_kW'],
                                        min_soc=project_params['min_soc'],
                                        max_soc=project_params['max_soc'],
                                        current_soc=project_params['current_soc'],
                                        one_way_efficiency=project_params['one_way_efficiency'],
                                        initial_demand_target=initial_demand_target)
    col_names = ['battery_power', 'battery_energy', 'net_load_after_storage', 'final_demand_target']
    results_df = pd.DataFrame(dict(zip(col_names, [results[0], results[1], results[2].values, results[3]])))
    results_df.index = daily_load.index
    results_df["net_load_after_pv"] = daily_load["net_load_after_pv"]
    results_df.to_csv(f"results/{project_params['project_name']}/2cd_opt_{project_params['project_name']}_{str(day)}.csv")
    # two_cd_opt_results.append(results_df)

    flex_results_2cd_opt, fig_2cd_opt = \
        calculate_flexibility(daily_load=results_df, date=day,
                                demand_target=results[3][0],
                                storage_size_kWh=project_params['battery_size_kWh'])
    flex_results_2cd_opt['date'] = day
    fig_2cd_opt.write_html(f"results/{project_params['project_name']}/2cd_opt_{project_params['project_name']}_{str(day)}.html")

    results_optimization = run_optimization(daily_load=daily_load.reset_index(),
                                            battery_size_kWh=project_params['battery_size_kWh'],
                                            battery_power_kW=project_params['battery_power_kW'],
                                            min_soc=project_params['min_soc'],
                                            max_soc=project_params['max_soc'],
                                            current_soc=project_params['current_soc'],
                                            one_way_efficiency=project_params['one_way_efficiency'],
                                            tariff_dict=tariff_dict)

    col_names = ['battery_power', 'battery_energy', 'net_load_after_storage', 'final_demand_target']
    results_optimization_df = pd.DataFrame(dict(zip(col_names, [results_optimization[0], results_optimization[1],
                                                                results_optimization[2].values,
                                                                results_optimization[3]])))
    results_optimization_df.index = daily_load.index
    results_optimization_df["net_load_after_pv"] = daily_load["net_load_after_pv"]
    results_optimization_df.to_csv(f"results/{project_params['project_name']}/optimization_{project_params['project_name']}_{str(day)}.csv")
    # optimization_results.append(results_optimization_df)

    flex_results_optimization, fig_optimization = \
        calculate_flexibility(daily_load=results_optimization_df, date=day,
                                demand_target=results_optimization[3][0],
                                storage_size_kWh=project_params['battery_size_kWh'])
    flex_results_optimization['date'] = day
    fig_optimization.write_html(f"results/{project_params['project_name']}/optimization_{project_params['project_name']}_{str(day)}.html")
    return flex_results_2cd_opt, fig_2cd_opt, fig_optimization, flex_results_optimization

if __name__ == "__main__":
    
    with open('data/project_params.json') as json_file:
        config = json.load(json_file)
    project_params = config['project_params']
    analysis_params = config['analysis_params']
    analysis_params['weekday_only'] = strtobool(analysis_params['weekday_only'])
    analysis_params['peak_hours_only'] = strtobool(analysis_params['peak_hours_only'])
    analysis_params['peak_days'] = strtobool(analysis_params['peak_days'])

    if not os.path.exists(f"results/{project_params['project_name']}"):
        os.makedirs(f"results/{project_params['project_name']}")
    
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
    # two_cd_opt_results = []
    # optimization_results = []
    flex_df_2cd_opt = []
    flex_df_optimization = []
    for day in peak_days:
        flex_results_2cd_opt, fig_2cd_opt, fig_optimization, flex_results_optimization =\
            get_day_results(load_data=load_data, 
                            day=day,
                            project_params=project_params,
                            tariff_dict=tariff_dict)
        flex_df_2cd_opt.append(flex_results_2cd_opt)
        flex_df_optimization.append(flex_results_optimization)
    
    flex_df_optimization = pd.DataFrame.from_dict(flex_df_optimization, orient='columns')
    flex_df_2cd_opt = pd.DataFrame.from_dict(flex_df_2cd_opt, orient='columns')
    flex_df_optimization.to_csv(f"results/{project_params['project_name']}/optimization_{project_params['project_name']}_flexibility.csv")
    flex_df_2cd_opt.to_csv(f"results/{project_params['project_name']}/2cd_opt_{project_params['project_name']}_flexibility.csv")

    peak_data = load_data.loc[np.isin(load_data.index.date, peak_days)]
    try:
        cor_results, fig_hist, fig_peak = \
            calculate_predictability(peak_data=peak_data, start_date=None, end_date=None,
                                    weekday_only=analysis_params['weekday_only'])
        fig_hist.write_html(f"results/{project_params['project_name']}/{project_params['project_name']}_predictability_hist.html")
        fig_peak.write_html(f"results/{project_params['project_name']}/{project_params['project_name']}_predictability_peak.html")
    except:
        print("only one day available in peak. cannot run calculate predictability")
    
    peak_distribution, fig = get_peak_statistics(load_data=load_data,
                                                 peak_level=analysis_params['peak_level'],
                                                 peak_days=analysis_params['peak_days'],
                                                 weekday_only=analysis_params['weekday_only'],
                                                 peak_hours_only=analysis_params['peak_hours_only'],
                                                 peak_hours=(analysis_params['peak_hour_start'],
                                                             analysis_params['peak_hour_end']))
    fig.write_html(f"results/{project_params['project_name']}/{project_params['project_name']}_peak_distribution.html")
