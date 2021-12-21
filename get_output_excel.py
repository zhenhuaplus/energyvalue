from numpy.lib.function_base import append
import pandas as pd
import numpy as np
import json
from distutils.util import strtobool
import os
import datetime

from financial_analysis import calculate_irr
from rule_based import run_rule_based, logger
from optimization_solar import run_optimization
from load_analysis_clustering import run_unsupervised

pd.set_option('mode.chained_assignment', None)


def calculate_energy_bill(tariff_dict, load, resolution):
    tariff = pd.Series(load.index) \
        .map(lambda x: tariff_dict['energy_charge']['hours'][str(x.hour)]) \
        .map(lambda k: tariff_dict['energy_charge']['price'][k])
    tariff.index = load.index
    energy_charges = tariff * load * resolution / 60
    energy_charges = energy_charges.sum()
    return - energy_charges


def calculate_demand_bill(tariff_dict, load, billing_type, transformer_capacity):
    if billing_type == "peak_demand":
        demand_price = tariff_dict['demand_charge']['demand_rmb_per_kW_month']
        demand_charges = demand_price * load.max()
    elif billing_type == "transformer_capacity":
        demand_charges = transformer_capacity * \
                         tariff_dict['demand_charge']['trans_rmb_per_kVA_month']
    else:
        logger.warning(
            "Billing type is not defined. Assuming peak_demand as bill type")
        demand_charges = - \
            calculate_demand_bill(
                tariff_dict, load, "peak_demand", transformer_capacity)
    return - demand_charges


def obtain_results(config, load, tariff_dict):
    project_params = config['project_params']
    pv_params = config['pv_params']

    dir = f"results/{str(project_params['project_name']) + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    output_file_prefix = f"{config['mode']}_pv_{pv_params['simulate_pv']}_solar_to_battery_{pv_params['solar_to_battery']}_{project_params['battery_size_kWh']}kWh_{project_params['battery_power_kW']}kW.csv"

    load = load.sort_values(by="datetime")
    clustering_results, fig = run_unsupervised(load)

    load["datetime"] = pd.to_datetime(load["datetime"])
    load = load.set_index("datetime")
    load_resolution = int((load.index[1] - load.index[0]).seconds / 60)

    if 'pv' in load.columns:
        pv = pd.DataFrame(load['pv'])
    else:
        pv = pd.read_csv(
            f"data/{pv_params['pv_file']}",
            parse_dates=['datetime'],
            index_col='datetime',
            usecols=['datetime', 'pv']
        )
        pv = pv.resample(f"{str(int(load_resolution))}T").pad()
        pv = pv * pv_params['pv_capacity']

    col_list = ['date', 'simulate_pv', 'solar_to_battery',
                'solar_to_battery_purchase_price',
                'pv_size_kW', 'storage_size_kWh',
                'storage_power_kW', 'control_mode', 'weather_normalization',
                'peak_load', 'peak_pv', 'peak_net_load_after_pv',
                'peak_net_load_after_storage', 'original_energy_bill',
                'post_pv_energy_bill', 'post_pv&storage_energy_bill',
                'pv_energy_saving', 'pv_energy_saving_monthly', 'storage_energy_saving',
                'storage_energy_saving_monthly', 'original_demand_bill',
                'post_pv_demand_bill', 'post_pv&storage_demand_bill',
                'pv_demand_saving', 'storage_demand_saving', 'IRR',
                'pv_export_revenue', 'pv_import_charge',
                '0_8_max_net_load_after_pv', '12_17_max_net_load_after_pv',
                '8_10_min_net_load_after_pv', '8_12_min_net_load_after_pv',
                '17_21_min_net_load_after_pv', 'total_pv_production_kWh',
                'storage_per_kWh_revenue', '2cd_complete',
                'weekday']
    if os.path.exists(f"{dir}/output.csv"):
        output_excel = pd.read_csv(f"{dir}/output.csv")
    else:
        output_excel = pd.DataFrame(
            columns=col_list
        )
    _current_results = pd.DataFrame(
        columns=col_list
    )

    annual_results = pd.DataFrame()
    for day in np.unique(load.index.date):
        daily_load_before_pv = load.loc[load.index.date == day, [
            'net_load_before_pv']]
        if strtobool(pv_params['simulate_pv']):
            daily_pv = pv.loc[pv.index.date == day, ['pv']]
            if daily_pv.shape[0] == 0:
                daily_pv = pv.copy(deep=True)
                daily_pv.index = daily_load_before_pv.index
        else:
            daily_pv = pd.DataFrame(np.zeros(daily_load_before_pv.shape[0]))
        daily_pv.index = daily_load_before_pv.index
        daily_pv.columns = daily_load_before_pv.columns

        if config['mode'] in ["1cd", "2cd"]:
            results = run_rule_based(daily_load_before_pv=daily_load_before_pv,
                                     pv=daily_pv,
                                     project_params=project_params,
                                     tariff_dict=tariff_dict,
                                     pv_params=pv_params,
                                     simulate_mode=config['mode'])
        elif config['mode'] == "opt":
            results = run_optimization(daily_load_before_pv=daily_load_before_pv,
                                       pv=daily_pv,
                                       project_params=project_params,
                                       tariff_dict=tariff_dict,
                                       pv_params=pv_params)

        else:
            logger.error("Mode not defined")
            exit(1)
        # results.to_csv(f"{dir}/{config['mode']}_pv{pv_params['pv_capacity']}")

        original_energy_bill = calculate_energy_bill(
            tariff_dict,
            results['net_load_before_pv'],
            resolution=load_resolution
        )
        post_pv_energy_bill = calculate_energy_bill(
            tariff_dict,
            results['net_load_after_pv'],
            resolution=load_resolution
        )
        post_pv_storage_energy_bill = calculate_energy_bill(
            tariff_dict,
            results['net_load_after_storage'],
            resolution=load_resolution
        )

        pv_energy_saving = post_pv_energy_bill - original_energy_bill
        pv_energy_saving_monthly = pv_energy_saving * 30
        storage_energy_saving = post_pv_storage_energy_bill - post_pv_energy_bill
        storage_energy_saving_monthly = storage_energy_saving * 30

        original_demand_bill = calculate_demand_bill(
            tariff_dict,
            results['net_load_before_pv'],
            billing_type=project_params['billing_type'],
            transformer_capacity=project_params['transformer_capacity']
        )
        post_pv_demand_bill = calculate_demand_bill(
            tariff_dict,
            results['net_load_after_pv'],
            billing_type=project_params['billing_type'],
            transformer_capacity=project_params['transformer_capacity']
        )
        post_pv_storage_demand_bill = calculate_demand_bill(
            tariff_dict,
            results['net_load_after_storage'],
            billing_type=project_params['billing_type'],
            transformer_capacity=project_params['transformer_capacity']
        )
        pv_demand_saving = post_pv_demand_bill - original_demand_bill
        storage_demand_saving = post_pv_storage_demand_bill - post_pv_demand_bill

        pv_export_revenue = (pv_params['solar_to_grid_price'] * results['solar_to_grid']).sum() / (
                60 / load_resolution)
        pv_import_charge = -(pv_params['solar_to_battery_purchase_price']
                             * results['solar_to_battery']).sum() / (60 / load_resolution)

        full_irr, sub_irr = calculate_irr(
            config=config,
            energy_saving_per_day=storage_energy_saving,
            demand_saving_per_month=storage_demand_saving,
        )

        max_net_load_after_pv_0_8 = results.loc[results.index.hour < 8, 'net_load_after_pv'].mean()
        max_net_load_after_pv_12_17 = results.loc[(results.index.hour > 12) & (
                results.index.hour <= 17), 'net_load_after_pv'].mean()
        min_net_load_after_pv_8_10 = results.loc[(results.index.hour > 8) & (
                results.index.hour <= 10), 'net_load_after_pv'].min()
        min_net_load_after_pv_8_12 = results.loc[(results.index.hour > 8) & (
                results.index.hour <= 12), 'net_load_after_pv'].min()
        min_net_load_after_pv_17_21 = results.loc[(results.index.hour > 17) & (
                results.index.hour <= 21), 'net_load_after_pv'].min()
        total_pv_production_kWh = np.sum(results['pv'])
        storage_per_kWh_revenue = (storage_energy_saving + pv_import_charge) / project_params['battery_size_kWh']

        # complete_2cd = 1 if round(results.loc[results.index.hour ==
        #                                       12, 'battery_energy'].values[0]) == 0 else 0
        revenue_threshold = (tariff_dict['energy_charge']['price']['peak'] * 2 * project_params['one_way_efficiency']) - \
                            (tariff_dict['energy_charge']['price']['valley'] / project_params['one_way_efficiency']) - \
                            (tariff_dict['energy_charge']['price']
                             ['normal'] / project_params['one_way_efficiency'])
        complete_2cd = 1 if (storage_per_kWh_revenue -
                             revenue_threshold) >= - 0.001 else 0
        is_workday = clustering_results.loc[clustering_results["date"] == day, "labels"].values[0]

        op = pd.DataFrame([[day, pv_params['simulate_pv'], pv_params['solar_to_battery'],
                            pv_params['solar_to_battery_purchase_price'],
                            pv_params['pv_capacity'], project_params['battery_size_kWh'],
                            project_params['battery_power_kW'], config['mode'], 1,
                            daily_load_before_pv.max()['net_load_before_pv'],
                            -results['pv'].max(), results['net_load_after_pv'].max(),
                            results.max()['net_load_after_storage'],
                            original_energy_bill, post_pv_energy_bill,
                            post_pv_storage_energy_bill, pv_energy_saving,
                            pv_energy_saving_monthly,
                            storage_energy_saving, storage_energy_saving_monthly,
                            original_demand_bill,
                            post_pv_demand_bill, post_pv_storage_demand_bill,
                            pv_demand_saving, storage_demand_saving,
                            full_irr, pv_export_revenue, pv_import_charge,
                            max_net_load_after_pv_0_8, max_net_load_after_pv_12_17,
                            min_net_load_after_pv_8_10, min_net_load_after_pv_8_12,
                            min_net_load_after_pv_17_21, total_pv_production_kWh,
                            storage_per_kWh_revenue, complete_2cd,
                            is_workday
                            # 0 if day.weekday() >= 5 else 1
                            ]],
                          columns=col_list)
        output_excel = output_excel.append(op, ignore_index=True)
        _current_results = _current_results.append(op, ignore_index=True)
        results.to_csv(f"{dir}/load_flow_{output_file_prefix}_{str(day)}.csv")
        annual_results = annual_results.append(results)
    annual_results["date"] = pd.to_datetime(annual_results.index).date
    annual_results["time"] = pd.to_datetime(annual_results.index).time
    annual_results.to_csv(f"{dir}/output_load_flow.csv")
    output_excel.to_csv(f"{dir}/output.csv", index=False)
    output_excel.to_csv(
        f"{dir}/output_{output_file_prefix}.csv")

    revenue_threshold = (tariff_dict['energy_charge']['price']['peak'] * 2 * project_params['one_way_efficiency']) - \
                        (tariff_dict['energy_charge']['price']['valley'] / project_params['one_way_efficiency']) - \
                        (tariff_dict['energy_charge']['price']
                         ['normal'] / project_params['one_way_efficiency'])
    # _current_results.index = pd.to_datetime(_current_results['date'])
    summary_list = []
    for name, group in _current_results.groupby(_current_results['weekday']):
        current_results = group
        summary = {
            '储能规模kWh': project_params['battery_size_kWh'],
            '储能功率kW': project_params['battery_power_kW'],
            '是否工作日': "工作日" if name == 1 else "非工作日",
            '总天数': current_results.shape[0],

            # '完成两充两放': np.sum(current_results['2cd_complete']),
            '储能每度电的日收益上限': revenue_threshold,

            '0~8点超容天数': np.sum(current_results['0_8_max_net_load_after_pv'] >
                               project_params['transformer_capacity'] -
                               project_params['battery_power_kW']),
            '12～17超容天数': np.sum(current_results['12_17_max_net_load_after_pv'] >
                                project_params['transformer_capacity'] -
                                project_params['battery_power_kW']),
            '8～10放电受限天数': np.sum(current_results['8_10_min_net_load_after_pv'] <
                                 project_params['battery_power_kW']),
            '8～12放电受限天数': np.sum(current_results['8_12_min_net_load_after_pv'] <
                                 project_params['battery_power_kW']),
            '17～21放电受限天数': np.sum(current_results['17_21_min_net_load_after_pv'] <
                                  project_params['battery_power_kW']),

            '完成2cd天数': np.sum((current_results['storage_per_kWh_revenue'] - revenue_threshold) >=
                              - 0.001),
            '完成2cd天数平均每度电日收益': np.mean(current_results.loc[current_results['storage_per_kWh_revenue'] >=
                                                           revenue_threshold - 0.001,
                                                           'storage_per_kWh_revenue']),
            '未完成2cd天数': np.sum(current_results['storage_per_kWh_revenue'] < revenue_threshold -
                               0.001),
            '未完成2cd天数平均每度电日收益': np.mean(current_results.loc[current_results['storage_per_kWh_revenue'] <
                                                            revenue_threshold - 0.001,
                                                            'storage_per_kWh_revenue']),
            '平均每度电的日收益': np.mean(current_results['storage_per_kWh_revenue']),

            '有需量收益天数': np.sum(current_results['storage_demand_saving'] > 0),
            '有需量收益天数的平均月收益': np.round(np.mean(current_results.loc[current_results['storage_demand_saving'] > 0,
                                                                  'storage_demand_saving'])),
            '有需量罚款天数': np.sum(current_results['storage_demand_saving'] <= 0),
            '有需量罚款天数的平均月罚款': np.round(np.mean(current_results.loc[current_results['storage_demand_saving'] < 0,
                                                                  'storage_demand_saving'])),
            '平均需量月收益或罚款': np.round(np.mean(current_results['storage_demand_saving'])),

            '平均IRR': np.mean(current_results['IRR'])
        }
        summary_list.append(summary)
    if os.path.exists(f"{dir}/summary.csv"):
        summary_df = pd.read_csv(f"{dir}/summary.csv")
        summary_df = summary_df.append(summary_list, ignore_index=True)
    else:
        summary_df = pd.DataFrame().from_dict(summary_list)
    summary_df["是否工作日"] = pd.Categorical(summary_df["是否工作日"], categories=["工作日", "非工作日"], ordered=True)
    summary_df = summary_df.sort_values("是否工作日")
    summary_df = summary_df.set_index("是否工作日")
    summary_df.to_csv(f"{dir}/summary.csv", index=True, encoding='utf_8_sig')

    with open(f"{dir}/project_params.json", 'w') as a:
        json.dump(config, a)

    with open(f"{dir}/tariff.json", 'w') as b:
        json.dump(tariff_dict, b)

    return dir
