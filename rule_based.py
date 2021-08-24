
import logging
from collections import defaultdict
# from rich.logging import RichHandler
from datetime import date, datetime, timedelta
from distutils.util import strtobool
import pandas as pd
import numpy as np

level = logging.INFO
logger = logging.getLogger(__name__)
# shell_handler = RichHandler()
logger.setLevel(level)
fmt_shell = '%(message)s'
shell_formatter = logging.Formatter(fmt_shell)
# shell_handler.setFormatter(shell_formatter)
# logger.addHandler(shell_handler)


def charge(
    max_charge_rate_per_step,
    battery_power,
    transformer_capacity,
    current_load,
    power_limit=1,
    **kwargs
):
    logger.debug("charging")
    return - min(
        max_charge_rate_per_step,
        battery_power * power_limit,
        transformer_capacity - current_load
    ) if current_load >= 0 else 0


def discharge(
    max_discharge_rate_per_step,
    battery_power,
    current_load,
    solar_to_battery,
    max_charge_rate_per_step,
    transformer_capacity,
    power_limit=1,
    **kwargs
):
    logger.debug("discharging")
    return min(
        max_discharge_rate_per_step,
        battery_power * power_limit,
        current_load
    ) if current_load > 0 else -min(max_charge_rate_per_step,
                                    battery_power * power_limit,
                                    transformer_capacity - current_load,
                                    min(solar_to_battery,
                                        max_charge_rate_per_step))


def get_tariff(tariff_dict, load):
    tariff = pd.Series(load.index) \
        .map(lambda x: tariff_dict['energy_charge']['hours'][str(x.hour)]) \
        .map(lambda k: tariff_dict['energy_charge']['price'][k])
    tariff.index = load.index
    return tariff


def get_battery_power(
    current_hour,
    tariff_dict,
    **kwargs
):
    logger.debug("obtaining battery power")
    rules = defaultdict(list)
    {rules[v].append(k)
     for k, v in tariff_dict['energy_charge']['hours'].items()}
    rules = dict(rules)
    logger.debug(f"rules = {rules}")
    logger.debug(kwargs)
    if str(current_hour) in rules['valley'] + rules['normal']:
        logger.debug(f"trying to charge for hour {current_hour}")
        if current_hour >= 21:
            battery_power = 0
        else:
            battery_power = charge(**kwargs)
    elif str(current_hour) in rules['peak']:
        logger.debug(f"trying to discharge for hour {current_hour}")
        battery_power = discharge(**kwargs)
    else:
        logger.debug(f"hour {current_hour} not in rules")
        battery_power = 0
    logger.debug(f"returning battery power {battery_power}")
    return battery_power


def run_rule_based(daily_load_before_pv, pv, project_params, tariff_dict, pv_params):
    load_resolution = int(
        (daily_load_before_pv.index[1] - daily_load_before_pv.index[0]).seconds/60)
    results = pd.DataFrame(np.zeros((daily_load_before_pv.shape[0], 9)), index=daily_load_before_pv.index, columns=[
                           'battery_power', 'battery_energy', 'net_load_after_storage', 'net_load_before_pv',
                           'pv', 'net_load_after_pv', 'solar_to_load', 'solar_to_grid', 'solar_to_battery'])
    current_soc = project_params['current_soc']
    results.loc[daily_load_before_pv.index[0],
                'battery_energy'] = project_params['battery_size_kWh'] * current_soc

    daily_load_after_pv = np.maximum(0, daily_load_before_pv -
                                     pv)
    daily_load_after_pv.columns = ['net_load_after_pv']

    if strtobool(pv_params['solar_to_battery']):
        max_solar_to_battery = pd.DataFrame(
            np.maximum(0, pv - daily_load_before_pv).values,
            columns=['solar_to_battery'],
            index=daily_load_after_pv.index
        )
    else:
        max_solar_to_battery = pd.DataFrame(
            np.zeros(daily_load_after_pv.shape[0]),
            columns=['solar_to_battery'],
            index=daily_load_after_pv.index
        )
    daily_load = daily_load_after_pv.copy()
    chgable = 1
    dchgable = 1
    logger.debug("running rule based")
    logger.debug(f"solar to bettery = {max_solar_to_battery}")
    for index, row in daily_load.iterrows():
        if current_soc >= 1:
            chgable = 0
            dchgable = 1
        elif current_soc <= 0:
            dchgable = 0
            chgable = 1
        else:
            dchgable = 1
            chgable = 1
        current_hour = index.hour
        current_load = row['net_load_after_pv']
        max_charge_rate_per_step = project_params['battery_size_kWh'] * (1 - current_soc) \
            * (60 / load_resolution) / project_params['one_way_efficiency']
        max_discharge_rate_per_step = project_params['battery_size_kWh'] * current_soc \
            * (60 / load_resolution) * project_params['one_way_efficiency']
        logger.debug(f"current hour = {current_hour}, load = {current_load}")
        # logger.info(f"max_charge_rate_per_step = {max_charge_rate_per_step},\n\
        #     max_discharge_rate_per_step = {max_discharge_rate_per_step},\n\
        #         current_soc = {current_soc}")
        results.loc[index, 'battery_power'] = get_battery_power(
            current_hour=current_hour,
            tariff_dict=tariff_dict,
            max_charge_rate_per_step=max_charge_rate_per_step,
            max_discharge_rate_per_step=max_discharge_rate_per_step,
            battery_power=project_params['battery_power_kW'],
            transformer_capacity=project_params['transformer_capacity'],
            current_load=current_load,
            solar_to_battery=max_solar_to_battery.loc[index,
                                                      'solar_to_battery']
        )

        if results.loc[index, 'battery_power'] >= 0:
            results.loc[index, 'battery_power'] = 0 if abs(
                dchgable) <= 10 ** -2 else results.loc[index, 'battery_power']
        elif results.loc[index, 'battery_power'] <= 0:
            results.loc[index, 'battery_power'] = 0 if abs(
                chgable) <= 10 ** -2 else results.loc[index, 'battery_power']
        results.loc[index, 'battery_power'] = 0 if abs(dchgable) <= 10 ** -2 and abs(
            chgable) <= 10 ** -2 else results.loc[index, 'battery_power']

        logger.debug(
            f"updated battery power to {results.loc[index, 'battery_power']}")

        if index < daily_load.index.max():
            if results.loc[index, 'battery_power'] >= 0:  # discharge
                results.loc[index + timedelta(minutes=load_resolution), 'battery_energy'] = \
                    results.loc[index, 'battery_energy'] + \
                    results.loc[index, 'battery_power'] / project_params['one_way_efficiency'] \
                    * (load_resolution / 60)
            else:  # power < 0, charge
                results.loc[index + timedelta(minutes=load_resolution), 'battery_energy'] = \
                    results.loc[index, 'battery_energy'] + \
                    results.loc[index, 'battery_power'] * project_params['one_way_efficiency'] \
                    * (load_resolution / 60)
            current_soc = - results.loc[index + timedelta(
                minutes=load_resolution), 'battery_energy'] / project_params['battery_size_kWh']
        logger.debug(f"current soc updated to = {current_soc}")
        results.loc[index, 'net_load_before_pv'] = daily_load_before_pv.loc[index,
                                                                            'net_load_before_pv']
        results.loc[index, 'pv'] = pv.loc[index, 'net_load_before_pv']
        results.loc[index, 'net_load_after_pv'] = current_load
        results.loc[index, 'solar_to_load'] = np.minimum(
            results.loc[index, 'net_load_before_pv'], results.loc[index, 'pv'])
        # np.maximum(0, _pv.values - solar_to_load.values - np.minimum(solar_to_battery.values, - np.where(
        # results['battery_power'].values < 0, results['battery_power'].values, 0).reshape(-1, 1))),
        results.loc[index, 'solar_to_grid'] = \
            np.maximum(0, results.loc[index, 'pv'] -
                       results.loc[index, 'solar_to_load'] -
                       np.minimum(max_solar_to_battery.loc[index, 'solar_to_battery'],
                                  - results.loc[index, 'battery_power'] if results.loc[index, 'battery_power'] < 0 else 0))
        results.loc[index, 'solar_to_battery'] = \
            np.maximum(0, max_solar_to_battery.loc[index, 'solar_to_battery'] -
                       results.loc[index, 'solar_to_grid'])
        results.loc[index, 'net_load_after_storage'] = \
            current_load - \
            results.loc[index, 'battery_power'] - \
            results.loc[index, 'solar_to_battery']
        results.loc[index, 'battery_discharge'] = np.maximum(
            results.loc[index, 'battery_power'], 0)
        results.loc[index, 'battery_charge_from_solar'] = - \
            results.loc[index, 'solar_to_battery']
        results.loc[index, 'battery_charge_from_grid'] = np.minimum(
            results.loc[index, 'battery_power'], 0) - results.loc[index, 'battery_charge_from_solar']

    results['energy_price'] = get_tariff(
        tariff_dict, daily_load_after_pv).values
    results['solar_purchase_price'] = pv_params['solar_to_battery_purchase_price']
    results['solar_export_price'] = pv_params['solar_to_grid_price']
    results['battery_energy'] = - results['battery_energy']
    return results
