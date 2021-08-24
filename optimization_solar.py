import pandas as pd
import numpy as np
import cvxpy as cp
from distutils.util import strtobool


"""

    Optimization-based control based on tariff information

    Input:
        daily_load (dataframe): parsed daily load profile with any possible resolution
            e.g.:
                             datetime     net_load_after_pv
            0     2020-05-01 00:00:00  135.660000
            1     2020-05-01 00:01:00  136.520000
            2     2020-05-01 00:02:00  137.380000
            3     2020-05-01 00:03:00  138.240000
            4     2020-05-01 00:04:00  139.100000
            ...
        tariff_dict (dict): tariff dict from json file
        project_params['battery_size_kWh'] (float): battery size in kWh
        project_params['battery_power_kW'] (float): maximum battery charge/discharge power in kW
        project_params['min_soc'] (float): minimum SOC
        project_params['max_soc'] (float): maximum SOC
        project_params['current_soc'] (float): current SOC
        project_params['one_way_efficiency'] (float): battery charge/discharge efficiency
    Output:
        battery_power (array): battery charge/discharge amount
        battery_energy (array): battery energy level
        net_load_after_storage (array): net load after storage
        final_demand_target (array): final demand target in kW
        

"""


def run_optimization(
    daily_load_before_pv,
    pv,
    project_params,
    tariff_dict,
    pv_params
):
    # Parameters
    # pv_params['solar_to_grid_price'] = 0.39
    # pv_params['solar_to_battery_purchase_price'] = 0.41

    # Obtain parameters
    input_resolution = int(
        (daily_load_before_pv.index[1] - daily_load_before_pv.index[0]).seconds/60)

    daily_load = daily_load_before_pv.reset_index()
    # Make sure all variables have the right format
    daily_load['datetime'] = pd.to_datetime(daily_load['datetime'])
    daily_load['hour'] = pd.to_datetime(daily_load['datetime']).dt.hour

    battery_energy = np.zeros(daily_load.shape[0])
    battery_energy[0] = project_params['battery_size_kWh'] * \
        project_params['current_soc']
    battery_energy_min = project_params['battery_size_kWh'] * \
        project_params['min_soc']
    battery_energy_max = project_params['battery_size_kWh'] * \
        project_params['max_soc']

    load = daily_load['net_load_before_pv']
    pv.columns = ['pv']
    pv = pv.reset_index()['pv']
    net_load_after_pv = np.maximum(0, load - pv)
    solar_to_load = np.minimum(load, pv)
    # solar_to_battery_ = np.minimum(project_params['battery_power_kW'], np.maximum(0, pv - load))
    # solar_to_grid = pv - solar_to_load - solar_to_battery

    # Design optimization-based control policies
    # Define variables
    is_charging = cp.Variable(daily_load.shape[0], boolean=True)
    net_load_after_storage = cp.Variable(daily_load.shape[0])
    battery_charge = cp.Variable(daily_load.shape[0])
    battery_charge_from_grid = cp.Variable(daily_load.shape[0])
    battery_charge_from_solar = cp.Variable(daily_load.shape[0])
    battery_discharge = cp.Variable(daily_load.shape[0])
    battery_energy = cp.Variable(daily_load.shape[0])
    solar_to_battery = cp.Variable(daily_load.shape[0])
    solar_to_grid = cp.Variable(daily_load.shape[0])

    # Figure out other energy flow components
    # battery_charge_from_solar = - solar_to_battery
    # battery_charge_from_grid = battery_charge - battery_charge_from_solar

    # Initialize constraints
    constraints = []
    # Battery constraints
    constraints += [net_load_after_storage >= 0]
    constraints += [net_load_after_storage <=
                    project_params['transformer_capacity']]
    constraints += [0 <= battery_discharge, battery_discharge <=
                    cp.multiply(project_params['battery_power_kW'], 1 - is_charging)]
    constraints += [cp.multiply(- project_params['battery_power_kW'], is_charging) <= battery_charge_from_grid,
                    battery_charge_from_grid <= 0]
    constraints += [cp.multiply(- project_params['battery_power_kW'], is_charging) <= battery_charge_from_solar,
                    battery_charge_from_solar <= 0]
    constraints += [list(- np.maximum(0, pv - load))
                    <= battery_charge_from_solar]
    constraints += [battery_charge_from_solar +
                    battery_charge_from_grid == battery_charge]
    constraints += [battery_energy_min <= battery_energy,
                    battery_energy <= battery_energy_max]
    constraints += [battery_energy[0] ==
                    project_params['battery_size_kWh'] * project_params['current_soc']]
    constraints += [net_load_after_storage ==
                    list(net_load_after_pv) - battery_charge_from_grid - battery_discharge]

    constraints += [solar_to_battery == - battery_charge_from_solar]
    constraints += [solar_to_grid ==
                    list(pv - solar_to_load) - solar_to_battery]

    if not strtobool(pv_params['solar_to_battery']):
        constraints += [solar_to_battery == 0]

    for i in range(daily_load.shape[0] - 1):
        constraints += [battery_energy[i + 1] == battery_energy[i] -
                        battery_charge[i] * project_params['one_way_efficiency'] * (input_resolution / 60) -
                        battery_discharge[i] / project_params['one_way_efficiency'] * (input_resolution / 60)]
    constraints += [(battery_energy[-1] -
                     battery_charge[-1] * project_params['one_way_efficiency'] * (input_resolution / 60) -
                     battery_discharge[-1] / project_params['one_way_efficiency'] * (input_resolution / 60)) >= 0]

    energy_price = pd.Series(daily_load['datetime']) \
        .map(lambda x: tariff_dict['energy_charge']['hours'][str(x.hour)]) \
        .map(lambda k: tariff_dict['energy_charge']['price'][k])

    energy_charges = - sum(np.multiply(energy_price, load)
                           ) * (input_resolution / 60)
    energy_charges_after_pv = - \
        sum(np.multiply(energy_price, net_load_after_pv)) * \
        (input_resolution / 60)
    energy_charges_after_storage = - \
        sum(cp.multiply(energy_price, net_load_after_storage)) * \
        (input_resolution / 60)

    battery_charge_from_grid_cost = sum(cp.multiply(
        energy_price, battery_charge_from_grid)) * (input_resolution / 60)
    battery_discharge_revenues = sum(cp.multiply(
        energy_price, battery_discharge)) * (input_resolution / 60)

    pv_export_revenues = cp.sum(solar_to_grid) * \
        pv_params['solar_to_grid_price']
    battery_charge_from_pv_payments = - \
        cp.sum(solar_to_battery) * pv_params['solar_to_battery_purchase_price']

    energy_savings_from_pv = energy_charges_after_pv - energy_charges
    energy_savings_from_storage = energy_charges_after_storage - energy_charges_after_pv

    demand_charges = - \
        tariff_dict['demand_charge']['demand_rmb_per_kW_month'] * max(load)
    demand_charges_after_pv = - \
        tariff_dict['demand_charge']['demand_rmb_per_kW_month'] * \
        max(net_load_after_pv)
    demand_charges_after_storage = - tariff_dict['demand_charge']['demand_rmb_per_kW_month'] * cp.max(
        net_load_after_storage)
    demand_savings_from_pv = demand_charges_after_pv - demand_charges
    demand_savings_from_storage = demand_charges_after_storage - demand_charges_after_pv

    objective = cp.Maximize(energy_savings_from_storage +
                            demand_savings_from_storage + battery_charge_from_pv_payments)

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        print('Using GLPK_MI solver')
        result = problem.solve(solver=cp.GLPK_MI, verbose=False)
    except:
        try:
            print('Using ECOS solver')
            result = problem.solve(solver=cp.ECOS, verbose=False)
        except:
            print('Using SCS solver')
            result = problem.solve(solver=cp.SCS, verbose=False)

    # Check the problem status and determine the final demand target
    if problem.status not in ["infeasible", "unbounded"]:
        print('Total bill savings are {}'.format(problem.value))
        print(energy_charges,
              energy_charges_after_pv,
              energy_charges_after_storage.value,
              pv_export_revenues.value,
              battery_charge_from_pv_payments.value,
              energy_savings_from_pv,
              energy_savings_from_storage.value,
              battery_charge_from_grid_cost.value,
              battery_discharge_revenues.value,
              demand_charges,
              demand_charges_after_pv,
              demand_charges_after_storage.value,
              demand_savings_from_pv,
              demand_savings_from_storage.value)
    else:
        print('Problem {}'.format(problem.status))
    results = pd.DataFrame(np.zeros((daily_load_before_pv.shape[0], 9)), index=daily_load_before_pv.index, columns=[
                           'battery_power', 'battery_energy', 'net_load_after_storage', 'net_load_before_pv',
                           'pv', 'net_load_after_pv', 'solar_to_load', 'solar_to_grid', 'solar_to_battery'])
    results['battery_power'] = battery_charge.value + battery_discharge.value
    results['battery_energy'] = battery_energy.value
    results['net_load_after_storage'] = net_load_after_storage.value
    results['net_load_before_pv'] = load.values
    results['pv'] = pv.values
    results['net_load_after_pv'] = net_load_after_pv.values
    results['solar_to_load'] = solar_to_load.values
    results['solar_to_grid'] = solar_to_grid.value
    results['solar_to_battery'] = solar_to_battery.value

    return results
    # return battery_charge.value, battery_discharge.value, battery_energy.value, \
    #     net_load_after_pv, net_load_after_storage.value
