import numpy as np
import numpy_financial as npf


def calculate_irr(config, energy_saving_per_day, demand_saving_per_month):
    # User inputs
    project_params = config['project_params']
    pv_params = config['pv_params']
    battery_size_kWh = project_params["battery_size_kWh"]
    solar_purchase_cost = pv_params["solar_to_battery_purchase_price"]

    energy_saving_per_year = energy_saving_per_day * project_params["count_working_days"]
    solar_puchase_cost_per_year = solar_purchase_cost * project_params["count_working_days"]
    # demand_saving_per_month = 0
    demand_saving_per_year = demand_saving_per_month * 12
    # first year project revenues
    project_revenue = energy_saving_per_year + \
        solar_puchase_cost_per_year + demand_saving_per_year

    # Read multiyear outputs excel file
    project_years = project_params["project_years"]
    depreciation_years = project_params["depreciation_years"]
    salvage_value_ratio = 0
    battery_degradation_per_year = project_params["battery_degradation_per_year"]

    operation_cost_ratio = 0.1
    management_cost_rmb_per_kwh = 0
    management_cost_vat_rate = 0.13
    ess_cost_rmb_per_kwh = project_params["ess_cost_rmb_per_kwh"]
    ess_cost_vat_rate = 0.13
    epc_cost_rmb_per_kwh = 0
    epc_cost_vat_rate = 0.09
    settlement_tax_rate = 0.06
    corporate_income_tax_rate = 0.25

    # Dispatch ad saving results
    project_revenues = np.zeros(project_years)
    for i in range(project_years):
        project_revenues[i] = project_revenue * \
            (1 - battery_degradation_per_year * i)
    investor_revenues = project_revenues * 0.9 / (1 + settlement_tax_rate)

    # Investment and Operations calculations
    management_cost = - battery_size_kWh * management_cost_rmb_per_kwh
    ess_cost = - battery_size_kWh * ess_cost_rmb_per_kwh
    epc_cost = - battery_size_kWh * epc_cost_rmb_per_kwh
    project_investment_cost = management_cost + ess_cost + epc_cost
    project_vat = project_investment_cost - management_cost / (1 + management_cost_vat_rate) - \
        ess_cost / (1 + ess_cost_vat_rate) - epc_cost / (1 + epc_cost_vat_rate)
    project_investment_cost = project_investment_cost - project_vat
    operation_cost = - investor_revenues * operation_cost_ratio
    insurance_cost = - investor_revenues * 0.3 / 100
    stamp_duty = - (np.array(investor_revenues) -
                    np.array(operation_cost)) * 0.0003
    stamp_duty[0] = - (- project_investment_cost +
                       investor_revenues[0] - operation_cost[0]) * 0.0003
    ebitda = np.array(investor_revenues) + np.array(operation_cost) + \
        np.array(insurance_cost) + np.array(stamp_duty)

    # Tax and depreciate calculations
    output_tax = - investor_revenues * settlement_tax_rate
    input_tax_deductible = - project_vat
    input_tax_balance = [0] * project_years
    for i in range(len(input_tax_balance)):
        if i == 0:
            input_tax_balance[i] = max(0, output_tax[i] + input_tax_deductible)
        else:
            input_tax_balance[i] = max(
                0, output_tax[i] + input_tax_balance[i - 1])
    input_tax_balance = [0] + input_tax_balance
    vat = [0] * project_years
    for i in range(len(vat)):
        if input_tax_balance[i + 1] >= 0:
            vat[i] = 0
        else:
            vat[i] = output_tax[i] + input_tax_balance[i]
    depreciation = [0] * project_years
    for i in range(depreciation_years):
        depreciation[i] = project_investment_cost * \
            (1 - salvage_value_ratio) / depreciation_years

    # Cash flow calculations
    full_taxable_profit = np.array(ebitda) + np.array(depreciation)
    full_income_tax = [- max(0, i * corporate_income_tax_rate)
                       for i in full_taxable_profit]
    full_net_profit = np.array(
        vat) + np.array(full_taxable_profit) + np.array(full_income_tax)
    full_cash_flow = - np.array(depreciation) + np.array(full_net_profit)
    full_cash_flow = [project_investment_cost] + list(full_cash_flow)

    sub_taxable_profit = [i for i in full_taxable_profit]
    for i in range(len(sub_taxable_profit)):
        if i in [0, 1, 2]:
            sub_taxable_profit[i] = 0
        elif i in [3, 4, 5]:
            sub_taxable_profit[i] = full_taxable_profit[i] * 0.5
    sub_income_tax = [- max(0, i * corporate_income_tax_rate)
                      for i in sub_taxable_profit]
    sub_net_profit = np.array(ebitda) + np.array(depreciation) + \
        np.array(vat) + np.array(sub_income_tax)
    sub_cash_flow = - np.array(depreciation) + np.array(sub_net_profit)
    sub_cash_flow = [project_investment_cost] + list(sub_cash_flow)

    # IRR calculations
    full_irr = npf.irr(full_cash_flow)
    sub_irr = npf.irr(sub_cash_flow)

    return full_irr, sub_irr
