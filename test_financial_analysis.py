import numpy as np
import json
from financial_analysis import calculate_irr

with open('/Users/zhenhua/Documents/ei_database/ei_pre/data/project_params_template.json') as b:
    config = json.load(b)

print(calculate_irr(config, energy_saving_per_day=10000, demand_saving_per_month=0))
