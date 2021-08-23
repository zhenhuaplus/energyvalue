from get_output_excel import obtain_results
import pandas as pd
import json

load = pd.read_csv("./data/sample_data/jingleng_metering.csv")
with open('/Users/zhenhua/Documents/ei_database/ei_pre/data/tariffs/jiangsu_202101_1_10kV.json') as a:
    tariff_dict = json.load(a)
with open('/Users/zhenhua/Documents/ei_database/ei_pre/data/project_params_template.json') as b:
    config = json.load(b)

dir = obtain_results(config, load, tariff_dict)
