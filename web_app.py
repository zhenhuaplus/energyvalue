import streamlit as st
import pandas as pd
import numpy as np
import base64
import json
import os
from distutils.util import strtobool

from get_output_excel import obtain_results

from plotly import graph_objs as go
import plotly.express as px

st.set_page_config(page_title="EnergyValue", page_icon=":zap:")
st.title("EnergyValue V2.0")

st.markdown("储能系统负荷分析及收益测算工具 - 于2021年8月20日更新")
st.text("")
st.text("")


def get_table_download_link(df, file_name, button_name):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="{file_name}.csv">{button_name}</a>'


def plot_load_data(load_data_df, mandarin=True):
    figs = []

    # Figure for yearly load
    fig_a = go.Figure()
    load_data_df['datetime'] = pd.to_datetime(load_data_df['datetime'])
    fig_a.add_trace(go.Scatter(x=load_data_df['datetime'], y=load_data_df['net_load_before_pv'],
                               mode='lines', name='原负荷'))
    fig_a.add_trace(go.Scatter(x=load_data_df['datetime'], y=load_data_df['pv'],
                               mode='lines', name='光伏'))
    fig_a.add_trace(go.Scatter(x=load_data_df['datetime'], y=load_data_df['net_load_before_pv'] - load_data_df['pv'],
                               mode='lines', name='光伏后净负荷'))
    fig_a.update_layout(title="全年负荷曲线", yaxis_title="功率 (kW)", xaxis_title="时间")
    figs.append(fig_a)

    # Figure for hourly averages
    load_data_df['date'] = pd.to_datetime(load_data_df['datetime']).map(lambda x: x.date())
    load_data_df['time'] = pd.to_datetime(load_data_df['datetime']).dt.time
    load_data_df['is_weekend'] = load_data_df['date'].map(lambda x: x.isoweekday() >= 5)
    hourly_average = load_data_df[['is_weekend', 'time', 'net_load_before_pv']].groupby(
        by=['is_weekend', 'time']).mean().reset_index()
    hourly_average['is_weekend'] = hourly_average['is_weekend'].map(lambda x: "非工作日负荷" if x else "工作日负荷")
    hourly_pv_average = load_data_df[['is_weekend', 'time', 'pv']].groupby(by=['time']).mean().reset_index()
    fig_b = px.line(hourly_average, x='time', y='net_load_before_pv', color='is_weekend', labels={"is_weekend": "Legend"})
    fig_b.add_trace(go.Scatter(x=hourly_pv_average['time'], y=hourly_pv_average['pv'], mode='lines', name='光伏'))
    fig_b.update_layout(title="全年每天负荷小时平均", yaxis_title="功率 (kW)", xaxis_title="Hour of day")
    figs.append(fig_b)

    # Figure for daily profiles by date
    load_data_df['date'] = pd.to_datetime(load_data_df['datetime']).map(lambda x: x.date())
    load_dates = set(load_data_df['date'])
    fig_c = go.Figure()
    for date in load_dates:
        # color = "blue" if date.isoweekday() >= 5 else "red"
        data = load_data[load_data["date"] == date].reset_index(drop=True)
        fig_c.add_trace(go.Scatter(x=data['time'], y=data['net_load_before_pv'], mode='lines', name=str(date)))
    fig_c.update_layout(title="全年每天负荷分布", yaxis_title="功率 (kW)", xaxis_title="Hour of day")
    figs.append(fig_c)

    return figs


def plot_tariff_data(tariff_dict, mandarin=True):
    tariff = pd.DataFrame()
    tariff['hour'] = [i for i in range(24)]
    tariff['energy_charge_type'] = tariff['hour'].map(lambda x: tariff_dict['energy_charge']['hours'][str(x)])
    tariff['energy_charge'] = tariff['energy_charge_type'].map(lambda x: tariff_dict['energy_charge']['price'][x])
    fig = px.bar(tariff, x='hour', y='energy_charge', color='energy_charge_type', text='energy_charge')
    fig.update_layout(title="电价信息", yaxis_title="电度电费 (元/kWh)", xaxis_title="小时")

    return fig


st.markdown("**1. 基本信息**")
col1, col2 = st.columns(2)
with col1:
    project_name = st.text_input("项目名称", "自定义名称")
    project_scope = st.selectbox("项目类型", ["新安装光伏+储能", "自带光伏, 新加储能", "只加储能"], 2)
    simulate_pv = "True" if "光伏" in project_scope else "False"
    project_years = int(st.slider("项目年限 (年)", 5, 20, 15))
    ess_cost_rmb_per_kwh = int(st.number_input("储能系统造价 (元/kWh)", min_value=0, max_value=2000, value=1350, step=1))
    count_working_days = int(st.number_input("全年工作天数 (日)", min_value=250, value=365, step=1))
with col2:
    project_address = st.text_input("项目地址")
    salvage_value_ratio = st.number_input("折旧残值比率 (%)", min_value=0, max_value=100, value=5, step=1) / 100
    depreciation_years = int(st.slider("折旧年限 (年)", 5, 20, 15))
    battery_degradation_per_year = st.number_input("储能系统年衰减率 (%/年)", min_value=0.0, max_value=5.0,
                                                   value=3.0, step=0.1) / 100
    corporate_income_tax_rate = st.number_input("企业所得税税率 (%)", min_value=0, max_value=100, value=25, step=1) / 100
st.text("")
st.text("")


st.markdown("**2. 负荷数据分析**")
# Show sample dataframe
st.info('输入数据模版参考 (确保每一列名字符合要求)')
sample_df = pd.DataFrame({"datetime": ['3/1/20 0:00', '3/1/20 0:01', '3/1/20 0:02'],
                          'net_load_after_pv': [192, 191, 199],
                          'pv': [5, 40, 100]})
# data_sample = pd.read_csv("templates/data_sample.csv")
col1, col2 = st.columns((1, 2))
with col1:
    st.markdown(get_table_download_link(sample_df, file_name="输入数据模版", button_name="点击下载数据模版"),
                unsafe_allow_html=True)
with col2:
    st.dataframe(sample_df)
# Upload data
load_data = st.file_uploader("上传负荷和光伏数据 (CSV或XLSX格式)", type=["csv", "xlsx"])
if load_data is not None:
    load_data = pd.read_csv(load_data)
    # st.dataframe(load_data.loc[:10, :])
    load_figs = plot_load_data(load_data_df=load_data, mandarin=True)
    for fig in load_figs:
        st.plotly_chart(fig, use_container_width=True)
st.text("")
st.text("")


if load_data is not None:
    st.markdown("**3. 储能系统规模**")
    col1, col2 = st.columns(2)
    with col1:
        storage_size = st.number_input("500kWh-125kW数量", min_value=1, max_value=100, value=1, step=1)
        battery_size_kWh = storage_size * 500
        battery_power_kW = storage_size * 125
        min_soc = st.number_input("最小SOC (%)", min_value=0, max_value=100, value=0, step=1) / 100
        max_soc = st.number_input("最大SOC (%)", min_value=0, max_value=100, value=100, step=1) / 100

    with col2:
        one_way_efficiency = st.number_input("电池单程效率 (%)",
                                             min_value=0.0, max_value=100.0, value=92.0, step=0.1) / 100
        battery_degradation_rate = st.number_input("电池每年衰减率 (%)",
                                                   min_value=0.0, max_value=5.0, value=3.0, step=0.1) / 100
        simulate_mode = st.selectbox("储能运行策略", ["两充两放基本控制", "需量优化"], 0)
        simulate_mode = "2cd" if simulate_mode == "两充两放基本控制" else "opt"
st.text("")
st.text("")


if load_data is not None:
    st.markdown("**4. 电网接入信息**")
    col1, col2 = st.columns(2)
    with col1:
        tariff_choices = ["jiangsu_202101_1_10kV", "jiangsu_202101_20_35kV", "jiangsu_202101_35_110kV",
                          "jiangsu_202101_110_220kV", "jiangsu_202101_220kV"]
        tariff_selection = ["1~10kV", "20~35kV", "35~110kV", "110~220kV", ">220kV"]
        tariff_name = st.selectbox("国网进线等级 (kV, 电价依据)", tariff_selection)
        tariff = tariff_choices[tariff_selection.index(tariff_name)] + ".json"
        with open('./data/tariffs/' + tariff) as a:
            tariff_dict = json.load(a)
            revenue_threshold = (tariff_dict['energy_charge']['price']['peak'] * 2 * one_way_efficiency) - \
                                (tariff_dict['energy_charge']['price']['valley'] / one_way_efficiency) - \
                                (tariff_dict['energy_charge']['price']['normal'] / one_way_efficiency)
        revenue_threshold = st.number_input("每度电电度收益最大值 (元/kWh)", min_value=revenue_threshold,
                                            max_value=revenue_threshold, value=revenue_threshold)
    with col2:
        billing_type = st.selectbox("电费计算方式", ["需量结算", "容量结算"])
        billing_type = "transformer_capacity" if billing_type == "容量结算" else "peak_demand"
        transformer_capacity = st.number_input("变压器容量 (kVA)", min_value=0, max_value=999999, value=999999, step=1)
    tariff_fig = plot_tariff_data(tariff_dict, mandarin=True)
    st.plotly_chart(tariff_fig, use_container_width=True)
st.text("")
st.text("")


solar_to_battery = "False"
solar_to_battery_purchase_price = 0.35
solar_to_grid_price = 0.42
if strtobool(simulate_pv) and load_data is not None:
    st.markdown("**5. 光伏信息**")
    col1, col2 = st.columns(2)
    with col1:
        solar_to_battery = str(st.selectbox("光伏是否可以给储能充电", ["False", "True"], 0))
        if strtobool(solar_to_battery):
            solar_to_battery_purchase_price = st.number_input("储能收购光伏价格 (元/kWh)",
                                                              min_value=0.00, max_value=99.99, value=0.39, step=0.01)
    with col2:
        solar_to_grid_price = st.number_input("光伏倒送电网单价 (元/kWh)",
                                              min_value=0.00, max_value=99.99, value=0.42, step=0.01)
st.text("")
st.text("")


# Save ev inputs results
if load_data is not None:
    with open('./data/project_params_template.json') as json_file:
        ev_inputs = json.load(json_file)
        ev_inputs["mode"] = simulate_mode
        ev_inputs["project_params"]["project_name"] = project_name
        ev_inputs["project_params"]["project_years"] = project_years
        ev_inputs["project_params"]["depreciation_years"] = depreciation_years
        ev_inputs["project_params"]["tariff_name"] = tariff
        ev_inputs["project_params"]["min_soc"] = min_soc
        ev_inputs["project_params"]["max_soc"] = max_soc
        ev_inputs["project_params"]["battery_size_kWh"] = battery_size_kWh
        ev_inputs["project_params"]["battery_power_kW"] = battery_power_kW
        ev_inputs["project_params"]["one_way_efficiency"] = one_way_efficiency
        ev_inputs["project_params"]["battery_degradation_per_year"] = battery_degradation_per_year
        ev_inputs["project_params"]["ess_cost_rmb_per_kwh"] = ess_cost_rmb_per_kwh
        ev_inputs["project_params"]["transformer_capacity"] = transformer_capacity
        ev_inputs["project_params"]["billing_type"] = billing_type
        ev_inputs["project_params"]["count_working_days"] = count_working_days
        ev_inputs["pv_params"]["simulate_pv"] = simulate_pv
        ev_inputs["pv_params"]["solar_to_grid_price"] = solar_to_grid_price
        ev_inputs["pv_params"]["solar_to_battery"] = solar_to_battery
        ev_inputs["pv_params"]["solar_to_battery_purchase_price"] = solar_to_battery_purchase_price

# Submit
submit = False
dir = None
if load_data is not None:
    st.markdown("**6. 确认提交**")
    # project_name = st.text_input("本次模拟请求名称")
    # username = st.text_input("邮箱地址")
    # password = st.text_input("密码", type="password")
    submit = st.button("提交模拟请求")
if submit and load_data is not None:
    st.info("你已提交模拟请求")
    dir = obtain_results(config=ev_inputs, load=load_data, tariff_dict=tariff_dict)
    summary = pd.read_csv(f"{dir}/summary.csv")

    annual_results = pd.read_csv(f"{dir}/output_load_flow.csv")
    summary = pd.read_csv(f"{dir}/summary.csv").set_index("是否工作日")
    dates = [str(x) for x in np.unique(annual_results["date"])]
st.text("")
st.text("")

# Check or download results
if dir is not None:
    st.markdown("**7. 查看结果**")
    st.table(summary.iloc[:, :9])
    st.table(summary.iloc[:, 9:])
    submit = st.info("下载结果")
    st.markdown(get_table_download_link(annual_results, file_name="储能运行结果", button_name="储能运行结果"),
                unsafe_allow_html=True)
    st.markdown(get_table_download_link(summary, file_name="负荷收益分析结果", button_name="负荷收益分析结果"),
                unsafe_allow_html=True)

