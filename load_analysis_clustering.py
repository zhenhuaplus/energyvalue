import pandas as pd
import numpy as np
from plotly import graph_objs as go

from sklearn.cluster import KMeans


def run_unsupervised(load, cluster_size=2, clustering_method='KMeans'):
    # Pivot data
    load['datetime'] = pd.to_datetime(load['datetime'])
    load['date'] = load['datetime'].dt.date
    load['time'] = load['datetime'].dt.time
    pivoted_day = pd.pivot_table(load,
                                 values='net_load_before_pv', index=['date'], columns=['time'],
                                 aggfunc=np.sum)

    if clustering_method == 'KMeans':
        clustering_model = KMeans(n_clusters=cluster_size, random_state=1).fit(pivoted_day)

    colors_light = ['#AED6F1', '#F7CEC5', '#BE94F2', '#D5F5E3']
    colors_dark = ['#2E86C1', '#A93226', '#7D3C98', '#239B56']

    # Store model and clustering results
    clustering_results = pd.DataFrame()
    clustering_results['date'] = [x for x in pivoted_day.index]
    clustering_results['labels'] = clustering_model.labels_
    cluster_average = [np.mean(clustering_model.cluster_centers_[j]) for j in range(cluster_size)]
    max_cluster_average_index = int(np.array(cluster_average).argmax().item())
    clustering_results["labels"] = clustering_results["labels"].map(lambda x:
                                                                    "workday" if x == max_cluster_average_index
                                                                    else "non-workday")
    clustering_results["labels"] = clustering_results["labels"].map(lambda x: 1 if x == "workday" else 0)

    # Plot results
    fig = go.Figure()
    for i in range(len(pivoted_day.index)):
        label = clustering_results["labels"][i]
        fig.add_trace(go.Scatter(x=pivoted_day.columns, y=pivoted_day.iloc[i], mode='lines',
                                 name=str(pivoted_day.index[i]), showlegend=False,
                                 line=dict(color=colors_light[label], width=2, dash='dash')))
    for j in range(cluster_size):
        if j == max_cluster_average_index:
            fig.add_trace(go.Scatter(x=pivoted_day.columns, y=clustering_model.cluster_centers_[j], mode='lines',
                                     name="工作日负荷", line=dict(color=colors_dark[1], width=2)))
        else:
            fig.add_trace(go.Scatter(x=pivoted_day.columns, y=clustering_model.cluster_centers_[j], mode='lines',
                                     name="非工作日负荷", line=dict(color=colors_dark[0], width=2)))

    return clustering_results, fig

# test = pd.read_csv("/Users/zhenhua/Documents/EnergyValue_v2/data/sample_data/jingleng_metering.csv")
# clustering_results, fig = run_unsupervised(test)
# fig.show()
# print(clustering_results)
