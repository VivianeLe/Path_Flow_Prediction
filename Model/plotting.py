import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from statistics import mean
import networkx as nx
import pandas as pd
import pickle
import numpy as np
from multi.multi_helpers import aggregate_result
from single.single_helpers import aggregate_result_single

def read_file(filename):
  with open(filename, "rb") as file:
      stat = pickle.load(file)
      file.close()
  return stat

def plot_loss(train_loss, val_loss, epochs, chart_title):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs+1), train_loss, label='Training Loss')
    plt.plot(range(1, epochs+1), val_loss, label='Validating Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(chart_title)
    plt.legend()
    plt.grid(True)
    # plt.show(block=False)

def plot_error(Link_flow, Path_flow, chart_title):
    Link_abs = [i for df in Link_flow for i in df['abs_err']]
    Link_sqr = [i for df in Link_flow for i in df['sqr_err']]
    Path_abs = [i for df in Path_flow for i in df['abs_err']]
    Path_sqr = [i for df in Path_flow for i in df['sqr_err']]

    Link_abs_threshold = np.percentile(Link_abs, 95)
    # Link_sqr_threshold = np.percentile(Link_sqr, 95)
    Path_abs_threshold = np.percentile(Path_abs, 95)
    # Path_sqr_threshold = np.percentile(Path_sqr, 95)

    plt.figure(figsize=(14, 12))
    plt.title(chart_title)
    plt.subplot(2,2, 1)
    sns.histplot(Link_abs, bins=100, kde=True)
    plt.axvline(Link_abs_threshold, color='r', linestyle='--', label=f'95% threshold: {round(Link_abs_threshold,2)}')
    plt.xlabel('Link flow absolute error')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(2,2, 2)
    sns.histplot(Link_sqr, bins=100, kde=True)
    # plt.axvline(Link_sqr_threshold, color='r', linestyle='--', label=f'95% threshold: {round(Link_sqr_threshold,2)}')
    plt.xlabel('Link flow square error')
    plt.ylabel('Frequency')
    # plt.legend()

    plt.subplot(2,2, 3)
    sns.histplot(Path_abs, bins=100, kde=True)
    plt.axvline(Path_abs_threshold, color='r', linestyle='--', label=f'95% threshold: {round(Path_abs_threshold,2)}')
    # plt.ylim(0, 60000)
    plt.xlabel('Path flow absolute error')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(2,2, 4)
    sns.histplot(Path_sqr, bins=50, kde=True)
    # plt.axvline(Path_sqr_threshold, color='r', linestyle='--', label=f'95% threshold: {round(Path_sqr_threshold,2)}')
    # plt.ylim(0, 200000)
    plt.xlabel('Path flow square error')
    plt.ylabel('Frequency')
    # plt.legend()

    # plt.show(block=False)

def create_graph(edges):
    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge[0], edge[1], capacity=edge[2])
    return G

def plot_graph_with_heatmap(G, pos, chart_title):
    edge_capacities = [G[u][v]['capacity'] for u, v in G.edges]
    min_capacity = min(edge_capacities)
    max_capacity = max(edge_capacities)
    print("Min: ",min(edge_capacities))
    print("Max: ",max(edge_capacities))
    norm = plt.Normalize(vmin=min_capacity, vmax=max_capacity)

    plt.figure(figsize=(10, 8))
    cmap = plt.cm.RdYlGn_r
    edges = nx.draw_networkx_edges(G, pos, edge_color=edge_capacities,
                                   width = 3, arrows=True,
                                   edge_cmap=cmap,
                                   connectionstyle='arc3,rad=0.05',
                                   edge_vmin=min_capacity,
                                   edge_vmax=max_capacity)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=200, node_color='white', edgecolors='black')
    labels = nx.draw_networkx_labels(G, pos, font_color='black', font_size=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax = plt.gca()
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Link Flow MAE', fontsize=17)
    plt.title(chart_title)
    # plt.show(block=False)

def heatmap_link_mae(Link_flow, filename, pos_file, chart_title):
    link_abs = [{row['link_id']: row['abs_err']} for l in Link_flow for _, row in l.iterrows()]
    values_by_key = defaultdict(list)
    for d in link_abs:
        for key, value in d.items():
            values_by_key[key].append(value)
    link_mae = {key: mean(values) for key, values in values_by_key.items()}
    Link_mae_df = pd.DataFrame({'link_id': list(link_mae.keys()), 'link_mae': list(link_mae.values())})

    stat = read_file(filename)
    nodes = stat['data']['network'][['link_id', 'init_node', 'term_node']]
    Link_mae_df = pd.merge(nodes, Link_mae_df, on='link_id', how='left')
    edges = [(int(row['init_node']), int(row['term_node']), row['link_mae']) for _, row in Link_mae_df.iterrows()]
    G = create_graph(edges)

    pos = pd.read_csv(pos_file)
    pos = {row['Node']: (row['X'], row['Y']) for _, row in pos.iterrows()}
    plot_graph_with_heatmap(G, pos, chart_title)
    # return Link_mae_df

# FOR MULTI-CLASS NETWORK
def print_result(pred_c, pred_t, test_files, pos_file, chart_title_c, chart_title_t):
    # result, Link_flow, Path_flow, pred_mean_cost, UE_mean_path_cost, p, s = aggregate_result(pred_c, pred_t, test_files, type)
    result_c, result_t = aggregate_result(pred_c, pred_t, test_files)

    def print_result(result, chart_title):
        print(result[0])
        print("Avg predicted path cost: ", round(np.mean(result[3]), 2), "mins")
        print("Prediction average delay: ", round(result[5],2), "mins = ", round(result[5]/np.mean(result[3])*100, 2), "%")
        print("Solution average delay: ", round(result[6], 2), "mins = ", round(result[6]/np.mean(result[4])*100, 2), "%")
        print(f"Difference: {round(result[5] - result[6],2)} mins")
        plot_error(result[1], result[2], chart_title)
        heatmap_link_mae(result[1], test_files[0], pos_file, chart_title)
    print_result(result_c, chart_title_c)
    print_result(result_t, chart_title_t)

# FOR SINGLE CLASS NETWORK
def print_result_single(pred_tensor, test_files, pos_file, chart_title):
    result, Link_flow, Path_flow, pred_mean_cost, UE_mean_path_cost, p, s = aggregate_result_single(pred_tensor, test_files)

    print(result)
    print("Avg path cost: ", round(np.mean(pred_mean_cost), 4), "mins")
    print("Prediction average delay: ", round(p,4), "mins = ", round(p/np.mean(pred_mean_cost)*100, 2), "%")
    print("Solution average delay: ", round(s, 2), "mins = ", round(s/np.mean(UE_mean_path_cost)*100, 2), "%")
    print(f"Difference: {round(p-s,4)} mins")

    plot_error(Link_flow, Path_flow, chart_title)
    heatmap_link_mae(Link_flow, test_files[0], pos_file, chart_title)