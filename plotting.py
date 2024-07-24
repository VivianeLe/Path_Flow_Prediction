import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from statistics import mean
import networkx as nx
import pandas as pd
import pickle

def read_file(filename):
  with open(filename, "rb") as file:
      stat = pickle.load(file)
      file.close()
  return stat

def plot_loss(train_loss, val_loss, epochs):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs+1), train_loss, label='Training Loss')
    plt.plot(range(1, epochs+1), val_loss, label='Validating Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validating Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_error(Link_flow, Path_flow):
    Link_abs = [i for df in Link_flow for i in df['abs_err']]
    Link_sqr = [i for df in Link_flow for i in df['sqr_err']]
    Path_abs = [i for df in Path_flow for i in df['abs_err']]
    Path_sqr = [i for df in Path_flow for i in df['sqr_err']]

    plt.figure(figsize=(14, 12))
    plt.subplot(2,2, 1)
    sns.histplot(Link_abs, bins=100, kde=True)
    # plt.title('Absolute error of link flow')
    plt.xlabel('Link flow absolute error')
    plt.ylabel('Frequency')

    plt.subplot(2,2, 2)
    sns.histplot(Link_sqr, bins=100, kde=True)
    # plt.title('Histogram of square error of link flow')
    plt.xlabel('Link flow square error')
    plt.ylabel('Frequency')

    plt.subplot(2,2, 3)
    sns.histplot(Path_abs, bins=100, kde=True)
    # plt.ylim(0, 60000)
    # plt.title('Histogram of absolute error of path flow')
    plt.xlabel('Path flow absolute error')
    plt.ylabel('Frequency')

    plt.subplot(2,2, 4)
    sns.histplot(Path_sqr, bins=50, kde=True)
    # plt.ylim(0, 200000)
    # plt.title('Histogram of square error of path flow')
    plt.xlabel('Path flow square error')
    plt.ylabel('Frequency')

    plt.show()

def create_graph(edges):
    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge[0], edge[1], capacity=edge[2])
    return G

def plot_graph_with_heatmap(G, pos, ratio):
    edge_capacities = [G[u][v]['capacity'] for u, v in G.edges]
    norm = plt.Normalize(vmin=min(edge_capacities), vmax=max(edge_capacities))

    plt.figure(figsize=(12, 10))
    edges = nx.draw_networkx_edges(G, pos, edge_color=edge_capacities, 
                                   width = 3, arrows=True,
                                   edge_cmap=plt.cm.hot, 
                                   connectionstyle='arc3,rad=0.05',
                                   edge_vmin=min(edge_capacities), 
                                   edge_vmax=max(edge_capacities))
    nodes = nx.draw_networkx_nodes(G, pos, node_size=200, node_color='white', edgecolors='black')
    labels = nx.draw_networkx_labels(G, pos, font_color='black', font_size=8)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.copper, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Link flow MAE')

    # plt.title(f'Missing ratio = {ratio}%')
    plt.show()

def heatmap_link_mae(Link_flow, filename, ratio):
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
    # Link_mae_df['link_mae'][Link_mae_df['link_mae'] >12000] = Link_mae_df['link_mae']-7000
    edges = [(int(row['init_node']), int(row['term_node']), row['link_mae']) for _, row in Link_mae_df.iterrows()]
    G = create_graph(edges)

    pos = pd.read_csv('Generate_data/SiouxFalls/SiouxFalls_node.csv')
    pos = {row['Node']: (row['X'], row['Y']) for _, row in pos.iterrows()}
    plot_graph_with_heatmap(G, pos, ratio)
    return Link_mae_df