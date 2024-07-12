import numpy as np
import pickle
import tensorflow as tf
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd 
from collections import defaultdict

base_path = 'Output/5by5_Data'
# Create dictionary of all unique paths
def path_encoder():
    path_sample = []
    for i in range(10):
        file_name = ''.join([base_path, str(i)])
        with open(file_name, "rb") as file:
            stat = pickle.load(file)
        path_sample.append(stat["data"]["paths_link"])

    all_path_link = [path_sample[i].values() for i in range(len(path_sample))]
    unique_values_set = {tuple(p) for path_set in all_path_link for path in path_set for p in path}
    path_set_dict = {v: k for k, v in enumerate(unique_values_set, start=1)}
    return path_set_dict

unique_set = path_encoder()

def normalize(tensor, return_scaler=False):
    tensorY = tensor.numpy()
    scaler = MinMaxScaler()
    tensorY = scaler.fit_transform(tensorY)
    tensorY = tf.convert_to_tensor(tensorY, dtype=tf.float32)
    if return_scaler:
        return tensorY, scaler
    return tensorY

def get_Link_Path_adj(net):
    link_path = tf.zeros((net.shape[0], len(unique_set)), dtype=tf.float32)
    indices = []
    updates = []
    for p, index in unique_set.items():
        for link in p:
            indices.append([link, index - 1])
            updates.append(1.0)
    indices = tf.constant(indices, dtype=tf.int32)
    updates = tf.constant(updates, dtype=tf.float32)
    link_path = tf.tensor_scatter_nd_update(link_path, indices, updates)

    return link_path

def create_matrix(data, nodes):
    # data is an array, nodes is a set
    matrix = np.zeros((len(nodes), len(nodes)))
    for (o, d), v in data:
        o = int(o)
        d = int(d)
        matrix[o-1][d-1] = v
    return matrix

def create_single_tensor(data, nodes):
    matrix = create_matrix(data, nodes)
    tensor = tf.convert_to_tensor([matrix], dtype=tf.float32)
    tensor = tf.squeeze(tensor, axis=0) # 25,25
    tensor = tf.reshape(tensor, [-1]) # Flatten the matrix to a 1D tensor
    tensor = tf.expand_dims(tensor, axis=1) # TensorShape([625, 1])
    return tensor

def get_graphTensor(network, nodes):
    cap = np.array(network[['init_node', 'term_node', 'capacity']]\
                   .apply(lambda row: ((row['init_node'], row['term_node']), row['capacity']), axis=1).tolist(), dtype=object)
    length = np.array(network[['init_node', 'term_node', 'length']]\
                    .apply(lambda row: ((row['init_node'], row['term_node']), row['length']), axis=1).tolist(), dtype=object)
    fft = np.array(network[['init_node', 'term_node', 'free_flow_time']]\
                   .apply(lambda row: ((row['init_node'], row['term_node']), row['free_flow_time']), axis=1).tolist(), dtype=object)

    Cap = create_single_tensor(cap, nodes) # 625,1
    Length = create_single_tensor(length, nodes)
    Fft = create_single_tensor(fft, nodes)
    tensor = tf.concat([tf.cast(Cap, tf.float32), tf.cast(Length, tf.float32), tf.cast(Fft, tf.float32)], axis=1) # 625,3
    return tensor

def get_demandTensor(demand, nodes):
    tensor = np.array([(key, value) for key, value in demand.items()], dtype=object)
    tensor = create_single_tensor(tensor, nodes)
    return tensor

# Get 3 feasible paths for each OD pair, return tensor shape 625x3
def get_pathTensor(path_links, nodes):
    paths = np.array([(key, [tuple(path) for path in value]) for key, value in path_links.items()], dtype=object)
    p1, p2, p3 = [], [], []
    for od, [path1, path2, path3] in paths:
        p1.append((od, unique_set[path1]))
        p2.append((od, unique_set[path2]))
        p3.append((od, unique_set[path3]))

    p1 = create_single_tensor(p1, nodes)
    p2 = create_single_tensor(p2, nodes)
    p3 = create_single_tensor(p3, nodes)
    tensor = tf.concat([tf.cast(p1, tf.float32), tf.cast(p2, tf.float32), tf.cast(p3, tf.float32)], axis=1)
    return tensor

def get_pair_path_tensor(path_links, nodes):
    pair_path_dict = {k: [unique_set[tuple(path)] for path in v] for k, v in path_links.items()}
    pair_path_tensor = tf.zeros((len(nodes), len(nodes), len(unique_set)), dtype=tf.float32) # shape (25,25,1155)

    indices = []
    updates = []
    for (o, d), k_list in pair_path_dict.items():
        for k in k_list:
            indices.append([o-1, d-1, k-1])
            updates.append(1.0)
    indices_tensor = tf.constant(indices, dtype=tf.int64)
    updates_tensor = tf.constant(updates, dtype=tf.float32)
    pair_path_tensor = tf.tensor_scatter_nd_update(pair_path_tensor, indices_tensor, updates_tensor)
    pair_path_tensor = tf.reshape(pair_path_tensor, (len(nodes)**2, pair_path_tensor.shape[-1]))
    return pair_path_tensor

# Get path flow distribution (Y), return a tensor 625x3
def get_flowTensor(demand, path_flows, nodes, test_set=None):
    flows = np.array([(k, v) for k, v in zip(demand.keys(), path_flows)], dtype=object)
    p1, p2, p3 = [], [], []
    for od, [path1, path2, path3] in flows:
        p1.append((od, path1))
        p2.append((od, path2))
        p3.append((od, path3))
    p1 = create_single_tensor(p1, nodes)
    p2 = create_single_tensor(p2, nodes)
    p3 = create_single_tensor(p3, nodes)
    tensor = tf.concat([tf.cast(p1, tf.float32), tf.cast(p2, tf.float32), tf.cast(p3, tf.float32)], axis=1) # 625, 3
    return tensor

def create_mask(tensor):
    mask = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(tensor), axis=-1)),-1) # create mask for row
    return mask

def reduce_dimensionality(X, n_components=5):
    X_np = X.numpy()
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_np)

    # Convert back to TensorFlow tensor
    X_pca_tf = tf.convert_to_tensor(X_pca, dtype=tf.float32)
    return X_pca_tf

def get_X(Graph, OD_demand, Path_tensor, Adj):
    X = [Graph, OD_demand, Path_tensor, Adj]
    max_size = max([x.shape[0] for x in X])

    Graph = tf.pad(Graph, paddings=[[0, max_size - Graph.shape[0]], [0, 0]])
    OD_demand = tf.pad(OD_demand, paddings=[[0, max_size - OD_demand.shape[0]], [0, 0]])
    Path_tensor = tf.pad(Path_tensor, paddings=[[0, max_size - Path_tensor.shape[0]], [0, 0]])
    Adj = tf.pad(Adj, paddings=[[0, max_size - Adj.shape[0]], [0, 0]])

    X = tf.concat([tf.cast(Graph, tf.float32), tf.cast(OD_demand, tf.float32),
                    tf.cast(Path_tensor, tf.float32), tf.cast(Adj, tf.float32)], axis=1)
    return X

def generate_xy(file_name, test_set=None):
    with open(file_name, "rb") as file:
        stat = pickle.load(file)
        file.close()

    path_links = stat["data"]["paths_link"]
    demand = stat["data"]["demand"]
    path_flows = stat["path_flow"]
    nodes = stat["data"]["nodes"]
    net = stat["data"]["network"]

    # Get X
    Graph = get_graphTensor(net, nodes) # (625, 3)
    OD_demand = get_demandTensor(demand, nodes) # (625,1)
    Path_set = get_pathTensor(path_links, nodes) # (625, 3)
    Adj = get_Link_Path_adj(net)
    # X = tf.concat([tf.cast(Graph, tf.float32),
    #                tf.cast(OD_demand, tf.float32),
    #                tf.cast(Path_set, tf.float32)
    #                ], axis=1)
    X = get_X(Graph, OD_demand, Path_set, Adj) # 625x1162
    X_mask = create_mask(X)
    X = normalize(X)
    X = reduce_dimensionality(X)
    # X = tf.where(X == 0, tf.constant(-2**32+1, dtype=X.dtype), X)

    # Get Y
    Y = get_flowTensor(demand, path_flows, nodes)
    Y_mask = create_mask(Y)
    Y, scaler = normalize(Y, return_scaler=True)
    if test_set:
        return X, Y, X_mask, Y_mask, scaler
    return X, Y, X_mask, Y_mask

def plot_loss(train_loss, val_loss, epochs, learning_rate, train_time, N, d_model):
    plt.figure(figsize=(12, 6))
    train_loss = train_loss[0:100]
    val_loss = val_loss[0:100]
    plt.plot(range(1, 101), train_loss, label='Training Loss')
    plt.plot(range(1, 101), val_loss, label='Validating Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validating Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

"""
CHECK UE CONDITIONS OF PREDICTED OUTPUT
"""

def read_file(filename):
  with open(filename, "rb") as file:
      stat = pickle.load(file)
      file.close()
  return stat

def get_origin_path(stat):
    path_link = stat['data']['paths_link']
    od = [k for k in path_link.keys()]
    path1 = [tuple(p[0]) for p in path_link.values()]
    path2 = [tuple(p[1]) for p in path_link.values()]
    path3 = [tuple(p[2]) for p in path_link.values()]

    demand_dic = stat["data"]["demand"]
    demand = [v for v in demand_dic.values()]
    path_link_df = pd.DataFrame({"od": od, "demand":demand, "path1": path1, "path2": path2, "path3": path3})
    return path_link_df

def get_UE_link_cost(stat):
    # return a dataframe of link cost, link flow
    link = stat['data']['network'].copy()
    link['link_flow'] = stat['link_flow']
    # Calculate link cost
    link['link_cost'] = round(link['free_flow_time']*\
                            (1+link['b']*((link['link_flow']/link['capacity'])**4)), 2)
    return link

# Calculate path travel time for each od pair
def calculate_path_cost(row, link_df):
    sum_time = 0
    for l in row:
        sum_time += link_df.at[l, 'link_cost']
    return round(sum_time, 2)

# calculate each link flow based on path flow
def extract_link_flow(path_link, flows):
    # input: a dictionary of {od pair: path_link} and list of flow distribution
    # return a dictionary of link flow
    path_flow = {}
    for path_set, flow_set in zip(path_link.values(), flows):
        for path, flow in zip(path_set, flow_set):
            path_flow[tuple(path)] = flow

    aggregated_sums = defaultdict(float)
    for path, flow in path_flow.items():
        for link in path:
            aggregated_sums[link] += flow
    link_flow = dict(aggregated_sums)
    return link_flow

def extract_flow(pred_tensor):
    pred1 = tf.reshape(pred_tensor[:, 0], (25,25))
    pred2 = tf.reshape(pred_tensor[:, 1], (25,25))
    pred3 = tf.reshape(pred_tensor[:, 2], (25,25))

    dict1 = {(i+1, j+1): pred1[i, j].numpy() for i in range(pred1.shape[0]) for j in range(pred1.shape[1])}
    dict2 = {(i+1, j+1): pred2[i, j].numpy() for i in range(pred2.shape[0]) for j in range(pred2.shape[1])}
    dict3 = {(i+1, j+1): pred3[i, j].numpy() for i in range(pred3.shape[0]) for j in range(pred3.shape[1])}

    final_dict = {}
    for key in dict1.keys():
        final_dict[key] = [dict1[key], dict2[key], dict3[key]]
    final_dict = {k: v for k, v in final_dict.items() if not all(val == 0 for val in v)}
    return final_dict

def create_pred_df(tensor, stat):
  final_dict = extract_flow(tensor)  
  flow_df = pd.DataFrame.from_dict(final_dict, orient='index', columns=['flow1', 'flow2', 'flow3']).reset_index()
  flow_df.rename(columns={'index': 'od'}, inplace=True)
  pred_df = get_origin_path(stat)[['od', 'demand', 'path1', 'path2', 'path3']]
  pred_df = pd.merge(pred_df, flow_df, how='left', on='od')
  nan_val = pred_df['flow1'].isna().sum()
  nan_num = round(nan_val/len(stat['path_flow']),2)
  pred_df = pred_df.fillna(0)
  return pred_df, len(stat['path_flow']), len(final_dict), nan_num

# Calculate link flow from pred path flow
# def sum_pred_link_flow(pred_df, stat):
#     pred_path_flow = pred_df[['pred_f1', 'pred_f2', 'pred_f3']].values.tolist()
#     path_link = stat['data']['paths_link']

#     pred_link_flow = extract_link_flow(path_link, pred_path_flow)
#     pred_link_flow = pd.DataFrame.from_dict(pred_link_flow, orient='index', columns=['pred_link_flow']).sort_index(ascending=True).reset_index()
#     pred_link_flow.rename(columns={'index': 'link_id'}, inplace=True)
#     link = stat['data']['network'].copy()[['link_id', 'capacity', 'free_flow_time', 'b']]
#     output = pd.merge(link, pred_link_flow, how='left', on='link_id')
#     output = output.fillna(0)
#     output['link_cost'] = round(output['free_flow_time']*\
#                             (1+output['b']*((output['pred_link_flow']/output['capacity'])**1)), 2)
#     return output

def sum_pred_link_flow(pred_df, stat):
    pred_path_flow = pred_df[['flow1', 'flow2', 'flow3']].values.tolist()
    path_link = stat['data']['paths_link']

    pred_link_flow = extract_link_flow(path_link, pred_path_flow)
    pred_link_flow = pd.DataFrame.from_dict(pred_link_flow, orient='index', columns=['link_flow']).sort_index(ascending=True).reset_index()
    pred_link_flow.rename(columns={'index': 'link_id'}, inplace=True)
    link = stat['data']['network'].copy()[['link_id', 'capacity', 'free_flow_time', 'b']]
    output = pd.merge(link, pred_link_flow, how='left', on='link_id')
    output = output.fillna(0)
    output['link_cost'] = round(output['free_flow_time']*\
                            (1+output['b']*((output['link_flow']/output['capacity'])**4)), 2)
    return output

def calculate_delay(pred_df, pred_link_flow):
    pred_df['path1_cost'] = pred_df['path1'].apply(lambda x: calculate_path_cost(x, pred_link_flow))
    pred_df['path2_cost'] = pred_df['path2'].apply(lambda x: calculate_path_cost(x, pred_link_flow))
    pred_df['path3_cost'] = pred_df['path3'].apply(lambda x: calculate_path_cost(x, pred_link_flow))
    pred_df['min_path_cost'] = pred_df[['path1_cost', 'path2_cost', 'path3_cost']].min(axis=1)
    pred_df['delay'] = (
        pred_df['flow1'] * (pred_df['path1_cost'] - pred_df['min_path_cost']) +
        pred_df['flow2'] * (pred_df['path2_cost'] - pred_df['min_path_cost']) +
        pred_df['flow3'] * (pred_df['path3_cost'] - pred_df['min_path_cost'])
    )
    avg_delay = pred_df['delay'].sum()/pred_df['demand'].sum()
    #return average delay in minutes
    return avg_delay*60

def mean_path_cost(stat):
    path_link_df = get_origin_path(stat)
    UE_link = get_UE_link_cost(stat)

    path_link_df['path1_cost'] = path_link_df['path1'].apply(lambda x: calculate_path_cost(x, UE_link))
    path_link_df['path2_cost'] = path_link_df['path2'].apply(lambda x: calculate_path_cost(x, UE_link))
    path_link_df['path3_cost'] = path_link_df['path3'].apply(lambda x: calculate_path_cost(x, UE_link))

    flows = stat['path_flow']
    path_link_df['flow1'] = [f[0] for f in flows]
    path_link_df['flow2'] = [f[1] for f in flows]
    path_link_df['flow3'] = [f[2] for f in flows]

    avg_path_cost = (np.mean(path_link_df['path1_cost']) + np.mean(path_link_df['path2_cost']) + np.mean(path_link_df['path3_cost']))/3
    return UE_link, path_link_df, avg_path_cost

def single_avg_delay(pred_tensor, filename):
    """ len_origin: number of OD pair in origin dataset
    len_pred: number of OD pair in predicted value
    nan_num: number of nan value 
    """
    stat = read_file(filename)
    pred_df, len_origin, len_pred, nan_num = create_pred_df(pred_tensor, stat)
    pred_link_flow = sum_pred_link_flow(pred_df, stat)
    # Avg delay of predicted flow
    pred_avg_delay = calculate_delay(pred_df, pred_link_flow)
    # Avg delay of solution
    UE_link, path_link_df, avg_path_cost = mean_path_cost(stat)
    solution_avg_delay = calculate_delay(path_link_df, UE_link)
    return pred_avg_delay, solution_avg_delay, len_origin, len_pred, nan_num