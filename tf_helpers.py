import numpy as np
import pickle
import tensorflow as tf
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
from collections import defaultdict
import random

def load_files_from_folders(folders, max_files):
    file_list = []
    for folder in folders:
        for i in range(max_files):
            # file = ''.join([folder,str(f'/5by5_Data{i}')])
            file = ''.join([folder, str(i)])
            file_list.append(file)
    return file_list

def read_file(filename):
  with open(filename, "rb") as file:
      stat = pickle.load(file)
      file.close()
  return stat

def split_dataset(files, train_ratio, val_ratio):
    random.shuffle(files)

    total_files = len(files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)

    train_files = files[:train_size]
    val_files = files[train_size:train_size + val_size]
    test_files = files[train_size + val_size:]

    return train_files, val_files, test_files

# Need to load all files in dataset to get unique path dict
def path_encoder(files):
    path_sample = []
    for file_name in files:
        stat = read_file(file_name)
        path_sample.append(stat["data"]["paths_link"])

    all_path_link = [path_sample[i].values() for i in range(len(path_sample))]
    unique_values_set = {tuple(p) for path_set in all_path_link for path in path_set for p in path}
    path_set_dict = {v: k for k, v in enumerate(unique_values_set, start=1)}
    return path_set_dict

# path_encoded = path_encoder()

def normalize(tensor):
    scaler = MinMaxScaler()
    normed = scaler.fit_transform(tensor)
    return normed

# def normalizeY(tensor):
#     scaler = MinMaxScaler()
#     normed = scaler.fit_transform(tensor)
#     return normed, scaler

def normalizeY(tensor):
    # Normalize by row
    if not isinstance(tensor, np.ndarray):
        tensor = np.array(tensor)
    scaler = MinMaxScaler()
    normed = scaler.fit_transform(np.transpose(tensor))
    tensor = np.transpose(normed)
    tensor = tensor[:, :-1] # get 1st 3 columns, ignore the last column of demand
    return tensor, scaler

def create_matrix(data, nodes):
    # data is an array, nodes is a set
    # matrix = np.zeros((len(nodes), len(nodes)))
    matrix = np.zeros((24, 24))
    for (o, d), v in data:
        o = int(o)
        d = int(d)
        matrix[o-1][d-1] = v
    matrix = matrix.reshape(-1, 1).astype(float) # 625x1
    return matrix

def get_graphMatrix(network, nodes):
    # 625x3
    cap = np.array(network[['init_node', 'term_node', 'capacity']].apply(lambda row: ((row['init_node'], row['term_node']), row['capacity']), axis=1).tolist(), dtype=object)
    length = np.array(network[['init_node', 'term_node', 'length']].apply(lambda row: ((row['init_node'], row['term_node']), row['length']), axis=1).tolist(), dtype=object)
    fft = np.array(network[['init_node', 'term_node', 'free_flow_time']].apply(lambda row: ((row['init_node'], row['term_node']), row['free_flow_time']), axis=1).tolist(), dtype=object)

    # Cap = create_matrix(cap, nodes)
    Length = create_matrix(length, nodes)
    Fft = create_matrix(fft, nodes)

    # matrix = np.concatenate((normalize(Cap), np.log1p(Length), np.log1p(Fft)), axis=1)
    matrix = np.concatenate((np.log1p(Length), np.log1p(Fft)), axis=1)
    return matrix

def get_demandMatrix(demand, nodes):
    # 625x1
    tensor = np.array([(key, value) for key, value in demand.items()], dtype=object)
    tensor = create_matrix(tensor, nodes)
    return tensor

# Get 3 feasible paths for each OD pair, return tensor shape 625x3
def get_pathMatrix(path_links, nodes, unique_set):
    # 625x3
    paths = np.array([(key, [tuple(path) for path in value]) for key, value in path_links.items()], dtype=object)
    p1, p2, p3 = [], [], []
    for od, path_list in paths:
        path1 = path2 = path3 = 0

        if len(path_list) > 0:
            path1 = path_list[0]
        if len(path_list) > 1:
            path2 = path_list[1]
        if len(path_list) > 2:
            path3 = path_list[2]

        p1.append((od, unique_set[path1] if path1 != 0 else 0))
        p2.append((od, unique_set[path2] if path2 != 0 else 0))
        p3.append((od, unique_set[path3] if path3 != 0 else 0))
    p1 = create_matrix(p1, nodes)
    p2 = create_matrix(p2, nodes)
    p3 = create_matrix(p3, nodes)
    matrix = np.concatenate((p1, p2, p3), axis=1)
    return matrix

# Get path flow distribution (Y), return a tensor 625x3
def get_flowMatrix(demand, path_flows, nodes):
    # 625x3
    flows = np.array([(k, v) for k, v in zip(demand.keys(), path_flows)], dtype=object)
    p1, p2, p3 = [], [], []
    for od, flow in flows:
        path1 = path2 = path3 = 0
        if len(flow) > 0:
            path1 = flow[0]
        if len(flow) > 1:
            path2 = flow[1]
        if len(flow) > 2:
            path3 = flow[2]

        p1.append((od, path1 if path1 != 0 else 0))
        p2.append((od, path2 if path2 != 0 else 0))
        p3.append((od, path3 if path3 != 0 else 0))
    p1 = create_matrix(p1, nodes)
    p2 = create_matrix(p2, nodes)
    p3 = create_matrix(p3, nodes)

    matrix = np.concatenate((p1, p2, p3), axis=1)
    return matrix

def get_frequency(path_link):
    a = tuple(tuple(p) for path in path_link.values() for p in path)
    frequency_dict = {}
    for sublist in a:
        for value in sublist:
            if value in frequency_dict:
                frequency_dict[value] += 1
            else:
                frequency_dict[value] = 1
    frequency_dict = dict(sorted(frequency_dict.items()))
    return frequency_dict

def map_frequence(row, frequency_dict):
    if row in frequency_dict.keys():
        return frequency_dict[row]
    return 0

def get_frequenceMatrix(path_link, net, nodes):
    frequency_dict = get_frequency(path_link)
    net['frequence'] = net['link_id'].apply(lambda x: map_frequence(x, frequency_dict))
    frequence = np.array(net[['init_node', 'term_node', 'frequence']].apply(lambda row: ((row['init_node'], row['term_node']), row['frequence']), axis=1).tolist(), dtype=object)
    frequence = create_matrix(frequence, nodes)
    return frequence

def create_mask(matrix):
    tensor = tf.convert_to_tensor(matrix, dtype=tf.float32)
    mask = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(tensor), axis=-1)),-1) # create mask for row
    return mask

def to_percentage_list(lst):
    total = sum(lst)
    if total == 0:
        return [0.0, 0.0, 0.0]
    return [x / total for x in lst]

# Mask model
# def generate_xy(file_name, unique_set, test_set=None):
#     with open(file_name, "rb") as file:
#         stat = pickle.load(file)
#         file.close()

#     path_links = stat["data"]["paths_link"]
#     demand = stat["data"]["demand"]
#     path_flows = stat["path_flow"]
#     nodes = stat["data"]["nodes"]
#     net = stat["data"]["network"]

#     # Get X
#     Graph = get_graphMatrix(net, nodes)
#     OD_demand = get_demandMatrix(demand, nodes)
#     Path_tensor = get_pathMatrix(path_links, nodes, unique_set)
#     Frequence = get_frequenceMatrix(path_links, net, nodes)

#     X = np.concatenate((Graph, OD_demand, Path_tensor, Frequence), axis=1)
#     X_mask = create_mask(X) # type tensor shape 625x1
#     X = normalize(X)
#     X = tf.convert_to_tensor(X, dtype=tf.float32) # 625x8

#     # Get Y
#     Flow = get_flowMatrix(demand, path_flows, nodes)
#     Y = np.concatenate((Flow, OD_demand), axis=1)
#     Y_mask = create_mask(Y)
#     Y, scaler = normalizeY(Y)
#     Y = tf.convert_to_tensor(Y, dtype=tf.float32)

#     if test_set:
#         return X, Y, X_mask, Y_mask, scaler
#     return X, Y, X_mask, Y_mask

# No mask model
def generate_xy(file_name, unique_set, test_set=None):
    with open(file_name, "rb") as file:
        stat = pickle.load(file)
        file.close()

    path_links = stat["data"]["paths_link"]
    demand = stat["data"]["demand"]
    path_flows = stat["path_flow"]
    # path_flows = [to_percentage_list(inner_list) for inner_list in path_flows]
    nodes = stat["data"]["nodes"]
    net = stat["data"]["network"]

    # Get X
    Graph = get_graphMatrix(net, nodes) #return normalized data
    OD_demand = get_demandMatrix(demand, nodes)
    Path_tensor = get_pathMatrix(path_links, nodes, unique_set)

    X = np.concatenate((Graph, normalize(OD_demand), normalize(Path_tensor)), axis=1)
    # X = normalize(X)
    X = tf.convert_to_tensor(X, dtype=tf.float32) # 625x6

    # Get Y
    Y = get_flowMatrix(demand, path_flows, nodes)
    # a = np.ones_like(OD_demand)
    Y = np.concatenate((Y, OD_demand), axis=1)
    Y, scaler = normalizeY(Y)
    # Y = to_percentage_list(Y)
    Y = tf.convert_to_tensor(Y, dtype=tf.float32)

    if test_set:
        return X, Y, scaler
    return X, Y

"""
CHECK UE CONDITIONS OF PREDICTED OUTPUT
"""

def get_origin_path(stat):
    path_link = stat['data']['paths_link']
    od = [k for k in path_link.keys()]
    path1 = [tuple(p[0]) if len(p) > 0 else np.nan for p in path_link.values()]
    path2 = [tuple(p[1]) if len(p) > 1 else np.nan for p in path_link.values()]
    path3 = [tuple(p[2]) if len(p) > 2 else np.nan for p in path_link.values()]

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
    if pd.isna(row): 
        return np.nan
    if row == 0:
        return np.nan

    sum_time = 0
    for link in row:
        sum_time += link_df[link_df['link_id']==link]['link_cost'].iloc[0]
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
    x = int(np.sqrt(pred_tensor.shape[0]))
    pred1 = pred_tensor[:, 0].reshape(x, x)
    pred2 = pred_tensor[:, 1].reshape(x, x)
    pred3 = pred_tensor[:, 2].reshape(x, x)

    dict1 = {(i+1, j+1): pred1[i, j] for i in range(pred1.shape[0]) for j in range(pred1.shape[1])}
    dict2 = {(i+1, j+1): pred2[i, j] for i in range(pred2.shape[0]) for j in range(pred2.shape[1])}
    dict3 = {(i+1, j+1): pred3[i, j] for i in range(pred3.shape[0]) for j in range(pred3.shape[1])}

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
    pred_df.loc[pred_df['path1'] == 0, 'flow1'] = 0
    pred_df.loc[pred_df['path2'] == 0, 'flow2'] = 0
    pred_df.loc[pred_df['path3'] == 0, 'flow3'] = 0

    pred_df.loc[pred_df['flow1'] < 0, 'flow1'] = 0
    pred_df.loc[pred_df['flow2'] < 0, 'flow2'] = 0
    pred_df.loc[pred_df['flow3'] < 0, 'flow3'] = 0

    pred_df['flow1'] = round(pred_df['flow1'], 0)
    pred_df['flow2'] = round(pred_df['flow2'], 0)
    pred_df['flow3'] = round(pred_df['flow3'], 0)

    # pred_df['flow1'] = pred_df['flow1'] * pred_df['demand']
    # pred_df['flow2'] = pred_df['flow2'] * pred_df['demand']
    # pred_df['flow3'] = pred_df['flow3'] * pred_df['demand']

    return pred_df, len(stat['path_flow']), len(final_dict), nan_num

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
    pred_df = pred_df.fillna(0)
    pred_df['delay'] = (
        pred_df['flow1'] * (pred_df['path1_cost'] - pred_df['min_path_cost']) +
        pred_df['flow2'] * (pred_df['path2_cost'] - pred_df['min_path_cost']) +
        pred_df['flow3'] * (pred_df['path3_cost'] - pred_df['min_path_cost'])
    )
    avg_delay = pred_df['delay'].sum()/pred_df['demand'].sum()
    #return average delay in minutes
    return pred_df, avg_delay

def mean_path_cost(stat):
    path_link_df = get_origin_path(stat)
    UE_link = get_UE_link_cost(stat)

    path_link_df['path1_cost'] = path_link_df['path1'].apply(lambda x: calculate_path_cost(x, UE_link))
    path_link_df['path2_cost'] = path_link_df['path2'].apply(lambda x: calculate_path_cost(x, UE_link))
    path_link_df['path3_cost'] = path_link_df['path3'].apply(lambda x: calculate_path_cost(x, UE_link))

    flows = stat['path_flow']
    p1, p2, p3 = [], [], []
    for flow in flows:
        path1 = path2 = path3 = 0
        if len(flow) > 0:
            path1 = flow[0]
        if len(flow) > 1:
            path2 = flow[1]
        if len(flow) > 2:
            path3 = flow[2]

        p1.append((path1 if path1 != 0 else 0))
        p2.append((path2 if path2 != 0 else 0))
        p3.append((path3 if path3 != 0 else 0))
    path_link_df['flow1'] = p1
    path_link_df['flow2'] = p2
    path_link_df['flow3'] = p3

    avg_path_cost = (np.mean(path_link_df['path1_cost']) + np.mean(path_link_df['path2_cost']) + np.mean(path_link_df['path3_cost']))/3
    return UE_link, path_link_df, avg_path_cost

def compare_link_flow(UE_link, pred_link_flow):
    # Calculate abs err and sqr err of link flow
    UE_link = UE_link[['link_id', 'link_flow']]
    UE_link = UE_link.rename(columns={'link_flow': 'UE_flow'})
    link_flow = pd.merge(pred_link_flow, UE_link, on='link_id', how='right')
    link_flow = link_flow.drop(['capacity','free_flow_time', 'b', 'link_cost'], axis=1)
    link_flow['abs_err'] = (link_flow['link_flow'] - link_flow['UE_flow']).abs()
    link_flow['sqr_err'] = link_flow['abs_err']**2
    return link_flow

def get_all_path_flow(df):
    # Transform the table of 6 columns to 2 columns
    path_df = pd.melt(df, value_vars=['path1', 'path2', 'path3'], var_name='path_type', value_name='path')
    flow_df = pd.melt(df, value_vars=['flow1', 'flow2', 'flow3'], var_name='flow_type', value_name='flow')
    result_df = pd.concat([path_df['path'], flow_df['flow']], axis=1)
    return result_df

def compare_path_flow(path_link_df, pred_df):
    # Calculate abs err and sqr err of path flow
    pred_path_flow = get_all_path_flow(pred_df)
    UE_path_flow = get_all_path_flow(path_link_df)
    path_flow = pd.merge(UE_path_flow, pred_path_flow, on='path', how='left')
    path_flow = path_flow.rename(columns={'flow_x': 'UE_flow', 'flow_y': 'pred_flow'})
    path_flow['abs_err'] = (path_flow['pred_flow'] - path_flow['UE_flow']).abs()
    path_flow['sqr_err'] = path_flow['abs_err']**2
    path_flow = path_flow[~path_flow['abs_err'].isna()]
    return path_flow

def calculate_indicator(flowList):
    mse = np.mean([np.mean(flowList[x]['sqr_err']) for x in range(len(flowList))])
    mae = np.mean([np.mean(flowList[x]['abs_err']) for x in range(len(flowList))])
    rmse = np.sqrt(mse)
    mape = [df['abs_err'][df['UE_flow']!=0]/ df['UE_flow'][df['UE_flow']!=0] for df in flowList]
    mape = np.mean([j for i in mape for j in i])*100
    return [round(mae,2), round(rmse,2), round(mape,2)]

def single_avg_delay(pred_tensor, filename):
    """ len_origin: number of OD pair in origin dataset
    len_pred: number of OD pair in predicted value
    nan_num: number of nan value
    """
    stat = read_file(filename)
    pred_df, len_origin, len_pred, nan_num = create_pred_df(pred_tensor, stat)
    pred_link_flow = sum_pred_link_flow(pred_df, stat)
    # Avg delay of predicted flow
    pred_df, pred_avg_delay = calculate_delay(pred_df, pred_link_flow)
    # Avg delay of solution
    UE_link, path_link_df, avg_path_cost = mean_path_cost(stat)
    a, solution_avg_delay = calculate_delay(path_link_df, UE_link)

    link_flow = compare_link_flow(UE_link, pred_link_flow)
    path_flow = compare_path_flow(path_link_df, pred_df)
    return [link_flow, path_flow],[pred_avg_delay, solution_avg_delay], [len_pred, len_origin], nan_num, avg_path_cost