import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
from collections import defaultdict
import random
import ast
import os
import concurrent.futures

def load_files_from_folders(folder, max_files):
    files = os.listdir(folder)

    file_list = []
    for i, file in enumerate(files):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path) and i < max_files:
            file_list.append(file_path)
    return file_list

# def load_files_from_folders(folders, max_files):
#     file_list = []
#     for folder in folders:
#         for i in range(max_files):
#             file = ''.join([folder, str(i)])
#             file_list.append(file)
#     return file_list

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

def normalize(tensor):
    scaler = MinMaxScaler()
    normed = scaler.fit_transform(tensor)
    return normed

def normalizeY(tensor):
    scaler = MinMaxScaler()
    normed = scaler.fit_transform(tensor)
    return normed, scaler

# def normalizeY(tensor):
#     # Normalize by row
#     if not isinstance(tensor, np.ndarray):
#         tensor = np.array(tensor)
#     scaler = MinMaxScaler()
#     normed = scaler.fit_transform(np.transpose(tensor))
#     tensor = np.transpose(normed)
#     tensor = tensor[:, :-1] # get 1st 3 columns, ignore the last column of demand
#     return tensor, scaler

def create_matrix(data, nodes):
    matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
    # matrix = np.zeros((24, 24), dtype=float)
    for (o, d), v in data:
        matrix[int(o)-1][int(d)-1] = v
    return matrix.reshape(-1, 1)

def get_graphMatrix(network, nodes):
    # 625x4
    # cap = np.array(network[['init_node', 'term_node', 'capacity']].apply(lambda row: ((row['init_node'], row['term_node']), row['capacity']), axis=1).tolist(), dtype=object)
    length = np.array(network[['init_node', 'term_node', 'length']].apply(lambda row: ((row['init_node'], row['term_node']), row['length']), axis=1).tolist(), dtype=object)
    fft_c = np.array(network[['init_node', 'term_node', 'free_flow_time']].apply(lambda row: ((row['init_node'], row['term_node']), row['free_flow_time']), axis=1).tolist(), dtype=object)
    fft_t = np.array(network[['init_node', 'term_node', 'fft_t']].apply(lambda row: ((row['init_node'], row['term_node']), row['fft_t']), axis=1).tolist(), dtype=object)

    # Cap = create_matrix(cap, nodes)
    Length = create_matrix(length, nodes)
    Fft_c = create_matrix(fft_c, nodes)
    Fft_t = create_matrix(fft_t, nodes)

    # matrix = np.concatenate((normalize(Cap), np.log1p(Length), np.log1p(Fft_c), np.log1p(Fft_t)), axis=1)
    matrix = np.concatenate((np.log1p(Length), np.log1p(Fft_c), np.log1p(Fft_t)), axis=1)
    return matrix

def get_demandMatrix(demand, nodes):
    matrix_c = np.array([(key, value[0]) for key, value in demand.items()], dtype=object)
    matrix_t = np.array([(key, value[1]) for key, value in demand.items()], dtype=object)
    matrix_c = create_matrix(matrix_c, nodes)
    matrix_t = create_matrix(matrix_t, nodes)
    return matrix_c, matrix_t

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

# Get path flow distribution (Y), return 2 tensors of dimension 3
def to_percentage_list(lst):
    total = sum(lst)
    if total == 0:
        return [0.0, 0.0, 0.0]
    return [x / total for x in lst]

def process_flows(demand, path_flows):
    flows = [[(od, path[i] if i < len(path) else 0) for i in range(3)] 
             for od, path in zip(demand.keys(), path_flows)]
    # Change the value to percentage
    # flows = [[f[0], to_percentage_list(list(f[1]))] for f in flows]
    return np.array(flows, dtype=object)

def get_flowMatrix(demand, path_flows, nodes_size):
    flow_c = process_flows(demand, [[f[i][0] for i in range(len(f))] for f in path_flows])
    flow_t = process_flows(demand, [[f[i][1] for i in range(len(f))] for f in path_flows])

    matrices_c = [create_matrix(flow_c[:, i], nodes_size) for i in range(3)]
    matrices_t = [create_matrix(flow_t[:, i], nodes_size) for i in range(3)]

    matrix_c = np.concatenate(matrices_c, axis=1)
    matrix_t = np.concatenate(matrices_t, axis=1)
    
    return matrix_c, matrix_t

# No mask model
def generate_xy(file_name, unique_set, test_set=None):
    stat = read_file(file_name)
    path_links = stat["data"]["paths_link"]
    demand = stat["data"]["demand"]
    nodes = stat["data"]["nodes"]
    net = stat["data"]["network"]
    net['fft'] = net['fft'].apply(lambda x: ast.literal_eval(x))
    net['fft_t'] = net['fft'].apply(lambda x: int(x[1]))

    path_flows = stat["path_flow"]

    # Get X
    Graph = get_graphMatrix(net, nodes) #return normalized data
    matrix_c, matrix_t = get_demandMatrix(demand, nodes)
    OD_demand = np.concatenate((matrix_c, matrix_t), axis=1)
    Path_tensor = get_pathMatrix(path_links, nodes, unique_set)

    X = np.concatenate((Graph, normalize(OD_demand), normalize(Path_tensor)), axis=1)
    X = tf.convert_to_tensor(X, dtype=tf.float32) # (nodexnode) x 9

    # # Get Y
    Y_c, Y_t = get_flowMatrix(demand, path_flows, nodes)
    Y_c = [to_percentage_list(y) for y in Y_c]
    Y_t = [to_percentage_list(y) for y in Y_t]

    # Concat Y car and Y truck, normalize by column, return 1 scaler.
    Y = np.concatenate((Y_c, Y_t), axis=1)
    Y, scaler = normalizeY(Y)
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

    path_link_df['demand_c'] = path_link_df['demand'].apply(lambda x: x[0])
    path_link_df['demand_t'] = path_link_df['demand'].apply(lambda x: x[1])

    return path_link_df

def calculate_link_cost(row, type):
    if row.iloc[2] == 0:
        return np.nan
    elif type == 'c':
        return round(row.iloc[0]*(1+0.15*((row.iloc[1]/row.iloc[2])**4)), 2)
    else: 
        return round(row.iloc[0]*(1+0.15*((2.5 * row.iloc[1]/row.iloc[2])**4)), 2)

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

# get dictionary of path flow for each OD pair from predicted tensor
def extract_path_flow(pred_tensor):
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

def process_pred_df(final_dict, path_set, UE_path_flow, type):
    pred_df = pd.DataFrame.from_dict(final_dict, orient='index', columns=['flow1', 'flow2', 'flow3']).reset_index()
    pred_df.rename(columns={'index': 'od'}, inplace=True)
    pred_df = pd.merge(pred_df, path_set, how='right', on='od')

    pred_df = pred_df.fillna(0)
    pred_df.loc[pred_df['path1'] == 0, 'flow1'] = 0
    pred_df.loc[pred_df['path2'] == 0, 'flow2'] = 0
    pred_df.loc[pred_df['path3'] == 0, 'flow3'] = 0

    pred_df.loc[pred_df['flow1'] < 0, 'flow1'] = 0
    pred_df.loc[pred_df['flow2'] < 0, 'flow2'] = 0
    pred_df.loc[pred_df['flow3'] < 0, 'flow3'] = 0

    pred_df['flow1'] = pred_df['flow1'] * pred_df[f'demand_{type}']
    pred_df['flow2'] = pred_df['flow2'] * pred_df[f'demand_{type}']
    pred_df['flow3'] = pred_df['flow3'] * pred_df[f'demand_{type}']

    pred_df['flow1'] = round(pred_df['flow1'], 0)
    pred_df['flow2'] = round(pred_df['flow2'], 0)
    pred_df['flow3'] = round(pred_df['flow3'], 0)

    pred_df['UE_flow1'] = [f[0] if len(f)>0 else 0 for f in UE_path_flow]
    pred_df['UE_flow2'] = [f[1] if len(f)>1 else 0 for f in UE_path_flow]
    pred_df['UE_flow3'] = [f[2] if len(f)>2 else 0 for f in UE_path_flow]
    return pred_df

# 1
def create_pred_df(tensor_c, tensor_t, stat):
    # return a df of predicted path flow and UE path flow
    flows = stat['path_flow']
    UE_path_flow_c = [[f[i][0] for i in range(len(f))] for f in flows]
    UE_path_flow_t = [[f[i][1] for i in range(len(f))] for f in flows]

    final_dict_c = extract_path_flow(tensor_c) 
    final_dict_t = extract_path_flow(tensor_t)
    path_set = get_origin_path(stat)[['od', 'demand_c', 'demand_t', 'path1', 'path2', 'path3']]

    pred_df_c = process_pred_df(final_dict_c, path_set, UE_path_flow_c, 'c')
    pred_df_t = process_pred_df(final_dict_t, path_set, UE_path_flow_t, 't')

    return pred_df_c, pred_df_t

def compare_path_flow(pred_UE_path_flow):
    melted = pd.melt(pred_UE_path_flow,
                    value_vars=['path1', 'path2', 'path3', 'flow1', 'flow2', 'flow3', 'UE_flow1', 'UE_flow2', 'UE_flow3'],
                    var_name='variable', value_name='value')

    melted['path_type'] = melted['variable'].str.extract('(path|flow|UE_flow)')
    melted['id'] = melted.groupby('path_type').cumcount()

    compare = melted.pivot(index='id', columns='path_type', values='value').reset_index(drop=True)
    compare['abs_err'] = (compare['flow'] - compare['UE_flow']).abs()
    compare['sqr_err'] = compare['abs_err']**2
    compare['mape'] = compare[compare['UE_flow']!=0]['abs_err']/compare[compare['UE_flow']!=0]['UE_flow']*100
    # compare_path_flow[~compare_path_flow['mape'].isna()]['mape'].mean()
    return compare

# 2 - Get link flow and cost from predicted path flow
def compare_link_flow(pred_df, stat, type):
    pred_path_flow = pred_df[['flow1', 'flow2', 'flow3']].values.tolist()
    path_link = stat['data']['paths_link']

    pred_link_flow = extract_link_flow(path_link, pred_path_flow)
    pred_link_flow = pd.DataFrame.from_dict(pred_link_flow, orient='index', columns=['link_flow']).sort_index(ascending=True).reset_index()
    pred_link_flow.rename(columns={'index': 'link_id'}, inplace=True)

    link = stat['data']['network'].copy()
    link['fft'] = link['fft'].apply(lambda x: ast.literal_eval(x))
    if type == 'c':
        link['free_flow_time'] = link['fft'].apply(lambda x: int(x[0]))
    else: 
        link['free_flow_time'] = link['fft'].apply(lambda x: int(x[1]))

    output = pd.merge(link, pred_link_flow, how='left', on='link_id')
    output = output.fillna(0)
    output['link_cost'] = output[['free_flow_time','link_flow', 'capacity']].apply(lambda x: calculate_link_cost(x, type), axis=1)

    link_flow_c = [f[0] for f in stat['link_flow']]
    link_flow_t = [f[1] for f in stat['link_flow']]

    if type == 'c':
        output['UE_flow'] = output['link_id'].apply(lambda x: link_flow_c[x])

    else: 
        output['UE_flow'] = output['link_id'].apply(lambda x: link_flow_t[x])

    output['UE_link_cost'] = output[['free_flow_time','UE_flow', 'capacity']].apply(lambda x: calculate_link_cost(x, type), axis=1)
    output['abs_err'] = (output['link_flow'] - output['UE_flow']).abs()
    output['sqr_err'] = output['abs_err']**2
    output['mape'] = output[output['UE_flow']!=0]['abs_err']/output[output['UE_flow']!=0]['UE_flow']*100
    return output[['link_id', 'link_flow','link_cost','UE_flow','UE_link_cost','abs_err','sqr_err', 'mape']]
    # return output

def calculate_path_cost(row, link_df):
    if pd.isna(row): 
        return np.nan
    
    if isinstance(row, int):
        return np.nan

    sum_cost = 0
    for link in row:
        sum_cost += link_df[link_df['link_id']==link].iloc[:, 1].iloc[0]
        
    return round(sum_cost, 2)

# 3 - Avg delay of predicted/solution flow
def get_delay(path_flow, link_flow, type):
    # Calculate path cost for each od pair
    path_flow['path1_cost'] = path_flow['path1'].apply(lambda x: calculate_path_cost(x,link_flow))
    path_flow['path2_cost'] = path_flow['path2'].apply(lambda x: calculate_path_cost(x,link_flow))
    path_flow['path3_cost'] = path_flow['path3'].apply(lambda x: calculate_path_cost(x,link_flow))
    path_flow['min_path_cost'] = path_flow[['path1_cost', 'path2_cost', 'path3_cost']].min(axis=1)
    path_flow = path_flow.fillna(0)
    path_flow['delay'] = (
        path_flow['flow1'] * (path_flow['path1_cost'] - path_flow['min_path_cost']) +
        path_flow['flow2'] * (path_flow['path2_cost'] - path_flow['min_path_cost']) +
        path_flow['flow3'] * (path_flow['path3_cost'] - path_flow['min_path_cost'])
    )

    avg_delay = path_flow['delay'].sum()/path_flow[f'demand_{type}'].sum()
    mean_path_cost = (np.nanmean(path_flow['path1_cost']) + np.nanmean(path_flow['path2_cost']) + np.nanmean(path_flow['path3_cost']))/3
    return avg_delay, mean_path_cost

def single_avg_delay(pred_tensor_c, pred_tensor_t, filename):
    def get_flow_error(pred_UE_path_flow, stat, type):
        path_flow_err = compare_path_flow(pred_UE_path_flow)
        link_flow_err = compare_link_flow(pred_UE_path_flow, stat, type)

        avg_pred_delay, pred_mean_cost = get_delay(pred_UE_path_flow, link_flow_err[['link_id', 'link_cost']], type)
        UE_delay = pred_UE_path_flow[['od', f'demand_{type}', 'path1','path2','path3', 'UE_flow1','UE_flow2','UE_flow3']]
        UE_delay = UE_delay.rename(columns={'UE_flow1': 'flow1', 'UE_flow2': 'flow2', 'UE_flow3': 'flow3'})
        avg_UE_delay, UE_mean_cost = get_delay(UE_delay, link_flow_err[['link_id', 'UE_link_cost']], type)

        return link_flow_err, path_flow_err, avg_pred_delay, avg_UE_delay, pred_mean_cost, UE_mean_cost

    stat = read_file(filename)
    pred_UE_path_flow_c, pred_df_t = create_pred_df(pred_tensor_c, pred_tensor_t, stat)
    err_c = get_flow_error(pred_UE_path_flow_c, stat, 'c')
    err_t = get_flow_error(pred_df_t, stat, 't')    
    return err_c, err_t

def calculate_indicator(flowList):
    mse = np.mean([np.mean(flowList[x]['sqr_err']) for x in range(len(flowList))])
    mae = np.mean([np.mean(flowList[x]['abs_err']) for x in range(len(flowList))])
    rmse = np.sqrt(mse)
    mape = np.mean([np.mean(flowList[x]['mape']) for x in range(len(flowList))])
    return [round(mae,2), round(rmse,2), round(mape,2)]

def single_avg_delay_wrapper(c, t, filename):
    return single_avg_delay(c, t, filename)

def aggregate_result(pred_tensor_c, pred_tensor_t, test_files):
    def append_results(results, link_flow, path_flow, avg_delay, ue_avg_delay, pred_mean_cost, ue_mean_cost):
        link_flow.append(results[0])
        path_flow.append(results[1])
        avg_delay.append(results[2])
        ue_avg_delay.append(results[3])
        pred_mean_cost.append(results[4])
        ue_mean_cost.append(results[5])

    size = len(test_files)
    metrics_c = {'link_flow': [], 'path_flow': [], 'avg_delay': [], 'ue_avg_delay': [], 'pred_mean_cost': [], 'ue_mean_cost': []}
    metrics_t = {'link_flow': [], 'path_flow': [], 'avg_delay': [], 'ue_avg_delay': [], 'pred_mean_cost': [], 'ue_mean_cost': []}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(single_avg_delay_wrapper, c, t, filename): filename 
                   for c, t, filename in zip(pred_tensor_c[:size], pred_tensor_t[:size], test_files[:size])}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=size):
            err_c, err_t = future.result()

            # Append results for 'c' and 't'
            append_results(err_c, metrics_c['link_flow'], metrics_c['path_flow'], metrics_c['avg_delay'], 
                           metrics_c['ue_avg_delay'], metrics_c['pred_mean_cost'], metrics_c['ue_mean_cost'])

            append_results(err_t, metrics_t['link_flow'], metrics_t['path_flow'], metrics_t['avg_delay'], 
                           metrics_t['ue_avg_delay'], metrics_t['pred_mean_cost'], metrics_t['ue_mean_cost'])
            
    
    def summarize_metrics(metrics):
        avg_delay = np.mean(metrics['avg_delay'])
        ue_avg_delay = np.mean(metrics['ue_avg_delay'])
        link_indicator = calculate_indicator(metrics['link_flow'])
        path_indicator = calculate_indicator(metrics['path_flow'])

        rows = ['MAE', 'RMSE', 'MAPE']
        result_df = pd.DataFrame({'Indicator': rows, 
                                  'Link flow': link_indicator, 
                                  'Path flow': path_indicator})

        return result_df, metrics['link_flow'], metrics['path_flow'], metrics['pred_mean_cost'], metrics['ue_mean_cost'], avg_delay, ue_avg_delay

    # Summarize metrics for 'c' and 't'
    result_c = summarize_metrics(metrics_c)
    result_t = summarize_metrics(metrics_t)

    return result_c, result_t