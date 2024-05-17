import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tqdm.notebook import tqdm

# Create dictionary of all unique paths
def path_encoder():
    path_sample = []
    for i in range(10):
        file_name = f"Output/5by5_Data{i}"
        with open(file_name, "rb") as file:
            stat = pickle.load(file)
        path_sample.append(stat["data"]["paths_link"])

    all_path_link = [path_sample[i].values() for i in range(len(path_sample))]
    unique_values_set = {tuple(p) for path_set in all_path_link for path in path_set for p in path}
    path_set_dict = {v: k for k, v in enumerate(unique_values_set, start=1)}
    return path_set_dict

# Normalize tensor by Min-max scaling
def normalize(tensor):
    min_values = tf.reduce_min(tensor, axis=0)
    max_values = tf.reduce_max(tensor, axis=0)
    
    scaled_tensor = (tensor - min_values) / (max_values - min_values)
    mask = tf.equal(min_values, max_values)
    scaled_tensor = tf.where(mask, 1.0, scaled_tensor)
    return scaled_tensor

def get_Link_Path_adj(net, path_encoded):
    link_path = tf.zeros((net.shape[0], len(path_encoded)), dtype=tf.float32)
    for p, index in path_encoded.items():
        for link in p:
            link_path = link_path.write(link, link_path.read(link).scatter(tf.IndexedSlices(1.0, [index-1])))
    return link_path.stack()

def get_graphTensor(network):
    data_columns = ['init_node', 'term_node', 'capacity', 'length', 'free_flow_time']
    data = tf.convert_to_tensor(network[data_columns].values, dtype=tf.float32)
    return data

def convert_DictToTensor(OD_demand, nodes):
    matrix = np.zeros((len(nodes), len(nodes)))
    for (o, d), v in OD_demand.items():
        matrix[o-1][d-1] = v
    return tf.convert_to_tensor([matrix], dtype=tf.float32)

# Transform a dictionary to a tensor, flatten, transpose, and unsqueeze to get a 3D tensor
def preprocessTensor(OD_dict, nodes):
    tensor = convert_DictToTensor(OD_dict, nodes)  # size (1, 25, 25)
    tensor = tf.reshape(tensor, (1, -1))  # shape 1x625
    tensor = tf.transpose(tensor)  # shape 625x1
    return tensor

def preprocess_path(path_links, node, path_encoded):
    new_path_links = {k: [tuple(path) for path in v] for k, v in path_links.items()}
    
    # Encode paths
    p1, p2, p3 = {}, {}, {}
    for k, v in new_path_links.items():
        p1[k], p2[k], p3[k] = [path_encoded[path] for path in v]

    # Extract 3 path tensors
    p1 = preprocessTensor(p1, node)  # 625x1
    p2 = preprocessTensor(p2, node)
    p3 = preprocessTensor(p3, node)
    stack = tf.stack([p1, p2, p3], axis=1)
    stack = tf.squeeze(stack, -1)
    return stack

def preprocess_flow(demand, path_flows, nodes):
    flows = {k: v for k, v in zip(demand.keys(), path_flows)}
    f1, f2, f3 = {}, {}, {}
    for k, v in flows.items():
        f1[k], f2[k], f3[k] = v

    # Extract 3 flow tensors
    f1 = preprocessTensor(f1, nodes)
    f2 = preprocessTensor(f2, nodes)
    f3 = preprocessTensor(f3, nodes)
    stack = tf.stack([f1, f2, f3], axis=1)
    return stack

def get_X(Graph, OD_demand, Path_tensor):
    X = [Graph, OD_demand, Path_tensor]
    max_size = max([x.shape[0] for x in X])

    Graph = tf.pad(Graph, paddings=[[0, max_size - Graph.shape[0]], [0, 0]])
    OD_demand = tf.pad(OD_demand, paddings=[[0, max_size - OD_demand.shape[0]], [0, 0]])
    Path_tensor = tf.pad(Path_tensor, paddings=[[0, max_size - Path_tensor.shape[0]], [0, 0]])

    X = tf.concat([tf.cast(Graph, tf.float32), tf.cast(OD_demand, tf.float32), tf.cast(Path_tensor, tf.float32)], axis=1)
    return X

# Try standardize to replace normalize function
def standardize(tensor):
    mean = tf.reduce_mean(tensor, axis=0)
    std = tf.math.reduce_std(tensor, axis=0)
    std = tf.where(tf.equal(std, 0), 1.0, std)
    standardized_tensor = (tensor - mean) / std
    return standardized_tensor

def generate_xy(file_name, path_encoded, standard_norm):
    with open(file_name, "rb") as file:
        stat = pickle.load(file)
    
    path_links = stat["data"]["paths_link"]
    demand = stat["data"]["demand"]
    path_flows = stat["path_flow"]
    nodes = stat["data"]["nodes"]
    net = stat["data"]["network"]

    # Get X
    if standard_norm == 'standardize':
        Graph = standardize(get_graphTensor(net))
        OD_demand = standardize(preprocessTensor(demand, nodes))
        Path_tensor = standardize(preprocess_path(path_links, nodes, path_encoded))
        X = get_X(Graph, OD_demand, Path_tensor)  # 625 x 1164
    else:
        Graph = normalize(get_graphTensor(net))
        OD_demand = normalize(preprocessTensor(demand, nodes))
        Path_tensor = normalize(preprocess_path(path_links, nodes, path_encoded))
        X = get_X(Graph, OD_demand, Path_tensor)  # 625 x 1164

    # Get Y
    Flow_tensor = preprocess_flow(demand, path_flows, nodes)
    Flow_tensor = tf.squeeze(Flow_tensor, -1)  # 625x3
    return X, Flow_tensor

def create_attention_mask(input_tensor):
    batch_size, sequence_length, _ = input_tensor.shape
    # Create a mask tensor with 0s for valid positions and -inf for padding positions
    mask = tf.zeros([batch_size, 1, 1, sequence_length], dtype=tf.float32)
    mask = tf.where(input_tensor != 0, 1.0, float('-inf'), name='attention_mask')
    return mask