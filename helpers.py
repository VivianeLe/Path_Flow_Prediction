import pandas as pd
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import torch.utils.data as data
import torch.nn.functional as F

# Create dictionary of all unique paths
def path_encoder():
    path_sample = []
    for i in range(10):
        file_name = f"Output/5by5_Data{i}"
        file = open(file_name, "rb")
        stat = pickle.load(file)
        file.close()
        path_sample.append(stat["data"]["paths_link"])

    unique_values_set = set()
    all_path_link = [path_sample[i].values() for i in range (len(path_sample))]
    for path_set in all_path_link:
        for path in path_set:
            for p in path:
                unique_values_set.add(tuple(p))
    path_set_dict = {v: k for k, v in enumerate(unique_values_set, start=1)}
    return path_set_dict

# Normalize tensor by Min-max scaling
def normalize_tensor(tensor):
    # reshaped_tensor = tensor.view(-1, tensor.size(-1))
    min_values = torch.min(tensor, dim=0)[0]
    max_values = torch.max(tensor, dim=0)[0]
    scaled_tensor = (tensor - min_values) / (max_values - min_values)
    return scaled_tensor

def get_Link_Path_adj(net, path_encoded):
    link_path = torch.zeros(net.shape[0],len(path_encoded))
    for p, index in path_encoded.items():
        for link in p:
            link_path[link, index-1] = 1
    # link_path = link_path.unsqueeze(0)
    return link_path

def get_graphTensor(network):
    # convert data to numpy
    data = network[['link_id', 'init_node', 'term_node', 'capacity', 'length', 'free_flow_time']].to_numpy()
    # create network tensor 3D
    data = torch.tensor(data)
    return data

"""
    This function is to convert a dictionary to a tensor, exp OD demand dictionary with n nodes
    First create a zero matrix with size = node x node (25x25)
    Then, for each OD pair, we update the matrix with the demand value
    matrix index = node index - 1 (because node index start from 1)
    Finally convert matrix to a tensor type float (default)
"""
def convert_DictToTensor(OD_demand, nodes) :
    matrix = [ [0 for n in nodes] for n in nodes]
    for k,v in OD_demand.items() :
        o,d = k
        matrix[o-1][d-1] = v
    return torch.tensor([matrix], dtype=torch.float32)

# Tranform a dictionary to a tensor, flatten, transpose, and unsqueeze to get a 3D tensor
def preprocessTensor(OD_dict, nodes):
    tensor = convert_DictToTensor(OD_dict, nodes)# size (1, 25, 25)
    tensor = torch.flatten(tensor, start_dim=1) # shape 1x625
    tensor = torch.transpose(tensor, 0, 1) # shape 625x1
    # tensor = tensor.unsqueeze(0) # shape 1x625x1
    return tensor

def preprocess_path(path_links, node, path_encoded):
    new_path_links = {k: [tuple(path) for path in v] for k, v in path_links.items()}
    
    # Encode paths
    p1, p2, p3 = {}, {}, {}
    for k, v in new_path_links.items():
        p1[k], p2[k], p3[k] = [path_encoded[path] for path in v]

    # extract 3 path tensors
    p1 = preprocessTensor(p1, node) # 625x1
    p2 = preprocessTensor(p2, node)
    p3 = preprocessTensor(p3, node)
    stack = torch.stack([p1, p2, p3], dim=1).squeeze(-1)
    return stack

def preprocess_flow(demand, path_flows, nodes):
    flows = {k: v for k, v in zip(demand.keys(), path_flows)}
    f1, f2, f3 = {}, {}, {}
    for k, v in flows.items():
        f1[k], f2[k], f3[k] = v

    # extract 3 flow tensors
    f1 = preprocessTensor(f1, nodes)
    f2 = preprocessTensor(f2, nodes)
    f3 = preprocessTensor(f3, nodes)
    stack = torch.stack([f1,f2,f3], dim=1).squeeze(-1)
    return stack

def get_X(Graph, Link_Path_adj, OD_demand, Path_tensor):
    X = [Graph, Link_Path_adj,OD_demand,Path_tensor]

    # Tính toán kích thước lớn nhất của các ma trận (dim 0)
    max_size = max([x.size(0) for x in X])

    # Thêm padding cho mỗi ma trận để chúng có kích thước giống nhau
    Graph = F.pad(Graph, (0,0,0, max_size-Graph.size(0)))
    OD_demand = F.pad(OD_demand, (0,0,0, max_size-OD_demand.size(0)))
    Path_tensor = F.pad(Path_tensor, (0,0,0, max_size-Path_tensor.size(0)))
    Link_Path_adj = F.pad(Link_Path_adj,(0,0,0, max_size-Link_Path_adj.size(0)))

    X = torch.cat([Graph.float(), OD_demand.float(), Path_tensor.float(), Link_Path_adj.float()], dim=1) # 625x1165
    return X

def generate_xy(file_name, path_encoded):
    file = open(file_name, "rb")
    stat = pickle.load(file)
    file.close()
    
    path_links = stat["data"]["paths_link"]
    demand = stat["data"]["demand"]
    path_flows = stat["path_flow"]
    nodes = stat["data"]["nodes"]
    net = stat["data"]["network"]

    # Get X
    Link_Path_adj = get_Link_Path_adj(net, path_encoded)
    Graph = normalize_tensor(get_graphTensor(net))
    OD_demand = normalize_tensor(preprocessTensor(demand, nodes))
    Path_tensor = preprocess_path(path_links, nodes, path_encoded)
    X = get_X(Graph, Link_Path_adj,OD_demand,Path_tensor) # 625 x 1165

    # Get Y
    Flow_tensor = preprocess_flow(demand, path_flows, nodes) # 625x3
    return X, Flow_tensor