import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import networkx as nx
from itertools import chain
import random
import pickle
import time
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error



def readNet(fileN) : 
    net = pd.read_csv(fileN,delimiter='\t',skiprows=8)
    
    nodes = set(list(net['init_node'])+list(net['term_node']))
    
    links = int(net.shape[0])
    cap = [0 for i in range(links)]
    t0 = [0 for i in range(links)]
    alpha = [0 for i in range(links)]
    beta = [0 for i in range(links)]
    lengths = [0 for i in range(links)]
    
    net['link_id'] = net.index
    
    i = 0
    for capacityi,fftti,alphai,betai,leni in zip(net['capacity'],net['free_flow_time'],net['power'],net['b'], net['length']):
        cap[i] = capacityi
        t0[i] = fftti
        alpha[i] = alphai
        beta[i] = betai
        lengths[i] = leni
        i = i + 1
    return net, nodes, links, cap, t0, alpha, beta, lengths

# Get origin, destination from OD matrix
def get_origDest(OD_demand) : 
    orig = []
    dest = []
    for o,d in OD_demand.keys() :
        if o not in orig :
            orig.append(o)
        if d not in dest :
            dest.append(o)
    return orig, dest



def k_shortest_paths(G, source, target, k):
    try : 
        paths = list(islice(nx.shortest_simple_paths(G, source, target, weight="free_flow_time"), k))
    except : 
        paths = []
    return paths

def transform_paths(network, paths) : # transform node path to edge path
    paths_OD = []
    for path in paths:
        pathEdges = []
        for i in range(len(path)-1):
            mask = (network['init_node']==path[i]) & (network['term_node']==path[i+1])
            pathEdges.append(network.index[mask].tolist()[0])   
        paths_OD.append(pathEdges)
    return paths_OD


def find_paths_new(network, OD_Matrix,kk) :
    netG = nx.from_pandas_edgelist(network,source='init_node',target='term_node',edge_attr='free_flow_time',create_using=nx.DiGraph())
    # nx.draw(netG)
    print(network)
    paths = {}
    paths_N = {}
    dict_o = {}
    dict_d = {}
    for key in OD_Matrix.keys():
        paths_OD_N = []
        o,d = key[0],key[1]
        if o in dict_o.keys() :
            for v1, v2 in dict_o[o].items() :
                if d<v1 :
                    for v3 in v2 :
                        if d in v3 :
                            p = v3[0:v3.index(d)+1]
                            if p not in paths_OD_N :
                                paths_OD_N.append(p)
                            if len(paths_OD_N) == kk :
                                paths[(o,d)]=transform_paths(network, paths_OD_N)
                                paths_N[(o,d)]=paths_OD_N
                                dict_o[o].update({d:paths_OD_N})
                                dict_d[d]={o:paths_OD_N}
                                break
                if len(paths_OD_N) == kk :
                        break
        if d in dict_d.keys() :
            for v1, v2 in dict_d[d].items() :
                if o>v1 :
                    for v3 in v2 :
                        if o in v3 :
                            p = v3[v3.index(d):len(v3)+1]
                            if p not in paths_OD_N :
                                paths_OD_N.append(p)
                            if len(paths_OD_N) == kk :
                                paths[(o,d)]=transform_paths(network, paths_OD_N)
                                paths_N[(o,d)]=paths_OD_N
                                dict_o[o]={d:paths_OD_N}
                                dict_d[d].update({o:paths_OD_N})
                                break
                if len(paths_OD_N) == kk :
                        break
        if (o,d) not in paths.keys() : 
            try : 
                p = k_shortest_paths(netG,o,d,kk)
                paths[(o,d)]=transform_paths(network, p)
                paths_N[(o,d)]=p
                dict_o[o] = {d:p}
                dict_d[d] = {o:p}
            except :
                paths[(o,d)] = []
                paths_N[(o,d)]=[]
    
    return paths, paths_N


def find_paths(network, OD_Matrix,k) :
    netG = nx.from_pandas_edgelist(network,source='init_node',target='term_node',edge_attr='free_flow_time',create_using=nx.DiGraph())
    paths = {}
    paths_N = {}
    for key in OD_Matrix.keys():
        paths_OD = []
        o,d = key[0],key[1]
        try : 
            p = k_shortest_paths(netG,o,d,k)
            # print("p = ", p)
            paths_OD = transform_paths(network, p)
            paths[(o,d)]=paths_OD
            paths_N[(o,d)]=p
        except :
            paths[(o,d)] = []
            paths_N[(o,d)]=[]
            
    return paths, paths_N

def mergeDictionary(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = value + dict_1[key]
    return dict_3

def translate(nodes, OD_demand) :
    Q = [int(value) for value in OD_demand.values()]
    OD = len(OD_demand) 
    O_D = [ [0 for i in nodes] for n in range(OD)]
    n= 0
    
    for key in OD_demand.keys() :
        for i in nodes : 
            if i==key[0] : #origin node
                O_D[n][i-1] = -1
            if i==key[1] : #destination node
                O_D[n][i-1] = 1
        n = n + 1
    
    return Q, OD, O_D

def create_Adj(net_df, links, nodes) : 
    Adj = [ [0 for i in range(links)] for n in nodes]
    for n in nodes :
        init = net_df[net_df['init_node']==n]
        for j in init['link_id'].values : 
            Adj[n-1][j] = -1
        
        term = net_df[net_df['term_node']==n]
        for j in term['link_id'].values : 
            Adj[n-1][j] = 1
    return Adj
            


def create_delta(links, paths, od_matrix) :
    delta = [[[0 for i in range(links)] for p in paths[k]] for k in od_matrix.keys()]

    kk = 0
    for k in od_matrix.keys() : 
        value = paths[k]
        pp = 0
        for p in value : # p is a list of links
            for j in p : 
                delta[kk][pp][j]=1
            pp += 1
        kk += 1
    return delta

def generate_Random_OD_matrix(number_OD, OD_pairs) : 
    
    max_dem = 2000
    
    Matrix = {}
    total_demand = 0
    k = 0
    while k < number_OD :
        demand =  random.randint(50, max_dem)
        od_id = random.randint(0, len(OD_pairs)-1)
        o,d = OD_pairs[od_id]
        if (o,d) not in Matrix.keys() :
            Matrix[(o,d)] = demand
            total_demand = total_demand + demand
            k = k + 1
    
    return Matrix, total_demand

def generate_Random_ODs(dim1, dim2, nb_entries,origins, dest,OD_pairs,file_name) : 
    
    stats = {}
    k = 0            
    while k < nb_entries :   
        print("nb_entries : ",k)
        number_OD = random.randint(1, len(OD_pairs))
        
        M, TD = generate_Random_OD_matrix(number_OD, OD_pairs)
        
        if number_OD not in stats.keys() :
            stats[number_OD] = []
            stats[number_OD].append(M)
            k = k +1
            file = open("Data/{}by{}_Data{}_{}".format(dim1, dim2,k,file_name), "wb")
            pickle.dump(M, file)
            file.close()
        else : 
            if M not in stats[number_OD] :
                stats[number_OD].append(M)
                k = k +1
                
    a_file = open(file_name, "wb")
    pickle.dump(stats, a_file)
    a_file.close()
      
    return stats

def get_fullOD_pairs(dim1, dim2) : 
    origins = [i for i in range(1,dim1*dim2+1) if (i%dim1!=0)]
    dest = [i for i in range(1,dim1*dim2+1) if (i%dim1!=1)]

    OD_pairs = []
    for i in origins :
        for j in dest :
            if (i != j) and ( (i,j) not in OD_pairs) :
                OD_pairs.append((i,j))
    return OD_pairs

# Combine OD matrix from many files into 1 
def fuse_stats(dict1, dict2) : 
    for k,v in dict2.items() :
        if k not in dict1.keys() :
            dict1[k] = v
        else :
            for vv in v : 
                if vv not in dict1[k] : 
                    dict1[k].append(vv)
    return dict1

def get_data(Network, Nodes, links, cap, fft, alpha, beta, lengths, OD_mat ) : 
    """ OD_mat: OD demand matrix: {OD pair: demand}
    Q: value of demand (int)
    OD: number of OD pairs
    n: path length of each OD pair
    """
    num_paths = 3
    O,D = get_origDest(OD_mat)
    paths, paths_N = find_paths(Network, OD_mat, num_paths)
    print("Paths: ", paths)
    Q, OD, O_D = translate(Nodes, OD_mat)
    Adj = create_Adj(Network, links, Nodes)
    delta = create_delta(links, paths, OD_mat)
    n = [ len(paths[h]) for h in OD_mat.keys() ]
    ### Linearizing variables
    seg = 100 # try 100 instead of 1000
    Mflow = 10e4   

    # define the segments
    segments = set([i for i in range(0,seg+1)])          
    eta = [[ v for v in segments ] for i in range(links) ]    
    for i in range(links):
        cnt = 0
        step = Mflow/seg
        #step = cap[i]/seg
        for v in segments:
            eta[i][v] = cnt*step
            cnt += 1  
    #segments_p = segments.difference({0})    
    data = {'network' :Network, 'demand' :OD_mat, 'nodes':Nodes,'links':links,'orig':O,'dest':D,'fftt':fft,'capacity':cap, 'length': lengths, 'beta':beta,
        'approx':segments,'eta':eta,'paths_link':paths, 'paths_node':paths_N, 'delta':delta,'alpha':alpha, 'Adjacency_matrix' : Adj}
    return data, Q, OD, O_D,n

# Function to find path flows and link flows at UE conditions
def TA_UE(data, n, OD, Q):
    print(Q)
    model = gp.Model("UE")
    model.setParam("OutputFlag", 0)

    a = data['links']
    segments = data['approx']
    t0 = data['fftt']
    eta = data['eta']
    alpha = data['alpha']
    beta = data['beta']
    #OD_demand = data['demand']
    #paths = data['paths_link']
    #paths_N = data['paths_node']
    sigma = data['delta']
    cap = data['capacity']
    segments_p = segments.difference({0})
    
    # x = [model.addVar(vtype=GRB.CONTINUOUS) for j in range(a)]
    x = [model.addVar(vtype=GRB.CONTINUOUS) for i in range(a)]
    f = [ [model.addVar(vtype=GRB.CONTINUOUS) for i in range(n[k])] for k in range(OD)]
    # ll = [ [model.addVar(vtype=GRB.CONTINUOUS) for l in segments] for i in range(a)]
    # lr = [ [model.addVar(vtype=GRB.CONTINUOUS) for l in segments] for i in range(a)]
    # t = [model.addVar(vtype=GRB.CONTINUOUS, name=f"t_{i}") for i in range(a)]

    for i in range(a):
        model.addConstr(x[i] == sum( sum(f[k][p]*sigma[k][p][i] for p in range(n[k])) for k in range(OD)), "link-path%d" %i)
        # X.append(sum( sum(f[k][p]*sigma[k][p][i] for p in range(n[k])) for k in range(OD)))
        model.addConstr(x[i] >=0)
    #     model.addConstr(sum(ll[i][v] + lr[i][v] for v in segments_p) == 1, "Approx2%d" %i)
        # model.addConstr(x[i] >= 0, "integrality_x%d" %i)  
    #     for ss in segments:
    #         model.addConstr(ll[i][ss] >= 0, "integrality_ll%d%d" %(i,ss))
    #         model.addConstr(lr[i][ss] >= 0, "integrality_lr%d%d" %(i,ss))

    for k in range(OD) :
    #     model.addConstr( sum(f[k][p] for p in range(n[k])) == Q[k] , "FConservation%d" %k ) 
         for p in range(n[k]) : 
             model.addConstr(f[k][p] >= 0, "integrality%d%d" %(k,p))
    
    
    Z = gp.quicksum(t0[i] * x[i] for i in range(a))

    model.setObjective(Z, GRB.MINIMIZE)
    t1 = time.time()
    model.optimize()
    t2 = time.time()
    print('model solved in:',t2-t1)
    print(model.Status)
    print(n)
    flows =  [[f[k][p].X for p in range(n[k])] for k in range(OD)] # path flow: [[a, b, c]]
    print(flows)
    linkss = [x[i].X for i in range(a)]

    return flows, linkss, model.Status, t2-t1

# Version 2
# def TA_UE(data, n, OD, Q):
#     model = gp.Model("UE")
#     model.setParam("OutputFlag", 0)

#     a = data['links']
#     t0 = data['fftt']
#     alpha = data['alpha']
#     beta = data['beta']
#     sigma = data['delta']
#     cap = data['capacity']

#     # Decision variables
#     x = [model.addVar(vtype=GRB.CONTINUOUS, name=f"x_{j}") for j in range(a)]
#     f = [[model.addVar(vtype=GRB.CONTINUOUS, name=f"f_{k}_{p}") for p in range(n[k])] for k in range(OD)]
#     t = [model.addVar(vtype=GRB.CONTINUOUS, name=f"t_{i}") for i in range(a)]

#     # Constraints
#     for i in range(a):
#         # Flow conservation constraint
#         model.addConstr(
#             x[i] == gp.quicksum(f[k][p] * sigma[k][p][i] for k in range(OD) for p in range(n[k])), f"link_path_{i}"
#         )
#         # Non-negativity constraint for link flow
#         model.addConstr(x[i] >= 0, f"non_negativity_x_{i}")

#         # BPR travel time function using linear approximation
#         M = 10000  # A large constant to approximate the power function
#         model.addConstr(
#             t[i] >= t0[i] * (1 + alpha[i] * (x[i] / cap[i])) - M * (1 - x[i] / cap[i]),
#             f"BPR_travel_time_lower_{i}"
#         )
#         model.addConstr(
#             t[i] <= t0[i] * (1 + alpha[i] * (x[i] / cap[i])) + M * (1 - x[i] / cap[i]),
#             f"BPR_travel_time_upper_{i}"
#         )

#     for k in range(OD):
#         # Demand satisfaction constraint
#         model.addConstr(gp.quicksum(f[k][p] for p in range(n[k])) == Q[k], f"FConservation_{k}")
#         for p in range(n[k]):
#             # Non-negativity constraint for path flow
#             model.addConstr(f[k][p] >= 0, f"non_negativity_f_{k}_{p}")

#     # Objective function: minimize total travel time
#     Z = gp.quicksum(t[i] * x[i] for i in range(a))
#     model.setObjective(Z, GRB.MINIMIZE)

#     # Solve the model
#     t1 = time.time()
#     model.optimize()
#     t2 = time.time()
#     print('Model solved in:', t2 - t1)

#     # Retrieve the results if the model is feasible
#     if model.Status == GRB.OPTIMAL:
#         path_flow = [[f[k][p].X for p in range(n[k])] for k in range(OD)]
#         link_flow = [x[i].X for i in range(a)]
#         status = model.Status
#         sol_time = t2 - t1
#     else:
#         print(f"Model status: {model.Status}, model is not feasible.")
#         path_flow = [[] for _ in range(OD)]
#         link_flow = [0.0 for _ in range(a)]
#         status = model.Status
#         sol_time = t2 - t1

#     return path_flow, link_flow, status, sol_time



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

def getLinks_embedding(data) :
    links = []
    for i in range(data['links']) :
        li = []
        li.append(data['fftt'][i])
        li.append(data['capacity'][i])
        li.append(data['length'][i])
        li.append(data['beta'][i]) 
        li.append(data['alpha'][i])
        links.append(li)
    return torch.tensor([links], dtype=torch.float32)

def getOutputEncoding(dim1, dim2, max_num_paths, OD_demand, path_flows) : 
    Full_OD_pairs = get_fullOD_pairs(dim1, dim2)
    output_size = len(Full_OD_pairs)*max_num_paths # 1155
    output_tensor = torch.zeros([1, output_size], dtype=torch.float32)
    i = 0
    for k in OD_demand.keys() :
        od_ind = Full_OD_pairs.index(k)*max_num_paths
        flows = path_flows[i]
        j = 0
        for f in flows :
            output_tensor[0,od_ind+j] = f
            j += 1
        i += 1
    return output_tensor

def getOutputTensor(data, od_demand, nodes):
    # Get index of path used (flow != 0)
    used_path_index = [i+1 for sublist in data for i, val in enumerate(sublist) if val != 0]
    # Map OD pair with the path used index list
    used_path_dict = {key: indices for key, indices in zip(od_demand.keys(), used_path_index)}
    # Create used path tensor (this is output of the model)
    output = convert_DictToTensor(used_path_dict, nodes) # shape (1, 25, 25)
    return output

def create_datapoint(file_name,dim1, dim2, max_num_paths ) :
    file = open(file_name, "rb")
    stat = pickle.load(file)
    file.close()
    Nodes =  stat['data']['nodes']
    OD_demand = stat['data']['demand']
    
    T_demand = convert_DictToTensor(OD_demand, Nodes)
    T_Adj = torch.tensor( [stat['data']['Adjacency_matrix']], dtype=torch.float32 )  # tensor of adjacency matrix
    T_links = getLinks_embedding(stat['data'])
    T_path_flows = getOutputEncoding(dim1, dim2, max_num_paths, OD_demand, stat['path_flow']) # tensor of path flows

    # need to come up with a better way
    # flatten the multi-dimention to 1D tensor
    T_demand = torch.flatten(T_demand, start_dim=1)
    T_Adj = torch.flatten(T_Adj, start_dim=1)
    T_links = torch.flatten(T_links, start_dim=1)
    T_path_flows = torch.flatten(T_path_flows, start_dim=1)
    return torch.cat((T_demand,T_Adj,T_links),1) , T_path_flows  # X, Y

def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))