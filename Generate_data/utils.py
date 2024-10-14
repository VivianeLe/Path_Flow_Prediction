from fcntl import F_GETLK
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import networkx as nx
from itertools import islice
import random
import pickle
import time
import numpy as np
from tqdm.notebook import tqdm
from collections import defaultdict
import ast

def generate_gridNet(dim1, dim2, file_name, draw=True, target_links=80):
    G, pos = create_Grid_Net(dim1, dim2)
    G = reduce_links(G, target_links)
    i = 1
    mapping = {}
    for e in G.nodes:
        mapping[e] = i
        i = i + 1
    convert_net_to_file(G, file_name, mapping)
    if draw:
        nx.draw_networkx(G, pos=pos, with_labels=True, labels=mapping, font_size=9, font_color='white')
    return G, pos

def create_Grid_Net(dim1, dim2):
    G = nx.DiGraph()
    for n1 in range(dim1):
        for n2 in range(dim2):
            G.add_node((n1, n2))
    for n1 in range(dim1):
        for n2 in range(dim2):
            if n2 + 1 < dim2:
                G.add_edge((n1, n2), (n1, n2 + 1))
                G.add_edge((n1, n2 + 1), (n1, n2))  # bidirectional
            if n1 + 1 < dim1:
                G.add_edge((n1, n2), (n1 + 1, n2))
                G.add_edge((n1 + 1, n2), (n1, n2))  # bidirectional

    pos = {(x, y): (y, -x) for x, y in G.nodes()}  # for the drawing
    return G, pos

def reduce_links(G, target_links):
    current_links = len(G.edges)
    if target_links >= current_links:
        return G  # No reduction needed
    
    edges = list(G.edges)
    random.shuffle(edges)
    edges_to_remove = edges[:current_links - target_links]
    
    for edge in edges_to_remove:
        G.remove_edge(*edge)
        
    return G

def convert_net_to_file(net, file_name,labels_map) : 
    with open(file_name, 'w') as f:
        f.write("<NUMBER OF ZONES> "+str(0))
        f.write("\n")
        f.write("<NUMBER OF NODES> "+str(len(net.nodes)))
        f.write("\n")
        f.write("<FIRST THRU NODE> 1")
        f.write("\n")
        f.write("<NUMBER OF LINKS> "+str(len(net.edges)))
        f.write("\n")
        f.write("<ORIGINAL HEADER>~ \tInit node \tTerm node \tCapacity \tLength \tFree Flow Time \tB \tPower \tSpeed limit \tToll \tType \t;")
        f.write("\n")
        f.write("<END OF METADATA>")
        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("~ \tlink_id\tinit_node\tterm_node\tcapacity\tlength\tfree_flow_time\tb\tpower\tspeed\ttoll\tlink_type\t;")
        f.write("\n")
        
        link_id = 0
        for n1,n2 in net.edges :
            n1_id = labels_map[n1]
            n2_id = labels_map[n2]
            cap = np.random.randint(1000, 2001)
            leng = np.random.randint(20, 41)
            fft = round(np.random.uniform(0.5, 1.0),1)
            speed = np.random.choice([60, 70, 80, 90, 100])
            f.write("\t" + str(link_id)+"\t"+str(n1_id)+"\t"+str(n2_id)+"\t"+str(cap)+"\t"+str(leng)+"\t"+str(fft)+"\t0.15\t4\t"+str(speed)+"\t0\t0\t;")
            f.write("\n")
            link_id += 1

# def readNet(fileN) : 
#     net = pd.read_csv(fileN,delimiter='\t',skiprows=8)
    
#     nodes = set(list(net['init_node'])+list(net['term_node']))
    
#     links = int(net.shape[0])
#     cap = [0 for i in range(links)]
#     t0 = [[0 for j in range(2)] for i in range(links)]
#     alpha = [0 for i in range(links)]
#     beta = [0 for i in range(links)]
#     lengths = [0 for i in range(links)]
    
#     net['link_id'] = net.index
    
#     for i, (capacityi,fftti,alphai,betai,leni) in enumerate(zip(net['capacity'],net['fft'],net['power'],net['b'], net['length'])):
#         fftti = ast.literal_eval(fftti)
#         cap[i] = capacityi
#         for j in range(2):
#             t0[i][j] = fftti[j]
#         alpha[i] = alphai
#         beta[i] = betai
#         lengths[i] = leni

#     return net, nodes, links, cap, t0, alpha, beta, lengths

def readNet(fileN) : 
    net = pd.read_csv(fileN,delimiter='\t',skiprows=8)
    
    nodes = set(list(net['init_node'])+list(net['term_node']))
    
    links = int(net.shape[0])
    cap = [0 for i in range(links)]
    t0 = [0 for i in range(links)]
    alpha = [0 for i in range(links)]
    beta = [0 for i in range(links)]
    lengths = [0 for i in range(links)]
    
    i = 0
    for capacityi,fftti,alphai,betai,leni in zip(net['capacity'],net['free_flow_time'],net['power'],net['b'], net['length']):
        cap[i] = capacityi
        t0[i] = fftti
        alpha[i] = alphai
        beta[i] = betai
        lengths[i] = leni
        i = i + 1
    return net, nodes, links, cap, t0, alpha, beta, lengths

def k_shortest_paths(G, source, target, k):
    try : 
        paths = list(islice(nx.shortest_simple_paths(G, source, target, weight="free_flow_time"), k))
    except : 
        paths = []

    return paths

def transform_paths(network, paths) : # transform node path to edge path
    paths_OD = []
    for path in paths:
        pathEdges = [] # list of links in each path
        for i in range(len(path)-1):
            mask = (network['init_node']==path[i]) & (network['term_node']==path[i+1])
            pathEdges.append(network.loc[mask, 'link_id'].values[0])   
            # pathEdges.append(network.index[mask].tolist()[0])   
        paths_OD.append(pathEdges)
    return paths_OD

def generate_Random_OD_matrix(number_OD, OD_pairs) : 
    max_dem = 300
    Matrix = {}
    total_demand = 0
    k = 0
    while k < number_OD :
        demand =  random.randint(10, max_dem)
        od_id = random.randint(0, len(OD_pairs))
        o,d = OD_pairs[od_id]
        if (o,d) not in Matrix.keys() :
            Matrix[(o,d)] = demand
            total_demand = total_demand + demand
            k = k + 1
    
    return Matrix, total_demand

def generate_OD_demand(num_nodes, min_demand, max_demand, num_pairs):
    od_demand = {}
    # num_pairs = int((num_nodes**2)*0.8)
    # Generate unique OD pairs
    pairs = set()
    while len(pairs) < num_pairs:
        origin = random.randint(1, num_nodes)
        destination = random.randint(1, num_nodes)
        if origin != destination:  # Ensure origin is not equal to destination
            pairs.add((origin, destination))

    # Assign random demand values to each OD pair
    for origin, destination in pairs:
        demand_c = random.randint(min_demand, max_demand)
        demand_t = int(demand_c/2)
        od_demand[(origin, destination)] = [demand_c, demand_t]
    return od_demand

def generate_Random_ODs(dim1, dim2, nb_entries,origins, dest,OD_pairs,file_name) : 
    
    stats = {}
    k = 0            
    while k < nb_entries :   
        number_OD = random.randint(1, len(OD_pairs)-200)
        print("nb_entries : ",k, ". Number OD: ", number_OD)
        
        M, TD = generate_Random_OD_matrix(number_OD, OD_pairs)
        
        if number_OD not in stats.keys() :
            stats[number_OD] = []
            stats[number_OD].append(M)
            k = k +1
            file = open("../Data/{}by{}_Data{}_{}".format(dim1, dim2,k,file_name), "wb")
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

def fuse_stats(dict1, dict2) : 
    for k,v in dict2.items() :
        if k not in dict1.keys() :
            dict1[k] = v
        else :
            for vv in v : 
                if vv not in dict1[k] : 
                    dict1[k].append(vv)
    return dict1

def mergeDictionary(dict_1, dict_2):
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            dict_3[key] = value + dict_1[key]
    return dict_3

def get_origDest(OD_demand) : 
    orig = []
    dest = []
    for o,d in OD_demand.keys() :
        if o not in orig :
            orig.append(o)
        if d not in dest :
            dest.append(o)
    return orig, dest

def find_paths(network, OD_Matrix,k) :
    netG = nx.from_pandas_edgelist(network,source='init_node',target='term_node',edge_attr='free_flow_time',create_using=nx.DiGraph())
    paths = {}
    paths_N = {}
    for key in OD_Matrix.keys():
        paths_OD = []
        o,d = key[0],key[1]
        try : 
            p = k_shortest_paths(netG,o,d,k)
            paths_OD = transform_paths(network, p) # paths_OD is list of feasible path set, each path includes list of links
            paths[(o,d)]=paths_OD
            paths_N[(o,d)]=p
        except :
            paths[(o,d)] = []
            paths_N[(o,d)]=[]
            
    return paths, paths_N

def translate(nodes, OD_demand) :
    Q = [value for value in OD_demand.values()]
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

def create_Adj(net_df, links, nodes): 
    # links: number of links in the network (80, 75, 70, 65)
    # nodes: a list of all nodes (from 1 to 25)
    # Create adj matrix shape 25x80 for all scenarios
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
        value = paths[k] # get all paths of OD pair k
        pp = 0
        for p in value : # iterate each path of pair k
            for j in p : # iterate each link in path p
                delta[kk][pp][j]=1
            pp += 1
        kk += 1
    return delta

def get_data(Network, Nodes, links, cap, fft, alpha, beta, lengths, OD_mat, paths) : 
    O,D = get_origDest(OD_mat)
    Q, OD, O_D = translate(Nodes, OD_mat)
    Adj = create_Adj(Network, links, Nodes)
    delta = create_delta(links, paths, OD_mat) #Adj of OD pair - path - link
    n = [ len(paths[h]) for h in OD_mat.keys() ]
    ### Linearizing variables
    seg = 1000
    Mflow = 10e4

    # define the segments
    segments = set([i for i in range(0,seg+1)])          
    eta = [ [ v for v in segments ] for i in range(links) ]  
    step = Mflow/seg  
    for i in range(links):
        cnt = 0
        #step = cap[i]/seg
        for v in segments:
            eta[i][v] = cnt*step
            cnt += 1  
    #segments_p = segments.difference({0})    
    data = {'network' :Network, 'demand' :OD_mat, 'nodes':Nodes,'links':links,'orig':O,'dest':D,'fftt':fft,'capacity':cap, 'length': lengths, 'beta':beta,
        'approx':segments,'eta':eta,'paths_link':paths, 'delta':delta,'alpha':alpha, 'Adjacency_matrix' : Adj}
    return data, Q, OD, O_D,n

# def get_data_N(Network, Nodes, links, cap, fft, alpha, beta, lengths, OD_mat, paths) : 
#     # print("No of links: ", links)
#     O,D = get_origDest(OD_mat)
#     Q, OD, O_D = translate(Nodes, OD_mat)
#     Adj = create_Adj(Network, links, Nodes)
#     delta = create_delta(links, paths, OD_mat)
#     n = [ len(paths[h]) for h in OD_mat.keys() ] # number of path for each OD pair 
#     ### Linearizing variables
#     seg = 1000
#     Mflow = 10e4   

#     # define the segments
#     segments = set([i for i in range(0,seg+1)])          
#     eta = [[[ v for j in range(2)] for v in segments] for i in range(links)]    
#     for i in range(links):
#         cnt = 0
#         step = Mflow/seg
#         for v in segments:
#             for j in range(2):
#                 eta[i][v][j] = cnt*step
#             cnt += 1  
  
#     data = {'network' :Network, 'demand' :OD_mat, 'nodes':Nodes,'links':links,'orig':O,'dest':D,'fftt':fft, 'capacity':cap, 'length': lengths, 'beta':beta,
#         'approx':segments,'eta':eta,'paths_link':paths, 'delta':delta,'alpha':alpha, 'Adjacency_matrix' : Adj}
#     return data, Q, OD, O_D,n

#################### BRUE SOLVER ##########################
def BRUE(data, n, OD, Q):
    model = gp.Model("BRUE")
    model.setParam("OutputFlag", 1)
    model.setParam("LogFile", "gurobi_log.txt")
    model.setParam("NumericFocus", 3)
    # model.setParam("Method", 2)  # Barrier method

    a = data['links']
    segments = data['approx']
    t0 = data['fftt']
    eta = data['eta']
    alpha = data['alpha'] #4
    beta = data['beta'] #0.15
    sigma = data['delta'] # adj of od pair - path - link 
    cap = data['capacity']
    segments_p = segments.difference({0})
    M = 1e3
    
    x = [model.addVar(vtype=GRB.CONTINUOUS) for j in range(a)] # link flow
    x4 = [model.addVar(vtype=GRB.CONTINUOUS) for j in range(a)] # x^4
    link_cost = [model.addVar(vtype=GRB.CONTINUOUS) for j in range(a)]
    f = [ [model.addVar(vtype=GRB.CONTINUOUS) for i in range(n[k])] for k in range(OD)] # path flow
    path_cost = [ [model.addVar(vtype=GRB.CONTINUOUS) for i in range(n[k])] for k in range(OD)]
    
    ll = [ [model.addVar(vtype=GRB.CONTINUOUS) for l in segments] for i in range(a)]
    lr = [ [model.addVar(vtype=GRB.CONTINUOUS) for l in segments] for i in range(a)]

    y = [ [model.addVar(vtype=GRB.CONTINUOUS) for i in range(n[k])] for k in range(OD)] # BINARY VAR
    min_path_cost = [model.addVar(vtype=GRB.CONTINUOUS, name=f"min_path_cost_{k}") for k in range(OD)]

    for i in range(a):
        model.addConstr(x[i] == sum( sum(f[k][p]*sigma[k][p][i] for p in range(n[k])) for k in range(OD)), "link-path%d" %i)
        model.addConstr(x[i] == sum(ll[i][v]*eta[i][v-1] + lr[i][v]*eta[i][v] for v in segments_p), "Approx1%d" %i)
        model.addConstr(sum(ll[i][v] + lr[i][v] for v in segments_p) == 1, "Approx2%d" %i)
        model.addConstr(x[i] >= 0, "integrality_x%d" %i)  

        for ss in segments:
            model.addConstr(ll[i][ss] >= 0, "integrality_ll%d%d" %(i,ss))
            model.addConstr(lr[i][ss] >= 0, "integrality_lr%d%d" %(i,ss))
        
        model.addConstr(x4[i] == sum(eta[i][v-1]**alpha[i] * ll[i][v] + eta[i][v]**alpha[i] * lr[i][v] 
                                                     for v in segments_p))

        model.addConstr(link_cost[i] == t0[i] * (1 + beta[i]/(cap[i]**alpha[i]) * 
                                                 sum(eta[i][v-1]**alpha[i] * ll[i][v] + eta[i][v]**alpha[i] * lr[i][v] 
                                                     for v in segments_p)),
                                                 name=f"link_cost_BPR_{i}")

        # num_segments = 50
        # breakpoints = np.linspace(0, 10e4, num_segments)  # create 1000 segments from 0 to 10e4
        # values = t0[i] * (1 + beta[i] * (breakpoints / cap[i]) ** alpha[i])
        # model.addGenConstrPWL(x[i], link_cost[i], breakpoints.tolist(), values.tolist(), name=f"link_cost_BPR_PWL_{i}")


    for k in range(OD) :
        model.addConstr( sum(f[k][p] for p in range(n[k])) == Q[k] , "FConservation%d" %k ) 

        if (n[k] > 0):
            model.addConstr(min_path_cost[k] == path_cost[k][0], name=f"init_min_cost_{k}")

        for p in range(n[k]) : 
            model.addConstr(f[k][p] >= 0, "integrality%d%d" %(k,p))

            model.addConstr(y[k][p] >= 0)
            model.addConstr(y[k][p] <= 1)
            model.addConstr(f[k][p] <= M * y[k][p]) # if f[k][i] = 0 then y[k][i] = 0
            model.addConstr(f[k][p] >= 1e-6 * y[k][p]) # if f[k][i] > 0 then y[k][i] = 1
            # model.addConstr(f[k][p] * (1-y[k][p]) <= f[k][p])
            # model.addConstr(f[k][p] * y[k][p] == f[k][p]) # add this one make infeasible

            model.addConstr(path_cost[k][p] == sum(link_cost[i] * sigma[k][p][i] for i in range(a)), "path-cost%d%d" %(k,p)) # satisfied
            model.addConstr((min_path_cost[k] <= path_cost[k][p]), name=f"min_cost_{k}") # satisfied

            model.addConstr(path_cost[k][p] - min_path_cost[k] <= M* (1-y[k][p]) + min_path_cost[k]*0.05) #BRUE
            # model.addConstr(M * (path_cost[k][p] - min_path_cost[k]) >= 1-y[k][p])
        
    Z = sum(M * y[k][p] for p in range(n[k]) for k in range(OD))

    model.setObjective(Z, GRB.MINIMIZE)
    t1 = time.time()
    model.optimize()
    t2 = time.time()
    print('model solved in:',t2-t1)
    
    if model.Status == GRB.OPTIMAL:
        flows =  [ [ f[k][p].X  for p in range(n[k])] for k in range(OD)]
        linkss = [ x[i].X  for i in range(a)]
        x4 = [ x4[i].X  for i in range(a)]
        min_cost = [min_path_cost[i].X for i in range(OD)]
        path_cost = [[path_cost[k][p].X for p in range(n[k])] for k in range(OD)]
        link_cost = [link_cost[i].X for i in range(a)]
        return flows, linkss, path_cost, min_cost, link_cost, x4
    
    elif model.Status == GRB.INFEASIBLE:
        print(f"Model is infeasible, status {model.Status}")
        return None, None, None, None, None
    
    else:
        print(f"Model did not solve to optimality. Status: {model.Status}")
        return None, None, None, None, None

################ SINGLE CLASS SOLVER ####################
def TA_UE(data, n, OD, Q):
    model = gp.Model("UE")
    model.setParam("OutputFlag", 0)

    a = data['links']
    segments = data['approx']
    t0 = data['fftt']
    eta = data['eta']
    alpha = data['alpha']
    beta = data['beta']
    sigma = data['delta']
    cap = data['capacity']
    segments_p = segments.difference({0})
    
    x = [model.addVar(vtype=GRB.CONTINUOUS) for j in range(a)]
    f = [ [model.addVar(vtype=GRB.CONTINUOUS) for i in range(n[k])] for k in range(OD)]
    ll = [ [model.addVar(vtype=GRB.CONTINUOUS) for l in segments] for i in range(a)]
    lr = [ [model.addVar(vtype=GRB.CONTINUOUS) for l in segments] for i in range(a)]

    for i in range(a):
        model.addConstr(x[i] == sum( sum(f[k][p]*sigma[k][p][i] for p in range(n[k])) for k in range(OD)), "link-path%d" %i)
        model.addConstr(x[i] == sum(ll[i][v]*eta[i][v-1] + lr[i][v]*eta[i][v] for v in segments_p), "Approx1%d" %i)
        model.addConstr(sum(ll[i][v] + lr[i][v] for v in segments_p) == 1, "Approx2%d" %i)
        model.addConstr(x[i] >= 0, "integrality_x%d" %i)  
        for ss in segments:
            model.addConstr(ll[i][ss] >= 0, "integrality_ll%d%d" %(i,ss))
            model.addConstr(lr[i][ss] >= 0, "integrality_lr%d%d" %(i,ss))

    for k in range(OD) :
        model.addConstr( sum(f[k][p] for p in range(n[k])) == Q[k] , "FConservation%d" %k ) 
        for p in range(n[k]) : 
            model.addConstr(f[k][p] >= 0, "integrality%d%d" %(k,p))
    
    Z = sum( t0[i]*(x[i]+1/(alpha[i]+1)*beta[i]/(cap[i]**alpha[i])*
                    sum(eta[i][v-1]**(alpha[i]+1)*ll[i][v]+eta[i][v]**(alpha[i]+1)*lr[i][v] for v in segments_p))
                    for i in range(a) )    

    model.setObjective(Z, GRB.MINIMIZE)
    t1 = time.time()
    model.optimize()
    t2 = time.time()
    print('model solved in:',t2-t1)
    
    flows =  [ [ f[k][p].X  for p in range(n[k])] for k in range(OD)]
    linkss = [ x[i].X   for i in range(a)]

    return flows, linkss
    
############## MULTI CLASS UE SOLVER ###############
# def TA_UE(data, n, OD, Q):
#     # n: number of path of each OD pair
#     # OD: number of OD pair 
#     # Q: value of demand 
#     model = gp.Model("UE")
#     model.setParam("OutputFlag", 0)

#     a = data['links']
#     segments = data['approx']
#     t0 = data['fftt']
#     eta = data['eta']
#     alpha = data['alpha']
#     beta = data['beta']
#     sigma = data['delta']
#     cap = data['capacity']
#     segments_p = segments.difference({0})

#     pi = [[1, 2.5] for i in range(a)]
    
#     x = [[model.addVar(vtype=GRB.CONTINUOUS) for j in range(2)]for i in range(a)]
#     f = [ [[model.addVar(vtype=GRB.CONTINUOUS) for j in range(2)]for i in range(n[k])] for k in range(OD)]
#     ll = [ [[model.addVar(vtype=GRB.CONTINUOUS) for j in range(2)] for l in segments] for i in range(a)]
#     lr = [ [[model.addVar(vtype=GRB.CONTINUOUS) for j in range(2)] for l in segments] for i in range(a)]

#     for i in range(a):
#         # x[i][1] = x[i][1] * 2.5
#         for j in range(2):
#             model.addConstr(x[i][j] == sum(sum(f[k][p][j]*sigma[k][p][i] for p in range(n[k])) for k in range(OD)), "link-path%d" %i)
#             model.addConstr(x[i][j] == sum(ll[i][v][j]*eta[i][v-1][j] + lr[i][v][j]*eta[i][v][j] for v in segments_p), "Approx1%d" %i)

#             model.addConstr(sum(ll[i][v][j] + lr[i][v][j] for v in segments_p) == 1, "Approx2%d" %i)
#             model.addConstr(x[i][j] >= 0, "integrality_x%d" %i)   

#             for ss in segments:
#                 model.addConstr(ll[i][ss][j] >= 0, "integrality_ll%d%d" %(i,ss))
#                 model.addConstr(lr[i][ss][j] >= 0, "integrality_lr%d%d" %(i,ss))

#     for k in range(OD) :
#         for j in range(2):
#             model.addConstr( sum(f[k][p][j] for p in range(n[k])) == Q[k][j] , "FConservation%d" %k ) 
#             for p in range(n[k]) : 
#                 model.addConstr(f[k][p][j] >= 0, "integrality%d%d" %(k,p))
    
#     Z = sum( t0[i][j] * (x[i][j]+1/(alpha[i]+1)*beta[i]*(pi[i][j]**alpha[i])/(cap[i]**alpha[i])*
#                     sum(eta[i][v-1][j]**(alpha[i]+1)*ll[i][v][j] + eta[i][v][j]**(alpha[i]+1)*lr[i][v][j] for v in segments_p))
#                     for j in range(2)
#                     for i in range(a))

#     model.setObjective(Z, GRB.MINIMIZE)
#     t1 = time.time()
#     model.optimize()
#     t2 = time.time()
#     print('model solved in:',t2-t1)
    
#     if model.Status == GRB.OPTIMAL:
#         flows =  [[[f[k][p][j].X for j in range(2)] for p in range(n[k])] for k in range(OD)]
#         linkss = [[x[i][j].X for j in range(2)] for i in range(a)]
#         # linkss = None
#         return flows, linkss
    
#     elif model.Status == GRB.INFEASIBLE:
#         print(f"Model is infeasible, status {model.Status}")
#         model.computeIIS()
#         model.write("model.ilp")
#         return None, None
    
#     else:
#         print(f"Model did not solve to optimality. Status: {model.Status}")
#         return None, None

def read_file(filename):
  with open(filename, "rb") as file:
      stat = pickle.load(file)
      file.close()
  return stat

# This function get all feasible paths of all OD pairs.
# Each OD pair has 3 paths
# From this origin path set, when remove any link, we will remove the path containing that link, other paths keep no change.
def get_full_paths(demand_file, net_file, path_num):
    stat = read_file(demand_file)
    Network = pd.read_csv(net_file,delimiter='\t',skiprows=8)

    path_set = set()
    pair_path = defaultdict(list)
    for OD_matrix in tqdm(stat[:10]) :
        paths, paths_N = find_paths(Network, OD_matrix, path_num)
        path_set.add(p for path in paths.values() for p in path)
        for k, v in paths.items():
            pair_path[k] = [tuple(p) for p in v]
    pair_path = dict(pair_path)

    path_set = {tuple(p) for path in path_set for p in path}
    path_set_dict = {v: k for k, v in enumerate(path_set, start=1)}
    return path_set_dict, pair_path

def solve_UE(net_file, demand_file, pair_path, output_file, to_solve):
    stat = read_file(demand_file)
    Network, Nodes, links, cap, fft, alpha, beta, lengths = readNet(net_file)

    time = 0
    for OD_matrix in tqdm(stat[:to_solve]):
        print(time)
        paths = {k: (pair_path[k][:3] if len(pair_path[k]) >= 3 else pair_path[k]) for k in OD_matrix.keys()}
        data, Q, OD, O_D,n = get_data(Network, Nodes, links, cap, fft, alpha, beta, lengths, OD_matrix, paths)
        flows, linkss, path_cost, min_cost, link_cost = BRUE(data, n, OD, Q)
        dataa = {'data' : data, 'path_flow' : flows, 'link_flow' : linkss, 'path_cost': path_cost, 'min_cost': min_cost, 'link_cost': link_cost}
        # flows, linkss = BRUE(data, n, OD, Q)
        # dataa = {'data' : data, 'path_flow' : flows, 'link_flow' : linkss}
        file_data = open(output_file+str(time), "wb")
        pickle.dump(dataa , file_data)
        file_data.close()
        time +=1

# This function remove the link in the feasible path when that link is removed in the network
def remove_links_from_path(pair_path, remove_ids):
    remove_ids = set(remove_ids)
    new_dict = defaultdict(list)
    for key, value in pair_path.items():
        v = [p for p in value if not any(link in p for link in remove_ids)]
        new_dict[key] = v      
    return dict(new_dict)

def remove_links_from_tntp(input_file, output_file, remove_ids):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    remove_ids = set(remove_ids)

    modified_lines = []
    for line in lines:
        parts = line.split()
        if parts and parts[0].isdigit() and int(parts[0]) in remove_ids:
            # Replace all values from column 2 onward with '0'
            parts[1:] = ['0'] * (len(parts) - 2)
            modified_line = '\t' + '\t'.join(parts) + '\t;\n'
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)
    
    with open(output_file, 'w') as file:
        file.writelines(modified_lines)