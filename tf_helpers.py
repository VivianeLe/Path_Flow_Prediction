import numpy as np
import pickle
import tensorflow as tf
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

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
def get_flowTensor(demand, path_flows, nodes):
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
    # mask = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(tensor), axis=0)),0) # create mask for column
    return mask

def reduce_dimensionality(X, n_components):
    X_np = X.numpy()
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_np)

    # Convert back to TensorFlow tensor
    X_pca_tf = tf.convert_to_tensor(X_pca, dtype=tf.float32)
    return X_pca_tf

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
    # Pair_path = get_pair_path_tensor(unique_set,path_links,nodes) # (625, 1155)
    Path_set = get_pathTensor(path_links, nodes) # (625, 3)
    X = tf.concat([tf.cast(Graph, tf.float32),
                   tf.cast(OD_demand, tf.float32),
                   tf.cast(Path_set, tf.float32)
                #    tf.cast(Pair_path, tf.float32)
                   ], axis=1) # (625,1162)
    X_mask = create_mask(X)
    X = normalize(X)

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
    # plt.text(0.82, 0.75,
    #           f'Learning Rate: {learning_rate}\n'
    #             f'Training Time: {train_time/60:.2f}m\n'
    #             f'Layers number: {N}\n'
    #             f'D_model: {d_model}',
    #           transform=plt.gca().transAxes,
    #           fontsize=10
    #           )

    plt.show()

def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()