import numpy as np
import pickle
import tensorflow as tf
from tqdm.notebook import tqdm
from numba import jit
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import plotly.offline as py

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
    tensor = tf.squeeze(tensor, axis=0)
    tensor = tf.reshape(tensor, [-1]) # Flatten the matrix to a 1D tensor
    tensor = tf.expand_dims(tensor, axis=1) # TensorShape([625, 1])
    return tensor

def get_graphTensor(network, nodes):
    cap = np.array(network[['init_node', 'term_node', 'capacity']].apply(lambda row: ((row['init_node'], row['term_node']), row['capacity']), axis=1).tolist(), dtype=object)
    length = np.array(network[['init_node', 'term_node', 'length']].apply(lambda row: ((row['init_node'], row['term_node']), row['length']), axis=1).tolist(), dtype=object)
    fft = np.array(network[['init_node', 'term_node', 'free_flow_time']].apply(lambda row: ((row['init_node'], row['term_node']), row['free_flow_time']), axis=1).tolist(), dtype=object)

    Cap = create_single_tensor(cap, nodes)
    Length = create_single_tensor(length, nodes)
    Fft = create_single_tensor(fft, nodes)
    tensor = tf.concat([tf.cast(Cap, tf.float32), tf.cast(Length, tf.float32), tf.cast(Fft, tf.float32)], axis=1)
    return tensor

def get_demandTensor(demand, nodes):
    tensor = np.array([(key, value) for key, value in demand.items()], dtype=object)
    tensor = create_single_tensor(tensor, nodes)
    return tensor

def get_pathTensor(path_links, nodes, path_encoded):
    paths = np.array([(key, [tuple(path) for path in value]) for key, value in path_links.items()], dtype=object)
    p1, p2, p3 = [], [], []
    for od, [path1, path2, path3] in paths:
        p1.append((od, path_encoded[path1]))
        p2.append((od, path_encoded[path2]))
        p3.append((od, path_encoded[path3]))

    p1 = create_single_tensor(p1, nodes)
    p2 = create_single_tensor(p2, nodes)
    p3 = create_single_tensor(p3, nodes)
    tensor = tf.concat([tf.cast(p1, tf.float32), tf.cast(p2, tf.float32), tf.cast(p3, tf.float32)], axis=1)
    return tensor

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
    tensor = tf.concat([tf.cast(p1, tf.float32), tf.cast(p2, tf.float32), tf.cast(p3, tf.float32)], axis=1)
    return tensor

# Try standardize to replace normalize function
def standardize(tensor):
    mean = tf.reduce_mean(tensor, axis=0)
    std = tf.math.reduce_std(tensor, axis=0)
    std = tf.where(tf.equal(std, 0), 1.0, std)
    standardized_tensor = (tensor - mean) / std
    return standardized_tensor

# Tạo mask trên raw data chưa norm, tính tổng các giá trị theo chiều cuối cùng (dim = -1) 
#  tức tổng của 7 cột của mỗi hàng. Nếu hàng nào sum >0 thì trả về 1, sum = 0 thì trả về 0
def create_mask(tensor):
    mask = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(tensor), axis=-1)),-1)
    return mask

def generate_xy(file_name, path_encoded):
    with open(file_name, "rb") as file:
        stat = pickle.load(file)
    
    path_links = stat["data"]["paths_link"]
    demand = stat["data"]["demand"]
    path_flows = stat["path_flow"]
    nodes = stat["data"]["nodes"]
    net = stat["data"]["network"]

    # Get X
    Graph = get_graphTensor(net, nodes)
    OD_demand = get_demandTensor(demand, nodes)
    Path_tensor = get_pathTensor(path_links, nodes, path_encoded)
    X = tf.concat([Graph, OD_demand, Path_tensor], axis=1)
    X_mask = create_mask(X)
    X = tf.concat([normalize(Graph), normalize(OD_demand), normalize(Path_tensor)], axis=1)

    # Get Y
    Y = get_flowTensor(demand, path_flows, nodes)
    Y_mask = create_mask(Y)
    return X, Y, X_mask, Y_mask

def plot_loss(train_loss, val_loss, epochs, learning_rate, train_time, N, d_model):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs+1), train_loss, label='Training Loss')
    plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validating Loss')
    plt.legend()
    plt.grid(True)
    plt.text(0.82, 0.75,
              f'Learning Rate: {learning_rate}\n'
                f'Training Time: {train_time/60:.2f}m\n'
                f'Layers number: {N}\n'
                f'D_model: {d_model}',
            #   horizontalalignment='center', 
            #   verticalalignment='center', 
              transform=plt.gca().transAxes, 
              fontsize=10
              )

    plt.show()

def plot_loss_plotly(train_loss, val_loss, epochs, learning_rate, train_time):
    fig = go.Figure()
    # Add traces for training and validation loss
    fig.add_trace(go.Scatter(x=list(range(1, epochs + 1)), y=train_loss, mode='lines', name='Training Loss'))
    fig.add_trace(go.Scatter(x=list(range(1, epochs + 1)), y=val_loss, mode='lines', name='Validation Loss'))

    # Add title and labels
    fig.update_layout(
        title=f'Training and Validation Loss over {epochs} Epochs',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend=dict(x=0.01, y=0.99),
        annotations=[
            go.layout.Annotation(
                x=0.5,
                y=1,
                xref='paper',
                yref='paper',
                showarrow=False,
                text=f'Learning Rate: {learning_rate}<br>Training Time: {train_time / 60:.2f}m',
                align='center',
                bgcolor='white',
                opacity=0.6
            )
        ]
    )
    py.plot(fig, filename='Training-and-Validation-Loss')