import datetime
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from sklearn.cluster import SpectralClustering
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from numba import njit, prange
import warnings
from typing import Optional, Tuple, List

@njit
def isin_2d(a, b):
    """Numba-accelerated 2D array element check"""
    s = set((x, y) for x, y in b)
    return np.array([(x, y) in s for x, y in a])

def anomaly_generation(ini_graph_percent, anomaly_percent, data, n, m, seed=1):
    np.random.seed(seed)
    print(f'[{datetime.datetime.now()}] generating anomalous dataset...')
    print(f'[{datetime.datetime.now()}] initial network: {ini_graph_percent*100:.1f}%, anomaly: {anomaly_percent*100:.1f}%')

    np.random.seed(seed)
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.int32)
    
    train_num = int(ini_graph_percent * m)
    if train_num <= 0 or train_num >= m:
        raise ValueError(f"Invalid train_num {train_num} from ini_graph_percent {ini_graph_percent}")
        
    train = data[:train_num].copy()  # Create copy to prevent modifications
    test = data[train_num:].copy()

    # Optimized adjacency matrix creation
    adjacency_matrix = edgeList2Adj(data)
    
    # Cluster with sparse matrix input
    sc = SpectralClustering(
        42, 
        affinity='precomputed', 
        n_init=10,  # Restore original n_init
        assign_labels='discretize',  # Critical fix
        random_state=seed,
        n_jobs=-1
    )
    labels = sc.fit_predict(adjacency_matrix)

    # Vectorized fake edge generation
    idx_pool = np.arange(n)
    fake_edges = np.column_stack((
        np.random.choice(idx_pool, 2*m),
        np.random.choice(idx_pool, 2*m)
    )).astype(np.int32) + 1  # Match original 1-based indexing

    # Fast edge processing
    fake_edges = processEdges(fake_edges, data)
    
    # Anomaly injection with pre-allocation
    anomaly_num = int(anomaly_percent * len(test))
    anomalies = fake_edges[:anomaly_num]
    
    # Synthetic test construction
    synthetic_test = np.zeros((len(test) + anomaly_num, 3), dtype=np.int32)
    anomaly_pos = np.random.choice(len(synthetic_test), anomaly_num, False)
    
    # Vectorized assignments
    synthetic_test[anomaly_pos, :2] = anomalies
    synthetic_test[anomaly_pos, 2] = 1
    synthetic_test[~np.isin(np.arange(len(synthetic_test)), anomaly_pos), :2] = test
    
    # Optimized sparse matrix construction
    row = train[:, 0].ravel()
    col = train[:, 1].ravel()
    train_mat = coo_matrix((np.ones_like(row), (row, col)), shape=(n, n)).tocsr()
    train_mat = (train_mat + train_mat.T + sparse.eye(n)).tolil()
    
    return synthetic_test, train_mat, train

def anomaly_generation2(ini_graph_percent, anomaly_percent, data, n, m,seed = 1):
    """ generate anomaly
    split the whole graph into training network which includes parts of the
    whole graph edges(with ini_graph_percent) and testing edges that includes
    a ratio of manually injected anomaly edges, here anomaly edges mean that
    they are not shown in previous graph;
     input: ini_graph_percent: percentage of edges in the whole graph will be
                                sampled in the intitial graph for embedding
                                learning
            anomaly_percent: percentage of edges in testing edges pool to be
                              manually injected anomaly edges(previous not
                              shown in the whole graph)
            data: whole graph matrix in sparse form, each row (nodeID,
                  nodeID) is one edge of the graph
            n:  number of total nodes of the whole graph
            m:  number of edges in the whole graph
     output: synthetic_test: the testing edges with injected abnormal edges,
                             each row is one edge (nodeID, nodeID, label),
                             label==0 means the edge is normal one, label ==1
                             means the edge is abnormal;
             train_mat: the training network with square matrix format, the training
                        network edges for initial model training;
             train:  the sparse format of the training network, each row
                        (nodeID, nodeID)
    """
    # The actual generation method used for Netwalk(shown in matlab version)
    # Abort the SpectralClustering
    np.random.seed(seed)
    print('[%s] generating anomalous dataset...\n' % datetime.datetime.now())
    print('[%s] initial network edge percent: %.2f, anomaly percent: %.2f.\n' % (
        datetime.datetime.now(),
        ini_graph_percent,
        anomaly_percent
    ))

    # ini_graph_percent = 0.5;
    # anomaly_percent = 0.05;
    train_num = int(np.floor(ini_graph_percent * m))

    # select part of edges as in the training set
    train = data[0:train_num, :]

    # select the other edges as the testing set
    test = data[train_num:, :]

    #data to adjacency_matrix
    #adjacency_matrix = edgeList2Adj(data)

    # clustering nodes to clusters using spectral clustering
    # kk = 3 #3#10#42#42
    # sc = SpectralClustering(kk, affinity='precomputed', n_init=10, assign_labels = 'discretize',n_jobs=-1)
    # labels = sc.fit_predict(adjacency_matrix)


    # generate fake edges that are not exist in the whole graph, treat them as
    # anamalies
    # 真就直接随机生成
    idx_1 = np.expand_dims(np.transpose(np.random.choice(n, m)) , axis=1)
    idx_2 = np.expand_dims(np.transpose(np.random.choice(n, m)) , axis=1)
    fake_edges = np.concatenate((idx_1, idx_2), axis=1)

    ####### genertate abnormal edges ####
    #fake_edges = np.array([x for x in generate_edges if labels[x[0] - 1] != labels[x[1] - 1]])

    # 移除掉self-loop以及真实边
    fake_edges = processEdges(fake_edges, data)

    #anomaly_num = 12#int(np.floor(anomaly_percent * np.size(test, 0)))
    # 按比例圈定要的异常边
    anomaly_num = int(np.floor(anomaly_percent * np.size(test, 0)))
    anomalies = fake_edges[0:anomaly_num, :]

    # 按照总边数（测试正常+异常）圈定标签
    idx_test = np.zeros([np.size(test, 0) + anomaly_num, 1], dtype=np.int32)
    # randsample: sample without replacement
    # it's different from datasample!

    # 随机选择异常边的位置
    anomaly_pos = np.random.choice(np.size(idx_test, 0), anomaly_num, replace=False)

    #anomaly_pos = np.random.choice(100, anomaly_num, replace=False)+200
    # 选定的位置定为1
    idx_test[anomaly_pos] = 1

    # 汇总数据，按照起点，终点，label的形式填充，并且把对应的idx找出
    synthetic_test = np.concatenate((np.zeros([np.size(idx_test, 0), 2], dtype=np.int32), idx_test), axis=1)
    idx_anomalies = np.nonzero(idx_test.squeeze() == 1)
    idx_normal = np.nonzero(idx_test.squeeze() == 0)
    synthetic_test[idx_anomalies, 0:2] = anomalies
    synthetic_test[idx_normal, 0:2] = test

    # coo:efficient for matrix construction ;  csr: efficient for arithmetic operations
    # coo+to_csr is faster for small matrix, but nearly the same for large matrix (size: over 100M)
    #train_mat = csr_matrix((np.ones([np.size(train, 0)], dtype=np.int32), (train[:, 0] , train[:, 1])),shape=(n, n))
    train_mat = coo_matrix((np.ones([np.size(train, 0)], dtype=np.int32), (train[:, 0], train[:, 1])), shape=(n, n)).tocsr()
    # sparse(train(:,1), train(:,2), ones(length(train), 1), n, n)
    train_mat = train_mat + train_mat.transpose()

    return synthetic_test, train_mat, train

@njit(parallel=True)
def processEdges(fake_edges, data):
    """Optimized edge processing with numba acceleration"""
    if not isinstance(fake_edges, np.ndarray):
        fake_edges = np.array(fake_edges, dtype=np.int32)
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.int32)
        
    # Remove self-loops and sort edges in single pass
    valid_edges = np.zeros((len(fake_edges), 2), dtype=np.int32)
    valid_count = 0
    
    for i in prange(len(fake_edges)):
        u, v = fake_edges[i]
        if u != v:  # Not a self-loop
            # Order vertices
            if u > v:
                u, v = v, u
            valid_edges[valid_count] = [u, v]
            valid_count += 1
            
    # Resize array to actual valid edges
    valid_edges = valid_edges[:valid_count]
    
    # Remove duplicates efficiently
    sorted_idx = np.lexsort((valid_edges[:,1], valid_edges[:,0]))
    valid_edges = valid_edges[sorted_idx]
    
    # Remove edges that exist in original data
    mask = np.ones(len(valid_edges), dtype=np.bool_)
    for i in prange(len(data)):
        orig_u, orig_v = data[i]
        if orig_u > orig_v:
            orig_u, orig_v = orig_v, orig_u
        for j in prange(len(valid_edges)):
            if valid_edges[j,0] == orig_u and valid_edges[j,1] == orig_v:
                mask[j] = False
                
    return valid_edges[mask]


def edgeList2Adj(data):
    """Create adjacency matrix with proper bounds checking"""
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.int32)
        
    # Ensure non-negative indices
    if (data < 0).any():
        raise ValueError("Negative indices found in edge list")
        
    # Get dimensions without subtraction (assuming 0-based indexing)
    users = data[:, 0]
    items = data[:, 1]
    n = max(np.max(users), np.max(items)) + 1
    
    # Create sparse matrix efficiently
    rows = np.concatenate([users, items])
    cols = np.concatenate([items, users])
    
    return coo_matrix(
        (np.ones(2*len(data), dtype=np.int32), (rows, cols)),
        shape=(n, n),
        dtype=np.int32
    ).tocsr()

if __name__ == "__main__":
    data_path = "data/karate.edges"
    # data_path = './fb-messages2.txt'

    edges = np.loadtxt(data_path, dtype=float, comments='%',delimiter=',')
    edges = edges[:,0:2].astype(dtype=int)

    vertices = np.unique(edges)
    m = len(edges)
    n = len(vertices)

    synthetic_test, train_mat, train = anomaly_generation(0.5, 0.1, edges, n, m)

    print(train)
