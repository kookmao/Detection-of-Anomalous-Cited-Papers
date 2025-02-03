import datetime
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from sklearn.cluster import SpectralClustering
from numba import njit


@njit
def isin_2d(a, b):
    """Numba-accelerated 2D array membership check"""
    b_set = set((x, y) for x, y in b)
    return np.array([(x, y) in b_set for x, y in a])

@njit
def process_edges_batch(edges, start_idx, end_idx, n):
    """Process a batch of edges efficiently"""
    batch = edges[start_idx:end_idx]
    valid_mask = (batch[:, 0] < n) & (batch[:, 1] < n) & (batch[:, 0] >= 0) & (batch[:, 1] >= 0)
    return batch[valid_mask]

def anomaly_generation(ini_graph_percent, anomaly_percent, data, n, m, seed=1):
    """Generate anomalies with memory-efficient implementation"""
    np.random.seed(seed)
    print(f'[{datetime.datetime.now()}] generating anomalous dataset...')
    print(f'[{datetime.datetime.now()}] initial network: {ini_graph_percent*100:.1f}%, anomaly: {anomaly_percent*100:.1f}%')

    # Input validation
    data = np.asarray(data, dtype=np.int32)
    if np.any(data < 0) or np.any(data >= n):
        raise ValueError("Edge indices must be within [0, n)")

    # 1. Split training data
    train_num = int(ini_graph_percent * m)
    train = data[:train_num]
    test = data[train_num:]

    # 2. Generate adjacency matrix efficiently
    rows = data[:, 0]
    cols = data[:, 1]
    values = np.ones(len(rows), dtype=np.int32)
    adj = coo_matrix((values, (rows, cols)), shape=(n, n), dtype=np.int32).tocsr()
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)

    # 3. Generate fake edges in batches
    batch_size = 100000
    fake_edges = []
    required_fake_edges = 2 * m

    while len(fake_edges) < required_fake_edges:
        batch_edges = np.column_stack((
            np.random.choice(n, batch_size),
            np.random.choice(n, batch_size)
        ))
        
        # Remove self-loops and invalid edges
        mask = batch_edges[:, 0] != batch_edges[:, 1]
        batch_edges = batch_edges[mask]
        
        # Convert to set for efficient lookup
        existing_edges = set(map(tuple, data))
        batch_edges = np.array([
            edge for edge in batch_edges 
            if tuple(edge) not in existing_edges and 
               tuple(edge[::-1]) not in existing_edges
        ])
        
        fake_edges.extend(batch_edges)
        if len(fake_edges) >= required_fake_edges:
            fake_edges = np.array(fake_edges[:required_fake_edges])
            break

    # 4. Create synthetic test set
    anomaly_num = int(anomaly_percent * len(test))
    if len(fake_edges) < anomaly_num:
        raise ValueError(f"Not enough fake edges generated ({len(fake_edges)} < {anomaly_num})")

    anomalies = fake_edges[:anomaly_num]
    synthetic_test = np.zeros((len(test) + anomaly_num, 3), dtype=np.int32)
    
    # Insert anomalies at random positions
    anomaly_pos = np.random.choice(len(synthetic_test), anomaly_num, False)
    synthetic_test[anomaly_pos, :2] = anomalies
    synthetic_test[anomaly_pos, 2] = 1
    
    # Fill remaining positions with test edges
    normal_pos = np.setdiff1d(np.arange(len(synthetic_test)), anomaly_pos)
    synthetic_test[normal_pos, :2] = test

    # 5. Create training matrix efficiently
    train_mat = coo_matrix(
        (np.ones(len(train)), (train[:, 0], train[:, 1])),
        shape=(n, n)
    ).tocsr()
    
    # Make symmetric
    train_mat = train_mat + train_mat.T
    train_mat.data = np.ones_like(train_mat.data)

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

def processEdges(fake_edges, data):
    """Process edges with efficient set operations"""
    # Convert to numpy arrays
    fake_edges = np.asarray(fake_edges, dtype=np.int32)
    data = np.asarray(data, dtype=np.int32)
    
    # Remove self-loops
    mask = fake_edges[:, 0] != fake_edges[:, 1]
    fake_edges = fake_edges[mask]
    
    # Sort edges for consistency
    swap_mask = fake_edges[:, 0] > fake_edges[:, 1]
    fake_edges[swap_mask] = fake_edges[swap_mask][:, [1, 0]]
    
    # Remove duplicates
    fake_edges = np.unique(fake_edges, axis=0)
    
    # Remove existing edges efficiently
    data_set = set(map(tuple, data))
    mask = np.array([
        tuple(edge) not in data_set 
        for edge in fake_edges
    ])
    
    return fake_edges[mask]


def edgeList2Adj(data):
    """Convert edge list to adjacency matrix efficiently"""
    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=np.int32)
    
    n = data.max() + 1
    
    # Create sparse matrix directly
    adj = coo_matrix(
        (np.ones(len(data), dtype=np.int32), 
         (data[:, 0], data[:, 1])),
        shape=(n, n)
    ).tocsr()
    
    # Make symmetric
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    
    return adj

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
