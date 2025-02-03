from codes.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv
import pickle
import os
#import pyarrow.csv as pa


class DynamicDatasetLoader(dataset):
    c = 0.15
    k = 5
    eps = 0.001
    window_size = 1
    data = None
    batch_size = None
    dataset_name = None
    load_all_tag = False
    compute_s = False
    anomaly_per = 0.1
    train_per = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Add this line

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(DynamicDatasetLoader, self).__init__(dName, dDescription)
        

    def load_hop_wl_batch(self):  #load the "raw" WL/Hop/Batch dict
        print('Load WL Dictionary')
        f = open('./result/WL/' + self.dataset_name, 'rb')
        wl_dict = pickle.load(f)
        f.close()

        print('Load Hop Distance Dictionary')
        f = open('./result/Hop/hop_' + self.dataset_name + '_' + str(self.k) + '_' + str(self.window_size), 'rb')
        hop_dict = pickle.load(f)
        f.close()

        print('Load Subgraph Batches')
        f = open('./result/Batch/' + self.dataset_name + '_' + str(self.k) + '_' + str(self.window_size), 'rb')
        batch_dict = pickle.load(f)
        f.close()

        return hop_dict, wl_dict, batch_dict

    @staticmethod
    def normalize(mx, epsilon: float = 1e-6):
        """Row-normalize sparse matrix with epsilon for numerical stability"""
        if not sp.issparse(mx):
            mx = sp.csr_matrix(mx)
        rowsum = np.array(mx.sum(1)) + epsilon
        r_inv = np.power(rowsum, -1).flatten()
        r_mat_inv = sp.diags(r_inv)
        return r_mat_inv.dot(mx)

    @staticmethod
    def normalize_adj(adj, epsilon: float = 1e-6):
        """Symmetrically normalize adjacency matrix"""
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        rowsum = np.array(adj.sum(1)) + epsilon
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    @staticmethod
    def adj_normalize(mx, epsilon: float = 1e-6):
        """Row-normalize sparse matrix with handling for infinite values"""
        if not sp.issparse(mx):
            mx = sp.csr_matrix(mx)
        rowsum = np.array(mx.sum(1)) + epsilon
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        return r_mat_inv.dot(mx).dot(r_mat_inv)

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def sparse_to_tuple(self, sparse_mx):
        """Convert sparse matrix to tuple representation. (0226)"""

        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)
        return sparse_mx

    def preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation. (0226)"""
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj_np = np.array(adj.todense())
        adj_normalized = self.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_normalized = self.sparse_mx_to_torch_sparse_tensor(adj_normalized).to(self.device)
        return adj_normalized

    def get_adjs(self, rows, cols, weights, nb_nodes):
        """Create adjacency matrices with proper dimension checks"""
        try:
            eigen_file_name = os.path.join('data', 'eigen', 
                f'{self.dataset_name}_{self.train_per}_{self.anomaly_per}.pkl')
            
            # Find actual maximum node ID
            max_node = max(
                max(max(row) for row in rows if len(row) > 0),
                max(max(col) for col in cols if len(col) > 0)
            )
            
            # Adjust nb_nodes to accommodate all indices
            nb_nodes = max_node + 1
            
            generate_eigen = not os.path.exists(eigen_file_name)
            eigen_adjs = []
            
            if not generate_eigen:
                with open(eigen_file_name, 'rb') as f:
                    eigen_adjs_sparse = pickle.load(f)
                eigen_adjs = eigen_adjs_sparse
            
            adjs = []
            if generate_eigen:
                eigen_adjs_sparse = []
            
            for i in range(len(rows)):
                # Validate indices
                if len(rows[i]) != len(cols[i]) or len(rows[i]) != len(weights[i]):
                    raise ValueError(f"Dimension mismatch in snapshot {i}")
                    
                # Create sparse matrix with corrected dimensions
                adj = sp.csr_matrix(
                    (weights[i], (rows[i], cols[i])),
                    shape=(nb_nodes, nb_nodes),
                    dtype=np.float32
                ).tocoo()
                
                adjs.append(self.preprocess_adj(adj))
                
                if self.compute_s and generate_eigen:
                    # Rest of eigen computation remains the same
                    adj_normalized = self.adj_normalize(adj).tocsr()
                    B = (1 - self.c) * adj_normalized
                    
                    S = sp.eye(adj.shape[0], format='csr', dtype=np.float32)
                    current_term = B.copy()
                    
                    for _ in range(50):
                        if current_term.nnz == 0:
                            break
                        S += current_term
                        current_term = current_term.dot(B)
                        current_term.data[abs(current_term.data) < 1e-6] = 0
                        current_term.eliminate_zeros()
                    
                    eigen_adj = self.c * S
                    eigen_adj.setdiag(0)
                    eigen_adj = self.normalize(eigen_adj)
                    eigen_adjs_sparse.append(eigen_adj)
                    eigen_adjs.append(eigen_adj)
            
            if generate_eigen:
                with open(eigen_file_name, 'wb') as f:
                    pickle.dump(eigen_adjs_sparse, f, pickle.HIGHEST_PROTOCOL)
            
            return adjs, eigen_adjs
            
        except Exception as e:
            print(f"Error in get_adjs: {str(e)}")
        raise

    def load(self):
        try:
            file_path = os.path.join('data', 'percent', 
                f'{self.dataset_name}_{self.train_per}_{self.anomaly_per}.pkl')
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
                
            with open(file_path, 'rb') as f:
                rows, cols, labels, weights, headtail, train_size, test_size, nb_nodes, nb_edges, edge_data = pickle.load(f)
                
            # Validate data
            if not all(isinstance(x, list) for x in [rows, cols, labels, weights]):
                raise ValueError("Invalid data format: expected lists for rows, cols, labels, weights")
                
            # Convert to tensors on appropriate device
            edges = [np.vstack((rows[i], cols[i])).T for i in range(train_size + test_size)]
            adjs, eigen_adjs = self.get_adjs(rows, cols, weights, nb_nodes)
            labels = [torch.LongTensor(label).to(self.device, non_blocking=True) for label in labels]
            
            snap_train = list(range(train_size))
            snap_test = list(range(train_size, train_size + test_size))
            
            idx = np.array(range(nb_nodes))
            index_id_map = {i:i for i in idx}  # Maintain 0-based indexing
            
            return {
                'X': None,
                'A': adjs,
                'S': eigen_adjs,
                'index_id_map': index_id_map,
                'edges': edges,
                'y': labels,
                'idx': idx,
                'snap_train': snap_train,
                'degrees': np.array([len(x) for x in headtail]),
                'snap_test': snap_test,
                'num_snap': train_size + test_size,
                'edge_data': edge_data
            }
            
        except Exception as e:
            print(f"Error loading dataset {self.dataset_name}: {str(e)}")
            raise
