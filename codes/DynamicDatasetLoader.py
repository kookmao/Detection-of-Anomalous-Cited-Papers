from codes.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv
import pickle
import os
#import pyarrow.csv as pa
from scipy.sparse.linalg import inv


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def normalize(self, mx):
        """Row-normalize sparse matrix with proper epsilon handling"""
        epsilon = 1e-12  # Small epsilon to prevent division by zero
        mx = mx.tocsr()
        rowsum = np.array(mx.sum(1)).flatten()
        # Add epsilon to zero rows
        rowsum[rowsum == 0] = epsilon
        r_inv = np.power(rowsum, -1)
        r_mat_inv = sp.diags(r_inv)
        mx_norm = r_mat_inv.dot(mx)
        return mx_norm.tocsr()

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix"""
        epsilon = 1e-12
        adj = adj.tocoo()
        rowsum = np.array(adj.sum(1)).flatten()
        rowsum[rowsum == 0] = epsilon
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix with zero handling"""
        epsilon = 1e-12
        mx = mx.tocsr()
        rowsum = np.array(mx.sum(1)).flatten()
        rowsum[rowsum == 0] = epsilon
        r_inv = np.power(rowsum, -0.5)
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx.tocsr()

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert scipy sparse matrix to torch sparse tensor with error checking"""
        if not sp.isspmatrix_coo(sparse_mx):
            sparse_mx = sparse_mx.tocoo()
        
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data.astype(np.float32))
        shape = torch.Size(sparse_mx.shape)
        
        return torch.sparse_coo_tensor(indices, values, shape, 
                                     device=self.device)

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
        """Compute adjacency matrices with memory-efficient eigen decomposition"""
        eigen_file_name = f'data/eigen/{self.dataset_name}_{self.train_per}_{self.anomaly_per}.pkl'
        generate_eigen = not os.path.exists(eigen_file_name)
        
        print(f'{"Generating" if generate_eigen else "Loading"} eigen as: {eigen_file_name}')
        
        if not generate_eigen:
            with open(eigen_file_name, 'rb') as f:
                eigen_adjs_sparse = pickle.load(f)
            return [self.preprocess_adj(adj) for adj in eigen_adjs_sparse], eigen_adjs_sparse
        
        adjs = []
        eigen_adjs_sparse = []
        
        for i in range(len(rows)):
            # Construct sparse adjacency matrix
            adj = sp.csr_matrix(
                (weights[i], (rows[i], cols[i])), 
                shape=(nb_nodes, nb_nodes), 
                dtype=np.float32
            ).tocoo()
            
            # Process adjacency matrix
            adj_normalized = self.preprocess_adj(adj)
            adjs.append(adj_normalized)
            
            if self.compute_s:
                # Compute eigen_adj using iterative method
                adj_norm = self.adj_normalize(adj)
                B = (1 - self.c) * adj_norm
                S = sp.eye(nb_nodes, format='csr', dtype=np.float32)
                current_term = B
                
                for _ in range(50):  # Max iterations
                    S += current_term
                    current_term = current_term.dot(B)
                    # Cleanup small values
                    current_term.data[abs(current_term.data) < 1e-10] = 0
                    current_term.eliminate_zeros()
                    if current_term.nnz == 0:
                        break
                
                eigen_adj = self.c * S
                eigen_adj.setdiag(0)
                eigen_adj = self.normalize(eigen_adj)
                eigen_adjs_sparse.append(eigen_adj)
        
        if generate_eigen:
            with open(eigen_file_name, 'wb') as f:
                pickle.dump(eigen_adjs_sparse, f, pickle.HIGHEST_PROTOCOL)
        
        return adjs, eigen_adjs_sparse

    def load(self):
            """Load dynamic network dataset with proper error handling"""
            print(f'Loading {self.dataset_name} dataset...')
            
            file_path = f'data/percent/{self.dataset_name}_{self.train_per}_{self.anomaly_per}.pkl'
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
                
            with open(file_path, 'rb') as f:
                rows, cols, labels, weights, headtail, train_size, test_size, nb_nodes, nb_edges, edge_data = pickle.load(f)
            
            # Validate data
            if nb_nodes <= 0 or nb_edges <= 0:
                raise ValueError(f"Invalid graph dimensions: nodes={nb_nodes}, edges={nb_edges}")
                
            # Process data
            degrees = np.array([len(x) for x in headtail])
            num_snap = test_size + train_size
            edges = [np.vstack((rows[i], cols[i])).T for i in range(num_snap)]
            
            # Get adjacency matrices
            adjs, eigen_adjs = self.get_adjs(rows, cols, weights, nb_nodes)
            
            # Convert labels to tensors
            labels = [torch.LongTensor(label) for label in labels]
            
            # Create snapshots
            snap_train = list(range(num_snap))[:train_size]
            snap_test = list(range(num_snap))[train_size:]
            
            # Create node index mapping
            idx = np.arange(nb_nodes)
            index_id_map = {i:i for i in idx}
            
            return {
                'X': None,
                'A': adjs,
                'S': eigen_adjs,
                'index_id_map': index_id_map,
                'edges': edges,
                'y': labels,
                'idx': idx,
                'snap_train': snap_train,
                'degrees': degrees,
                'snap_test': snap_test,
                'num_snap': num_snap,
                'edge_data': edge_data
            }
