def validate_edge_list(edges, num_nodes=None):
    """Validate edge list format and indices"""
    if not isinstance(edges, (list, np.ndarray)):
        raise TypeError("edges must be list or numpy array")
        
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must be Nx2 array")
        
    if np.any(edges < 0):
        raise ValueError("negative indices found in edges")
        
    if num_nodes is not None:
        if np.any(edges >= num_nodes):
            raise ValueError(f"edge indices must be < {num_nodes}")
            
    return edges

def validate_sparse_matrix(matrix):
    """Validate sparse matrix properties"""
    if not sparse.issparse(matrix):
        raise TypeError("matrix must be scipy sparse matrix")
        
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
        
    if np.any(matrix.data < 0):
        raise ValueError("negative values found in matrix")
        
    return matrix

def validate_batch_dict(batch_dict, num_nodes):
    """Validate batch dictionary format"""
    if not isinstance(batch_dict, dict):
        raise TypeError("batch_dict must be dictionary")
        
    for key, entries in batch_dict.items():
        if not isinstance(entries, list):
            raise TypeError(f"entries for key {key} must be list")
            
        for entry in entries:
            if not isinstance(entry, tuple) or len(entry) != 4:
                raise TypeError(f"invalid entry format for key {key}")
            
            node, idx, hop, time = entry
            if not (isinstance(node, int) and 
                   isinstance(idx, int) and
                   isinstance(hop, int) and
                   isinstance(time, int)):
                raise TypeError(f"invalid entry types for key {key}")
                
            if not (0 <= node < num_nodes):
                raise ValueError(f"invalid node index {node} for key {key}")
                
            if hop < 0 or time < 0:
                raise ValueError(f"negative hop/time values for key {key}")
                
    return batch_dict

def validate_tensors(*tensors):
    """Validate tensor properties"""
    for i, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"input {i} must be torch.Tensor")
            
        if tensor.isnan().any():
            raise ValueError(f"NaN values found in tensor {i}")
            
        if tensor.isinf().any():
            raise ValueError(f"Inf values found in tensor {i}")
    
    return tensors