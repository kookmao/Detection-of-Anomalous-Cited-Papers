from codes.AnomalyGeneration import *
from scipy import sparse
import pickle
import time
import os
import argparse
import numpy as np
from pathlib import Path
from numba import njit, prange
import pandas as pd
import shutil
from tqdm import tqdm  # Added for progress bar


@njit(parallel=True)
def optimize_edge_order(edges):
    """Parallel edge ordering with bounds checking"""
    edges = edges.copy()  # Prevent modifying input
    
    # Verify no negative indices
    if np.any(edges < 0):
        raise ValueError("Negative indices found in edges")
    
    for i in prange(len(edges)):
        if edges[i, 0] > edges[i, 1]:
            temp = edges[i, 0]
            edges[i, 0] = edges[i, 1]
            edges[i, 1] = temp
    
    # Remove self-loops efficiently
    mask = edges[:, 0] != edges[:, 1]
    return edges[mask]

def clean_directories():
    """Delete all files in specified directories except 'i'."""
    directories = ["data/eigen", "data/interim", "data/mappings", "data/percent"]

    for dir_path in directories:
        if not os.path.exists(dir_path):
            continue  # Skip if directory doesn't exist
        
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            
            # Keep the file named 'i' (no extension)
            if os.path.isfile(file_path) and file == "i":
                continue

            # Delete file or directory
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove subdirectories
            else:
                os.remove(file_path)  # Remove files

    print("Cleaned directories while keeping 'i' files intact.")

@njit
def edge_to_key(edge):
    """Convert edge to unique key ensuring 0-based indexing"""
    max_id = np.iinfo(np.int32).max // 2  # Safe upper bound
    return min(edge[0], max_id) * max_id + min(edge[1], max_id)

@njit
def fast_unique_edges(edges):
    """Optimized unique edge finding with verification"""
    n = len(edges)
    if n == 0:
        return edges
        
    # Generate unique keys
    keys = np.zeros(n, dtype=np.int64)
    for i in range(n):
        keys[i] = edge_to_key(edges[i])
    
    # Sort based on keys
    sort_idx = np.argsort(keys)
    sorted_edges = edges[sort_idx]
    
    # Find unique edges with mask
    unique_mask = np.ones(n, dtype=np.bool_)
    for i in range(1, n):
        if (sorted_edges[i, 0] == sorted_edges[i-1, 0] and 
            sorted_edges[i, 1] == sorted_edges[i-1, 1]):
            unique_mask[i] = False
    
    return sorted_edges[unique_mask]

def read_dataset_efficient(file_path, dataset):
    """Read dataset with robust error handling"""
    delimiter = ' ' if dataset in ['digg', 'uci'] else ','
    header = None if dataset in ['digg', 'uci'] else 0
    usecols = [0, 1] if dataset in ['digg', 'uci'] else ['fromNode', 'toNode']
    
    chunks = []
    chunk_size = 500000
    
    try:
        for chunk in pd.read_csv(
            file_path,
            usecols=usecols,
            delimiter=delimiter,
            header=header,
            dtype=np.int32,
            engine='c',
            chunksize=chunk_size
        ):
            # Verify no negative values
            if chunk.values.min() < 0:
                raise ValueError(f"Negative indices found in chunk from {file_path}")
            chunks.append(chunk.values)
        
        if not chunks:
            raise ValueError(f"No data read from {file_path}")
            
        data = np.vstack(chunks)
        
        # Final verification
        if data.dtype != np.int32:
            data = data.astype(np.int32)
        if np.any(data < 0):
            raise ValueError(f"Negative indices found after processing {file_path}")
            
        return data
        
    except Exception as e:
        raise RuntimeError(f"Error reading {file_path}: {str(e)}")

@njit(parallel=True)
def process_edge_chunk(chunk, node_array):
    """Process edge chunk with Numba acceleration"""
    result = np.empty_like(chunk)
    for i in prange(len(chunk)):
        result[i, 0] = node_array[chunk[i, 0]]
        result[i, 1] = node_array[chunk[i, 1]]
    return result

def preprocessDataset(dataset):
    print(f'Preprocess dataset: {dataset}')
    t0 = time.time()
    
    # Create output directories
    for dir_path in ['data/mappings', 'data/interim', 'data/percent']:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Determine input file path
    file_mapping = {
        'btc_alpha': 'data/raw/soc-sign-bitcoinalpha.csv',
        'btc_otc': 'data/raw/soc-sign-bitcoinotc.csv',
        'year_1992': 'data/raw/1992_remapped.csv',
        'year_1993': 'data/raw/1993_remapped.csv',
        'five_year': 'data/raw/five_year.csv'
    }
    file_name = file_mapping.get(dataset, f'data/raw/{dataset}')
    
    # Read and process edges efficiently
    print("Reading dataset...")
    edges = read_dataset_efficient(file_name, dataset)
    print(f"Read {len(edges)} edges")
    
    print("Optimizing edge order...")
    edges = optimize_edge_order(edges)
    print(f"After ordering: {len(edges)} edges")
    
    print("Finding unique edges...")
    edges = fast_unique_edges(edges)
    print(f"After deduplication: {len(edges)} edges")
    
    # Create node mapping with verification
    print("Creating node mapping...")
    unique_vertices = np.unique(edges.ravel())
    if np.any(unique_vertices < 0):
        raise ValueError("Negative vertex indices found after processing")
    
    node_mapping = np.arange(len(unique_vertices), dtype=np.int32)
    node_mapping_full = np.zeros(unique_vertices.max() + 1, dtype=np.int32)
    node_mapping_full[unique_vertices] = node_mapping
    
    # Process edges in parallel chunks
    print("Processing edges in chunks...")
    modified_edges = np.empty_like(edges)
    chunk_size = 500000
    num_chunks = (len(edges) + chunk_size - 1) // chunk_size
    
    for i in tqdm(range(num_chunks), desc="Processing chunks"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(edges))
        chunk = edges[start_idx:end_idx]
        modified_edges[start_idx:end_idx] = process_edge_chunk(chunk, node_mapping_full)
    
    # Create edge mapping
    print("Creating edge mapping...")
    edge_mapping = {tuple(edge): (int(edge[0]), int(edge[1])) 
                   for edge in modified_edges}
    
    # Save mappings
    print("Saving mappings...")
    with open(f'data/mappings/{dataset}_edge_mapping.pkl', 'wb') as f:
        pickle.dump({'edge_mapping': edge_mapping}, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f'vertex: {len(unique_vertices)}, edge: {len(modified_edges)}')
    
    # Save processed edges
    with open(f'data/interim/{dataset}.pkl', 'wb') as f:
        pickle.dump(modified_edges, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f'Preprocess finished! Time: {time.time() - t0:.2f}s')
    return modified_edges, edge_mapping, unique_vertices.max() + 1

@njit(parallel=True)
def process_data_chunk(data, start_loc, end_loc):
    """Process data chunk with Numba acceleration"""
    chunk = data[start_loc:end_loc]
    rows = chunk[:, 0].astype(np.int32)
    cols = chunk[:, 1].astype(np.int32)
    
    if chunk.shape[1] > 2:
        labs = chunk[:, 2].astype(np.int32)
    else:
        labs = np.zeros(end_loc - start_loc, dtype=np.int32)
    
    weis = np.ones(end_loc - start_loc, dtype=np.int32)
    return rows, cols, labs, weis

def generateDataset(dataset, snap_size, train_per=0.5, anomaly_per=0.01):
    print(f'Generating data with anomaly for Dataset: {dataset}')
    
    # Preprocess and get needed values directly
    edges, edge_data, n = preprocessDataset(dataset)
    m = len(edges)
    t0 = time.time()
    synthetic_test, train_mat, train = anomaly_generation(train_per, anomaly_per, edges, n, m, seed=1)
    print(f"Anomaly Generation finish! Time: {time.time()-t0:.2f}s")
    
    t0 = time.time()
    
    # Efficient sparse matrix operations
    train_mat_coo = sparse.coo_matrix(train_mat)
    matrix_size = train_mat_coo.shape[0]
    train_mat = (sparse.csr_matrix(train_mat_coo) + 
                sparse.csr_matrix(train_mat_coo.T) + 
                sparse.eye(matrix_size)).tolil()
    headtail = train_mat.rows
    del train_mat
    
    train_size = int(len(train) / snap_size + 0.5)
    test_size = int(len(synthetic_test) / snap_size + 0.5)
    print(f"Train size: {len(train)} {train_size} Test size: {len(synthetic_test)} {test_size}")
    
    # Process chunks in parallel using Numba
    rows, cols, labs, weis = [], [], [], []
    
    # Process training data
    for ii in range(train_size):
        start_loc = ii * snap_size
        end_loc = min((ii + 1) * snap_size, len(train))
        r, c, l, w = process_data_chunk(train, start_loc, end_loc)
        rows.append(r)
        cols.append(c)
        labs.append(l)
        weis.append(w)
    
    # Process test data
    for i in range(test_size):
        start_loc = i * snap_size
        end_loc = min((i + 1) * snap_size, len(synthetic_test))
        r, c, l, w = process_data_chunk(synthetic_test, start_loc, end_loc)
        rows.append(r)
        cols.append(c)
        labs.append(l)
        weis.append(w)
    
    print(f"Data processing finished! Time: {time.time()-t0:.2f}s")
    
    # Save in original pkl format for compatibility
    output_file = f'data/percent/{dataset}_{train_per}_{anomaly_per}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump((
            rows, cols, labs, weis, headtail, 
            train_size, test_size, n, m, 
            {'edge_mapping': edge_data}
        ), f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    clean_directories()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, 
                       choices=['uci', 'digg', 'btc_alpha', 'btc_otc', 'year_1992', 'year_1993', 'five_year'],
                       default='uci')
    parser.add_argument('--anomaly_per', type=float, choices=[0.01, 0.05, 0.1, 0.2], default=None)
    parser.add_argument('--train_per', type=float, default=0.5)
    args = parser.parse_args()
    
    snap_size_dict = {
        'uci': 1000,
        'digg': 6000,
        'btc_alpha': 1000,
        'btc_otc': 2000,
        'year_1992': 300,
        'year_1993': 300,
        'five_year': 4000
    }
    
    anomaly_pers = [args.anomaly_per] if args.anomaly_per is not None else [0.01, 0.05, 0.10]
    
    for anomaly_per in anomaly_pers:
        generateDataset(args.dataset, snap_size_dict[args.dataset], 
                       train_per=args.train_per, anomaly_per=anomaly_per)