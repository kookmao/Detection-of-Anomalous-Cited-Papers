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
def optimize_edge_order(edges_input):
    """Parallel edge ordering with Numba acceleration"""
    # Convert input to numpy array if needed
    edges = np.asarray(edges_input, dtype=np.int32)
    
    # Initialize output array
    n_edges = len(edges)
    ordered_edges = np.empty_like(edges)
    
    # Process edges in parallel
    for i in prange(n_edges):
        u, v = edges[i, 0], edges[i, 1]
        if u > v:
            ordered_edges[i, 0] = v
            ordered_edges[i, 1] = u
        else:
            ordered_edges[i, 0] = u
            ordered_edges[i, 1] = v
            
    # Remove self-loops
    mask = ordered_edges[:, 0] != ordered_edges[:, 1]
    return ordered_edges[mask]

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
    """Convert edge to unique key"""
    return edge[0] * 1000000 + edge[1]

@njit(parallel=True)
def fast_unique_edges(edges):
    """Optimized unique edge finder"""
    n = len(edges)
    keys = np.zeros(n, dtype=np.int64)
    
    # Calculate keys in parallel
    for i in prange(n):
        keys[i] = edge_to_key(edges[i])
    
    # Sort based on keys
    sort_idx = np.argsort(keys)
    sorted_edges = edges[sort_idx]
    
    # Find unique edges
    unique_mask = np.ones(n, dtype=np.bool_)
    for i in range(1, n):
        if (sorted_edges[i, 0] == sorted_edges[i-1, 0] and 
            sorted_edges[i, 1] == sorted_edges[i-1, 1]):
            unique_mask[i] = False
            
    return sorted_edges[unique_mask]

def read_dataset_efficient(file_path, dataset):
    total_rows = sum(1 for _ in open(file_path))
    estimated_chunks = (total_rows // 500000) + 1
    chunks = np.empty((estimated_chunks * 500000, 2), dtype=np.int32)
    current_idx = 0
    
    delimiter = ' ' if dataset in ['digg', 'uci'] else ','
    header = None if dataset in ['digg', 'uci'] else 0
    usecols = [0, 1] if dataset in ['digg', 'uci'] else ['fromNode', 'toNode']
    
    for chunk in pd.read_csv(
        file_path,
        usecols=usecols,
        delimiter=delimiter,
        header=header,
        dtype=np.int32,
        engine='c',
        chunksize=500000
    ):
        chunk_size = len(chunk)
        chunks[current_idx:current_idx + chunk_size] = chunk.values
        current_idx += chunk_size
    
    return chunks[:current_idx]

@njit(parallel=True)
def process_edge_chunk(chunk, node_array):
    try:
        result = np.empty_like(chunk)
        for i in prange(len(chunk)):
            idx0, idx1 = chunk[i, 0], chunk[i, 1]
            if 0 <= idx0 < len(node_array) and 0 <= idx1 < len(node_array):
                result[i, 0] = node_array[idx0]
                result[i, 1] = node_array[idx1]
            else:
                result[i] = [-1, -1]  # Mark invalid indices
        return result[result[:, 0] != -1]  # Remove invalid entries
    except Exception as e:
        print(f"Error in process_edge_chunk: {str(e)}")
        return np.array([])

def preprocessDataset(dataset):
    """Preprocess dataset with efficient edge processing"""
    print('Preprocess dataset: ' + dataset)
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
    
    print("Reading dataset...")
    edges = read_dataset_efficient(file_name, dataset)
    print(f"Read {len(edges)} edges")
    
    print("Optimizing edge order...")
    edges = optimize_edge_order(edges)
    print(f"After ordering: {len(edges)} edges")
    
    print("Finding unique edges...")
    edges = fast_unique_edges(edges)
    print(f"After deduplication: {len(edges)} edges")
    
    # Create node mapping
    print("Creating node mapping...")
    unique_vertices = np.unique(edges.ravel())
    node_mapping = np.arange(len(unique_vertices), dtype=np.int32)
    node_mapping_full = np.zeros(unique_vertices.max() + 1, dtype=np.int32)
    node_mapping_full[unique_vertices] = node_mapping
    
    # Process edges with updated mapping
    print("Processing edges...")
    edge_mapping = {tuple(edge): (edge[0], edge[1]) for edge in edges}
    
    # Save mappings
    print("Saving mappings...")
    with open(f'data/mappings/{dataset}_edge_mapping.pkl', 'wb') as f:
        pickle.dump({'edge_mapping': edge_mapping}, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f'vertex: {len(unique_vertices)}, edge: {len(edges)}')
    print(f'Preprocess finished! Time: {time.time() - t0:.2f}s')
    
    return edges, edge_mapping, len(unique_vertices)

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
    train_mat = (sparse.csr_matrix(train_mat_coo) + sparse.csr_matrix(train_mat_coo.T) + sparse.eye(n)).tolil()
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