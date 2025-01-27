from codes.AnomalyGeneration import *
from scipy import sparse
import pickle
import time
import os
import argparse
import pandas as pd

def preprocessDataset(dataset):
    print('Preprocess dataset: ' + dataset)
    t0 = time.time()
    if dataset in ['digg', 'uci']:
        edges = np.loadtxt(
            'data/raw/' +
            dataset,
            dtype=float,
            comments='%',
            delimiter=' ')
        edges = edges[:, 0:2].astype(dtype=int)
    elif dataset in ['btc_alpha', 'btc_otc','year_1992','year_1993','five_year']:
        if dataset == 'btc_alpha':
            file_name = 'data/raw/' + 'soc-sign-bitcoinalpha.csv'
        elif dataset =='btc_otc':
            file_name = 'data/raw/' + 'soc-sign-bitcoinotc.csv'
        elif dataset =='year_1992':
            file_name = 'data/raw/' + '1992_remapped.csv'
        elif dataset =='year_1993':
            file_name = 'data/raw/' + '1993_remapped.csv'
        elif dataset =='five_year':
            file_name = 'data/raw/' + 'five_year.csv'
            
        df = pd.read_csv(file_name)
        edges = df[['fromNode', 'toNode']].values

    # Reorder nodes in edges
    for ii in range(len(edges)):
        x0 = edges[ii][0]
        x1 = edges[ii][1]
        if x0 > x1:
            edges[ii][0] = x1
            edges[ii][1] = x0

    edges = edges[np.nonzero([x[0] != x[1] for x in edges])].tolist()
    aa, idx = np.unique(edges, return_index=True, axis=0)
    edges = np.array(edges)
    edges = edges[np.sort(idx)]

    # get unique vertices and create mapping
    unique_vertices = np.unique(edges)
    node_mapping = {original_id: modified_id for modified_id, original_id in enumerate(unique_vertices)}
    mapping_data = pd.DataFrame({
        'original_id': list(node_mapping.keys()),
        'modified_id': list(node_mapping.values())
    })
    
    os.makedirs('data/mappings', exist_ok=True)
    mapping_data.to_csv(f'data/mappings/{dataset}_node_mapping.csv', index=False)
    modified_edges = np.array([[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in edges])

    # Create edge mapping
    edge_mapping = {}
    for i, edge in enumerate(modified_edges):
        edge_mapping[tuple(edge)] = (edge[0], edge[1])
    
    # Save edge mapping
    mapping_file = f'data/mappings/{dataset}_edge_mapping.pkl'
    with open(mapping_file, 'wb') as f:
        pickle.dump({'edge_mapping': edge_mapping}, f)
    
    print('Edge mapping saved to:', mapping_file)

    print('vertex:', len(unique_vertices), ' edge: ', len(modified_edges))
    np.savetxt(
        'data/interim/' +
        dataset,
        X=modified_edges,
        delimiter=' ',
        comments='%',
        fmt='%d')
    print('Preprocess finished! Time: %.2f s' % (time.time() - t0))


def generateDataset(dataset, snap_size, train_per=0.5, anomaly_per=0.01):
    print('Generating data with anomaly for Dataset: ', dataset)
    
    preprocessDataset(dataset)
        
    # Load edge mapping after preprocessing
    mapping_file = 'data/mappings/' + dataset + '_edge_mapping.pkl'
    with open(mapping_file, 'rb') as f:
        edge_data = pickle.load(f)
        
    edges = np.loadtxt(
        'data/interim/' +
        dataset,
        dtype=float,
        comments='%',
        delimiter=' ')
    edges = edges[:, 0:2].astype(dtype=int)
    vertices = np.unique(edges)
    m = len(edges)
    n = len(vertices)

    t0 = time.time()
    synthetic_test, train_mat, train = anomaly_generation(train_per, anomaly_per, edges, n, m, seed=1)

    print("Anomaly Generation finish! Time: %.2f s"%(time.time()-t0))
    t0 = time.time()

    train_mat = (train_mat + train_mat.transpose() + sparse.eye(n)).tolil()
    headtail = train_mat.rows
    del train_mat

    train_size = int(len(train) / snap_size + 0.5)
    test_size = int(len(synthetic_test) / snap_size + 0.5)
    print("Train size:%d  %d  Test size:%d %d" %
          (len(train), train_size, len(synthetic_test), test_size))
    rows = []
    cols = []
    weis = []
    labs = []
    for ii in range(train_size):
        start_loc = ii * snap_size
        end_loc = (ii + 1) * snap_size

        row = np.array(train[start_loc:end_loc, 0], dtype=np.int32)
        col = np.array(train[start_loc:end_loc, 1], dtype=np.int32)
        lab = np.zeros_like(row, dtype=np.int32)
        wei = np.ones_like(row, dtype=np.int32)

        rows.append(row)
        cols.append(col)
        weis.append(wei)
        labs.append(lab)

    print("Training dataset construction finish! Time: %.2f s" % (time.time()-t0))
    t0 = time.time()

    for i in range(test_size):
        start_loc = i * snap_size
        end_loc = (i + 1) * snap_size

        row = np.array(synthetic_test[start_loc:end_loc, 0], dtype=np.int32)
        col = np.array(synthetic_test[start_loc:end_loc, 1], dtype=np.int32)
        lab = np.array(synthetic_test[start_loc:end_loc, 2], dtype=np.int32)
        wei = np.ones_like(row, dtype=np.int32)

        rows.append(row)
        cols.append(col)
        weis.append(wei)
        labs.append(lab)

    print("Test dataset finish constructing! Time: %.2f s" % (time.time()-t0))

    # Make sure the output directory exists
    os.makedirs('data/percent', exist_ok=True)

    with open('data/percent/' + dataset + '_' + str(train_per) + '_' + str(anomaly_per) + '.pkl', 'wb') as f:
        pickle.dump((rows, cols, labs, weis, headtail, train_size, test_size, n, m, edge_data), f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['uci', 'digg', 'btc_alpha', 'btc_otc','year_1992','year_1993','five_year'], default='uci')
    parser.add_argument('--anomaly_per' ,choices=[0.01, 0.05, 0.1] , type=float, default=None)
    parser.add_argument('--train_per', type=float, default=0.5)
    args = parser.parse_args()

    snap_size_dict = {'uci':1000, 'digg':6000, 'btc_alpha':1000, 'btc_otc':2000,'year_1992':300,'year_1993':300,'five_year':2000}

    if args.anomaly_per is None:
        anomaly_pers = [0.01, 0.05, 0.10]
    else:
        anomaly_pers = [args.anomaly_per]

    for anomaly_per in anomaly_pers:
        generateDataset(args.dataset, snap_size_dict[args.dataset], train_per=args.train_per, anomaly_per=anomaly_per)