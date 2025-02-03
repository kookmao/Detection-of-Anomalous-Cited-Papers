import numpy as np
import torch
import igraph
from functools import lru_cache
from typing import List, Dict, Tuple, Set, Optional


@lru_cache(maxsize=1024)
def WL_setting_init(node_list: np.ndarray, link_list: np.ndarray) -> Tuple[Dict, Dict]:
    """Initialize WL settings with efficient caching"""
    node_color_dict = {int(node): 1 for node in node_list}
    node_neighbor_dict = {int(node): {} for node in node_list}

    for u1, u2 in link_list:
        u1, u2 = int(u1), int(u2)
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    return node_color_dict, node_neighbor_dict



def compute_zero_WL(node_list: np.ndarray, link_list: np.ndarray) -> Dict:
    """Compute zero-iteration WL with validation"""
    return {int(i): 0 for i in node_list}
#max_hop = 999  # Set maximum hop distance

@lru_cache(maxsize=2048)
def get_hop(graph: igraph.Graph, source: int, target: int, max_hop: int = 99) -> int:
    """Calculate hop distance with caching and validation"""
    try:
        if source >= graph.vcount() or target >= graph.vcount():
            return max_hop
        distance = graph.shortest_paths(source=source, target=[target])[0][0]
        return min(int(distance) if distance != float('inf') else max_hop, max_hop)
    except Exception:
        return max_hop

# batching + hop + int + time
def compute_batch_hop(node_list: np.ndarray,
                     edges_all: List[np.ndarray],
                     num_snap: int,
                     Ss: List[np.ndarray],
                     k: int = 5,
                     window_size: int = 1) -> List[Dict]:
    """Compute batch hop distances efficiently"""
    batch_hop_dicts = [None] * (window_size-1)
    s_ranking = [0] + list(range(k+1))

    # Create igraph objects once
    Gs = []
    for snap in range(num_snap):
        g = igraph.Graph()
        g.add_vertices(len(node_list))
        g.add_edges([(int(e[0]), int(e[1])) for e in edges_all[snap]])
        Gs.append(g)

    for snap in range(window_size - 1, num_snap):
        batch_hop_dict = {}
        edges = edges_all[snap]

        for edge in edges:
            u = int(edge[0].item() if isinstance(edge[0], torch.Tensor) else edge[0])
            v = int(edge[1].item() if isinstance(edge[1], torch.Tensor) else edge[1])
            
            if u >= len(node_list) or v >= len(node_list):
                continue
                
            edge_idx = f"{snap}_{u}_{v}"
            batch_hop_dict[edge_idx] = []

            for lookback in range(window_size):
                s = Ss[snap - lookback][u] + Ss[snap - lookback][v]
                s[u] = -1000
                s[v] = -1000
                
                top_k_neighbor_index = s.argsort()[-k:][::-1]
                indices = np.hstack((
                    np.array([u, v]),
                    top_k_neighbor_index.cpu().numpy() if isinstance(top_k_neighbor_index, torch.Tensor)
                    else top_k_neighbor_index
                ))

                for i, neighbor_index in enumerate(indices):
                    hop1 = get_hop(Gs[snap-lookback], u, neighbor_index)
                    hop2 = get_hop(Gs[snap-lookback], v, neighbor_index)
                    hop = min(hop1, hop2)
                    batch_hop_dict[edge_idx].append((
                        int(neighbor_index),
                        s_ranking[i],
                        hop,
                        lookback
                    ))

        batch_hop_dicts.append(batch_hop_dict)

    return batch_hop_dicts

def compute_batch_hop_gpu(idx, edges, num_snap, S_list, k, window_size):
    """GPU-accelerated version of compute_batch_hop"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_hop_dict_list = [None] * num_snap  
    
    for current_snap in range(num_snap):
        if current_snap < window_size - 1:
            continue
            
        # Convert edges for current window to GPU tensors
        window_edges = []
        for snap_idx in range(max(0, current_snap - window_size + 1), current_snap + 1):
            if isinstance(edges[snap_idx], torch.Tensor):
                window_edges.append(edges[snap_idx].to(device))
            else:
                window_edges.append(torch.tensor(edges[snap_idx], device=device))
        
        # Create adjacency matrices on GPU
        n = len(idx)
        window_adjs = []
        for edge_tensor in window_edges:
            adj = torch.zeros((n, n), device=device)
            adj[edge_tensor[:, 0], edge_tensor[:, 1]] = 1
            adj[edge_tensor[:, 1], edge_tensor[:, 0]] = 1  # Make symmetric
            window_adjs.append(adj)
        
        # Compute intimacy scores using matrix operations
        S = S_list[current_snap].to(device) if S_list[current_snap] is not None else None
        
        # Process current snapshot edges
        current_edges = window_edges[-1]
        batch_hop_dict = {}
        
        for edge_idx in range(len(current_edges)):
            source, target = current_edges[edge_idx]
            edge_key = f'edge_{source.item()}_{target.item()}'
            
            # Initialize tensors for k-hop neighborhood
            neighbor_scores = torch.zeros(n, device=device)
            visited = torch.zeros(n, dtype=torch.bool, device=device)
            current_neighbors = torch.tensor([source.item(), target.item()], device=device)
            visited[current_neighbors] = True
            
            # BFS for k hops using matrix operations
            hop_neighbors = []
            for hop in range(k):
                # Get next hop neighbors using matrix multiplication
                next_neighbors = torch.zeros(n, dtype=torch.bool, device=device)
                for adj in window_adjs:
                    hop_adj = adj[current_neighbors]
                    next_neighbors = next_neighbors | (hop_adj.sum(0) > 0)
                
                # Filter out already visited nodes
                next_neighbors = next_neighbors & (~visited)
                current_neighbors = torch.where(next_neighbors)[0]
                
                if len(current_neighbors) == 0:
                    break
                    
                visited[current_neighbors] = True
                hop_neighbors.extend([(node.item(), hop + 1) for node in current_neighbors])
            
            # Compute intimacy scores using S matrix if available
            if S is not None:
                for node, hop in hop_neighbors:
                    neighbor_scores[node] = S[source.item(), node].item()
            
            # Sort neighbors by intimacy score
            neighbor_data = []
            for node, hop in hop_neighbors:
                score = neighbor_scores[node].item()
                time_delta = 0  # Assuming current timestep
                neighbor_data.append((node, score, hop, time_delta))
            
            neighbor_data.sort(key=lambda x: x[1], reverse=True)
            batch_hop_dict[edge_key] = neighbor_data
            
        batch_hop_dict_list[current_snap] = batch_hop_dict
        
        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return batch_hop_dict_list

# Dict to embeddings
def dicts_to_embeddings(feats: Optional[np.ndarray],
                       batch_hop_dicts: List[Dict],
                       wl_dict: Dict,
                       num_snap: int,
                       use_raw_feat: bool = False,
                       device: Optional[torch.device] = None,
                       window_size: int = 2,
                       k: int = 4) -> Tuple[None, None, List, List, List]:
    """Convert dictionaries to embeddings with validation"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expected_length = window_size * (2 + k)
    int_embeddings = []
    hop_embeddings = []
    time_embeddings = []

    for snap in range(num_snap):
        if batch_hop_dicts[snap] is None:
            int_embeddings.append(None)
            hop_embeddings.append(None)
            time_embeddings.append(None)
            continue

        # Process entries
        int_data = []
        hop_data = []
        time_data = []

        for edge_key in batch_hop_dicts[snap]:
            entries = batch_hop_dicts[snap][edge_key][:expected_length]
            
            # Pad if needed
            if len(entries) < expected_length:
                entries.extend([(0, 0, 99, 0)] * (expected_length - len(entries)))

            int_vals = [e[1] for e in entries]
            hop_vals = [e[2] for e in entries]
            time_vals = [e[3] for e in entries]

            int_data.append(int_vals)
            hop_data.append(hop_vals)
            time_data.append(time_vals)

        # Create tensors with proper device placement
        if int_data:
            int_embeddings.append(torch.LongTensor(int_data).to(device))
            hop_embeddings.append(torch.LongTensor(hop_data).to(device))
            time_embeddings.append(torch.LongTensor(time_data).to(device))
        else:
            int_embeddings.append(None)
            hop_embeddings.append(None)
            time_embeddings.append(None)

    return None, None, hop_embeddings, int_embeddings, time_embeddings
