from functools import lru_cache
import os
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import igraph as ig

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from codes.BaseModel import BaseModel

import time
import numpy as np

from sklearn import metrics
from codes.utils import dicts_to_embeddings, compute_batch_hop, compute_zero_WL
from codes.visualization_utils import AnomalyVisualizer
from anomaly_tracking_utils import AnomalyTracker
from torch.cuda.amp import GradScaler, autocast
#from timing_utils import TimingContext, print_tensor_device
import torch_geometric
from torch_geometric.utils import k_hop_subgraph

def check_gpu_memory(n: int, batch_size: int) -> bool:
    """Check if GPU has enough memory with proper cleanup"""
    if not torch.cuda.is_available():
        return False
    
    try:
        # Estimate memory for adjacency matrices
        adj_bytes = n * n * 4  # Float32 adjacency matrix
        total_bytes = adj_bytes * batch_size
        
        # Get GPU memory info
        gpu = torch.cuda.current_device()
        gpu_props = torch.cuda.get_device_properties(gpu)
        free_memory = gpu_props.total_memory - torch.cuda.memory_allocated(gpu)
        
        # Check if we have enough memory (with 20% buffer)
        return total_bytes * 1.2 < free_memory
    except Exception as e:
        print(f"GPU memory check error: {e}")
        return False
    finally:
        # Force cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compute_batch_hop_gpu(node_list, edges_all, num_snap, Ss, k=5, window_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_hop_dicts = [None] * (window_size-1)
    
    # Preconvert S matrices
    Ss_gpu = [torch.as_tensor(S.toarray() if hasattr(S, 'toarray') else S, 
                            device=device, dtype=torch.float32) for S in Ss]
    
    # Preconvert edges
    edge_tensors = [torch.tensor(edges, dtype=torch.long, device=device) 
                   for edges in edges_all]

    for snap in range(window_size-1, num_snap):
        batch_hop_dict = {}
        current_edges = edge_tensors[snap]
        sources = current_edges[:, 0]
        targets = current_edges[:, 1]

        for lookback in range(window_size):
            sl = snap - lookback
            S = Ss_gpu[sl]
            
            # Vectorized scoring
            scores = S[sources] + S[targets]
            scores.scatter_(1, sources.unsqueeze(1), -1000)
            scores.scatter_(1, targets.unsqueeze(1), -1000)
            _, top_k = torch.topk(scores, k, dim=1)

            # Process all edges
            for idx in range(len(current_edges)):
                u = sources[idx].item()
                v = targets[idx].item()
                key = f"{snap}_{u}_{v}"
                
                # Get neighbor nodes
                neighbors = [u, v] + top_k[idx].tolist()
                
                # Get precomputed hop distances from S matrix
                hop_u = S[u].cpu().numpy()[neighbors]
                hop_v = S[v].cpu().numpy()[neighbors]
                hops = np.minimum(hop_u, hop_v).astype(int).tolist()
                
                # Build entries with proper indexing
                entries = []
                for i, node in enumerate(neighbors):
                    entries.append((
                        int(node),
                        i,  # Original ranking
                        min(hops[i], 99),  # Clamp to max hop
                        lookback
                    ))
                
                batch_hop_dict[key] = entries

        batch_hop_dicts.append(batch_hop_dict)
    
    return batch_hop_dicts

def compute_batch_hop_with_fallback(idx, edges, num_snap, S_list, k, window_size):
    """Wrapper function to choose between GPU and CPU implementation"""
    n = len(idx)
    has_gpu_memory = check_gpu_memory(n, window_size)
    
    if has_gpu_memory:
        print("Using GPU implementation for batch_hop computation")
        return compute_batch_hop_gpu(idx, edges, num_snap, S_list, k, window_size)
    else:
        print("Insufficient GPU memory, falling back to CPU implementation")
        from codes.utils import compute_batch_hop
        return compute_batch_hop(idx, edges, num_snap, S_list, k, window_size)

class DynADModel(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    spy_tag = True

    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config, args):
        super(DynADModel, self).__init__(config)
        self.args = args
        self.config = config
        self.transformer = BaseModel(config)
        self.cls_y = torch.nn.Linear(config.hidden_size, 1)
        self.weight_decay = config.weight_decay
        self.init_weights()
        
        # Initialize tracking lists
        self.train_losses = []
        self.train_aucs = []
        self.test_aucs = []
        
        # Set up plotting
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(20, 7))
        
        # Configure axes
        self.axes[0].set_xlabel('Epoch')
        self.axes[0].set_ylabel('Loss')
        self.axes[0].set_title('Training Loss')
        
        self.axes[1].set_xlabel('Epoch')
        self.axes[1].set_ylabel('AUC')
        self.axes[1].set_title('Training AUC')
        
        self.axes[2].set_xlabel('Epoch')
        self.axes[2].set_ylabel('AUC')
        self.axes[2].set_title('Test AUC')
        
        plt.tight_layout()
        plt.show()
        
        self.anomaly_tracker = AnomalyTracker(persistence_threshold=0.8)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.cuda()
            print(f"Model initialized on {next(self.parameters()).device}")
        else:
            print("Model initialized on CPU")

    # def evaluate_test_set(self):
    #     """Evaluate model on test set"""
    #     self.eval()
    #     all_true = []
    #     all_pred = []
        
    #     raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
    #         self.generate_embedding([self.data['edges'][snap] for snap in self.data['snap_test']])

    #     for idx, snap in enumerate(self.data['snap_test']):
    #         if int_embeddings[idx] is None:
    #             continue
                
    #         test_edges = self.data['edges'][snap]
    #         test_labels = self.data['y'][snap]
            
    #         with torch.no_grad():
    #             output = self.forward(int_embeddings[idx], 
    #                                 hop_embeddings[idx],
    #                                 time_embeddings[idx]).squeeze()
    #             pred_scores = torch.sigmoid(output).cpu().numpy()
            
    #         all_true.append(test_labels.cpu().numpy())
    #         all_pred.append(pred_scores)
        
    #     if len(all_true) == 0:
    #         return 0.5
            
    #     all_true = np.concatenate(all_true)
    #     all_pred = np.concatenate(all_pred)
    #     test_auc = metrics.roc_auc_score(all_true, all_pred)
        
    #     return test_auc

    def evaluate_test_set(self) -> float:
        """Evaluate model on test set with proper memory management"""
        self.eval()
        test_aucs = []
        
        try:
            raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
                self.generate_embedding([self.data['edges'][snap] for snap in self.data['snap_test']])
            
            device = next(self.parameters()).device
            
            for idx, snap in enumerate(self.data['snap_test']):
                if int_embeddings[idx] is None:
                    continue
                
                try:
                    with torch.no_grad():
                        int_embed = int_embeddings[idx].to(device)
                        hop_embed = hop_embeddings[idx].to(device)
                        time_embed = time_embeddings[idx].to(device)
                        
                        output = self.forward(int_embed, hop_embed, time_embed).squeeze()
                        pred_scores = torch.sigmoid(output).cpu().numpy()
                        true_labels = self.data['y'][snap].cpu().numpy()
                        
                        if len(np.unique(true_labels)) > 1:
                            auc = metrics.roc_auc_score(true_labels, pred_scores)
                            test_aucs.append(auc)
                        
                    # Explicit cleanup
                    del int_embed, hop_embed, time_embed, output, pred_scores
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM in test snapshot {snap}. Skipping...")
                        continue
                    raise e
                    
            return np.mean(test_aucs) if test_aucs else 0.5
            
        finally:
            # Ensure cleanup
            del raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def update_plots(self, epoch):
        """Update all training and testing plots"""
        epochs = list(range(len(self.train_losses)))
        
        # Clear axes
        for ax in self.axes.flatten():
            ax.clear()
        
        # Plot training loss
        self.axes[0].plot(epochs, self.train_losses, 'b-')
        self.axes[0].set_xlabel('Epoch')
        self.axes[0].set_ylabel('Loss')
        self.axes[0].set_title('Training Loss')
        
        # Plot training AUC
        self.axes[1].plot(epochs, self.train_aucs, 'r-')
        self.axes[1].set_xlabel('Epoch')
        self.axes[1].set_ylabel('AUC')
        self.axes[1].set_title('Training AUC')
        
        # Plot testing AUC
        self.axes[2].plot(epochs, self.test_aucs, 'g-')
        self.axes[2].set_xlabel('Epoch')
        self.axes[2].set_ylabel('AUC')
        self.axes[2].set_title('Test AUC')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Save periodic results
        if (epoch + 1) % 10 == 0:
            output_dir = f'anomaly_analysis_epoch_{epoch+1}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save plots
            plt.savefig(f'{output_dir}/training_testing_progress.png')
            
            # Generate anomaly reports
            self.anomaly_tracker.generate_reports(output_dir=output_dir)

    def forward(self, init_pos_ids, hop_dis_ids, time_dis_ids, idx=None):

        outputs = self.transformer(init_pos_ids, hop_dis_ids, time_dis_ids)

        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output = outputs[0].mean(dim=1)  # Average across all sequence positions

        sequence_output /= float(self.config.k+1)

        output = self.cls_y(sequence_output)

        return output.squeeze()

    def batch_cut(self, idx_list):
        batch_list = []
        for i in range(0, len(idx_list), self.config.batch_size):
            batch_list.append(idx_list[i:i + self.config.batch_size])
        return batch_list

    def evaluate(self, trues, preds, edges):
        """Evaluate model performance and track anomalies"""
        aucs = {}
        predicted_labels = []
        
        for snap in range(len(self.data['snap_test'])):
            auc = metrics.roc_auc_score(trues[snap], preds[snap])
            aucs[snap] = auc
            predicted_labels.append((preds[snap] >= 0.5).astype(int))
            
            self.anomaly_tracker.track_snapshot(
                edges=edges[snap],
                scores=preds[snap],
                true_labels=trues[snap],
                timestamp=snap
            )

        trues_full = np.hstack(trues)
        preds_full = np.hstack(preds)
        auc_full = metrics.roc_auc_score(trues_full, preds_full)
        
        return aucs, auc_full, predicted_labels
            
    def generate_embedding(self, edges):
        edges_tuple = tuple(tuple(e.tolist() if isinstance(e, torch.Tensor) else e) for e in edges)

        num_snap = len(edges)
        
        # Edge processing
        edges_cpu = []
        for e in edges_tuple[:7]:
            if isinstance(e, torch.Tensor):
                edges_cpu.append(e.cpu().numpy())
            else:
                edges_cpu.append(e)
        
        
        # WL Dict computation
        
        WL_dict = compute_zero_WL(self.data['idx'], np.vstack(edges_cpu))
       
        
        
        # Use GPU-optimized batch hop computation
        
        batch_hop_dicts = compute_batch_hop_gpu(
            self.data['idx'], 
            edges, 
            num_snap, 
            self.data['S'], 
            self.config.k, 
            self.config.window_size
        )
        
        # Final embeddings
        
        device = next(self.parameters()).device

        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
            dicts_to_embeddings(
                self.data['X'], 
                batch_hop_dicts, 
                WL_dict, 
                num_snap,
                window_size=self.config.window_size,
                k=self.config.k,
                device=device
            )

        
        
        
        
        return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings

    # def negative_sampling(self, edges):
    #     """Improved negative sampling using temporal neighborhood constraints"""
    #     negative_edges = []
    #     node_list = self.data['idx']
    #     num_node = node_list.shape[0]
        
    #     # Get recent nodes from last window_size snapshots
    #     recent_nodes = np.unique(np.vstack(edges[-self.config.window_size:])) if len(edges) > 0 else node_list
        
    #     for snap_edge in edges:
    #         num_edge = snap_edge.shape[0]
    #         negative_edge = snap_edge.copy()

    #         # Strategy 1: Sample from recent nodes (temporal proximity)
    #         fake_idx = np.random.choice(recent_nodes, num_edge)
            
    #         # Strategy 2: Preserve structural context (50% chance to replace either node)
    #         fake_position = np.random.choice([0,1], num_edge, p=[0.5,0.5])
            
    #         # Ensure no self-loops
    #         for i in range(num_edge):
    #             original_pair = snap_edge[i]
    #             new_node = fake_idx[i]
    #             if fake_position[i] == 0:
    #                 while new_node == original_pair[1]:
    #                     new_node = np.random.choice(recent_nodes)
    #             else:
    #                 while new_node == original_pair[0]:
    #                     new_node = np.random.choice(recent_nodes)
                
    #             negative_edge[i, fake_position[i]] = new_node

    #         negative_edges.append(negative_edge)
        
    #     return negative_edges

    def negative_sampling(self, edges):
        neg_edges = []
        num_nodes = len(self.data['idx'])  # Total nodes in the graph
        
        for snap in self.data['snap_train']:
            adj_matrix = self.data['A'][snap].to_dense().cpu()
            disconnected = (adj_matrix == 0).nonzero(as_tuple=False)
            
            # Filter invalid edges (nodes ≥ num_nodes)
            valid = (disconnected[:, 0] < num_nodes) & (disconnected[:, 1] < num_nodes)
            disconnected_valid = disconnected[valid]
            
            # Ensure we don't sample more than available
            sample_size = min(len(edges[snap]), len(disconnected_valid))
            sampled_idx = torch.randint(0, len(disconnected_valid), (sample_size,))
            neg_edges.append(disconnected_valid[sampled_idx].numpy())
        
        return neg_edges


    def train_model(self, max_epoch):
        """Train model with optimized memory usage and gradient scaling"""
        device = next(self.parameters()).device
        print(f"Training on device: {device}")
        
        # Initialize optimizer and gradient scaler
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scaler = GradScaler()
        
        # Generate initial embeddings
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
            self.generate_embedding(self.data['edges'])
        
        self.data['raw_embeddings'] = None
        
        # Early stopping setup
        best_test_auc = 0
        patience = 10
        epochs_no_improve = 0
        
        for epoch in range(max_epoch):
            epoch_start = time.time()
            
            # Generate negative samples
            negatives = self.negative_sampling(self.data['edges'][:max(self.data['snap_train']) + 1])
            _, _, hop_embeddings_neg, int_embeddings_neg, time_embeddings_neg = \
                self.generate_embedding(negatives)
            
            self.train()
            loss_train = 0
            all_true = []
            all_pred = []
            
            for snap in self.data['snap_train']:
                if int_embeddings[snap] is None:
                    continue
                    
                try:
                    # Prepare batch data
                    int_embedding = torch.vstack((
                        int_embeddings[snap], 
                        int_embeddings_neg[snap]
                    )).to(device)
                    
                    hop_embedding = torch.vstack((
                        hop_embeddings[snap], 
                        hop_embeddings_neg[snap]
                    )).to(device)
                    
                    time_embedding = torch.vstack((
                        time_embeddings[snap], 
                        time_embeddings_neg[snap]
                    )).to(device)
                    
                    y = torch.hstack((
                        self.data['y'][snap].float(), 
                        torch.ones(int_embeddings_neg[snap].size()[0])
                    )).to(device)
                    
                    # Forward pass with automatic mixed precision
                    optimizer.zero_grad(set_to_none=True)
                    with autocast():
                        output = self.forward(
                            int_embedding, 
                            hop_embedding, 
                            time_embedding
                        ).squeeze()
                        
                        loss = F.binary_cross_entropy_with_logits(output, y)
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Metrics tracking
                    loss_train += loss.item()
                    pred_scores = torch.sigmoid(output).detach().cpu().numpy()
                    true_labels = y.detach().cpu().numpy()
                    all_true.append(true_labels)
                    all_pred.append(pred_scores)
                    
                    # Clean up GPU memory
                    del int_embedding, hop_embedding, time_embedding, y, output
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM in snapshot {snap}, skipping...")
                        torch.cuda.empty_cache()
                        continue
                    raise e
            
            # Calculate epoch metrics
            train_loss = loss_train / len(self.data['snap_train'])
            all_true = np.concatenate(all_true)
            all_pred = np.concatenate(all_pred)
            train_auc = metrics.roc_auc_score(all_true, all_pred)
            
            # Evaluate on test set
            test_auc = self.evaluate_test_set()
            
            # Update tracking metrics
            self.train_losses.append(train_loss)
            self.train_aucs.append(train_auc)
            self.test_aucs.append(test_auc)
            
            # Early stopping check
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                    break
            
            # Print epoch summary
            print(f'Epoch: {epoch + 1}, '
                f'Loss: {train_loss:.4f}, '
                f'Train AUC: {train_auc:.4f}, '
                f'Test AUC: {test_auc:.4f}, '
                f'Time: {time.time() - epoch_start:.4f}s')
            
            # Update plots
            self.update_plots(epoch)
            
            # Generate periodic reports
            if (epoch + 1) % 10 == 0:
                output_dir = f'anomaly_analysis_epoch_{epoch+1}'
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(f'{output_dir}/training_testing_progress.png')
                self.anomaly_tracker.generate_reports(output_dir=output_dir)
        
        return self.learning_record_dict

    def _train_epoch(self, raw_embeddings, wl_embeddings, hop_embeddings, 
                    int_embeddings, time_embeddings, optimizer):
        """Helper method to handle single training epoch"""
        negatives = self.negative_sampling(self.data['edges'][:max(self.data['snap_train']) + 1])
        _, _, hop_embeddings_neg, int_embeddings_neg, time_embeddings_neg = self.generate_embedding(negatives)

        loss_train = 0
        all_true = []
        all_pred = []

        for snap in self.data['snap_train']:
            if wl_embeddings[snap] is None:
                continue

            int_embedding = torch.vstack((int_embeddings[snap], int_embeddings_neg[snap]))
            hop_embedding = torch.vstack((hop_embeddings[snap], hop_embeddings_neg[snap]))
            time_embedding = torch.vstack((time_embeddings[snap], time_embeddings_neg[snap]))
            y = torch.hstack((self.data['y'][snap].float(), 
                            torch.ones(int_embeddings_neg[snap].size()[0])))

            optimizer.zero_grad()
            output = self.forward(int_embedding, hop_embedding, time_embedding).squeeze()
            
            # Add L2 regularization
            l2_lambda = 0.01
            l2_reg = torch.tensor(0., requires_grad=True)
            for param in self.parameters():
                l2_reg = l2_reg + torch.norm(param, 2)
            
            loss = F.binary_cross_entropy_with_logits(output, y) + l2_lambda * l2_reg
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            pred_scores = torch.sigmoid(output).detach().cpu().numpy()
            true_labels = y.detach().cpu().numpy()
            
            all_true.append(true_labels)
            all_pred.append(pred_scores)

        train_loss = loss_train / len(self.data['snap_train'])
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        train_auc = metrics.roc_auc_score(all_true, all_pred)
        
        return train_loss, train_auc

    def run(self):
        self.train_model(self.max_epoch)
        return self.learning_record_dict