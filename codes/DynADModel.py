from functools import lru_cache
import os
from matplotlib import pyplot as plt
import scipy as sp
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

def check_gpu_memory(n, batch_size):
    """Check if GPU has enough memory for computation"""
    if not torch.cuda.is_available():
        return False
    
    try:
        torch.cuda.empty_cache()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_used = torch.cuda.memory_allocated()
        gpu_memory_free = gpu_memory - gpu_memory_used
        
        # Estimate memory needed for essential operations
        estimated_memory = n * batch_size * 4 * 3  # Basic tensor operations
        
        # Check if we have enough memory with 20% buffer
        return estimated_memory * 1.2 < gpu_memory_free
    except:
        return False


def compute_batch_hop_gpu(node_list, edges, num_snap, Ss, k=5, window_size=1):
    """Memory-efficient GPU batch hop computation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_hop_dicts = [None] * (window_size-1)

    # Verify inputs
    if not isinstance(node_list, np.ndarray):
        node_list = np.array(node_list)
    max_node_id = node_list.max()

    # Convert S matrices to sparse tensors once
    sparse_Ss = []
    for S in Ss:
        if S is None:
            sparse_Ss.append(None)
            continue

        if not sp.issparse(S):
            S = sp.csr_matrix(S)

        # Convert to COO format for efficient GPU transfer
        S_coo = S.tocoo()
        indices = torch.from_numpy(
            np.vstack((S_coo.row, S_coo.col))
        ).long().to(device)
        values = torch.from_numpy(S_coo.data).float().to(device)
        size = torch.Size(S_coo.shape)
        sparse_Ss.append(
            torch.sparse_coo_tensor(indices, values, size, device=device)
        )

    for snap in range(window_size-1, num_snap):
        batch_hop_dict = {}
        current_edges = edges[snap]

        # Ensure edges are on correct device
        if not isinstance(current_edges, torch.Tensor):
            current_edges = torch.tensor(current_edges, device=device)

        # Process in batches to manage memory
        batch_size = min(1000, len(current_edges))
        for start_idx in range(0, len(current_edges), batch_size):
            end_idx = min(start_idx + batch_size, len(current_edges))
            batch = current_edges[start_idx:end_idx]

            # Process each edge in the batch
            for edge_idx, (u, v) in enumerate(batch.cpu().numpy()):
                edge_key = f"{snap}_{u}_{v}"
                entries = []

                # Process each lookback window
                for lookback in range(window_size):
                    sl = snap - lookback
                    if sparse_Ss[sl] is None:
                        continue

                    S = sparse_Ss[sl]

                    # Get neighbor information
                    u_row = S[u].coalesce()
                    v_row = S[v].coalesce()

                    # Combine neighbors efficiently
                    u_indices = u_row.indices()[1]  # Use dimension 1 for column indices
                    v_indices = v_row.indices()[1]
                    u_values = u_row.values()
                    v_values = v_row.values()

                    # Create unified neighbor list
                    all_neighbors = torch.unique(torch.cat([
                        u_indices, v_indices,
                        torch.tensor([u, v], device=device)
                    ]))

                    # Calculate combined scores
                    neighbor_scores = torch.zeros(len(all_neighbors), device=device)
                    for idx, neighbor in enumerate(all_neighbors):
                        u_mask = u_indices == neighbor
                        v_mask = v_indices == neighbor
                        u_score = u_values[u_mask].sum() if torch.any(u_mask) else 0
                        v_score = v_values[v_mask].sum() if torch.any(v_mask) else 0
                        neighbor_scores[idx] = u_score + v_score

                    # Remove source and target nodes
                    mask = (all_neighbors != u) & (all_neighbors != v)
                    valid_neighbors = all_neighbors[mask]
                    valid_scores = neighbor_scores[mask]

                    # Select top k neighbors
                    if len(valid_scores) > k:
                        _, top_indices = valid_scores.topk(k)
                        selected = valid_neighbors[top_indices]
                    else:
                        selected = valid_neighbors

                    # Add base nodes first
                    entries.extend([
                        (int(u), 0, 0, lookback),
                        (int(v), 1, 0, lookback)
                    ])

                    # Add selected neighbors
                    for idx, node in enumerate(selected.cpu().numpy()):
                        # Calculate hop distance using sparse matrix structure
                        node_row = S[node].coalesce()
                        hop = min(len(node_row.indices()[1]), 99)
                        entries.append((
                            int(node),
                            idx + 2,
                            hop,
                            lookback
                        ))

                batch_hop_dict[edge_key] = entries

            # Memory management
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        batch_hop_dicts.append(batch_hop_dict)

        # Memory cleanup after each snapshot
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    def __init__(self, config, args):
        super(DynADModel, self).__init__(config)
        self.args = args
        self.config = config
        self.transformer = BaseModel(config)
        self.cls_y = torch.nn.Linear(config.hidden_size, 1)
        self.weight_decay = config.weight_decay
        self.init_weights()
        
        self.train_losses = []
        self.train_aucs = []
        self.test_aucs = []
        
        # Plotting setup
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(20, 7))
        self._setup_axes()
        
        # Initialize tracking
        self.anomaly_tracker = AnomalyTracker(persistence_threshold=0.8)
        
        # Device setup with verification
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"Model initialized on {self.device}")
        
        # Memory tracking
        if torch.cuda.is_available():
            self.initial_memory = torch.cuda.memory_allocated()
            print(f"Initial GPU memory: {self.initial_memory/1e9:.2f} GB")
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.cuda()
            print(f"Model initialized on {next(self.parameters()).device}")
        else:
            print("Model initialized on CPU")

    def _setup_axes(self):
            """Configure plot axes"""
            titles = ['Training Loss', 'Training AUC', 'Test AUC']
            ylabels = ['Loss', 'AUC', 'AUC']
            
            for ax, title, ylabel in zip(self.axes, titles, ylabels):
                ax.set_xlabel('Epoch')
                ax.set_ylabel(ylabel)
                ax.set_title(title)
            
            plt.tight_layout()
            plt.show()
    
    def evaluate_test_set(self):
        self.eval()
        all_true = []
        all_pred = []
        
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
            self.generate_embedding([self.data['edges'][snap] for snap in self.data['snap_test']])
            
        with torch.no_grad():
            for idx, snap in enumerate(self.data['snap_test']):
                if int_embeddings[idx] is None:
                    continue
                    
                test_edges = self.data['edges'][snap]
                test_labels = self.data['y'][snap]
                
                int_embed = int_embeddings[idx].to(self.device)
                hop_embed = hop_embeddings[idx].to(self.device)
                time_embed = time_embeddings[idx].to(self.device)
                
                output = self.forward(int_embed, hop_embed, time_embed).squeeze()
                pred_scores = torch.sigmoid(output).cpu().numpy()
                
                all_true.append(test_labels.cpu().numpy())
                all_pred.append(pred_scores)
                
                # Clear cache after each snapshot
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if not all_true:
            print("No test samples processed")
            return 0.5
            
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        
        return metrics.roc_auc_score(all_true, all_pred)

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
        sequence_output = torch.mean(outputs[0], dim=1)
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
        """Memory-efficient negative sampling"""
        neg_edges = []
        num_nodes = len(self.data['idx'])
        
        for snap in self.data['snap_train']:
            adj = self.data['A'][snap].to_dense().cpu()
            disconnected = (adj == 0).nonzero(as_tuple=False)
            
            # Filter valid edges
            valid_mask = (disconnected[:, 0] < num_nodes) & (disconnected[:, 1] < num_nodes)
            valid_edges = disconnected[valid_mask]
            
            # Sample edges
            if len(valid_edges) > 0:
                sample_size = min(len(edges[snap]), len(valid_edges))
                indices = torch.randperm(len(valid_edges))[:sample_size]
                neg_edges.append(valid_edges[indices].numpy())
            else:
                print(f"Warning: No valid negative edges for snapshot {snap}")
                neg_edges.append(np.empty((0, 2), dtype=np.int64))
            
            del adj
            torch.cuda.empty_cache()
        
        return neg_edges

    # def train_model(self, max_epoch):
    #      # Initialize Focal Loss parameters
    #     focal_alpha = 0.15  # Weight for anomaly class (adjust based on class imbalance)
    #     focal_gamma = 2.0   # Focusing parameter for hard examples
        
    #     optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    #     raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = self.generate_embedding(self.data['edges'])
    #     self.data['raw_embeddings'] = None
    #     scaler = GradScaler()

    #     # Early stopping setup
    #     best_test_auc = 0
    #     patience = 10
    #     epochs_no_improve = 0

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     print(f"Using device: {device}")
    #     self.to(device)  # Add this line

    #     for epoch in range(max_epoch):
    #         t_epoch_begin = time.time()
    #         negatives = self.negative_sampling(self.data['edges'][:max(self.data['snap_train']) + 1])
    #         _, _, hop_embeddings_neg, int_embeddings_neg, time_embeddings_neg = self.generate_embedding(negatives)

    #         self.train()
    #         loss_train = 0
    #         all_true = []
    #         all_pred = []

    #         for snap in self.data['snap_train']:
    #             if wl_embeddings[snap] is None:
    #                 continue

    #             # Prepare batch data
    #             int_embedding = torch.vstack((int_embeddings[snap], int_embeddings_neg[snap])).to(device)
    #             hop_embedding = torch.vstack((hop_embeddings[snap], hop_embeddings_neg[snap])).to(device)
    #             time_embedding = torch.vstack((time_embeddings[snap], time_embeddings_neg[snap])).to(device)
    #             y = torch.hstack((self.data['y'][snap].float().to(device), 
    #                             torch.ones(int_embeddings_neg[snap].size()[0]).to(device)))

    #             optimizer.zero_grad()
                
    #             # Use autocast for mixed precision training
    #             with autocast():
    #                 output = self.forward(int_embedding, hop_embedding, time_embedding).squeeze()
                    
    #                 # Focal Loss implementation
    #                 bce_loss = F.binary_cross_entropy_with_logits(output, y, reduction='none')
    #                 pt = torch.exp(-bce_loss)
    #                 focal_loss = (focal_alpha * (1 - pt) ** focal_gamma * bce_loss).mean()
                    
    #                 # Add L2 regularization
    #                 l2_reg = torch.tensor(0., device=device)
    #                 for param in self.parameters():
    #                     l2_reg += torch.norm(param, 2)
    #                 loss = focal_loss + self.config.weight_decay * l2_reg

    #             # Backward pass with gradient scaling
    #             scaler.scale(loss).backward()
                
    #             # Gradient clipping
    #             scaler.unscale_(optimizer)
    #             torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                
    #             # Optimizer step with scaler
    #             scaler.step(optimizer)
    #             scaler.update()

    #             # Track metrics
    #             loss_train += loss.item()
    #             pred_scores = torch.sigmoid(output).detach().cpu().numpy()
    #             true_labels = y.detach().cpu().numpy()
                
    #             all_true.append(true_labels)
    #             all_pred.append(pred_scores)

    #         # Calculate epoch metrics
    #         train_loss = loss_train / len(self.data['snap_train'])
    #         all_true = np.concatenate(all_true)
    #         all_pred = np.concatenate(all_pred)
    #         train_auc = metrics.roc_auc_score(all_true, all_pred) if len(np.unique(all_true)) > 1 else 0.5
            
    #         # Test set evaluation
    #         test_auc = self.evaluate_test_set()

    #         # Generate reports every 10 epochs
    #         if (epoch + 1) % 10 == 0:
    #             output_dir = f'anomaly_analysis_epoch_{epoch+1}'
    #             os.makedirs(output_dir, exist_ok=True)
    #             plt.savefig(f'{output_dir}/training_testing_progress.png')
    #             self.anomaly_tracker.generate_reports(output_dir=output_dir)

    #         # Early stopping check
    #         if test_auc > best_test_auc:
    #             best_test_auc = test_auc
    #             epochs_no_improve = 0
    #         else:
    #             epochs_no_improve += 1
    #             if epochs_no_improve == patience:
    #                 print(f'Early stopping at epoch {epoch+1}')
    #                 break

    #         # Update tracking and plots
    #         self.train_losses.append(train_loss)
    #         self.train_aucs.append(train_auc)
    #         self.test_aucs.append(test_auc)
            
    #         print(f'Epoch: {epoch + 1}, Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, '
    #             f'Test AUC: {test_auc:.4f}, Time: {time.time() - t_epoch_begin:.4f}s')
            
    #         self.update_plots(epoch)

    #     return self.learning_record_dict

    def train_model(self, max_epoch):
        print("\n=== Starting Training ===")
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scaler = GradScaler()
        
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
            self.generate_embedding(self.data['edges'])
            
        self.data['raw_embeddings'] = None
        
        best_test_auc = 0
        patience = 10
        epochs_no_improve = 0
        
        for epoch in range(max_epoch):
            epoch_start = time.time()
            self.train()
            
            # Generate negatives
            negatives = self.negative_sampling(
                self.data['edges'][:max(self.data['snap_train']) + 1]
            )
            
            _, _, hop_embeddings_neg, int_embeddings_neg, time_embeddings_neg = \
                self.generate_embedding(negatives)
            
            # Training loop
            loss_train = 0
            batch_metrics = []
            
            for snap in self.data['snap_train']:
                if int_embeddings[snap] is None:
                    continue
                
                # Prepare batch data
                int_embed = torch.vstack((
                    int_embeddings[snap], 
                    int_embeddings_neg[snap]
                )).to(self.device)
                
                hop_embed = torch.vstack((
                    hop_embeddings[snap], 
                    hop_embeddings_neg[snap]
                )).to(self.device)
                
                time_embed = torch.vstack((
                    time_embeddings[snap], 
                    time_embeddings_neg[snap]
                )).to(self.device)
                
                y = torch.hstack((
                    self.data['y'][snap].float(),
                    torch.ones(int_embeddings_neg[snap].size(0))
                )).to(self.device)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass with mixed precision
                with autocast():
                    output = self.forward(int_embed, hop_embed, time_embed).squeeze()
                    loss = F.binary_cross_entropy_with_logits(output, y)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # Metrics
                loss_train += loss.item()
                pred_scores = torch.sigmoid(output).detach().cpu().numpy()
                true_labels = y.detach().cpu().numpy()
                batch_metrics.append((true_labels, pred_scores))
                
                # Clear cache
                torch.cuda.empty_cache()
            
            # Calculate epoch metrics
            train_loss = loss_train / len(self.data['snap_train'])
            
            all_true = np.concatenate([m[0] for m in batch_metrics])
            all_pred = np.concatenate([m[1] for m in batch_metrics])
            train_auc = metrics.roc_auc_score(all_true, all_pred)
            
            # Test evaluation
            test_auc = self.evaluate_test_set()
            
            # Early stopping check
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                    break
            
            # Update tracking
            self.train_losses.append(train_loss)
            self.train_aucs.append(train_auc)
            self.test_aucs.append(test_auc)
            
            print(f'Epoch: {epoch + 1}, Loss: {train_loss:.4f}, '
                  f'Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}, '
                  f'Time: {time.time() - epoch_start:.2f}s')
            
            self.update_plots(epoch)
            
            # Generate reports periodically
            if (epoch + 1) % 10 == 0:
                self._generate_epoch_reports(epoch)
            
            # Memory stats
            if torch.cuda.is_available():
                max_memory = torch.cuda.max_memory_allocated()
                print(f"Max GPU memory: {max_memory/1e9:.2f} GB")
                torch.cuda.reset_peak_memory_stats()
        
        return self.learning_record_dict

    def _generate_epoch_reports(self, epoch):
            """Generate periodic reports and visualizations"""
            output_dir = f'anomaly_analysis_epoch_{epoch+1}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save plots
            plt.savefig(f'{output_dir}/training_progress.png')
            
            # Generate anomaly reports
            self.anomaly_tracker.generate_reports(output_dir=output_dir)

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
        return self.train_model(self.max_epoch)