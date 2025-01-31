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
from timing_utils import TimingContext, print_tensor_device
import torch_geometric
from torch_geometric.utils import k_hop_subgraph

def check_gpu_memory(n, batch_size):
    """Check if GPU has enough memory for the computation"""
    if not torch.cuda.is_available():
        return False
    
    try:
        # Estimate memory needed for main adjacency matrices
        adj_memory = n * n * 4  # Float32 adjacency matrix
        total_memory = adj_memory * batch_size
        
        # Get available GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_used = torch.cuda.memory_allocated()
        gpu_memory_free = gpu_memory - gpu_memory_used
        
        # Check if we have enough memory (with 20% buffer)
        return total_memory * 1.2 < gpu_memory_free
    except:
        return False


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

    def evaluate_test_set(self):
        """Evaluate model on test set with timing diagnostics"""
        eval_start = time.time()
        self.eval()
        all_true = []
        all_pred = []
        
        with TimingContext("Test set embedding generation"):
            raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
                self.generate_embedding([self.data['edges'][snap] for snap in self.data['snap_test']])
            print_tensor_device(int_embeddings, "Test int_embeddings")

        device = next(self.parameters()).device
        
        for idx, snap in enumerate(self.data['snap_test']):
            if int_embeddings[idx] is None:
                continue
                
            with TimingContext(f"Processing test snapshot {snap}"):
                test_edges = self.data['edges'][snap]
                test_labels = self.data['y'][snap]
                print_tensor_device(test_labels, "Test labels")
                
                with torch.no_grad():
                    int_embed = int_embeddings[idx].to(device)
                    hop_embed = hop_embeddings[idx].to(device)
                    time_embed = time_embeddings[idx].to(device)
                    
                    output = self.forward(int_embed, hop_embed, time_embed).squeeze()
                    pred_scores = torch.sigmoid(output).cpu().numpy()
                
                all_true.append(test_labels.cpu().numpy())
                all_pred.append(pred_scores)
        
        if len(all_true) == 0:
            print("No test samples processed")
            return 0.5
                
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        test_auc = metrics.roc_auc_score(all_true, all_pred)
        
        print(f"Total evaluation time: {time.time() - eval_start:.4f} seconds")
        return test_auc

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
        """Cached version of generate_embedding"""
        print("\nDetailed embedding generation profiling:")
        total_start = time.time()
        edges_tuple = tuple(tuple(e.tolist() if isinstance(e, torch.Tensor) else e) for e in edges)

        num_snap = len(edges)
        
        # Edge processing
        edge_start = time.time()
        edges_cpu = []
        for e in edges_tuple[:7]:
            if isinstance(e, torch.Tensor):
                edges_cpu.append(e.cpu().numpy())
            else:
                edges_cpu.append(e)
        edge_time = time.time() - edge_start
        print(f"Edge processing: {edge_time:.4f}s")
        
        # WL Dict computation
        wl_start = time.time()
        WL_dict = compute_zero_WL(self.data['idx'], np.vstack(edges_cpu))
        wl_time = time.time() - wl_start
        print(f"WL Dict computation: {wl_time:.4f}s")
        
        # Use GPU-optimized batch hop computation
        hop_start = time.time()
        batch_hop_dicts = compute_batch_hop_gpu(
            self.data['idx'], 
            edges, 
            num_snap, 
            self.data['S'], 
            self.config.k, 
            self.config.window_size
        )
        hop_time = time.time() - hop_start
        print(f"Batch hop computation: {hop_time:.4f}s")
        
        # Final embeddings
        embed_start = time.time()
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

        embed_time = time.time() - embed_start
        print(f"Final embeddings: {embed_time:.4f}s")
        
        total_time = time.time() - total_start
        print(f"Total embedding generation: {total_time:.4f}s")
        
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
        print("\n=== Starting Training With Timing and Device Diagnostics ===")
        device = next(self.parameters()).device
        print(f"Training on device: {device}")
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scaler = GradScaler()
        
        with TimingContext("Initial embedding generation"):
            raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
                self.generate_embedding(self.data['edges'])
            print_tensor_device(int_embeddings, "int_embeddings")
            print_tensor_device(hop_embeddings, "hop_embeddings")
            print_tensor_device(time_embeddings, "time_embeddings")
        
        self.data['raw_embeddings'] = None
        
        for epoch in range(max_epoch):
            epoch_start = time.time()
            
            with TimingContext("Negative sampling"):
                negatives = self.negative_sampling(self.data['edges'][:max(self.data['snap_train']) + 1])
                print(f"Generated {len(negatives)} negative samples")
            
            with TimingContext("Negative embedding generation"):
                _, _, hop_embeddings_neg, int_embeddings_neg, time_embeddings_neg = \
                    self.generate_embedding(negatives)
                print_tensor_device(int_embeddings_neg, "int_embeddings_neg")
            
            self.train()
            loss_train = 0
            all_true = []
            all_pred = []
            
            batch_times = []
            for snap in self.data['snap_train']:
                # Remove wl_embeddings check since we're not using it
                if int_embeddings[snap] is None:  # Changed from wl_embeddings
                    continue
                    
                batch_start = time.time()
                
                with TimingContext(f"Data preparation for snap {snap}"):
                    int_embedding = torch.vstack((int_embeddings[snap], int_embeddings_neg[snap])).to(device)
                    hop_embedding = torch.vstack((hop_embeddings[snap], hop_embeddings_neg[snap])).to(device)
                    time_embedding = torch.vstack((time_embeddings[snap], time_embeddings_neg[snap])).to(device)
                    y = torch.hstack((self.data['y'][snap].float(), 
                                    torch.ones(int_embeddings_neg[snap].size()[0]))).to(device)
                    print_tensor_device(int_embedding, "Combined int_embedding")
                
                optimizer.zero_grad()
                
                with TimingContext("Forward pass"):
                    with autocast():
                        output = self.forward(int_embedding, hop_embedding, time_embedding).squeeze()
                        loss = F.binary_cross_entropy_with_logits(output, y)
                    print_tensor_device(output, "Model output")
                
                with TimingContext("Backward pass"):
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                with TimingContext("Metrics calculation"):
                    loss_train += loss.item()
                    pred_scores = torch.sigmoid(output).detach().cpu().numpy()
                    true_labels = y.detach().cpu().numpy()
                    all_true.append(true_labels)
                    all_pred.append(pred_scores)
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                print(f"Batch for snap {snap} took {batch_time:.4f} seconds")
            
            with TimingContext("Test evaluation"):
                test_auc = self.evaluate_test_set()
            
            # Calculate epoch metrics
            train_loss = loss_train / len(self.data['snap_train'])
            all_true = np.concatenate(all_true)
            all_pred = np.concatenate(all_pred)
            train_auc = metrics.roc_auc_score(all_true, all_pred)
            
            self.train_losses.append(train_loss)
            self.train_aucs.append(train_auc)
            self.test_aucs.append(test_auc)
            
            avg_batch_time = np.mean(batch_times)
            print(f'Epoch: {epoch + 1}')
            print(f'Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}')
            print(f'Total Time: {time.time() - epoch_start:.4f}s')
            print(f'Average Batch Time: {avg_batch_time:.4f}s')
            if torch.cuda.is_available():
                print(f'GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB')
                print(f'GPU Memory Cached: {torch.cuda.memory_reserved()/1024**2:.1f}MB')
            
            self.update_plots(epoch)

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