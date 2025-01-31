import os
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers.modeling_bert import BertPreTrainedModel
from codes.BaseModel import BaseModel

import time
import numpy as np

from sklearn import metrics
from codes.utils import dicts_to_embeddings, compute_batch_hop, compute_zero_WL
from codes.visualization_utils import AnomalyVisualizer
from anomaly_tracking_utils import AnomalyTracker


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

        self.focal_alpha = 0.75  # Weight for rare class (anomalies)
        self.focal_gamma = 2.0   # Focusing parameter

        # Initialize tracking lists
        self.train_losses = []
        self.train_aucs = []
        self.test_aucs = []  # Add this line

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

    def evaluate_test_set(self):
        """Evaluate model on test set"""
        self.eval()
        all_true = []
        all_pred = []
        
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
            self.generate_embedding([self.data['edges'][snap] for snap in self.data['snap_test']])

        for idx, snap in enumerate(self.data['snap_test']):
            if int_embeddings[idx] is None:
                continue
                
            test_edges = self.data['edges'][snap]
            test_labels = self.data['y'][snap]
            
            with torch.no_grad():
                output = self.forward(int_embeddings[idx], 
                                    hop_embeddings[idx],
                                    time_embeddings[idx]).squeeze()
                pred_scores = torch.sigmoid(output).cpu().numpy()
            
            all_true.append(test_labels.cpu().numpy())
            all_pred.append(pred_scores)
        
        if len(all_true) == 0:
            return 0.5
            
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        test_auc = metrics.roc_auc_score(all_true, all_pred)
        
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
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        output = self.cls_y(sequence_output)

        return output

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
        num_snap = len(edges)
        # WL_dict = compute_WL(self.data['idx'], np.vstack(edges[:7]))
        WL_dict = compute_zero_WL(self.data['idx'],  np.vstack(edges[:7]))
        batch_hop_dicts = compute_batch_hop(self.data['idx'], edges, num_snap, self.data['S'], self.config.k, self.config.window_size)
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
            dicts_to_embeddings(self.data['X'], batch_hop_dicts, WL_dict, num_snap)
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
        """Paper-compliant sampling: disconnected nodes in training period"""
        neg_edges = []
        for snap in self.data['snap_train']:
            adj_matrix = self.data['adj'][snap].to_dense()
            disconnected = (adj_matrix == 0).nonzero(as_tuple=False)
            sampled_idx = torch.randint(0, len(disconnected), (len(edges[snap]),))
            neg_edges.append(disconnected[sampled_idx])
        return neg_edges

    def train_model(self, max_epoch):
         # Initialize Focal Loss parameters
        focal_alpha = 0.15  # Weight for anomaly class (adjust based on class imbalance)
        focal_gamma = 2.0   # Focusing parameter for hard examples
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = self.generate_embedding(self.data['edges'])
        self.data['raw_embeddings'] = None

        # Early stopping setup
        best_test_auc = 0
        patience = 10
        epochs_no_improve = 0

        for epoch in range(max_epoch):
            t_epoch_begin = time.time()
            negatives = self.negative_sampling(self.data['edges'][:max(self.data['snap_train']) + 1])
            _, _, hop_embeddings_neg, int_embeddings_neg, time_embeddings_neg = self.generate_embedding(negatives)

            self.train()
            loss_train = 0
            all_true = []
            all_pred = []

            for snap in self.data['snap_train']:
                if wl_embeddings[snap] is None:
                    continue

                # Prepare batch data
                int_embedding = torch.vstack((int_embeddings[snap], int_embeddings_neg[snap]))
                hop_embedding = torch.vstack((hop_embeddings[snap], hop_embeddings_neg[snap]))
                time_embedding = torch.vstack((time_embeddings[snap], time_embeddings_neg[snap]))
                y = torch.hstack((self.data['y'][snap].float(), 
                                torch.ones(int_embeddings_neg[snap].size()[0])))

                optimizer.zero_grad()
                
                # Forward pass
                output = self.forward(int_embedding, hop_embedding, time_embedding).squeeze()
                
                # Focal Loss implementation
                bce_loss = F.binary_cross_entropy_with_logits(output, y, reduction='none')
                pt = torch.exp(-bce_loss)
                focal_loss = (focal_alpha * (1 - pt) ** focal_gamma * bce_loss).mean()
                
                # Add L2 regularization
                l2_reg = torch.tensor(0., device=output.device)
                for param in self.parameters():
                    l2_reg += torch.norm(param, 2)
                loss = focal_loss + self.config.weight_decay * l2_reg

                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                
                optimizer.step()

                # Track metrics
                loss_train += loss.item()
                pred_scores = torch.sigmoid(output).detach().cpu().numpy()
                true_labels = y.detach().cpu().numpy()
                
                all_true.append(true_labels)
                all_pred.append(pred_scores)

            # Calculate epoch metrics
            train_loss = loss_train / len(self.data['snap_train'])
            all_true = np.concatenate(all_true)
            all_pred = np.concatenate(all_pred)
            train_auc = metrics.roc_auc_score(all_true, all_pred) if len(np.unique(all_true)) > 1 else 0.5
            
            # Test set evaluation
            test_auc = self.evaluate_test_set()

            # Generate reports every 10 epochs
            if (epoch + 1) % 10 == 0:
                output_dir = f'anomaly_analysis_epoch_{epoch+1}'
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(f'{output_dir}/training_testing_progress.png')
                self.anomaly_tracker.generate_reports(output_dir=output_dir)

            # Early stopping check
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

            # Update tracking and plots
            self.train_losses.append(train_loss)
            self.train_aucs.append(train_auc)
            self.test_aucs.append(test_auc)
            
            print(f'Epoch: {epoch + 1}, Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, '
                f'Test AUC: {test_auc:.4f}, Time: {time.time() - t_epoch_begin:.4f}s')
            
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