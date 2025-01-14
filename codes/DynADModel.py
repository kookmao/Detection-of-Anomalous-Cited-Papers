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

        # Initialize plotting
        self.train_losses = []
        self.train_aucs = []  # Changed from accuracies to AUCs
        
        # Create figure for plotting
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Initialize empty lines
        self.line1, = self.ax1.plot([], [], 'b-', label='Loss')
        self.line2, = self.ax2.plot([], [], 'r-', label='AUC')  # Changed label
        
        # Configure axes
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Loss')
        self.ax1.legend()
        
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('AUC')  # Changed label
        self.ax2.set_title('Training AUC')  # Changed title
        self.ax2.legend()
        
        plt.tight_layout()
        plt.show()

        # Initialize anomaly tracker
        self.anomaly_tracker = AnomalyTracker(persistence_threshold=0.8)

    def update_plots(self, epoch):
        """Update the training plots with new data"""
        epochs = list(range(len(self.train_losses)))
        
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.plot(epochs, self.train_losses, 'b-')
        self.ax2.plot(epochs, self.train_aucs, 'r-')
        
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Loss')
        
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('AUC')  # Changed label
        self.ax2.set_title('Training AUC')  # Changed title
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Save periodic results every 10 epochs
        if (epoch + 1) % 10 == 0:
            output_dir = f'anomaly_analysis_epoch_{epoch+1}'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save training plots
            plt.savefig(f'{output_dir}/training_progress.png')
            
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

    def negative_sampling(self, edges):
        negative_edges = []
        node_list = self.data['idx']
        num_node = node_list.shape[0]
        for snap_edge in edges:
            num_edge = snap_edge.shape[0]

            negative_edge = snap_edge.copy()
            fake_idx = np.random.choice(num_node, num_edge)
            fake_position = np.random.choice(2, num_edge).tolist()
            fake_idx = node_list[fake_idx]
            negative_edge[np.arange(num_edge), fake_position] = fake_idx

            negative_edges.append(negative_edge)
        return negative_edges

    def train_model(self, max_epoch):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = self.generate_embedding(self.data['edges'])
        self.data['raw_embeddings'] = None

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

                int_embedding = torch.vstack((int_embeddings[snap], int_embeddings_neg[snap]))
                hop_embedding = torch.vstack((hop_embeddings[snap], hop_embeddings_neg[snap]))
                time_embedding = torch.vstack((time_embeddings[snap], time_embeddings_neg[snap]))
                y = torch.hstack((self.data['y'][snap].float(), torch.ones(int_embeddings_neg[snap].size()[0])))

                optimizer.zero_grad()
                output = self.forward(int_embedding, hop_embedding, time_embedding).squeeze()
                loss = F.binary_cross_entropy_with_logits(output, y)
                loss.backward()
                optimizer.step()

                loss_train += loss.item()
                pred_scores = torch.sigmoid(output).detach().cpu().numpy()
                true_labels = y.detach().cpu().numpy()
                
                all_true.append(true_labels)
                all_pred.append(pred_scores)
                
                # Track anomalies if it's a 10th epoch
                if (epoch + 1) % 10 == 0:
                    edges = self.data['edges'][snap]
                    self.anomaly_tracker.track_snapshot(
                        edges=edges,
                        scores=pred_scores,
                        true_labels=true_labels,
                        timestamp=snap
                    )

            # Calculate average loss and AUC
            train_loss = loss_train / len(self.data['snap_train'])
            all_true = np.concatenate(all_true)
            all_pred = np.concatenate(all_pred)
            train_auc = metrics.roc_auc_score(all_true, all_pred)

            self.train_losses.append(train_loss)
            self.train_aucs.append(train_auc)

            print(f'Epoch: {epoch + 1}, Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Time: {time.time() - t_epoch_begin:.4f}s')
            self.update_plots(epoch)

    def run(self):
        self.train_model(self.max_epoch)
        return self.learning_record_dict