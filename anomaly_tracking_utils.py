import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os
import seaborn as sns
from sklearn import metrics

class AnomalyTracker:
    def __init__(self, persistence_threshold=0.8):
        self.edge_history = defaultdict(list)  # Tracks edge states over time
        self.edge_scores = defaultdict(list)   # Tracks edge scores over time
        self.timestamp_scores = defaultdict(list)  # All scores for each timestamp
        self.timestamp_aucs = {}  # AUC scores per timestamp
        self.persistence_threshold = persistence_threshold
        self.transitions = []  # List to store edge transitions
        
    def track_snapshot(self, edges, scores, true_labels, timestamp):
        """
        Track a single snapshot's worth of anomaly data
        
        Args:
            edges: List of (source, target) tuples
            scores: Anomaly scores for each edge
            true_labels: True labels for AUC calculation
            timestamp: Current timestamp
        """
        # Calculate AUC for this timestamp
        auc_score = metrics.roc_auc_score(true_labels, scores)
        self.timestamp_aucs[timestamp] = auc_score
        
        # Track scores distribution
        self.timestamp_scores[timestamp] = scores.tolist()
        
        # Track individual edges
        for idx, (edge, score, label) in enumerate(zip(edges, scores, true_labels)):
            edge_id = f"{edge[0]}-{edge[1]}"
            is_anomaly = score >= 0.5  # Current threshold for anomaly
            
            # Check for state transition
            if self.edge_history[edge_id]:
                prev_state = self.edge_history[edge_id][-1][1]
                if prev_state != is_anomaly:
                    self.transitions.append({
                        'edge_id': edge_id,
                        'timestamp': timestamp,
                        'old_state': 'anomalous' if prev_state else 'normal',
                        'new_state': 'anomalous' if is_anomaly else 'normal',
                        'score': score,
                        'auc': auc_score
                    })
            
            # Store state and score
            self.edge_history[edge_id].append((timestamp, is_anomaly))
            self.edge_scores[edge_id].append((timestamp, score))

    def generate_reports(self, output_dir='anomaly_analysis'):
        """Generate all visualizations and reports"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Analyze persistent anomalies
        self._analyze_persistent_anomalies(output_dir)
        
        # 2. Save edge transitions to CSV
        self._save_transitions(output_dir)
        
        # 3. Generate timestamp histograms
        self._generate_timestamp_histograms(output_dir)
        
        print(f"Analysis complete. Results saved to {output_dir}/")

    def _analyze_persistent_anomalies(self, output_dir):
        """Generate histogram of persistently anomalous edges"""
        persistence_counts = {}
        
        for edge_id, history in self.edge_history.items():
            anomaly_ratio = sum(1 for _, is_anomaly in history if is_anomaly) / len(history)
            if anomaly_ratio >= self.persistence_threshold:
                persistence_counts[edge_id] = len(history)
        
        if persistence_counts:
            # Sort edges by count for better visualization
            sorted_items = sorted(persistence_counts.items(), key=lambda x: x[1], reverse=True)
            edge_ids, counts = zip(*sorted_items)
            
            # Create figure with larger size and better spacing
            plt.figure(figsize=(15, 8))
            
            # Create the bar plot
            bars = plt.bar(range(len(counts)), counts)
            
            # Customize the plot
            plt.title(f'Edges Anomalous in >{self.persistence_threshold*100}% of Snapshots',
                    fontsize=12, pad=20)
            plt.xlabel('Edge ID', fontsize=10)
            plt.ylabel('Number of Snapshots', fontsize=10)
            
            # Adjust x-axis labels
            plt.xticks(range(len(edge_ids)), 
                    edge_ids,
                    rotation=90,
                    ha='center',
                    fontsize=8)
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom',
                        fontsize=8)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(f'{output_dir}/persistent_anomalies.png', 
                    bbox_inches='tight',
                    dpi=300)
            plt.close()

    def _save_transitions(self, output_dir):
        """Save edge transitions to CSV"""
        if self.transitions:
            df = pd.DataFrame(self.transitions)
            df.to_csv(f'{output_dir}/edge_transitions.csv', index=False)

    def _generate_timestamp_histograms(self, output_dir):
        """Generate histograms of anomaly scores for each timestamp"""
        hist_dir = f'{output_dir}/histograms'
        os.makedirs(hist_dir, exist_ok=True)
        
        for timestamp, scores in self.timestamp_scores.items():
            plt.figure(figsize=(10, 6))
            sns.histplot(scores, bins='auto')
            auc = self.timestamp_aucs.get(timestamp, 'N/A')
            plt.title(f'Anomaly Score Distribution - Timestamp {timestamp}\nAUC: {auc:.4f}')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(f'{hist_dir}/histogram_t{timestamp}.png')
            plt.close()