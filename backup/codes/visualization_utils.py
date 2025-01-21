import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

class AnomalyVisualizer:
    def __init__(self):
        plt.style.use('seaborn')
        self.edge_history = defaultdict(list)
        self.edge_scores = defaultdict(list)
        self.timestamp_scores = defaultdict(list)
        
    def track_edge(self, edge, score, is_anomaly, timestamp):
        """Track edge behavior over time"""
        edge_key = f"{edge[0]}-{edge[1]}"
        self.edge_history[edge_key].append((timestamp, is_anomaly))
        self.edge_scores[edge_key].append((timestamp, score))
        self.timestamp_scores[timestamp].append(score)

    def plot_persistent_anomalies(self, min_anomaly_occurrences=3):
        """Plot histogram of edges that remain anomalous across multiple snapshots"""
        plt.figure(figsize=(12, 6))
        
        # Count anomaly occurrences per edge
        anomaly_counts = {}
        for edge, history in self.edge_history.items():
            anomaly_count = sum(1 for _, is_anomaly in history if is_anomaly)
            if anomaly_count >= min_anomaly_occurrences:
                anomaly_counts[edge] = anomaly_count
        
        if not anomaly_counts:
            plt.text(0.5, 0.5, 'No persistent anomalies found', 
                    horizontalalignment='center', verticalalignment='center')
        else:
            plt.bar(range(len(anomaly_counts)), list(anomaly_counts.values()))
            plt.xticks(range(len(anomaly_counts)), list(anomaly_counts.keys()), 
                      rotation=45, ha='right')
            
        plt.title('Persistent Anomalous Edges')
        plt.xlabel('Edge (Node Pairs)')
        plt.ylabel('Number of Times Marked Anomalous')
        plt.tight_layout()
        return plt.gcf()

    def plot_edge_transitions(self):
        """Plot edges that transition between normal and anomalous states"""
        plt.figure(figsize=(12, 6))
        
        transitions = []
        for edge, history in self.edge_history.items():
            if len(history) < 2:
                continue
                
            transition_count = sum(1 for i in range(len(history)-1) 
                                 if history[i][1] != history[i+1][1])
            
            if transition_count > 0:
                transitions.append((edge, transition_count))
        
        if not transitions:
            plt.text(0.5, 0.5, 'No state transitions found', 
                    horizontalalignment='center', verticalalignment='center')
        else:
            edges, counts = zip(*sorted(transitions, key=lambda x: x[1], reverse=True))
            plt.bar(range(len(edges)), counts)
            plt.xticks(range(len(edges)), edges, rotation=45, ha='right')
            
        plt.title('Edge State Transitions (Normal ↔ Anomalous)')
        plt.xlabel('Edge (Node Pairs)')
        plt.ylabel('Number of State Transitions')
        plt.tight_layout()
        return plt.gcf()

    def plot_anomaly_score_distributions(self):
        """Plot histogram of anomaly scores for each timestamp"""
        num_timestamps = len(self.timestamp_scores)
        cols = min(3, num_timestamps)
        rows = (num_timestamps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, (timestamp, scores) in enumerate(sorted(self.timestamp_scores.items())):
            if i < len(axes):
                sns.histplot(scores, bins=30, ax=axes[i])
                axes[i].set_title(f'Timestamp {timestamp}')
                axes[i].set_xlabel('Anomaly Score')
                axes[i].set_ylabel('Count')
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        return plt.gcf()

    def save_visualizations(self, output_dir='anomaly_visualizations'):
        """Save all visualizations to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save persistent anomalies plot
        fig = self.plot_persistent_anomalies()
        fig.savefig(f'{output_dir}/persistent_anomalies.png')
        plt.close(fig)
        
        # Save edge transitions plot
        fig = self.plot_edge_transitions()
        fig.savefig(f'{output_dir}/edge_transitions.png')
        plt.close(fig)
        
        # Save anomaly score distributions
        fig = self.plot_anomaly_score_distributions()
        fig.savefig(f'{output_dir}/anomaly_scores_distribution.png')
        plt.close(fig)