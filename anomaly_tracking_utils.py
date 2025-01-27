import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os
import seaborn as sns
from sklearn import metrics
from typing import Dict, List, Tuple

class AnomalyTracker:
    def __init__(self, persistence_threshold=0.8, high_variance_threshold=0.15):
        self.edge_history = defaultdict(list) 
        self.edge_scores = defaultdict(list)  
        self.timestamp_scores = defaultdict(list) 
        self.timestamp_aucs = {} 
        self.persistence_threshold = persistence_threshold
        self.high_variance_threshold = high_variance_threshold  
        self.transitions = []  
        
        # Edge mapping 
        self.edge_mapping = None  
        self.edge_id_mapping = {}  
        self.reverse_edge_mapping = {}  
        
        
        self.edge_score_series: Dict[str, List[float]] = defaultdict(list)  
        self.edge_timestamps: Dict[str, List[int]] = defaultdict(list)      
        self.edge_std_devs: Dict[str, float] = {}  
        
        # REMOVE THIS PART NOT RELEVANT ANYMORE

        # paper titles at initialization
        try:
            df = pd.read_csv('five_year.csv')
            self.paper_titles = dict(zip(df['fromNode'].astype(str), df['title']))
            # Store all valid node IDs
            self.valid_nodes = set(df['fromNode'].unique()) | set(df['toNode'].unique())
        except:
            print("Warning: Could not load paper titles from five_year.csv")
            self.paper_titles = {}
            self.valid_nodes = set()

    def set_edge_mapping(self, edge_data):
        """Set up edge mappings from the original data"""
        if edge_data and 'edge_mapping' in edge_data:
            self.edge_mapping = edge_data['edge_mapping']
            # create reverse mapping (modified -> original)
            for orig_idx, orig_edge in self.edge_mapping.items():
                modified_edge_id = f"{orig_idx}-{orig_idx}" 
                original_edge_id = f"{orig_edge[0]}-{orig_edge[1]}"
                self.edge_id_mapping[modified_edge_id] = original_edge_id
                self.reverse_edge_mapping[original_edge_id] = modified_edge_id

    def get_original_edge_id(self, edge_id):
        """Convert a modified edge ID to its original edge ID"""
        return self.edge_id_mapping.get(edge_id, edge_id)

    def track_snapshot(self, edges, scores, true_labels, timestamp):
        """Track a single snapshot's worth of anomaly data"""
        auc_score = metrics.roc_auc_score(true_labels, scores)
        self.timestamp_aucs[timestamp] = auc_score
        
        self.timestamp_scores[timestamp] = scores.tolist()
        
        for idx, (edge, score, label) in enumerate(zip(edges, scores, true_labels)):
            if self.edge_mapping is not None:
                edge_tuple = tuple(edge)
                if edge_tuple in self.edge_mapping:
                    original_edge = self.edge_mapping[edge_tuple]
                    edge_id = f"{original_edge[0]}-{original_edge[1]}"
                else:
                    edge_id = f"{edge[0]}-{edge[1]}"
            else:
                edge_id = f"{edge[0]}-{edge[1]}"
                
            is_anomaly = score >= 0.5

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
            
            self.edge_history[edge_id].append((timestamp, is_anomaly))
            self.edge_scores[edge_id].append((timestamp, score))
            self.edge_score_series[edge_id].append(score)
            self.edge_timestamps[edge_id].append(timestamp)

    def analyze_score_variations(self):
        """Calculate standard deviations for all edge scores"""
        for edge_id, scores in self.edge_score_series.items():
            if len(scores) > 1:  #at least 2 points for std dev
                self.edge_std_devs[edge_id] = np.std(scores)
    
    def get_high_variance_edges(self) -> List[Tuple[str, float]]:
        """Get edges with standard deviation above the threshold"""
        high_var_edges = [(edge_id, std_dev) 
                         for edge_id, std_dev in self.edge_std_devs.items() 
                         if std_dev > self.high_variance_threshold]
        return sorted(high_var_edges, key=lambda x: x[1], reverse=True)
    
    def plot_top_edges_score_variation(self, output_dir: str, top_n: int = 10):
        """Plot score variations using original edge IDs"""
        self.analyze_score_variations()
        high_var_edges = self.get_high_variance_edges()
        
        edges_to_plot = []
        for edge_id, std_dev in high_var_edges:
            original_edge_id = self.get_original_edge_id(edge_id)
            from_node = original_edge_id.split('-')[0]
            if from_node in self.valid_nodes:  # include edges with valid nodes
                edges_to_plot.append((original_edge_id, std_dev))
            if len(edges_to_plot) >= top_n:
                break
                
        if not edges_to_plot:
            print("No edges found or no high-variance edges. Skipping variation plots.")
            return
        
        variation_dir = os.path.join(output_dir, "score_variation_plots")
        os.makedirs(variation_dir, exist_ok=True)

        for i, (edge_id, std_dev) in enumerate(edges_to_plot, start=1):
            scores = self.edge_score_series[edge_id]
            ts = self.edge_timestamps[edge_id]
            
            combined = sorted(zip(ts, scores), key=lambda x: x[0])
            sorted_ts = [c[0] for c in combined]
            sorted_scores = [c[1] for c in combined]
            
            variations = []
            variation_ts = []
            for j in range(1, len(sorted_scores)):
                diff = abs(sorted_scores[j] - sorted_scores[j-1])
                variations.append(diff)
                variation_ts.append(sorted_ts[j])

            source_node = edge_id.split('-')[0]
            paper_title = self.paper_titles.get(source_node, '')
            if not paper_title:
                plot_title = f"Edge {edge_id} (No title found)"
            else:
                plot_title = f"{paper_title}\n(Edge {edge_id})"

            plt.figure(figsize=(8, 4))
            plt.plot(variation_ts, variations, marker='o', linestyle='-',
                    linewidth=2, markersize=6, label='Score Variation')
            
            plt.title(plot_title, fontsize=11, pad=15)
            plt.xlabel('Timestamp')
            plt.ylabel('Score Variation')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            for x_val, y_val in zip(variation_ts, variations):
                plt.annotate(f"{y_val:.2f}",
                            (x_val, y_val),
                            xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=8)
            
            y_max = max(variations) if variations else 0
            plt.ylim([0, max(1.0, y_max*1.1)])

            plt.tight_layout()
            
            plot_path = os.path.join(variation_dir, f"variation_{i}_{edge_id}.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()

            print(f"Saved variation plot for edge {edge_id} (std_dev={std_dev:.3f}) to {plot_path}")
    def save_edge_score_history(self, output_dir: str):
        """Save the complete history of anomaly scores for each edge across all timestamps.
        
        Outputs a CSV with columns:
        - edge_id: Original edge identifier
        - timestamp: Time point when score was recorded
        - anomaly_score: The score assigned to the edge at that timestamp
        - sequence: Order in which scores were recorded
        """
        import pandas as pd
        import os
        
        history_data = []
        
        for edge_id, scores in self.edge_scores.items():
            original_edge_id = self.get_original_edge_id(edge_id)
            
            for idx, (timestamp, score) in enumerate(scores):
                history_data.append({
                    'edge_id': original_edge_id,
                    'timestamp': timestamp,
                    'anomaly_score': score,
                    'sequence': idx
                })
        
        if history_data:
            df = pd.DataFrame(history_data)
            df = df.sort_values(['edge_id', 'timestamp', 'sequence'])
            
            output_path = os.path.join(output_dir, 'edge_score_history.csv')
            df.to_csv(output_path, index=False)
            print(f"Saved edge score history to {output_path}")
        else:
            print("No edge score history data available to save")

    def plot_score_variation_histogram(self, output_dir: str):
        """Generate histogram of score standard deviations with original edge IDs"""
        plt.figure(figsize=(12, 6))
        std_devs = list(self.edge_std_devs.values())
        
        sns.histplot(std_devs, bins='auto')
        plt.axvline(x=self.high_variance_threshold, color='r', linestyle='--', 
                   label=f'High Variance Threshold ({self.high_variance_threshold})')
        
        plt.title('Distribution of Edge Score Standard Deviations')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Count')
        plt.legend()
        
        plt.savefig(f'{output_dir}/score_variation_distribution.png')
        plt.close()
        
        std_dev_data = []
        for edge_id, std_dev in self.edge_std_devs.items():
            original_edge_id = self.get_original_edge_id(edge_id)
            std_dev_data.append({
                'edge_id': original_edge_id,
                'std_dev': std_dev
            })
        
        df = pd.DataFrame(std_dev_data)
        df.to_csv(f'{output_dir}/edge_score_variations.csv', index=False)

    def _analyze_persistent_anomalies(self, output_dir):
        """Generate histogram of persistently anomalous edges"""
        persistence_counts = {}
        
        for edge_id, history in self.edge_history.items():
            anomaly_ratio = sum(1 for _, is_anomaly in history if is_anomaly) / len(history)
            if anomaly_ratio >= self.persistence_threshold:
                original_edge_id = self.get_original_edge_id(edge_id)
                persistence_counts[original_edge_id] = len(history)
        
        if persistence_counts:
            sorted_items = sorted(persistence_counts.items(), key=lambda x: x[1], reverse=True)
            edge_ids, counts = zip(*sorted_items)
            
            plt.figure(figsize=(15, 8))
            
            bars = plt.bar(range(len(counts)), counts)
            
            plt.title(f'Edges Anomalous in >{self.persistence_threshold*100}% of Snapshots',
                    fontsize=12, pad=20)
            plt.xlabel('Edge ID', fontsize=10)
            plt.ylabel('Number of Snapshots', fontsize=10)
            plt.xticks(range(len(edge_ids)), 
                    edge_ids,
                    rotation=90,
                    ha='center',
                    fontsize=8)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom',
                        fontsize=8)
            
            plt.tight_layout()
            
            plt.savefig(f'{output_dir}/persistent_anomalies.png', 
                    bbox_inches='tight',
                    dpi=300)
            plt.close()

    def _save_transitions(self, output_dir):
        """Save edge transitions to CSV with original edge IDs"""
        if self.transitions:
            transitions_copy = []
            for transition in self.transitions:
                transition_copy = transition.copy()
                transition_copy['edge_id'] = self.get_original_edge_id(transition_copy['edge_id'])
                transitions_copy.append(transition_copy)
            
            df = pd.DataFrame(transitions_copy)
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

    def generate_reports(self, output_dir='anomaly_analysis'):
        """Generate all visualizations and reports"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.edge_score_series:
            print("No data collected yet. Skipping report generation.")
            return
            
        self.analyze_score_variations()
        
        print(f"Analyzing {len(self.edge_score_series)} edges...")
        print(f"Found {len(self.edge_std_devs)} edges with variance data...")
        
        self._analyze_persistent_anomalies(output_dir)
        self._save_transitions(output_dir)
        self._generate_timestamp_histograms(output_dir)
        self.plot_score_variation_histogram(output_dir)
        self.plot_top_edges_score_variation(output_dir)
        self.save_edge_score_history(output_dir)
        
        print(f"Analysis complete. Results saved to {output_dir}/")