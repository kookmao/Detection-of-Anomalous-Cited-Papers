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
        self.edge_history = defaultdict(list)  # Tracks edge states over time
        self.edge_scores = defaultdict(list)   # Tracks edge scores over time
        self.timestamp_scores = defaultdict(list)  # All scores for each timestamp
        self.timestamp_aucs = {}  # AUC scores per timestamp
        self.persistence_threshold = persistence_threshold
        self.high_variance_threshold = high_variance_threshold  # Threshold for high standard deviation
        self.transitions = []  # List to store edge transitions
        
        # New attributes for score variation analysis
        self.edge_score_series: Dict[str, List[float]] = defaultdict(list)  # Time series of scores for each edge
        self.edge_timestamps: Dict[str, List[int]] = defaultdict(list)      # Timestamps for each edge
        self.edge_std_devs: Dict[str, float] = {}  # Standard deviations of scores for each edge
        
        # Load paper titles at initialization
        try:
            df = pd.read_csv('five_year.csv')
            self.paper_titles = dict(zip(df['fromNode'].astype(str), df['title']))
        except:
            print("Warning: Could not load paper titles from five_year.csv")
            self.paper_titles = {}
        
    def track_snapshot(self, edges, scores, true_labels, timestamp):
        """
        Track a single snapshot's worth of anomaly data
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
            
            # Store score time series data with epoch as timestamp
            self.edge_score_series[edge_id].append(score)
            self.edge_timestamps[edge_id].append(timestamp)

    def analyze_score_variations(self):
        """Calculate standard deviations for all edge scores"""
        for edge_id, scores in self.edge_score_series.items():
            if len(scores) > 1:  # Need at least 2 points for std dev
                self.edge_std_devs[edge_id] = np.std(scores)
    
    def get_high_variance_edges(self) -> List[Tuple[str, float]]:
        """Get edges with standard deviation above the threshold"""
        high_var_edges = [(edge_id, std_dev) 
                         for edge_id, std_dev in self.edge_std_devs.items() 
                         if std_dev > self.high_variance_threshold]
        return sorted(high_var_edges, key=lambda x: x[1], reverse=True)
    
    def plot_top_edges_score_variation(self, output_dir: str, top_n: int = 10):
        """
        Plot, for each of the top-N edges with highest overall score standard deviation,
        the 'anomaly score variation' vs. timestamp.
        
        The 'variation' is the absolute difference between consecutive scores.
        X-axis: timestamp (except the very first, since variation is from a prior score).
        Y-axis: anomaly score difference from the previous timestamp.
        The plot title uses the source paper's title (from five_year.csv).
        """
        # Make sure we have calculated self.edge_std_devs
        self.analyze_score_variations()
        
        # Get edges sorted by descending std dev
        high_var_edges = self.get_high_variance_edges()  # returns [(edge_id, std_dev), ...]

        # Limit to top N
        edges_to_plot = high_var_edges[:top_n]
        if not edges_to_plot:
            print("No edges found or no high-variance edges. Skipping variation plots.")
            return
        
        # Directory for saving these plots
        variation_dir = os.path.join(output_dir, "score_variation_plots")
        os.makedirs(variation_dir, exist_ok=True)

        for i, (edge_id, std_dev) in enumerate(edges_to_plot, start=1):
            # Retrieve the anomaly scores and timestamps
            scores = self.edge_score_series[edge_id]
            ts = self.edge_timestamps[edge_id]
            
            # Sort by timestamp if needed, so the line is chronological
            # (If your timestamps are already in ascending order, you can skip this.)
            combined = sorted(zip(ts, scores), key=lambda x: x[0])
            sorted_ts = [c[0] for c in combined]
            sorted_scores = [c[1] for c in combined]
            
            # Compute the "variation" (difference) array
            # variation[t_i] = |score[i] - score[i-1]|
            variations = []
            variation_ts = []
            for j in range(1, len(sorted_scores)):
                diff = abs(sorted_scores[j] - sorted_scores[j-1])
                variations.append(diff)
                variation_ts.append(sorted_ts[j])  # Variation is at the "later" timestamp

            # Identify the source node
            source_node = edge_id.split('-')[0]
            
            # Get the paper title if available
            paper_title = self.paper_titles.get(source_node, '')
            if not paper_title:
                plot_title = f"Edge {edge_id} (No title found)"
            else:
                plot_title = f"{paper_title}\n(Edge {edge_id})"

            # Plot
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.plot(variation_ts, variations, marker='o', linestyle='-',
                    linewidth=2, markersize=6, label='Score Variation')
            
            plt.title(plot_title, fontsize=11, pad=15)
            plt.xlabel('Timestamp')
            plt.ylabel('Score Variation')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Annotate each point
            for x_val, y_val in zip(variation_ts, variations):
                plt.annotate(f"{y_val:.2f}",
                            (x_val, y_val),
                            xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=8)
            
            # Some room above highest variation
            y_max = max(variations) if variations else 0
            plt.ylim([0, max(1.0, y_max*1.1)])

            plt.tight_layout()
            
            # Save figure
            plot_path = os.path.join(variation_dir, f"variation_{i}_{edge_id}.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()

            print(f"Saved variation plot for edge {edge_id} (std_dev={std_dev:.3f}) to {plot_path}")
    def plot_score_variation_histogram(self, output_dir: str):
        """Generate histogram of score standard deviations"""
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
        
        # Save std dev data to CSV
        df = pd.DataFrame(list(self.edge_std_devs.items()), 
                         columns=['edge_id', 'std_dev'])
        df.to_csv(f'{output_dir}/edge_score_variations.csv', index=False)

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

    def generate_reports(self, output_dir='anomaly_analysis'):
        """Generate all visualizations and reports"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Only generate reports if we have data
        if not self.edge_score_series:
            print("No data collected yet. Skipping report generation.")
            return
            
        # Calculate score variations
        self.analyze_score_variations()
        
        print(f"Analyzing {len(self.edge_score_series)} edges...")
        print(f"Found {len(self.edge_std_devs)} edges with variance data...")
        
        # 1. Analyze persistent anomalies
        #self._analyze_persistent_anomalies(output_dir)
        
        # 2. Save edge transitions to CSV
        self._save_transitions(output_dir)
        
        # 3. Generate timestamp histograms
        self._generate_timestamp_histograms(output_dir)
        
        # 4. Generate score variation analysis
        self.plot_score_variation_histogram(output_dir)

        
        
        print(f"Analysis complete. Results saved to {output_dir}/")

    