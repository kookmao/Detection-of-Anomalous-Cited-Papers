import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_top_edges_history(history_file, top_edges_file, output_dir='edge_history_plots'):
    """
    Create plots showing the anomaly score history for top edges.
    
    Args:
        history_file (str): Path to edge_score_history.csv
        top_edges_file (str): Path to top_10.csv
        output_dir (str): Directory to save the plots
    """
    history_df = pd.read_csv(history_file)
    top_edges_df = pd.read_csv(top_edges_file)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn')
    
    for _, edge_data in top_edges_df.iterrows():
        edge_id = edge_data['edge_id']
        title = edge_data['title']
        variance = edge_data['variance']
        
        edge_history = history_df[history_df['edge_id'] == edge_id]
        
        if edge_history.empty:
            print(f"No history found for edge {edge_id}")
            continue
            
    
        plt.figure(figsize=(12, 6))
        plt.plot(edge_history['sequence'], edge_history['anomaly_score'], 
                marker='o', linestyle='-', linewidth=2, markersize=4)
        

        plt.title(f"{title}\n(Edge: {edge_id}, Variance: {variance:.3f})", 
                 fontsize=11, pad=15, wrap=True)
        plt.xlabel('Timestamp')
        plt.ylabel('Anomaly Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        for x, y in zip(edge_history['sequence'], edge_history['anomaly_score']):
            plt.annotate(f'{y:.2f}', 
                        (x, y),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center',
                        fontsize=8)
        
   
        plt.ylim(0, max(1.0, edge_history['anomaly_score'].max() * 1.1))
   
        plt.tight_layout()
        safe_title = edge_id.replace('/', '_').replace('\\', '_')
        plt.savefig(f'{output_dir}/history_{safe_title}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot for edge {edge_id}")

if __name__ == "__main__":
    plot_top_edges_history(
        history_file='anomaly_analysis_epoch_100/edge_score_history.csv',
        top_edges_file='top_10_highest_variations.csv',
        output_dir='edge_history_plots'
    )