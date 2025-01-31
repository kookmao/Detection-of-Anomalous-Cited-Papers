import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np
from datetime import datetime

def analyze_citation_dataset(data_path="data/raw/five_year.csv"):
    # Read the data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print("\n=== Dataset Overview ===")
    print(f"Total citations: {len(df)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Total unique papers: {len(set(df['fromNode'].unique()) | set(df['toNode'].unique()))}")
    
    # Temporal distribution
    print("\n=== Temporal Analysis ===")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    yearly_citations = df.groupby('year').size()
    print("\nCitations per year:")
    print(yearly_citations)
    
    # Monthly pattern
    monthly_avg = df.groupby('month').size() / df['year'].nunique()
    print("\nAverage citations per month:")
    print(monthly_avg)
    
    # Citation patterns
    print("\n=== Citation Patterns ===")
    citing_papers = df['fromNode'].value_counts()
    cited_papers = df['toNode'].value_counts()
    
    print("\nCiting patterns (papers making citations):")
    print(f"Mean citations made per paper: {citing_papers.mean():.2f}")
    print(f"Median citations made per paper: {citing_papers.median():.2f}")
    print(f"Max citations made by a paper: {citing_papers.max()}")
    
    print("\nCited patterns (papers being cited):")
    print(f"Mean citations received per paper: {cited_papers.mean():.2f}")
    print(f"Median citations received per paper: {cited_papers.median():.2f}")
    print(f"Max citations received by a paper: {cited_papers.max()}")
    
    # Snapshot analysis
    total_days = (df['date'].max() - df['date'].min()).days
    print(f"\nTotal timespan: {total_days} days")
    
    def analyze_snapshot_size(size):
        df_sorted = df.sort_values('date')
        num_complete = len(df) // size
        remainder = len(df) % size
        time_points = []
        
        for i in range(num_complete):
            snapshot = df_sorted.iloc[i*size:(i+1)*size]
            time_points.append((snapshot['date'].min(), snapshot['date'].max()))
        
        if remainder:
            last_snapshot = df_sorted.iloc[num_complete*size:]
            time_points.append((last_snapshot['date'].min(), last_snapshot['date'].max()))
        
        return time_points
    
    # Test different snapshot sizes
    print("\n=== Snapshot Size Analysis ===")
    for size in [1000, 1500, 2000, 2500]:
        time_points = analyze_snapshot_size(size)
        print(f"\nSnapshot size: {size}")
        print(f"Number of snapshots: {len(time_points)}")
        print("Snapshot time spans:")
        for i, (start, end) in enumerate(time_points, 1):
            span_days = (end - start).days
            print(f"  Snapshot {i}: {start.date()} to {end.date()} ({span_days} days)")
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Citations per year
    plt.subplot(2, 2, 1)
    yearly_citations.plot(kind='bar')
    plt.title('Citations per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Citations')
    
    # Citations per month
    plt.subplot(2, 2, 2)
    monthly_avg.plot(kind='bar')
    plt.title('Average Citations per Month')
    plt.xlabel('Month')
    plt.ylabel('Average Citations')
    
    # Distribution of citations made
    plt.subplot(2, 2, 3)
    plt.hist(citing_papers.values, bins=50, log=True)
    plt.title('Distribution of Citations Made (log scale)')
    plt.xlabel('Number of Citations Made')
    plt.ylabel('Count')
    
    # Distribution of citations received
    plt.subplot(2, 2, 4)
    plt.hist(cited_papers.values, bins=50, log=True)
    plt.title('Distribution of Citations Received (log scale)')
    plt.xlabel('Number of Citations Received')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_citation_dataset()