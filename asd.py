import torch
from torch_geometric.datasets import Planetoid
import networkx as nx
import matplotlib.pyplot as plt

# Load the CORA dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Select the first 50 nodes and filter edges accordingly
num_nodes = 50
subgraph_nodes = set(range(num_nodes))  # Nodes 0 to 49
subgraph_edges = [(src, dst) for src, dst in data.edge_index.T.tolist()
                  if src in subgraph_nodes and dst in subgraph_nodes]

# Create a NetworkX graph
G = nx.DiGraph()
G.add_nodes_from(subgraph_nodes)
G.add_edges_from(subgraph_edges)

# Add labels (topics) to nodes
labels = {i: f"Paper {i}\nLabel: {data.y[i].item()}" for i in subgraph_nodes}

# Plot the graph
plt.figure(figsize=(12, 8))
pos = nx.circular_layout(G)  # Layout for better visualization
nx.draw(G, pos, with_labels=False, node_size=500, node_color="lightblue", edge_color="black", arrowsize=10)
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="black")
plt.title("CORA Subgraph with 50 Nodes and Edges")
plt.show()
