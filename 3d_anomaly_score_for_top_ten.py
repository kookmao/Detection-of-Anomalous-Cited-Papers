import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

table1_path = 'top_10_edges.csv'  
table2_path = 'anomaly_analysis_epoch_100/edge_score_history.csv' 

# CSV files
table1 = pd.read_csv(table1_path)
table2 = pd.read_csv(table2_path)

# 'edge_id' is treated as a string in both tables to avoid conversion errors
table1['edge_id'] = table1['edge_id'].astype(str)
table2['edge_id'] = table2['edge_id'].astype(str)

filtered_table2 = table2[table2['edge_id'].apply(lambda x: any(iden in x for iden in table1['edge_id']))]
pivot_data = filtered_table2.pivot_table(index='edge_id', columns='sequence', values='anomaly_score', aggfunc='first')

x = []
y = []
z = []
dx = []  # width of the bars
dy = []  # depth of the bars
dz = []  # height of the bars (anomaly score)

# map edge_ids to numeric values (indices)
edge_id_mapping = {edge_id: index for index, edge_id in enumerate(pivot_data.index)}
# map edge_ids to their corresponding titles from table1 (taking the first two words)
edge_id_to_title = {row['edge_id']: ' '.join(row['title'].split()[:2]) for _, row in table1.iterrows()}

# populate the lists for 3D bars
for edge_id in pivot_data.index:
    for timestamp in pivot_data.columns:
        x.append(timestamp)
        y.append(edge_id_mapping[edge_id])  
        z.append(0)  
        dx.append(1)
        dy.append(1)  
        dz.append(pivot_data.at[edge_id, timestamp])

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(x, y, z, dx, dy, dz, color=plt.cm.viridis(dz))  # Color based on anomaly score

# anomaly_score = 0.6
threshold = 0.6

x_plane = np.linspace(min(x), max(x), len(pivot_data.columns))
y_plane = np.linspace(min(y), max(y), len(pivot_data.index))
x_plane, y_plane = np.meshgrid(x_plane, y_plane)


z_plane = np.full_like(x_plane, threshold)

#ax.plot_surface(x_plane, y_plane, z_plane, color='r', alpha=0.2)

ax.set_xlabel('Timestamp', fontsize=14)
#ax.set_ylabel('Edge ID', fontsize=14)   
ax.set_zlabel('Anomaly Score', fontsize=14)  
ax.tick_params(axis='both', which='major', labelsize=12)

y_ticks = list(edge_id_mapping.values())
y_labels = [edge_id_to_title[edge_id] for edge_id in pivot_data.index]
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=10)


plt.tight_layout(pad=5)

#fig.subplots_adjust(top=0.9)  # Adjust the top margin (0.9 means 90% of the figure's height is used)
#plt.show()
plt.savefig("3d.png", dpi=300)
plt.close()