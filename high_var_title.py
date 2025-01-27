import pandas as pd

#csv files
five_year_df = pd.read_csv(r'./data/raw/five_year.csv')
five_year_node_mapping_df = pd.read_csv(r'./data/mappings/five_year_node_mapping.csv')
edge_score_variations_df = pd.read_csv(r'./anomaly_analysis_epoch_100/edge_score_variations.csv')

# get the top 10 highest variations from edge_score_variations.csv
top_10_variations = edge_score_variations_df.nlargest(10, 'std_dev')

top_10_variations[['fromNode', 'toNode']] = top_10_variations['edge_id'].str.split('-', expand=True)
top_10_variations['fromNode'] = top_10_variations['fromNode'].astype(int)
top_10_variations['toNode'] = top_10_variations['toNode'].astype(int)

node_mapping = five_year_node_mapping_df.set_index('original_id')['modified_id'].to_dict()

titles = []
original_ids = []
variances = []
relative_ranks = []

# maximum variance
max_variance = top_10_variations['std_dev'].max()

# lookup the titles and original_ids for both fromNode and toNode for each edge
for index, row in top_10_variations.iterrows():
    # get the modified_ids from the mapping
    fromNode_modified = row['fromNode']
    toNode_modified = row['toNode']
    
    # lookup original_id from modified_id
    fromNode_original = five_year_node_mapping_df.loc[five_year_node_mapping_df['modified_id'] == fromNode_modified, 'original_id'].values
    toNode_original = five_year_node_mapping_df.loc[five_year_node_mapping_df['modified_id'] == toNode_modified, 'original_id'].values

    # try to find the title for fromNode first
    title_found = None
    original_id_found = None
    if fromNode_original.size > 0:
        fromNode_title = five_year_df[five_year_df['fromNode'] == fromNode_original[0]]['title'].values
        if fromNode_title.size > 0:
            title_found = fromNode_title[0]
            original_id_found = fromNode_original[0]

    # if not found for fromNode, try for toNode
    if not title_found and toNode_original.size > 0:
        toNode_title = five_year_df[five_year_df['fromNode'] == toNode_original[0]]['title'].values
        if toNode_title.size > 0:
            title_found = toNode_title[0]
            original_id_found = toNode_original[0]

    titles.append(title_found)
    original_ids.append(original_id_found)
    variances.append(row['std_dev'])
    
   
    if max_variance > 0:
        relative_rank = row['std_dev'] / max_variance
    else:
        relative_rank = 0
    relative_ranks.append(relative_rank)

top_10_variations['title'] = titles
top_10_variations['original_id'] = original_ids
top_10_variations['variance'] = variances
top_10_variations['relative_rank'] = relative_ranks

result_df = top_10_variations[['edge_id', 'title', 'original_id', 'variance', 'relative_rank']]

# save result
result_df.to_csv('top_10_highest_variations.csv', index=False)

print("Process complete. The result has been saved to 'top_10_highest_variations.csv'.")
