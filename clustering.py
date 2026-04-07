import pandas as pd
import networkx as nx
import os
import numpy as np

def build_connected_components(df_matches):
    G = nx.Graph()
    for index, row in df_matches.iterrows():
        G.add_edge(row['id_A'], row['id_B'], weight=row['similarity_score'])
    clusters = list(nx.connected_components(G))
    return _format_clusters(clusters)

def build_center_clustering(df_matches):
    G = nx.Graph()
    for index, row in df_matches.iterrows():
        G.add_edge(row['id_A'], row['id_B'], weight=row['similarity_score'])
    clusters = []
    H = G.copy()
    
    while H.edges():
        degrees = dict(H.degree())
        center_node = max(degrees, key=degrees.get)
        cluster_nodes = [center_node] + list(H.neighbors(center_node))
        clusters.append(cluster_nodes)
        H.remove_nodes_from(cluster_nodes)
        
    for node in list(H.nodes()):
        clusters.append([node])
        
    return _format_clusters(clusters)

def _format_clusters(clusters):
    cluster_records = []
    for cluster_id, node_set in enumerate(clusters):
        for node in node_set:
            cluster_records.append({
                'cluster_id': cluster_id,
                'entity_id': node
            })
    return pd.DataFrame(cluster_records)

def merge_cluster_attributes(df_clusters, source1_path, source2_path):
    df1 = pd.read_csv(source1_path)
    df2 = pd.read_csv(source2_path)
    df_all_data = pd.concat([df1, df2], ignore_index=True)
    df_merged = df_clusters.merge(df_all_data, left_on='entity_id', right_on='id', how='left')
    canonical_records = []
    
    for cluster_id, group in df_merged.groupby('cluster_id'):
        
        def get_longest_string(col_name):
            valid_strings = group[col_name].dropna().astype(str)
            if valid_strings.empty: return ""
            return max(valid_strings, key=len)
            
        def get_best_price():
            valid_prices = group['price'].replace(0.0, np.nan).dropna()
            if valid_prices.empty: return 0.0
            return valid_prices.iloc[0] 
            
        record = {
            'cluster_id': cluster_id,
            'title': get_longest_string('title'),
            'description': get_longest_string('description'),
            'manufacturer': get_longest_string('manufacturer'),
            'price': get_best_price()
        }
        canonical_records.append(record)
        
    return pd.DataFrame(canonical_records)

file_path = "output/mock/amazon_google/match_results_jaccard.csv"
source1 = "data/cleaned/amazon_google/cleaned_source1.csv"
source2 = "data/cleaned/amazon_google/cleaned_source2.csv"

if os.path.exists(file_path) and os.path.exists(source1):
    df = pd.read_csv(file_path)
    df_matches = df[df['is_match'] == 1]
    
    df_clusters = build_center_clustering(df_matches)
    df_final_entities = merge_cluster_attributes(df_clusters, source1, source2)
    
    print(df_final_entities.head())
    
    df_clusters.to_csv("output/mock/amazon_google/clusters.csv", index=False)
    df_final_entities.to_csv("output/mock/amazon_google/merged_entities.csv", index=False)
    print("\n✅ SUCCESS! clusters.csv and merged_entities.csv have been saved.")
else:
    print("❌ Could not find the required files.")