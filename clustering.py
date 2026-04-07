import pandas as pd
import networkx as nx
import os

def build_connected_components(df_matches):
    """Groups all connected nodes into clusters."""
    G = nx.Graph()
    for index, row in df_matches.iterrows():
        G.add_edge(row['id_A'], row['id_B'], weight=row['similarity_score'])

    clusters = list(nx.connected_components(G))
    return _format_clusters(clusters)

def build_center_clustering(df_matches):
    """Groups nodes by picking a highly-connected center and its direct neighbors."""
    G = nx.Graph()
    for index, row in df_matches.iterrows():
        G.add_edge(row['id_A'], row['id_B'], weight=row['similarity_score'])

    clusters = []
    H = G.copy() # Work on a copy so we can remove nodes as we cluster them

    while H.edges():
        # 1. Find the node with the highest degree (most connections)
        degrees = dict(H.degree())
        center_node = max(degrees, key=degrees.get)

        # 2. Get the center and its direct neighbors
        cluster_nodes = [center_node] + list(H.neighbors(center_node))
        clusters.append(cluster_nodes)

        # 3. Remove these nodes from the graph so they aren't clustered twice
        H.remove_nodes_from(cluster_nodes)

    # 4. Any remaining isolated nodes get their own individual cluster
    for node in list(H.nodes()):
        clusters.append([node])

    return _format_clusters(clusters)

def _format_clusters(clusters):
    """Helper function to format the output DataFrame."""
    cluster_records = []
    for cluster_id, node_set in enumerate(clusters):
        for node in node_set:
            cluster_records.append({
                'cluster_id': cluster_id,
                'entity_id': node
            })
    return pd.DataFrame(cluster_records)

# --- Let's test both methods ---
file_path = "output/mock/amazon_google/match_results_jaccard.csv"

if os.path.exists(file_path):
    # Load the data once and filter for matches
    df = pd.read_csv(file_path)
    df_matches = df[df['is_match'] == 1]
    
    print("\n--- 1. Connected Components Output ---")
    df_cc = build_connected_components(df_matches)
    print(df_cc)

    print("\n--- 2. Center Clustering Output ---")
    df_center = build_center_clustering(df_matches)
    print(df_center)
    
    # Uncomment below to save your final choice! 
    # df_cc.to_csv("output/mock/amazon_google/clusters.csv", index=False)
else:
    print(f"❌ Could not find {file_path}.")