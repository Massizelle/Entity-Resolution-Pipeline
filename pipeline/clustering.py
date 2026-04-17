import pandas as pd
import os
import numpy as np
import json

from pipeline.progress import ProgressBar

def _build_graph_edges(df_matches, source1_name="s1", source2_name="s2", *, progress_label: str | None = None):
    """
    Build edges using compound node keys (``"<source>:<entity_id>"``) so that
    the same integer ID from different sources is treated as a distinct node.
    Without this, datasets like dblp_acm (where both sources use IDs 0-N)
    would collapse different entities onto the same graph node.
    """
    edges = []
    progress = ProgressBar(progress_label, len(df_matches), color="34", unit_label="matches") if progress_label else None
    for idx, (_, row) in enumerate(df_matches.iterrows(), start=1):
        edges.append(
            (
                f"{source1_name}:{row['id_A']}",
                f"{source2_name}:{row['id_B']}",
                float(row["similarity_score"]),
            )
        )
        if progress:
            progress.update(idx)
    if progress:
        progress.close()
    nodes = set()
    for left, right, _weight in edges:
        nodes.add(left)
        nodes.add(right)
    return nodes, edges


def _connected_components_without_networkx(df_matches, source1_name, source2_name):
    nodes, edges = _build_graph_edges(
        df_matches,
        source1_name,
        source2_name,
        progress_label="  [CLUST] Construction du graphe",
    )
    adjacency = {node: set() for node in nodes}
    for left, right, _weight in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    visited = set()
    clusters = []
    for node in sorted(nodes):
        if node in visited:
            continue
        stack = [node]
        component = []
        visited.add(node)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        clusters.append(component)
    return _format_clusters(clusters)


def build_connected_components(df_matches, source1_name="s1", source2_name="s2"):
    return _connected_components_without_networkx(df_matches, source1_name, source2_name)

def _format_clusters(clusters):
    """
    Convert a list of node-sets into a DataFrame.

    Node values are expected to be compound keys (``"<source>:<entity_id>"``).
    The compound key is split and stored in separate ``entity_id`` and
    ``source`` columns so that downstream code can join unambiguously.
    """
    cluster_records = []
    progress = ProgressBar("  [CLUST] Formatage des clusters", len(clusters), color="34", unit_label="clusters")
    for cluster_id, node_set in enumerate(clusters, start=1):
        for node in node_set:
            node_str = str(node)
            if ":" in node_str:
                source, entity_id = node_str.split(":", 1)
            else:
                source, entity_id = "", node_str
            cluster_records.append({
                "cluster_id": cluster_id - 1,
                "entity_id": entity_id,
                "source": source,
            })
        progress.update(cluster_id)
    progress.close()
    return pd.DataFrame(cluster_records)

def merge_cluster_attributes(df_clusters, source1_path, source2_path):
    """
    Merge entity attributes within each cluster into one canonical record.

    Fully schema-agnostic: columns are discovered from the source CSVs at
    runtime, so this works with any dataset without modification.

    Merging heuristics per column:
      - Numeric columns  → first non-null, non-zero value found in the cluster
      - Text columns     → longest non-empty string found in the cluster
    """
    df1 = pd.read_csv(source1_path)
    df2 = pd.read_csv(source2_path)
    df_clusters = df_clusters.copy()
    df_clusters["entity_id"] = df_clusters["entity_id"].astype(str)
    df1["id"] = df1["id"].astype(str)
    df2["id"] = df2["id"].astype(str)
    df_all_data = pd.concat([df1, df2], ignore_index=True)

    # Columns that are pipeline metadata, not entity attributes
    meta_cols = {'id', 'source', 'ttl_source'}
    attr_cols = [c for c in df_all_data.columns if c not in meta_cols]

    # Detect which attribute columns are predominantly numeric
    numeric_cols: set[str] = set()
    for col in attr_cols:
        converted = pd.to_numeric(df_all_data[col], errors='coerce')
        if converted.notna().sum() > df_all_data[col].notna().sum() * 0.5:
            numeric_cols.add(col)

    if df_clusters.empty:
        return pd.DataFrame(columns=['cluster_id'] + attr_cols)

    # When clusters carry a 'source' column (produced by the compound-key
    # clustering path), join on BOTH entity_id AND source to avoid
    # cross-source ID collisions (e.g. dblp_acm where both sources use 0-N
    # integer IDs).  Fall back to entity_id-only join for legacy files that
    # lack the column.
    has_source_col = "source" in df_clusters.columns and df_clusters["source"].ne("").any()
    if has_source_col:
        df_clusters["source"] = df_clusters["source"].astype(str)
        df_all_data["source"] = df_all_data["source"].astype(str)
        df_joined = df_clusters.merge(
            df_all_data,
            left_on=["entity_id", "source"],
            right_on=["id", "source"],
            how="left",
        )
    else:
        df_joined = df_clusters.merge(df_all_data, left_on="entity_id", right_on="id", how="left")
    canonical_records = []

    grouped = list(df_joined.groupby('cluster_id'))
    progress = ProgressBar("  [MERGE] Fusion des clusters", len(grouped), color="34", unit_label="clusters")
    for idx, (cluster_id, group) in enumerate(grouped, start=1):
        record: dict = {'cluster_id': cluster_id}

        for col in attr_cols:
            if col not in group.columns:
                record[col] = None
                continue

            if col in numeric_cols:
                vals = pd.to_numeric(group[col], errors='coerce').replace(0.0, np.nan).dropna()
                record[col] = vals.iloc[0] if not vals.empty else None
            else:
                vals = (
                    group[col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .replace('', np.nan)
                    .dropna()
                )
                record[col] = max(vals, key=len) if not vals.empty else None

        canonical_records.append(record)
        progress.update(idx)
    progress.close()

    return pd.DataFrame(canonical_records)


def _normalize_matches_for_clustering(df_matches):
    """
    Normalize a match DataFrame to the contract expected by the clustering stage.
    """
    df = df_matches.copy()
    if "final_score" in df.columns and "similarity_score" not in df.columns:
        df = df.rename(columns={"final_score": "similarity_score"})
    required = {"id_A", "id_B", "is_match"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing clustering columns: {missing}")
    if "similarity_score" not in df.columns:
        df["similarity_score"] = 1.0
    df["id_A"] = df["id_A"].astype(str)
    df["id_B"] = df["id_B"].astype(str)
    df["is_match"] = df["is_match"].astype(int)
    return df[df["is_match"] == 1].copy()


def materialize_incremental_clusters(
    matches_df,
    source1_path,
    source2_path,
    output_dir,
    *,
    completed=False,
    processed_rows=0,
    total_rows=0,
    promote_final=False,
):
    """
    Build checkpointed clustering artifacts from the current matching state.

    This function is intentionally simple:
    - it consumes the currently accumulated match results
    - rebuilds clusters from the confirmed-match subgraph
    - writes checkpoint files that can be inspected while Step 3 is still running
    """
    df_matches = _normalize_matches_for_clustering(matches_df)

    # Derive source names from the cleaned source CSVs so compound node keys
    # are consistent with the rest of the pipeline.
    try:
        _s1 = pd.read_csv(source1_path, usecols=["source"], nrows=1)["source"].iloc[0]
        _s2 = pd.read_csv(source2_path, usecols=["source"], nrows=1)["source"].iloc[0]
    except Exception:
        _s1, _s2 = "s1", "s2"

    if df_matches.empty:
        df_clusters = pd.DataFrame(columns=["cluster_id", "entity_id", "source"])
        df_merged = merge_cluster_attributes(df_clusters, source1_path, source2_path)
    else:
        df_clusters = build_connected_components(df_matches, source1_name=_s1, source2_name=_s2)
        df_merged = merge_cluster_attributes(df_clusters, source1_path, source2_path)

    os.makedirs(output_dir, exist_ok=True)
    clusters_incremental = os.path.join(output_dir, "clusters_incremental.csv")
    merged_incremental = os.path.join(output_dir, "merged_entities_incremental.csv")
    status_path = os.path.join(output_dir, "cluster_status.json")

    df_clusters.to_csv(clusters_incremental, index=False)
    df_merged.to_csv(merged_incremental, index=False)

    status_payload = {
        "algorithm": "connected_components",
        "completed": bool(completed),
        "processed_rows": int(processed_rows),
        "total_rows": int(total_rows),
        "confirmed_matches": int(len(df_matches)),
        "n_clusters": int(df_clusters["cluster_id"].nunique()) if not df_clusters.empty else 0,
        "n_merged_entities": int(len(df_merged)),
    }
    with open(status_path, "w", encoding="utf-8") as handle:
        json.dump(status_payload, handle, indent=2)

    if promote_final:
        df_clusters.to_csv(os.path.join(output_dir, "clusters.csv"), index=False)
        df_merged.to_csv(os.path.join(output_dir, "merged_entities.csv"), index=False)

    return {
        "clusters_path": clusters_incremental,
        "merged_path": merged_incremental,
        "status_path": status_path,
        "confirmed_matches": int(len(df_matches)),
        "n_clusters": int(df_clusters["cluster_id"].nunique()) if not df_clusters.empty else 0,
        "n_merged_entities": int(len(df_merged)),
        "completed": bool(completed),
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Member 4 — standalone clustering test")
    parser.add_argument("--dataset", default="amazon_google",
                        help="Dataset key (default: amazon_google)")
    parser.add_argument("--mock", action="store_true",
                        help="Use output/mock/<dataset>/ paths")
    args = parser.parse_args()

    root = f"output/mock/{args.dataset}" if args.mock else f"output/{args.dataset}"
    file_path = f"{root}/match_results_jaccard.csv"
    if args.mock:
        source1 = f"{root}/cleaned_source1.csv"
        source2 = f"{root}/cleaned_source2.csv"
    else:
        source1 = f"data/cleaned/{args.dataset}/cleaned_source1.csv"
        source2 = f"data/cleaned/{args.dataset}/cleaned_source2.csv"

    if not os.path.exists(file_path):
        print(f"Could not find match file: {file_path}")
    elif not os.path.exists(source1) or not os.path.exists(source2):
        print(f"Could not find source files under {root}/")
    else:
        df = pd.read_csv(file_path)
        df_matches = df[df['is_match'] == 1]
        df_clusters = build_connected_components(df_matches)
        df_final_entities = merge_cluster_attributes(df_clusters, source1, source2)
        print(df_final_entities.head())
        df_clusters.to_csv(f"{root}/clusters.csv", index=False)
        df_final_entities.to_csv(f"{root}/merged_entities.csv", index=False)
        print(f"\n SUCCESS! Written to {root}/")
