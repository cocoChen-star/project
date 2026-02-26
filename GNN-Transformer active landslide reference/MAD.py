# import torch
# from sklearn.metrics import pairwise_distances
# import numpy as np
# from torch_geometric.data import Data
# import torch.nn.functional as F
# import pandas as pd
# from torch_sparse import SparseTensor
# import os
#
# def get_n_hop_neighbors_sparse(adj, target_node, n):
#     """Find n-hop neighbors of a target node using sparse adjacency matrix."""
#     neighbors = {target_node}
#     current_neighbors = {target_node}
#     for _ in range(n):
#         new_neighbors = set()
#         row, col, _ = adj.coo()
#         for node in current_neighbors:
#             new_neighbors.update(col[row == node].tolist())
#         neighbors.update(new_neighbors)
#         current_neighbors = new_neighbors
#     return neighbors
#
# def get_neighbors_in_hop_range(adj, target_node, min_hop, max_hop, feature):
#     """Get neighbors within a specified hop range.
#
#     Args:
#         adj (SparseTensor): Sparse adjacency matrix.
#         target_node (int): Target node.
#         min_hop (int): Minimum hop (inclusive).
#         max_hop (int): Maximum hop (inclusive).
#         feature (DataFrame): Node features with 'ID' column.
#
#     Returns:
#         list: Neighbors within the hop range.
#     """
#     current_neighbors = {target_node}
#     all_neighbors = set()
#     visited = {target_node}
#     for hop in range(1, max_hop + 1):
#         new_neighbors = set()
#         row, col, _ = adj.coo()
#         for node in current_neighbors:
#             new_neighbors.update(col[row == node].tolist())
#         new_neighbors = new_neighbors - visited
#         visited.update(new_neighbors)
#         current_neighbors = new_neighbors
#         if hop >= min_hop:
#             all_neighbors.update(new_neighbors)
#     neighbors_list = list(all_neighbors)
#     valid_ids = set(feature['ID'])
#     filtered_neighbors = [id for id in neighbors_list if id in valid_ids]
#     missing_ids = [id for id in neighbors_list if id not in valid_ids]
#     if missing_ids:
#         print(f"Nodes not found in dataset, removed: {missing_ids}")
#     return filtered_neighbors  # Do not include target node
#
# def build_neighbor_matrix(adj, neighbors):
#     """Build a sparse subgraph adjacency matrix for given neighbors."""
#     neighbors_set = torch.tensor(neighbors, dtype=torch.long)
#     sub_adj = adj.index_select(0, neighbors_set).index_select(1, neighbors_set).coalesce()
#     return sub_adj
#
# def save_subgraph_adj_to_csv(sub_adj, filename):
#     """Save sparse subgraph adjacency matrix as CSV."""
#     row, col, _ = sub_adj.coo()
#     edge_list = torch.stack([row, col], dim=1)
#     edge_df = pd.DataFrame(edge_list.numpy(), columns=["source", "target"])
#     os.makedirs(os.path.dirname(filename), exist_ok=True)
#     edge_df.to_csv(filename, index=False)
#     print(f"Subgraph adjacency matrix saved to {filename}")
#
# def mad_value_with_sparse(sub_adj, feature, neighbors, target_idx=None, distance_metric='cosine', digt_num=4):
#     """Calculate MAD using sparse adjacency matrix and sorted features."""
#     neighbors_tensor = torch.tensor(neighbors)
#     filtered_df = feature[feature['ID'].isin(neighbors_tensor.numpy())]
#     filtered_df_sorted = filtered_df.set_index('ID').loc[neighbors_tensor.numpy()].reset_index()
#     node_features = torch.tensor(filtered_df_sorted.iloc[:, 1:].to_numpy(), dtype=torch.float32)
#     if distance_metric == 'cosine':
#         norms = F.normalize(node_features, p=2, dim=1)
#         norms = torch.round(norms * 1e8) / 1e8
#         dist_arr = 1 - torch.matmul(norms, norms.T)
#         dist_arr = torch.round(dist_arr * 1e8) / 1e8
#     else:
#         dist_arr = torch.tensor(pairwise_distances(node_features.cpu().numpy(), metric=distance_metric))
#     row, col, _ = sub_adj.coo()
#     mask_dist = torch.zeros_like(dist_arr)
#     mask_dist[row, col] = dist_arr[row, col]
#     divide_arr = (mask_dist != 0).sum(1) + 1e-8
#     node_dist = mask_dist.sum(1) / divide_arr
#     mad = round(torch.mean(node_dist).item(), digt_num)
#     return mad
#
# if __name__ == '__main__':
#     # Configuration
#     edges_file = "../Data/YRB_edges.csv"
#     center_nodes = [1031648, 959988, 685134, 1533427]
#     hop_ranges = [(0, 3), (10, 15)]  # Hop ranges for MAD calculation
#     output_file = "../Results/mad_gap.csv"
#     graph_models = ["GCN", "GAT", "SAGE", "Proposed"]  # Models for 10-15 hops and MADGap
#
#     # Load edges
#     if not os.path.exists(edges_file):
#         raise FileNotFoundError(f"Edges file not found: {edges_file}")
#     edges_df = pd.read_csv(edges_file, header=None).sort_values(by=1)
#     E = torch.tensor(edges_df.iloc[:, 0].values, dtype=torch.long)
#     G = torch.tensor(edges_df.iloc[:, 1].values, dtype=torch.long)
#     edge_index = torch.stack([E, G], dim=0)
#     edge_index = torch.cat([edge_index, torch.stack([G, E], dim=0)], dim=1)  # Undirected graph
#
#     # Process each model
#     models = [
#         {"name": "SVM", "file": "../Data/MAD/SVM_features.csv"},
#         {"name": "RF", "file": "../Data/MAD/RF_features.csv"},
#         {"name": "MLP", "file": "../Data/MAD/MLP_features.csv"},
#         {"name": "GCN", "file": "../Data/MAD/GCN_features.csv"},
#         {"name": "GAT", "file": "../Data/MAD/GAT_features.csv"},
#         {"name": "SAGE", "file": "../Data/MAD/SAGE_features.csv"},
#         {"name": "Proposed", "file": "../Data/MAD/Proposed_features.csv"},
#     ]
#
#     # Store results for CSV output
#     results = []
#
#     # Process each center node for 0-3 hop subgraph
#     for target_node in center_nodes:
#         # Use a temporary feature set for subgraph computation (first model's features)
#         temp_content_file = models[0]["file"]
#         if not os.path.exists(temp_content_file):
#             print(f"Warning: Feature file for {models[0]['name']} not found: {temp_content_file}")
#             continue
#         temp_content = pd.read_csv(temp_content_file)
#         temp_content.rename(columns={temp_content.columns[0]: "ID"}, inplace=True)
#         temp_num_nodes = temp_content.shape[0]
#
#         # Filter edge_index for temporary adjacency matrix
#         temp_valid_ids = set(temp_content['ID'])
#         temp_valid_ids_tensor = torch.tensor(list(temp_valid_ids), dtype=torch.long)
#         temp_mask = (edge_index[0] < temp_num_nodes) & (edge_index[1] < temp_num_nodes) & torch.isin(edge_index[0], temp_valid_ids_tensor) & torch.isin(edge_index[1], temp_valid_ids_tensor)
#         temp_filtered_edge_index = edge_index[:, temp_mask]
#         temp_adj = SparseTensor.from_edge_index(temp_filtered_edge_index, sparse_sizes=(temp_num_nodes, temp_num_nodes))
#
#         # Get 0-3 hop neighbors and save subgraph
#         neighbors_0_3 = get_neighbors_in_hop_range(temp_adj, target_node, 0, 3, temp_content)
#         if neighbors_0_3:
#             sub_adj_0_3 = build_neighbor_matrix(temp_adj, neighbors_0_3)
#             subgraph_file = f"../Data/MAD/subgraph_adj_{target_node}.csv"
#             save_subgraph_adj_to_csv(sub_adj_0_3, subgraph_file)
#         else:
#             print(f"No 0-3 hop neighbors found for center node {target_node}")
#
#     # Process each model for MAD calculations
#     for model in models:
#         content_file = model["file"]
#         model_name = model["name"]
#         if not os.path.exists(content_file):
#             print(f"Warning: Feature file for {model_name} not found: {content_file}")
#             continue
#
#         # Load features
#         content = pd.read_csv(content_file)
#         content.rename(columns={content.columns[0]: "ID"}, inplace=True)
#         num_nodes = content.shape[0]
#
#         # Filter edge_index to valid nodes
#         valid_ids = set(content['ID'])
#         valid_ids_tensor = torch.tensor(list(valid_ids), dtype=torch.long)
#         mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes) & torch.isin(edge_index[0], valid_ids_tensor) & torch.isin(edge_index[1], valid_ids_tensor)
#         filtered_edge_index = edge_index[:, mask]
#         print(f"{model_name}: Original edge_index shape: {edge_index.shape}, Filtered edge_index shape: {filtered_edge_index.shape}")
#
#         # Build adjacency matrix
#         adj = SparseTensor.from_edge_index(filtered_edge_index, sparse_sizes=(num_nodes, num_nodes))
#
#         # Process each center node
#         for target_node in center_nodes:
#             mad_values = {}
#             for min_hop, max_hop in hop_ranges:
#                 # Skip 10-15 hops for non-graph models
#                 if (min_hop, max_hop) == (10, 15) and model_name not in graph_models:
#                     mad_values["10-15"] = None
#                     continue
#
#                 # Get neighbors
#                 neighbors = get_neighbors_in_hop_range(adj, target_node, min_hop, max_hop, content)
#                 print(f"{model_name} hop range [{min_hop}, {max_hop}] neighbors (center node {target_node}): {neighbors}")
#
#                 if not neighbors:
#                     print(f"No neighbors found for {model_name}, center node {target_node}, hop range [{min_hop}, {max_hop}]")
#                     mad_values[f"{min_hop}-{max_hop}"] = None
#                     continue
#
#                 # Build subgraph
#                 sub_adj = build_neighbor_matrix(adj, neighbors)
#
#                 # Calculate MAD
#                 mad = mad_value_with_sparse(sub_adj, content, neighbors)
#                 mad_values[f"{min_hop}-{max_hop}"] = mad
#                 print(f"{model_name} hop range [{min_hop}, {max_hop}] MAD value (center node {target_node}): {mad}")
#
#             # Calculate MAD difference (10-15 minus 0-3) for graph models only
#             mad_0_3 = mad_values.get("0-3")
#             mad_10_15 = mad_values.get("10-15")
#             mad_diff = None if (mad_0_3 is None or mad_10_15 is None or model_name not in graph_models) else round(mad_10_15 - mad_0_3, 4)
#             if mad_diff is not None:
#                 print(f"{model_name} MADGap (10-15 minus 0-3) for center node {target_node}: {mad_diff}")
#
#             # Store result
#             results.append({
#                 "Model": model_name,
#                 "Center_Node": target_node,
#                 "MAD_0-3": mad_0_3,
#                 "MAD_10-15": mad_10_15,
#                 "MAD_Diff": mad_diff
#             })
#
#     # Save results to CSV
#     if results:
#         os.makedirs(os.path.dirname(output_file), exist_ok=True)
#         results_df = pd.DataFrame(results)
#         results_df.to_csv(output_file, index=False)
#         print(f"MAD results saved to {output_file}")
#     else:
#         print("No results to save.")
#
#




import torch
from sklearn.metrics import pairwise_distances
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
import pandas as pd
from torch_sparse import SparseTensor
import os

def get_n_hop_neighbors_sparse(adj, target_node, n):
    """Find n-hop neighbors of a target node using sparse adjacency matrix."""
    neighbors = {target_node}
    current_neighbors = {target_node}
    for _ in range(n):
        new_neighbors = set()
        row, col, _ = adj.coo()
        for node in current_neighbors:
            new_neighbors.update(col[row == node].tolist())
        neighbors.update(new_neighbors)
        current_neighbors = new_neighbors
    return neighbors

def get_neighbors_in_hop_range(adj, target_node, min_hop, max_hop, feature):
    """Get neighbors within a specified hop range.

    Args:
        adj (SparseTensor): Sparse adjacency matrix.
        target_node (int): Target node.
        min_hop (int): Minimum hop (inclusive).
        max_hop (int): Maximum hop (inclusive).
        feature (DataFrame): Node features with 'ID' column.

    Returns:
        list: Neighbors within the hop range, excluding the target node.
    """
    current_neighbors = {target_node}
    all_neighbors = set()
    visited = {target_node}
    for hop in range(1, max_hop + 1):
        new_neighbors = set()
        row, col, _ = adj.coo()
        for node in current_neighbors:
            new_neighbors.update(col[row == node].tolist())
        new_neighbors = new_neighbors - visited
        visited.update(new_neighbors)
        current_neighbors = new_neighbors
        if hop >= min_hop:
            all_neighbors.update(new_neighbors)
    neighbors_list = list(all_neighbors)
    valid_ids = set(feature['ID'])
    filtered_neighbors = [id for id in neighbors_list if id in valid_ids]
    missing_ids = [id for id in neighbors_list if id not in valid_ids]
    if missing_ids:
        print(f"Nodes not found in dataset, removed: {missing_ids}")
    return filtered_neighbors

def build_neighbor_matrix(adj, neighbors):
    """Build a sparse subgraph adjacency matrix for given neighbors."""
    neighbors_set = torch.tensor(neighbors, dtype=torch.long)
    sub_adj = adj.index_select(0, neighbors_set).index_select(1, neighbors_set).coalesce()
    return sub_adj

def save_subgraph_adj_to_csv(sub_adj, filename):
    """Save sparse subgraph adjacency matrix as CSV."""
    row, col, _ = sub_adj.coo()
    edge_list = torch.stack([row, col], dim=1)
    edge_df = pd.DataFrame(edge_list.numpy(), columns=["source", "target"])
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    edge_df.to_csv(filename, index=False)
    print(f"Subgraph adjacency matrix saved to {filename}")

def mad_value_with_sparse(sub_adj, feature, neighbors, target_idx=None, distance_metric='cosine', digt_num=4):
    """Calculate MAD using sparse adjacency matrix and sorted features."""
    neighbors_tensor = torch.tensor(neighbors)
    filtered_df = feature[feature['ID'].isin(neighbors_tensor.numpy())]
    filtered_df_sorted = filtered_df.set_index('ID').loc[neighbors_tensor.numpy()].reset_index()
    node_features = torch.tensor(filtered_df_sorted.iloc[:, 1:].to_numpy(), dtype=torch.float32)
    if distance_metric == 'cosine':
        norms = F.normalize(node_features, p=2, dim=1)
        norms = torch.round(norms * 1e8) / 1e8
        dist_arr = 1 - torch.matmul(norms, norms.T)
        dist_arr = torch.round(dist_arr * 1e8) / 1e8
    else:
        dist_arr = torch.tensor(pairwise_distances(node_features.cpu().numpy(), metric=distance_metric))
    row, col, _ = sub_adj.coo()
    mask_dist = torch.zeros_like(dist_arr)
    mask_dist[row, col] = dist_arr[row, col]
    divide_arr = (mask_dist != 0).sum(1) + 1e-8
    node_dist = mask_dist.sum(1) / divide_arr
    mad = round(torch.mean(node_dist).item(), digt_num)
    return mad

def mad_value_with_sparse_origin(sub_adj, feature, neighbors, target_idx=None, distance_metric='cosine', digt_num=4):
    """Calculate MAD for origin graph using sparse adjacency matrix and sorted features."""
    neighbors_tensor = torch.tensor(neighbors)
    filtered_df = feature[feature['ID'].isin(neighbors_tensor.numpy())]
    filtered_df_sorted = filtered_df.set_index('ID').loc[neighbors_tensor.numpy()].reset_index()
    node_features = torch.tensor(filtered_df_sorted.iloc[:, 1:-2].to_numpy(), dtype=torch.float32)
    if distance_metric == 'cosine':
        norms = F.normalize(node_features, p=2, dim=1)
        dist_arr = 1 - torch.matmul(norms, norms.T)
    else:
        dist_arr = torch.tensor(pairwise_distances(node_features.cpu().numpy(), metric=distance_metric))
    row, col, _ = sub_adj.coo()
    mask_dist = torch.zeros_like(dist_arr)
    mask_dist[row, col] = dist_arr[row, col]
    divide_arr = (mask_dist != 0).sum(1) + 1e-8
    node_dist = mask_dist.sum(1) / divide_arr
    mad = round(torch.mean(node_dist).item(), digt_num)
    return mad

if __name__ == '__main__':
    # Configuration
    edges_file = "../Data/YRB_edges.csv"
    content_file = "../Data/YRB_contents.csv"
    center_nodes = [1031648, 959988, 685134, 1533427]
    hop_ranges = [(0, 3), (10, 15)]  # Hop ranges for MAD calculation
    output_file = "../Results/mad_gap.csv"
    graph_models = ["GCN", "GAT", "SAGE", "Proposed"]  # Models for 10-15 hops and MADGap

    # Load edges
    if not os.path.exists(edges_file):
        raise FileNotFoundError(f"Edges file not found: {edges_file}")
    edges_df = pd.read_csv(edges_file, header=None).sort_values(by=1)
    E = torch.tensor(edges_df.iloc[:, 0].values, dtype=torch.long)
    G = torch.tensor(edges_df.iloc[:, 1].values, dtype=torch.long)
    edge_index = torch.stack([E, G], dim=0)
    edge_index = torch.cat([edge_index, torch.stack([G, E], dim=0)], dim=1)  # Undirected graph

    # Initialize results list
    results = []

    # Process origin graph (0-3 hops only, no subgraph saving)
    if not os.path.exists(content_file):
        print(f"Warning: Feature file for origin graph not found: {content_file}")
    else:
        # Load origin features
        content = pd.read_csv(content_file, index_col=0)
        content.rename(columns={content.columns[0]: "ID"}, inplace=True)
        num_nodes = content.shape[0]

        # Build adjacency matrix
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))

        # Process each center node for origin graph
        for target_node in center_nodes:
            if target_node not in content['ID'].values:
                print(f"Error: Target node {target_node} not found in origin feature dataset IDs")
                results.append({
                    "Model": "Origin",
                    "Center_Node": target_node,
                    "MAD_0-3": None,
                    "MAD_10-15": None,
                    "MAD_Diff": None
                })
                continue

            # Get 0-3 hop neighbors
            neighbors_0_3 = get_neighbors_in_hop_range(adj, target_node, 0, 3, content)
            print(f"Origin hop range [0, 3] neighbors (center node {target_node}): {neighbors_0_3}")

            if not neighbors_0_3:
                print(f"No 0-3 hop neighbors found for origin, center node {target_node}")
                results.append({
                    "Model": "Origin",
                    "Center_Node": target_node,
                    "MAD_0-3": None,
                    "MAD_10-15": None,
                    "MAD_Diff": None
                })
                continue

            # Build subgraph (not saved)
            sub_adj = build_neighbor_matrix(adj, neighbors_0_3)

            # Calculate MAD for 0-3 hops
            mad_0_3 = mad_value_with_sparse_origin(sub_adj, content, neighbors_0_3)
            print(f"Origin hop range [0, 3] MAD value (center node {target_node}): {mad_0_3}")

            # Store result
            results.append({
                "Model": "Origin",
                "Center_Node": target_node,
                "MAD_0-3": mad_0_3,
                "MAD_10-15": None,
                "MAD_Diff": None
            })

    # Process other models
    models = [
        {"name": "SVM", "file": "../Data/MAD/SVM_features.csv"},
        {"name": "RF", "file": "../Data/MAD/RF_features.csv"},
        {"name": "MLP", "file": "../Data/MAD/MLP_features.csv"},
        {"name": "GCN", "file": "../Data/MAD/GCN_features.csv"},
        {"name": "GAT", "file": "../Data/MAD/GAT_features.csv"},
        {"name": "SAGE", "file": "../Data/MAD/SAGE_features.csv"},
        {"name": "Proposed", "file": "../Data/MAD/Proposed_features.csv"},
    ]

    # Process each center node for 0-3 hop subgraph
    for target_node in center_nodes:
        temp_content_file = models[0]["file"]
        if not os.path.exists(temp_content_file):
            print(f"Warning: Feature file for {models[0]['name']} not found: {temp_content_file}")
            continue
        temp_content = pd.read_csv(temp_content_file)
        temp_content.rename(columns={temp_content.columns[0]: "ID"}, inplace=True)
        temp_num_nodes = temp_content.shape[0]

        # Filter edge_index for temporary adjacency matrix
        temp_valid_ids = set(temp_content['ID'])
        temp_valid_ids_tensor = torch.tensor(list(temp_valid_ids), dtype=torch.long)
        temp_mask = (edge_index[0] < temp_num_nodes) & (edge_index[1] < temp_num_nodes) & torch.isin(edge_index[0], temp_valid_ids_tensor) & torch.isin(edge_index[1], temp_valid_ids_tensor)
        temp_filtered_edge_index = edge_index[:, temp_mask]
        temp_adj = SparseTensor.from_edge_index(temp_filtered_edge_index, sparse_sizes=(temp_num_nodes, temp_num_nodes))

        if target_node not in temp_content['ID'].values:
            print(f"Error: Target node {target_node} not found in {models[0]['name']} feature dataset IDs")
            continue

        neighbors_0_3 = get_neighbors_in_hop_range(temp_adj, target_node, 0, 3, temp_content)
        if neighbors_0_3:
            sub_adj_0_3 = build_neighbor_matrix(temp_adj, neighbors_0_3)
            subgraph_file = f"../Data/MAD/subgraph_adj_{target_node}.csv"
            save_subgraph_adj_to_csv(sub_adj_0_3, subgraph_file)
        else:
            print(f"No 0-3 hop neighbors found for center node {target_node}")

    # Process each model for MAD calculations
    for model in models:
        content_file = model["file"]
        model_name = model["name"]
        if not os.path.exists(content_file):
            print(f"Warning: Feature file for {model_name} not found: {content_file}")
            continue

        # Load features
        content = pd.read_csv(content_file)
        content.rename(columns={content.columns[0]: "ID"}, inplace=True)
        num_nodes = content.shape[0]

        # Filter edge_index to valid nodes
        valid_ids = set(content['ID'])
        valid_ids_tensor = torch.tensor(list(valid_ids), dtype=torch.long)
        mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes) & torch.isin(edge_index[0], valid_ids_tensor) & torch.isin(edge_index[1], valid_ids_tensor)
        filtered_edge_index = edge_index[:, mask]
        print(f"{model_name}: Original edge_index shape: {edge_index.shape}, Filtered edge_index shape: {filtered_edge_index.shape}")

        # Build adjacency matrix
        adj = SparseTensor.from_edge_index(filtered_edge_index, sparse_sizes=(num_nodes, num_nodes))

        for target_node in center_nodes:
            if target_node not in content['ID'].values:
                print(f"Error: Target node {target_node} not found in {model_name} feature dataset IDs")
                results.append({
                    "Model": model_name,
                    "Center_Node": target_node,
                    "MAD_0-3": None,
                    "MAD_10-15": None,
                    "MAD_Diff": None
                })
                continue

            mad_values = {}
            for min_hop, max_hop in hop_ranges:
                if (min_hop, max_hop) == (10, 15) and model_name not in graph_models:
                    mad_values["10-15"] = None
                    continue

                neighbors = get_neighbors_in_hop_range(adj, target_node, min_hop, max_hop, content)
                print(f"{model_name} hop range [{min_hop}, {max_hop}] neighbors (center node {target_node}): {neighbors}")

                if not neighbors:
                    print(f"No neighbors found for {model_name}, center node {target_node}, hop range [{min_hop}, {max_hop}]")
                    mad_values[f"{min_hop}-{max_hop}"] = None
                    continue

                sub_adj = build_neighbor_matrix(adj, neighbors)
                mad = mad_value_with_sparse(sub_adj, content, neighbors)
                mad_values[f"{min_hop}-{max_hop}"] = mad
                print(f"{model_name} hop range [{min_hop}, {max_hop}] MAD value (center node {target_node}): {mad}")

            mad_0_3 = mad_values.get("0-3")
            mad_10_15 = mad_values.get("10-15")
            mad_diff = None if (mad_0_3 is None or mad_10_15 is None or model_name not in graph_models) else round(mad_10_15 - mad_0_3, 4)
            if mad_diff is not None:
                print(f"{model_name} MADGap (10-15 minus 0-3) for center node {target_node}: {mad_diff}")

            results.append({
                "Model": model_name,
                "Center_Node": target_node,
                "MAD_0-3": mad_0_3,
                "MAD_10-15": mad_10_15,
                "MAD_Diff": mad_diff
            })

    # Save results to CSV
    if results:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"MAD results saved to {output_file}")
    else:
        print("No results to save.")



