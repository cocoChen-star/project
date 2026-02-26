import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import xlrd
import math
import random
import os

def load_data_S2(data='YRB', path="../Data/"):
    """Load and process YRB dataset to determine node connections."""
    content_filename = f"{path}{data}_contents_raw.txt"
    print(f"Reading content: {data}")
    content = pd.read_csv(content_filename)

    colname_list = ['ID', 'dem', 'ndvi', 'rain_max', 'slope', 'twi',
                    'water_lake', 'water_line', 'fault', 'roads', 'aspect', 'landuse', 'lithology',
                    'soiltype', 'label']
    if data == 'YRB':
        col = ['VALUE', 'dem', 'ndvi', 'rain_max', 'slope',
               'twi', 'water_lake', 'water_line', 'fault', 'roads', 'aspect', 'landuse', 'lithology',
               'soiltype', 'landslide']
        content = content.loc[:, col]
        content.columns = colname_list

    print(f"Modifying data: {data}")
    content = content.dropna(subset=colname_list[1:])
    content.loc[(content['label'] == 1) & (content['slope'] < 11), 'slope'] = 10

    print(f"Creating graph: {data}")
    idl = np.where(content['label'] > 0)[0]  # Labeled samples
    idn = np.where(content['label'] == 0)[0].flatten()  # Unlabeled samples
    np.random.shuffle(idn)

    # Build feature data
    col_choose = colname_list
    if idl.shape[0] == 0:
        print('No labeled data')
        return None
    features = content[col_choose]
    features.index = range(len(features))
    os.makedirs("../Data/Graph", exist_ok=True)
    features.to_csv(f"../Data/Graph/{data}_normal.csv")  # Save as CSV data

    # Map features using Excel table
    features, cw_dict = data_excle(features, path=f"{path}YRB_feature_table.xls")
    features = features.astype(np.int32)

    # Generate one-hot encodings and edge/node features
    f_onehot, f_edge, f_node = edge_creat1(features, cw_dict)

    # Batch process edge creation
    batch_size = 6000
    node_num = features.shape[0]
    batch_start = 0
    while batch_start < node_num:
        batch_end = min(batch_start + batch_size, node_num)
        batch_start = edge_creat2(f_edge, f_node, path="../Data/Graph/", data=data, batch_start=batch_start, batch_end=batch_end, node_num=node_num)
        if batch_end == node_num:
            break

    features['ENV'] = f_onehot['ENV']
    features.to_csv(f"../Data/Graph/{data}_class.csv")
    print(f"Saving graph: {data}")
    outF = pd.concat([features['ID'], f_onehot, features['label']], axis=1)
    outF.to_csv(f"../Data/Graph/{data}_contents.csv")

    # Note: Merge edge files and create graph object for training
    return None

def encode_onehot(features, cw_list):
    """Generate one-hot encodings and edge/node features."""
    node_num = features.shape[0]
    f_onehot = np.zeros([node_num, 1], dtype=int)
    f_edge = np.zeros([node_num, 1], dtype=int)
    f_node = np.zeros([node_num, 1], dtype=int)
    e_class = np.zeros([node_num, 1], dtype=int)
    list_edgeclass = ['water_lake', 'water_line', 'fault', 'slope', 'lithology', 'soiltype']
    onehot_name = []
    w_edge = []
    w_node = []
    for k, cw in cw_list.items():
        classes_dict = {c: (np.identity(len(cw.keys()))[c - 1, :]) for c, w in cw.items()}
        labels_onehot = np.array(list(map(classes_dict.get, features[k])), dtype=np.float64)
        f_onehot = np.c_[f_onehot, labels_onehot]
        onehot_name = onehot_name + list(f"{k}{i}" for i in np.array(list(cw.keys()), dtype=str))
        if k in list_edgeclass:
            for i in sorted(cw):
                w_edge.append(cw[i])
            f_edge = np.c_[f_edge, labels_onehot]
        else:
            for i in sorted(cw):
                w_node.append(cw[i])
            f_node = np.c_[f_node, labels_onehot]
    f_onehot = pd.DataFrame(f_onehot[:, 1:], columns=onehot_name)
    f_edge = f_edge[:, 1:] * w_edge

    # Environment classification
    edge_pd = pd.DataFrame(f_edge)
    edge_gp = edge_pd.groupby(edge_pd.columns.to_list())
    i = 0
    for key, value in edge_gp.indices.items():
        eachEnv = value.tolist()
        e_class[eachEnv] = i
        i += 1
    f_onehot['ENV'] = e_class
    return f_onehot

def edge_creat1(features, cw_list):
    """Generate one-hot encodings and weighted edge/node features."""
    node_num = features.shape[0]
    f_onehot = np.zeros([node_num, 1], dtype=int)
    f_edge = np.zeros([node_num, 1], dtype=int)
    f_node = np.zeros([node_num, 1], dtype=int)
    e_class = np.zeros([node_num, 1], dtype=int)
    list_edgeclass = ['water_lake', 'water_line', 'fault', 'slope', 'lithology', 'soiltype']
    onehot_name = []
    w_edge = []
    w_node = []
    for k, cw in cw_list.items():
        classes_dict = {c: (np.identity(len(cw.keys()))[c - 1, :]) for c, w in cw.items()}
        labels_onehot = np.array(list(map(classes_dict.get, features[k])), dtype=np.float64)
        f_onehot = np.c_[f_onehot, labels_onehot]
        onehot_name = onehot_name + list(f"{k}{i}" for i in np.array(list(cw.keys()), dtype=str))
        if k in list_edgeclass:
            for i in sorted(cw):
                w_edge.append(cw[i])
            f_edge = np.c_[f_edge, labels_onehot]
        else:
            for i in sorted(cw):
                w_node.append(cw[i])
            f_node = np.c_[f_node, labels_onehot]
    f_onehot = pd.DataFrame(f_onehot[:, 1:], columns=onehot_name)
    f_edge = f_edge[:, 1:] * w_edge
    f_node = f_node[:, 1:] * w_node

    # Environment classification
    edge_pd = pd.DataFrame(f_edge)
    edge_gp = edge_pd.groupby(edge_pd.columns.to_list())
    i = 0
    for key, value in edge_gp.indices.items():
        eachEnv = value.tolist()
        e_class[eachEnv] = i
        i += 1
    f_onehot['ENV'] = e_class
    return f_onehot, f_edge, f_node

def edge_creat2(f_edge, f_node, path, data, batch_start, batch_end, node_num):
    """Create edges for a batch of nodes using similarity constraints."""
    f_edge_tensor = torch.tensor(f_edge, dtype=torch.float32).cuda()
    f_node_tensor = torch.tensor(f_node, dtype=torch.float32).cuda()
    print(f"f_edge: {f_edge_tensor.device}")
    print(f"f_node: {f_node_tensor.device}")

    edges = torch.empty(0, f_edge.shape[1] + 4, dtype=torch.float32).cuda()
    print(f"edges: {edges.device}")
    for self_idx in tqdm(range(batch_start, batch_end), desc="Processing batch"):
        pt = self_idx * 100 / node_num
        print(f"\r{pt:.4f}%", end='')

        # Compute feature similarity
        dist_edge_tensor = 1 - F.cosine_similarity(f_edge_tensor, f_edge_tensor[self_idx].view(1, -1), dim=1)
        dist_node_tensor = 1 - F.cosine_similarity(f_node_tensor, f_node_tensor[self_idx].view(1, -1), dim=1)

        # Edge constraint: similarity < 0.1
        indices = torch.nonzero(dist_edge_tensor < 0.1).squeeze()
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)

        dist_edge_m = torch.cat((indices.unsqueeze(1), dist_edge_tensor[indices].unsqueeze(1)), dim=1)
        unique_values = torch.unique(dist_edge_m[:, 1])
        group_dict = {}
        for value in unique_values:
            indices = dist_edge_m[:, 1] == value
            group_dict[value.item()] = dist_edge_m[indices, 0].tolist()

        if not group_dict:
            print("\nNo connected nodes")
            continue

        max_edge_num = 10
        closeENid = []
        for key, value in group_dict.items():
            closeEid = list(map(int, value))
            closeEid_tensor = torch.tensor(closeEid).cuda()

            # Node constraint: similarity <= 0.11
            filtered_indices = dist_node_tensor[closeEid_tensor] <= 0.11
            filtered_closeEid_tensor = closeEid_tensor[filtered_indices]

            sorted_indices = torch.argsort(dist_node_tensor[filtered_closeEid_tensor])
            values = filtered_closeEid_tensor[sorted_indices]
            closeENid.extend(values[:max_edge_num - len(closeENid)].cpu().numpy())

            if len(closeENid) >= 8:
                break

        if len(closeENid) < 8:
            continue

        if len(closeENid) < max_edge_num:
            max_edge_num = len(closeENid)
        closeENid_tensor = torch.tensor(closeENid).to(f_edge_tensor.device)

        # Get weights
        weight = f_edge_tensor[closeENid_tensor]
        selflist = torch.ones(max_edge_num, dtype=torch.int) * self_idx
        dist_node_selected = 1 - dist_node_tensor[closeENid_tensor]
        dist_edge_selected = 1 - dist_edge_tensor[closeENid_tensor]

        selflist = selflist.to(closeENid_tensor.device)
        dist_node_selected = dist_node_selected.to(closeENid_tensor.device)
        dist_edge_selected = dist_edge_selected.to(closeENid_tensor.device)
        weight = weight.to(closeENid_tensor.device)

        edge = torch.cat((closeENid_tensor.view(-1, 1), selflist.unsqueeze(1), dist_node_selected.view(-1, 1),
                          dist_edge_selected.view(-1, 1), weight), dim=1)
        edges = torch.cat((edges, edge), dim=0)

    edges_np = edges.cpu().numpy()
    outE = pd.DataFrame(data=edges_np)
    outE.to_csv(f"{path}{data}_edges_{batch_end}.csv", index=False, header=False)
    return batch_end

def data_excle(features, path="../Data/YRB_feature_table.xls"):
    """Map features using an Excel table."""
    data = xlrd.open_workbook(path)
    sheets = data.sheet_names()
    cw_dict = {}
    for column in features:
        if column in data.sheet_names():
            table = data.sheet_by_name(column)
            k = np.array(table.col_values(0), dtype=np.int64)
            v = np.array(table.col_values(1), dtype=np.int64)
            v2 = np.array(table.col_values(2), dtype=np.float64)
            class_w = dict(zip(v, v2))
            index = np.digitize(features[column], k, right=True)
            features[column] = v[index]
            cw_dict[column] = class_w
    return features, cw_dict

if __name__ == '__main__':
    load_data_S2(data='YRB', path="../Data/")