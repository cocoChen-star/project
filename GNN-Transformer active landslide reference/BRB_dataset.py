#import torch_geometric.data as Data
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import remove_self_loops


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def to(self, device):
        # Move graph data to the specified device
        self.graph['edge_index'] = self.graph['edge_index'].to(device)
        if self.graph['edge_feat'] is not None:
            self.graph['edge_feat'] = self.graph['edge_feat'].to(device)
        if self.graph['node_feat'] is not None:
            self.graph['node_feat'] = self.graph['node_feat'].to(device)
        self.label = self.label.to(device)
        return self

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))
def YRB_Graph(content_file, edges_file):
    content = pd.read_csv(content_file, index_col=0)
    edges_weights = pd.read_csv(edges_file, header=None)

    E = torch.tensor(edges_weights.iloc[:, 0].values, dtype=torch.long)
    G = torch.tensor(edges_weights.iloc[:, 1].values, dtype=torch.long)

    num_nodes = len(content)
    x = torch.tensor(content.iloc[:, 1:-2].values, dtype=torch.float)

    y = torch.tensor(content.iloc[:, -2].values, dtype=torch.long)
    y = F.one_hot(y, num_classes=2).to(torch.float)

    edge_index = torch.stack([E, G], dim=0)
    edge_index = torch.cat([edge_index, torch.stack([G, E], dim=0)], dim=1)

    data = Data(num_nodes=num_nodes, x=x, edge_index=edge_index, y=y)
    ids = content['ID']

    features = content.iloc[:, 1:-2]

    features_n = features.to_numpy()
    features_n = torch.Tensor(features_n)


    label = content.iloc[:, -2]
    label = label.to_list()
    label = np.array(label)
    label = torch.from_numpy(label)
    label = F.one_hot(label.to(torch.int64), num_classes=2)
    label = label.type(torch.float32)

    print(data)



    # 数据集划分
    hp = content[content['label'] == 1]
    fhp = content[content['label'] == 0]
    idx_train_list = []
    idx_test_list = []
    seed = 6 * 10
    testr = 0.3
    env = content['ENV']

    id_f_train = []
    id_f_test = []
    id_hp_train = []
    id_hp_test = []
    for i in range(env.max()):
        mid_hp = hp[hp['ENV'] == i].index
        mid_fhp = fhp[fhp['ENV'] == i].index
        hp_num = mid_hp.shape[0]
        fhp_num = mid_fhp.shape[0]
        if hp_num == 0:
            continue
        elif hp_num == 1:
            hp_train = mid_hp
        else:
            hp_train, hp_test = train_test_split(mid_hp, test_size=testr, random_state=seed)  # i=8
            id_hp_test = id_hp_test + hp_test.to_list()
        if hp_num < fhp_num:  # 以正样本为准，取尽可能相等的负样本
            _, mid_fhp = train_test_split(mid_fhp, test_size=mid_hp.shape[0] / mid_fhp.shape[0], random_state=seed)
            fhp_num = mid_fhp.shape[0]
        if fhp_num < 2:
            f_train = mid_fhp
        else:
            f_train, f_test = train_test_split(mid_fhp, test_size=testr, random_state=seed)  #
            id_f_test = id_f_test + f_test.to_list()
        id_hp_train = id_hp_train + hp_train.to_list()
        id_f_train = id_f_train + f_train.to_list()
    idx_train = np.r_[id_f_train, id_hp_train]  # ,id_xp_train,id_bt_train]
    np.random.seed(seed)
    np.random.shuffle(idx_train)
    idx_test = np.r_[id_f_test, id_hp_test]  # ,id_xp_test,id_bt_test]
    np.random.seed(seed)
    np.random.shuffle(idx_test)
    idx_train_list.append(idx_train)
    idx_test_list.append(idx_test)
    print('positive:{} negative:{}'.format(len(id_hp_train) + len(id_hp_test), len(id_f_train) + len(id_f_test)))

    index_train = idx_train_list[0].tolist()
    index_test = idx_test_list[0].tolist()
    print("train_data:", len(index_train), "test_data:", len(index_test))
    print("output...")
    train_data = content.loc[index_train]
    test_data = content.loc[index_test]

    idx_train = torch.LongTensor(index_train)
    idx_val = torch.LongTensor(index_test)
    idx_test = torch.LongTensor(index_test)

    filename = 'YRB'
    dataset = NCDataset(filename)

    features_n = torch.as_tensor(features_n)
    edge_index = torch.as_tensor(edge_index)
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': features_n,
                     'num_nodes': num_nodes}
    #label = torch.tensor(label, dtype=torch.long).squeeze()
    dataset.label = label
    #dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])



    return data, ids, edges_weights, idx_train, idx_val, idx_test, dataset