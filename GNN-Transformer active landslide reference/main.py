# import argparse
# import json
# import os
# import numpy as np
# import pandas as pd
# import random
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import torch
# import argparse
# import time
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn import metrics
# from torch_geometric.utils import degree, remove_self_loops, add_self_loops
# from torch_geometric.transforms import SIGN
# from scipy.interpolate import interp1d
# import json
# import optuna
# from train import train_and_evaluate, hyperparameter_optimization
# from utils import fix_seed
# from YRB_dataset import YRB_Graph  # 假设提供此模块或记录其要求
#
# def parser_add_main_args(parser):
#     """为 Proposed 模型添加命令行参数。"""
#     parser.add_argument("--gpu", type=int, default=0, help="GPU 设备 ID，-1 表示使用 CPU")
#     parser.add_argument("--n-epochs", type=int, default=800, help="训练轮次")
#     parser.add_argument('--seed', type=int, default=123, help="随机种子")
#     parser.add_argument("--hidden_channels", type=int, default=32, help="隐藏层大小")
#     parser.add_argument('--use_graph', action='store_true', default=True, help="是否使用图卷积")
#     parser.add_argument('--aggregate', type=str, default='add', choices=['add', 'cat'], help="聚合方式")
#     parser.add_argument('--graph_weight', type=float, default=0.5, help="图卷积权重")
#     parser.add_argument('--trans_num_heads', type=int, default=1, help="Transformer 头数")
#     parser.add_argument('--trans_use_weight', action='store_true', default=True, help="Transformer 是否使用权重")
#     parser.add_argument('--trans_use_bn', action='store_true', default=True, help="是否使用批归一化")
#     parser.add_argument('--trans_use_residual', action='store_true', default=True, help="是否使用残差连接")
#     parser.add_argument('--trans_use_act', action='store_true', default=True, help="是否使用激活函数")
#     parser.add_argument('--trans_num_layers', type=int, default=1, help="Transformer 层数")
#     parser.add_argument('--trans_dropout', type=float, default=0.5, help="Transformer 丢弃率")
#     parser.add_argument('--trans_weight_decay', type=float, default=1e-4, help="Transformer 权重衰减")
#     parser.add_argument('--gnn_num_layers', type=int, default=3, help="GNN 层数")
#     parser.add_argument('--gnn_dropout', type=float, default=0.5, help="GNN 丢弃率")
#     parser.add_argument('--gnn_weight_decay', type=float, default=1e-4, help="GNN 权重衰减")
#     parser.add_argument('--lr', type=float, default=0.001, help="学习率")
#     parser.add_argument("--node_features", type=str, default=r"C:\YRB\test200\YRB_contents_归一化.csv", help="节点特征 CSV 文件路径")
#     parser.add_argument("--edges", type=str, default=r"C:\YRB\test200\YRB_edges.csv", help="边数据 CSV 文件路径")
#     parser.add_argument('--sign_num_layers', type=int, default=3, help="SIGN 层数")
#     parser.add_argument('--mlp_hidden', type=int, default=32, help="MLP 隐藏层大小")
#     # SCR 参数
#     parser.add_argument("--scr", action="store_true", default=True, help="是否使用 SCR 训练")
#     parser.add_argument("--ema_decay", type=float, default=0, help="EMA 衰减率")
#     parser.add_argument("--adap", action="store_true", help="是否使用自适应 EMA")
#     parser.add_argument("--sup_lam", type=float, default=1.0, help="监督损失权重")
#     parser.add_argument("--kl", action="store_true", help="SCR 中是否使用 KL 散度")
#     parser.add_argument("--kl_lam", type=float, default=0.025, help="KL 损失权重")
#     parser.add_argument("--top", type=float, default=0.5, help="SCR 上限阈值")
#     parser.add_argument("--down", type=float, default=0.4, help="SCR 下限阈值")
#     parser.add_argument("--warm_up", type=int, default=150, help="预热轮次")
#     parser.add_argument("--gap", type=int, default=20, help="SCR 更新间隔")
#     parser.add_argument("--tem", type=float, default=0.5, help="SCR 温度参数")
#     parser.add_argument("--lam", type=float, default=0.1, help="一致性损失权重")
#     # 运行参数
#     parser.add_argument("--n_runs", type=int, default=5, help="独立运行次数")
#     parser.add_argument("--optuna_trials", type=int, default=30, help="Optuna 试验次数")
#     parser.add_argument("--optuna_epochs", type=int, default=400, help="Optuna 每试验的轮次")
#     parser.add_argument("--best_params", type=str, default=r"D:\YRB_代码整理\参数\best_params1.json", help="最佳参数 JSON 文件路径")
#     parser.add_argument("--output_dir", type=str, default="results", help="输出文件保存目录")
#     parser.add_argument("--mode", type=str, default="scr", choices=["optuna", "scr", "both"],
#                         help="运行模式：optuna（仅超参数搜索），scr（仅 SCR 验证），both（两者皆运行）")
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Proposed: 图 Transformer 与自一致性精炼")
#     parser_add_main_args(parser)
#     args = parser.parse_args()
#
#     # 参数验证
#     if args.mode == "scr" and not args.best_params:
#         parser.error("--mode scr 要求提供 --best_params 参数，指定最佳参数 JSON 文件路径。")
#     if args.best_params and not os.path.exists(args.best_params):
#         parser.error(f"最佳参数文件 {args.best_params} 不存在。")
#
#     fix_seed(args.seed)
#     device = torch.device('cuda' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
#     pggg = YRB_Graph(args.node_features, args.edges)
#
#     if args.mode in ["optuna", "both"]:
#         best_params = hyperparameter_optimization(pggg, device, args, args.output_dir)
#         for key in best_params:
#             setattr(args, key, best_params[key])
#     elif args.mode == "scr":
#         with open(args.best_params, 'r') as f:
#             best_params = json.load(f)
#         for key in ['lr', 'hidden_channels', 'trans_dropout', 'gnn_dropout', 'graph_weight',
#                     'mlp_hidden', 'trans_num_layers', 'trans_num_heads', 'sign_num_layers']:
#             setattr(args, key, best_params[key])
#         print(f"已加载最佳参数: {best_params}")
#
#     if args.mode in ["scr", "both"]:
#         train_and_evaluate(pggg, args, args.output_dir)


import argparse
import json
import os
import numpy as np
import pandas as pd
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch_geometric.utils import degree, remove_self_loops, add_self_loops
from torch_geometric.transforms import SIGN
from scipy.interpolate import interp1d
import optuna
from train import train_and_evaluate, hyperparameter_optimization
from utils import fix_seed
from YRB_dataset import YRB_Graph  # Assumes this module is provided or its requirements are documented

def parser_add_main_args(parser):
    """Add command-line arguments for the Proposed model."""
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID, -1 for CPU")
    parser.add_argument("--n-epochs", type=int, default=800, help="Number of training epochs")
    parser.add_argument('--seed', type=int, default=123, help="Random seed")
    parser.add_argument("--hidden_channels", type=int, default=32, help="Hidden layer size")
    parser.add_argument('--use_graph', action='store_true', default=True, help="Use graph convolution")
    parser.add_argument('--aggregate', type=str, default='add', choices=['add', 'cat'], help="Aggregation method")
    parser.add_argument('--graph_weight', type=float, default=0.5, help="Graph convolution weight")
    parser.add_argument('--trans_num_heads', type=int, default=1, help="Number of Transformer heads")
    parser.add_argument('--trans_use_weight', action='store_true', default=True, help="Use weights in Transformer")
    parser.add_argument('--trans_use_bn', action='store_true', default=True, help="Use batch normalization")
    parser.add_argument('--trans_use_residual', action='store_true', default=True, help="Use residual connections")
    parser.add_argument('--trans_use_act', action='store_true', default=True, help="Use activation function")
    parser.add_argument('--trans_num_layers', type=int, default=1, help="Number of Transformer layers")
    parser.add_argument('--trans_dropout', type=float, default=0.5, help="Transformer dropout rate")
    parser.add_argument('--trans_weight_decay', type=float, default=1e-4, help="Transformer weight decay")
    parser.add_argument('--gnn_num_layers', type=int, default=3, help="Number of GNN layers")
    parser.add_argument('--gnn_dropout', type=float, default=0.5, help="GNN dropout rate")
    parser.add_argument('--gnn_weight_decay', type=float, default=1e-4, help="GNN weight decay")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument("--node_features", type=str, default="../Data/YRB_contents.csv", help="Path to node features CSV file")
    parser.add_argument("--edges", type=str, default="../Data/YRB_edges.csv", help="Path to edges CSV file")
    parser.add_argument('--sign_num_layers', type=int, default=3, help="Number of SIGN layers")
    parser.add_argument('--mlp_hidden', type=int, default=32, help="MLP hidden layer size")
    # SCR parameters
    parser.add_argument("--scr", action="store_true", default=True, help="Use SCR training")
    parser.add_argument("--ema_decay", type=float, default=0, help="EMA decay rate")
    parser.add_argument("--adap", action="store_true", help="Use adaptive EMA")
    parser.add_argument("--sup_lam", type=float, default=1.0, help="Supervised loss weight")
    parser.add_argument("--kl", action="store_true", help="Use KL divergence in SCR")
    parser.add_argument("--kl_lam", type=float, default=0.025, help="KL loss weight")
    parser.add_argument("--top", type=float, default=0.5, help="SCR upper threshold")
    parser.add_argument("--down", type=float, default=0.4, help="SCR lower threshold")
    parser.add_argument("--warm_up", type=int, default=150, help="Warm-up epochs")
    parser.add_argument("--gap", type=int, default=20, help="SCR update interval")
    parser.add_argument("--tem", type=float, default=0.5, help="SCR temperature parameter")
    parser.add_argument("--lam", type=float, default=0.1, help="Consistency loss weight")
    # Run parameters
    parser.add_argument("--n_runs", type=int, default=5, help="Number of independent runs")
    parser.add_argument("--optuna_trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--optuna_epochs", type=int, default=400, help="Epochs per Optuna trial")
    parser.add_argument("--best_params", type=str, default="../Results/best_params.json", help="Path to best parameters JSON file")
    parser.add_argument("--output_dir", type=str, default="../Results", help="Output directory")
    parser.add_argument("--mode", type=str, default="scr", choices=["optuna", "scr", "both"],
                        help="Run mode: optuna (hyperparameter search only), scr (SCR validation only), both (both modes)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proposed: Graph Transformer ")
    parser_add_main_args(parser)
    args = parser.parse_args()

    # Argument validation
    if args.mode == "scr" and not args.best_params:
        parser.error("--mode scr requires --best_params to specify the best parameters JSON file path.")
    if args.best_params and not os.path.exists(args.best_params):
        parser.error(f"Best parameters file {args.best_params} does not exist.")

    fix_seed(args.seed)
    device = torch.device('cuda' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    pggg = YRB_Graph(args.node_features, args.edges)

    if args.mode in ["optuna", "both"]:
        best_params = hyperparameter_optimization(pggg, device, args, args.output_dir)
        for key in best_params:
            setattr(args, key, best_params[key])
    elif args.mode == "scr":
        with open(args.best_params, 'r') as f:
            best_params = json.load(f)
        for key in ['lr', 'hidden_channels', 'trans_dropout', 'gnn_dropout', 'graph_weight',
                    'mlp_hidden', 'trans_num_layers', 'trans_num_heads', 'sign_num_layers']:
            setattr(args, key, best_params[key])
        print(f"Loaded best parameters: {best_params}")

    if args.mode in ["scr", "both"]:
        train_and_evaluate(pggg, args, args.output_dir)