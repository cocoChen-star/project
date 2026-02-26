import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import json
import os
import optuna
import numpy as np
import pandas as pd
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import logging
from sklearn import metrics
from models import Proposed
from utils import fix_seed, accuracy, aprf, plot_training_log, plot_roc
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.transforms import SIGN
from scipy.interpolate import interp1d

def setup_logging(log_file):
    """Initialize logging to file and console."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def consis_loss_kl(p_t, p_s, temp, lam):
    """Compute self-consistency loss with KL divergence."""
    p_t = torch.exp(F.log_softmax(p_t, dim=-1))
    sharp_p_t = (torch.pow(p_t, 1. / temp) / torch.sum(torch.pow(p_t, 1. / temp), dim=1, keepdim=True)).detach()
    p_s = F.softmax(p_s, dim=1)
    log_sharp_p_t = torch.log(sharp_p_t + 1e-8)
    loss = lam * torch.mean(torch.sum(torch.pow(p_s - sharp_p_t, 2), dim=1, keepdim=True))
    kl = torch.mean(torch.sum(p_s * (torch.log(p_s + 1e-8) - log_sharp_p_t), dim=1, keepdim=True))
    return loss, kl

def train_scr(model, teacher_model, labels, device, loss_fcn, optimizer, args, global_step, features, edge_index, enhance_idx, xs):
    """Train the model using Self-Consistency Refinement (SCR)."""
    model.train()
    output_att = model(features, edge_index, xs)
    L1 = loss_fcn(output_att[args.train_mask], labels[args.train_mask])

    with torch.no_grad():
        mean_t_output = teacher_model(features, edge_index, xs)[enhance_idx]
    student_output = output_att[enhance_idx]

    loss_consis, kl_loss = consis_loss_kl(mean_t_output, student_output, args.tem, args.lam)
    loss_supervised = args.sup_lam * L1
    kl_loss = args.kl_lam * kl_loss
    loss_train = loss_supervised + (kl_loss if args.kl else loss_consis)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    alpha = min(1 - 1 / (global_step + 1), args.ema_decay) if args.adap else args.ema_decay
    for mean_param, param in zip(teacher_model.parameters(), model.parameters()):
        mean_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    return loss_train, accuracy(output_att[args.train_mask], labels[args.train_mask])

def objective(trial, pggg, device, args_template):
    """Optuna hyperparameter optimization objective function (without SCR)."""
    g, ids, edges_weights, train_mask, val_mask, test_mask, dataset = pggg
    features = dataset.graph['node_feat'].to(device)
    n = dataset.graph['num_nodes']
    labels = dataset.label.to(device)
    edge_index, _ = remove_self_loops(dataset.graph['edge_index'])
    edge_index, _ = add_self_loops(edge_index, num_nodes=n)
    edge_index = edge_index.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Define hyperparameter search space
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    hidden_channels = trial.suggest_int('hidden_channels', 16, 128, step=16)
    trans_dropout = trial.suggest_float('trans_dropout', 0.2, 0.8)
    gnn_dropout = trial.suggest_float('gnn_dropout', 0.2, 0.8)
    graph_weight = trial.suggest_float('graph_weight', 0.2, 0.8)
    mlp_hidden = trial.suggest_int('mlp_hidden', 16, 128, step=16)
    trans_num_layers = trial.suggest_int('trans_num_layers', 1, 3)
    trans_num_heads = trial.suggest_int('trans_num_heads', 1, 2)
    sign_num_layers = trial.suggest_int('sign_num_layers', 2, 10)

    data = SIGN(sign_num_layers)(g.to(device))
    xs = [data.x] + [data[f'x{i}'] for i in range(1, sign_num_layers + 1)]

    in_feats = features.shape[1]
    n_classes = labels.shape[1]

    model = Proposed(
        in_channels=in_feats, hidden_channels=hidden_channels, out_channels=n_classes,
        trans_num_layers=trans_num_layers, trans_num_heads=trans_num_heads,
        trans_dropout=trans_dropout, trans_use_bn=True, trans_use_residual=True,
        trans_use_weight=True, trans_use_act=True, gnn_num_layers=3,
        gnn_dropout=gnn_dropout, use_graph=True, graph_weight=graph_weight,
        aggregate='add', mlp_hidden=mlp_hidden, sign_num_layers=sign_num_layers
    ).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args_template.trans_weight_decay},
        {'params': model.params2, 'weight_decay': args_template.gnn_weight_decay}
    ], lr=lr)
    loss_fcn = nn.BCEWithLogitsLoss()

    trlog = np.zeros([args_template.optuna_epochs, 4])
    best_val_acc = 0
    for epoch in range(args_template.optuna_epochs):
        model.train()
        logits = model(features, edge_index, xs)
        train_loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(features, edge_index, xs)
            eval_loss = loss_fcn(logits[val_mask], labels[val_mask])
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        trlog[epoch] = [train_loss.item(), accuracy(logits[train_mask], labels[train_mask]), eval_loss.item(), val_acc]
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    logging.info(f"Trial {trial.number}: Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc

def hyperparameter_optimization(pggg, device, args, output_dir):
    """Perform hyperparameter optimization using Optuna (without SCR)."""
    args_template = argparse.Namespace(**vars(args))
    study = optuna.create_study(direction='maximize')
    logging.info("Starting hyperparameter optimization (without SCR)...")
    study.optimize(lambda trial: objective(trial, pggg, device, args_template), n_trials=args.optuna_trials)

    best_params = {
        'lr': study.best_params['lr'],
        'hidden_channels': study.best_params['hidden_channels'],
        'trans_dropout': study.best_params['trans_dropout'],
        'gnn_dropout': study.best_params['gnn_dropout'],
        'graph_weight': study.best_params['graph_weight'],
        'mlp_hidden': study.best_params['mlp_hidden'],
        'trans_num_layers': study.best_params['trans_num_layers'],
        'trans_num_heads': study.best_params['trans_num_heads'],
        'sign_num_layers': study.best_params['sign_num_layers']
    }

    best_params_path = os.path.join(output_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f)
    logging.info(f"Best hyperparameters saved to: {best_params_path}")
    logging.info(f"Best validation accuracy: {study.best_value:.4f}")
    logging.info(f"Best hyperparameters: {best_params}")
    return best_params

def train_and_evaluate(pggg, args, output_dir='../Results'):
    """Train and evaluate the Proposed model with SCR (five independent runs)."""
    # Initialize logging
    setup_logging(os.path.join(output_dir, 'training_log.txt'))
    logging.info(f"Starting training and evaluation, output directory: {output_dir}")

    fix_seed(args.seed)
    device = torch.device('cuda' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    g, ids, edges_weights, train_mask, val_mask, test_mask, dataset = pggg
    features = dataset.graph['node_feat'].to(device)
    n = dataset.graph['num_nodes']
    labels = dataset.label.to(device)
    edge_index, _ = remove_self_loops(dataset.graph['edge_index'])
    edge_index, _ = add_self_loops(edge_index, num_nodes=n)
    edge_index = edge_index.to(device)
    args.train_mask = train_mask.to(device)
    args.val_mask = val_mask.to(device)
    args.test_mask = test_mask.to(device)
    data = SIGN(args.sign_num_layers)(g.to(device))
    xs = [data.x] + [data[f'x{i}'] for i in range(1, args.sign_num_layers + 1)]

    in_feats = features.shape[1]
    n_classes = labels.shape[1]
    accuracies, precisions, recalls, f1_scores, roc_aucs = [], [], [], [], []
    fpr_list, tpr_list = [], []
    dur = []

    for run in range(args.n_runs):
        logging.info(f"\nRun {run + 1}/{args.n_runs}...")
        fix_seed(args.seed + run)

        model = Proposed(
            in_channels=in_feats, hidden_channels=args.hidden_channels, out_channels=n_classes,
            trans_num_layers=args.trans_num_layers, trans_num_heads=args.trans_num_heads,
            trans_dropout=args.trans_dropout, trans_use_bn=args.trans_use_bn,
            trans_use_residual=args.trans_use_residual, trans_use_weight=args.trans_use_weight,
            trans_use_act=args.trans_use_act, gnn_num_layers=args.gnn_num_layers,
            gnn_dropout=args.gnn_dropout, use_graph=args.use_graph, graph_weight=args.graph_weight,
            aggregate=args.aggregate, mlp_hidden=args.mlp_hidden, sign_num_layers=args.sign_num_layers
        ).to(device)

        teacher_model = Proposed(
            in_channels=in_feats, hidden_channels=args.hidden_channels, out_channels=n_classes,
            trans_num_layers=args.trans_num_layers, trans_num_heads=args.trans_num_heads,
            trans_dropout=args.trans_dropout, trans_use_bn=args.trans_use_bn,
            trans_use_residual=args.trans_use_residual, trans_use_weight=args.trans_use_weight,
            trans_use_act=args.trans_use_act, gnn_num_layers=args.gnn_num_layers,
            gnn_dropout=args.gnn_dropout, use_graph=args.use_graph, graph_weight=args.graph_weight,
            aggregate=args.aggregate, mlp_hidden=args.mlp_hidden, sign_num_layers=args.sign_num_layers
        ).to(device)
        for param in teacher_model.parameters():
            param.detach_()

        optimizer = torch.optim.Adam([
            {'params': model.params1, 'weight_decay': args.trans_weight_decay},
            {'params': model.params2, 'weight_decay': args.gnn_weight_decay}
        ], lr=args.lr)
        loss_fcn = nn.BCEWithLogitsLoss()

        trlog = np.zeros([args.n_epochs, 4])
        best_val, best_model_state, best_epoch = 0, None, 0
        global_step = 0

        for epoch in range(args.n_epochs):
            t0 = time.time() if epoch >= 3 else 0
            if epoch < args.warm_up + 1:
                model.train()
                logits = model(features, edge_index, xs)
                train_loss = loss_fcn(logits[args.train_mask], labels[args.train_mask])
                train_acc = accuracy(logits[args.train_mask], labels[args.train_mask])
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
            else:
                if epoch == args.warm_up + 1:
                    logging.info("Starting SCR training...")
                if (epoch - 1) % args.gap == 0 or epoch == args.warm_up + 1:
                    model.eval()
                    with torch.no_grad():
                        preds = model(features, edge_index, xs)
                        prob_teacher = preds.softmax(dim=1)
                        threshold = args.top - (args.top - args.down) * epoch / args.n_epochs
                        indices = torch.arange(len(prob_teacher), device=device)
                        confident_nid = indices[prob_teacher.max(1)[0] > threshold]
                        enhance_idx = confident_nid[torch.tensor([idx.item() in args.test_mask for idx in confident_nid], device=device)]
                train_loss, train_acc = train_scr(model, teacher_model, labels, device, loss_fcn, optimizer, args, global_step, features, edge_index, enhance_idx, xs)
                global_step += 1

            model.eval()
            with torch.no_grad():
                logits = model(features, edge_index, xs)
                eval_loss = loss_fcn(logits[args.val_mask], labels[args.val_mask])
                eval_acc = accuracy(logits[args.val_mask], labels[args.val_mask])
            trlog[epoch] = [train_loss.item(), train_acc, eval_loss.item(), eval_acc]

            # Log results every 10 epochs
            if epoch % 10 == 0:
                logging.info(f"Run {run + 1} - Epoch {epoch}: Train Loss: {train_loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Loss: {eval_loss.item():.4f}, Val Acc: {eval_acc:.4f}")

            # Check for best model every 10 epochs after 400 epochs
            if epoch > 400 and epoch % 10 == 0 and eval_acc > best_val:
                best_val, best_model_state, best_epoch = eval_acc, model.state_dict(), epoch
                model_path = os.path.join(output_dir, f'SCR_best_run{run+1}.pkl')
                torch.save(best_model_state, model_path)
                logging.info(f"Run {run + 1} - Epoch {epoch}: Best validation accuracy: {best_val:.4f}, model saved to {model_path}")

            if epoch >= 3:
                dur.append(time.time() - t0)

        # Load best model if it exists
        model_path = os.path.join(output_dir, f'SCR_best_run{run+1}.pkl')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            logging.warning(f"Run {run + 1}: Best model not found, using final model for evaluation")
            best_model_state = model.state_dict()
            best_epoch = args.n_epochs - 1

        model.eval()
        with torch.no_grad():
            logits = model(features, edge_index, xs)
            pred_rst = F.softmax(logits, dim=1)
            test_acc, prec, rec, f1 = aprf(logits[args.test_mask], labels[args.test_mask])
            fpr, tpr, _ = metrics.roc_curve(labels[args.test_mask, 1].cpu().numpy(), pred_rst[args.test_mask, 1].cpu().numpy())
            roc_auc = metrics.auc(fpr, tpr)

        accuracies.append(test_acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

        logging.info(f"\nRun {run + 1} Results (Best Epoch: {best_epoch}):")
        logging.info(f"Test Accuracy: {test_acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        logging.info(f"Average training time per epoch: {np.mean(dur[-10:]):.2f} seconds")

        pd.DataFrame(pred_rst.cpu().tolist(), index=ids).to_csv(os.path.join(output_dir, f'pred_df_run{run+1}.csv'))
        plot_training_log(trlog, test_acc, os.path.join(output_dir, f'train_process_run{run+1}.png'))

    accuracies, precisions, recalls, f1_scores, roc_aucs = map(np.array, [accuracies, precisions, recalls, f1_scores, roc_aucs])
    logging.info("\n=== Average Results Over Multiple Runs ===")
    for metric, values in zip(['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'], [accuracies, precisions, recalls, f1_scores, roc_aucs]):
        logging.info(f"{metric}: {values.mean():.4f} (± {values.std():.4f})")

    max_fpr_len = max(len(fpr) for fpr in fpr_list)
    mean_fpr = np.linspace(0, 1, max_fpr_len)
    mean_tpr = np.zeros_like(mean_fpr)
    for fpr, tpr in zip(fpr_list, tpr_list):
        mean_tpr += interp1d(fpr, tpr, bounds_error=False, fill_value=(0, 1))(mean_fpr)
    mean_tpr /= args.n_runs
    mean_auc = np.mean(roc_aucs)

    roc_df = pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr, 'AUC': [mean_auc] * len(mean_fpr)})
    roc_df.to_csv(os.path.join(output_dir, 'SCR_mean_roc.csv'), index=False)
    plot_roc(os.path.join(output_dir, 'SCR_mean_roc.png'), mean_auc, mean_fpr, mean_tpr)
    logging.info(f"Mean ROC curve saved to: {os.path.join(output_dir, 'SCR_mean_roc.csv')}")