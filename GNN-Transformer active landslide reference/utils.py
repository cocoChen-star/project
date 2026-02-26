import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')  # Enable non-GUI backend (uncomment if needed)
import matplotlib.pyplot as plt
import torch
from sklearn import metrics
from scipy.interpolate import interp1d
import random

def fix_seed(seed):
    """Set random seed for experiment reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def accuracy(logits, labels):
    """Compute classification accuracy."""
    _, indices = torch.max(logits, dim=1)
    _, indices2 = torch.max(labels, dim=1)
    return (torch.sum(indices == indices2).item() * 1.0 / len(labels))

def aprf(logits, labels):
    """Compute accuracy, precision, recall, and F1 score."""
    _, indices = torch.max(logits, dim=1)
    _, indices2 = torch.max(labels, dim=1)
    TP = ((indices == 1) * (indices2 == 1)).sum().item()
    FP = ((indices == 1) * (indices2 == 0)).sum().item()
    FN = ((indices == 0) * (indices2 == 1)).sum().item()
    TN = ((indices == 0) * (indices2 == 0)).sum().item()
    Accuracy = (TP + TN) / (TP + FN + FP + TN + 1e-8)
    Precision = TP / (TP + FP + 1e-8)
    Recall = TP / (TP + FN + 1e-8)
    F_measures = 2 * Precision * Recall / (Precision + Recall + 1e-8)
    return Accuracy, Precision, Recall, F_measures

def plot_roc(path, roc_auc, fpr, tpr):
    """Plot and save ROC curve."""
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(path)
    plt.close()

def plot_training_log(trlog, acc=0, savepath=''):
    """Plot and save training loss and accuracy curves."""
    epoch = trlog.shape[0]
    train_loss, train_acc, val_loss, val_acc = trlog[:, 0], trlog[:, 1], trlog[:, 2], trlog[:, 3]
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(epoch), train_loss, 'coral', label='Train Loss')
    ax1.plot(range(epoch), val_loss, 'red', label='Validation Loss')
    ax2.plot(range(epoch), train_acc, 'yellowgreen', label='Train Accuracy')
    ax2.plot(range(epoch), val_acc, 'green', label='Validation Accuracy')
    ax1.set_xlabel(f'Epoch\nAccuracy = {acc:.4f}')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    ax1.legend()
    ax2.legend()
    ax1.set_title(f"Accuracy = {acc:.4f}")
    plt.savefig(savepath)
    plt.close()