# =============================================================================
# PLOTS
#
# last major rev. 2020/04
#
# Filippo Maria Castelli
# LENS Biophotonics Group
# castelli@lens.unifi.it
# =============================================================================

from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def training_metrics_plot(hist_dict,
                          loss_lims=[0, 0.1],
                          accuracy_lims=[0.8,1],
                          jaccard_lims=[0,1],
                          dice_lims=[0,1],
                          legend_outside=False,
                          figsize=(18,9),
                          dpi=300,
                          colormap=None,
                          exp_window=None,
                          param="lr"):
    
    hist_dict = OrderedDict(sorted(hist_dict.items()))
    values = np.array(list(hist_dict.keys()))
    normalize = mcolors.Normalize(vmin=np.min(values),
                                  vmax=np.max(values))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    fig.suptitle("Training Metrics, variable {}".format(param))
    axes = [ax1, ax2, ax3, ax4]
    ax1.set_title("Loss")
    ax2.set_title("Accuracy")
    ax3.set_title("Jaccard Index")
    ax4.set_title("Dice Coefficient")
    
    ax1.set_ylim(loss_lims)
    ax2.set_ylim(accuracy_lims)
    ax3.set_ylim(jaccard_lims)
    ax4.set_ylim(dice_lims)
    
    for lr, hist in hist_dict.items():
        series_lengths = []
        # fig.suptitle('Sharing x per column, y per row')
        if colormap is not None:
            color = colormap(normalize(lr))
        else:
            color = None
        #loss = [hist["loss"] if exp_window is None else ewma(np.array(hist["loss"]), exp_window)]
        
        ax1.plot(hist["loss"], color=color, label="{} {}".format(param, lr))
        ax2.plot(hist["accuracy"], color=color, label="{} {}".format(param, lr))
        ax3.plot(hist["jaccard_index"], color=color, label="{} {}".format(param, lr))
        ax4.plot(hist["dice_coefficient"], color=color, label="{} {}".format(param, lr))
        
        series_lengths.append(len(hist["loss"]))
        
    x_len = np.array(series_lengths).max()
    
    for ax in axes:
        if legend_outside:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:        
            ax.legend()
        ax.set_xlim([-1, x_len])
        ax.set_xlabel("epochs")
        for tick in ax.get_xticklabels():
            tick.set_visible(True)
            
            
def training_validation_plot(hist_dict,
                             figsize=(15,18),
                             dpi=300,
                             n_columns=3,
                             metric = "loss",
                             param = "batch_size",
                             lims=None):
    
    hist_dict = OrderedDict(sorted(hist_dict.items()))
    hist_dict_keys = list(hist_dict.keys())
    n_rows = int(np.ceil(len(hist_dict_keys)/n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=figsize, dpi=dpi)
    fig.suptitle("{}, Variable {}".format(metric, param), y=1.01)
    axes = list(axes.flat)
    
    for i, key in enumerate(hist_dict_keys):
        hist = hist_dict[key]
        ax = axes[i]
        metric_plot(ax=ax, hist=hist, base_filters=key, metric=metric, lims=lims, param=param)
    plt.tight_layout()

def metric_plot(ax, hist, base_filters, metric, lims=None, param="batch size"):
    val_metric = "val_" + metric
    ax.set_title("{}: {} {}".format(metric, param, base_filters))
    ax.plot(hist[metric], label=metric)
    ax.plot(hist[val_metric], label=val_metric)
    ax.legend()
    if lims is not None:
        ax.set_ylim(lims)
    else:  
        ax.set_ylim(0.01, np.array(hist[metric]).max())
    ax.set_xlabel("epochs")
    ax.set_ylabel(metric)
    for tick in ax.get_xticklabels():
        tick.set_visible(True)
        
        
def pr_plot(metrics_dict,
            figsize=(18,10),
            dpi=300,
            lims=None):
    "Precision-Recall plot"
    pr_curve = metrics_dict["pr_curve"]
    fig = plt.figure(figsize=figsize,
                     dpi=dpi)
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(pr_curve["recall"],
            pr_curve["precision"],
            label="pr_curve")
    
    ax.set_ylim(lims)
    
    ax.set_title("Precision-Recall plot")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    

def roc_plot(metrics_dict,
             figsize=(18,10),
             dpi=300,
             lims=None):
    "ROC plot"
    roc_curve = metrics_dict["roc_curve"]
    fig = plt.figure(figsize=figsize,
                     dpi=dpi)
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(roc_curve["fpr"],
            roc_curve["tpr"],
            label="pr_curve")
    
    ax.set_ylim(lims)
    ax.set_title("Receiver Operator characteristic curve plot")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    
    
def test_metrics_plot(metrics_dict,
                          roc_lims=[0, 1],
                          pr_lims=[0,1],
                          figsize=(18,9),
                          dpi=300,
                          colormap=None,
                          exp_window=None,
                          param="lr"):
    
    metrics_dict = OrderedDict(sorted(metrics_dict.items()))
    values = np.array(list(metrics_dict.keys()))
    normalize = mcolors.Normalize(vmin=np.min(values),
                                  vmax=np.max(values))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.suptitle("Test metrics, variable {}".format(param))
    ax1.set_title("Receiver Operator characteristic curve plot")
    ax2.set_title("Precision-Recall plot")

    ax1.set_ylim(roc_lims)
    ax2.set_ylim(pr_lims)
    
    for lr, metrics in metrics_dict.items():
        series_lengths = []
        # fig.suptitle('Sharing x per column, y per row')
        if colormap is not None:
            color = colormap(normalize(lr))
        else:
            color = None
        #loss = [hist["loss"] if exp_window is None else ewma(np.array(hist["loss"]), exp_window)]
        ax1.plot(metrics["roc_curve"]["fpr"],metrics["roc_curve"]["tpr"], color=color, label="{} {}".format(param, lr))
        ax2.plot(metrics["pr_curve"]["recall"],metrics["pr_curve"]["precision"], color=color, label="{} {}".format(param, lr))
        
        series_lengths.append(len(metrics["roc_curve"]))

    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")