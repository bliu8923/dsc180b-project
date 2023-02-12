#Courtesy of GraphGPS: graphgps/loss/multilabel_classification_loss.py
import torch.nn as nn

def multilabel_cross_entropy(pred, true):
    """Multilabel cross-entropy loss.
    """
    bce_loss = nn.BCEWithLogitsLoss()
    is_labeled = true == true  # Filter our nans.
    return bce_loss(pred[is_labeled], true[is_labeled].float()), pred