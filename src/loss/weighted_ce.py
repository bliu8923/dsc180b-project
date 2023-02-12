#Courtesy of GraphGPS: graphgps/loss/weighed_cross_entropy.py
import torch
import torch.nn.functional as F

def weighted_cross_entropy(pred, y):
    """Weighted cross-entropy for unbalanced classes.
    """
    # calculating label weights for weighted loss computation
    if len(y.shape) > 1:
        true = y[0]
    else:
        true = y
    V = true.size(0)
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    
    #print(V, label_count, cluster_sizes, weight, pred, pred.ndim)
    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight), pred
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                  weight=weight[true])
        return loss, torch.sigmoid(pred)