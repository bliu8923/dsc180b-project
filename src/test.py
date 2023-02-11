import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score
def test(test_loader, metric):

    model.eval()
    scores = []
    for data in tqdm(test_loader):
        data = data.to(device)
        try:
            pred = model(data.x, data.edge_index).argmax(dim=-1)
        except:
            pred = model(data)
        if metric == f1_score:
            _, pred = torch.max(F.softmax(pred, dim=1), 1)
        elif metric == average_precision_score:
            pred = F.softmax(pred, dim=1)
        scores.append(metric(data.y.cpu().tolist(), pred.cpu().tolist(), average='macro'))
    return np.mean(scores)