import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm

from src.models.san import SAN


def test(test_loader, metric, model, device):

    model.eval()
    scores = []
    for data in tqdm(test_loader):
        data = data.to(device)
        pred = model(data)
        if metric == f1_score:
            _, pred = torch.max(F.log_softmax(pred, dim=1), 1)
        elif metric == average_precision_score:
            pred = F.softmax(pred, dim=1)
        scores.append(metric(data.y.cpu().tolist(), pred.cpu().tolist(), average='macro'))
    return np.mean(scores)