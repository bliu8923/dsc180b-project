import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm

from src.models.san import SAN


def train(loader, val, model, optimizer, criterion, device, metric):
    model.to(device)
    model.train()
    total_loss = []
    trainscores = []
    scores = []

    for data in tqdm(loader):
        model.train()
        data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss, pred_score = criterion(out, data.y)
        if metric == f1_score:
            _, pred = torch.max(F.log_softmax(out, dim=1), 1)
        elif metric == average_precision_score:
            pred = F.log_softmax(out, dim=1)
        loss.backward()
        optimizer.step()
        total_loss.append(float(loss))
        trainscores.append(metric(data.y.cpu().tolist(), pred.cpu().tolist(), average='macro'))

    for data in tqdm(val):
        model.eval()
        data.to(device)
        pred = model(data)
        if metric == f1_score:
            _, pred = torch.max(F.log_softmax(pred, dim=1), 1)
        elif metric == average_precision_score:
            pred = F.log_softmax(pred, dim=1)
        scores.append(metric(data.y.cpu().tolist(), pred.cpu().tolist(), average='macro'))
    return np.mean(total_loss), np.mean(trainscores), np.mean(scores), model