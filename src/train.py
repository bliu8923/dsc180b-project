import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score
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

        try:
            out = model(data.x, data.edge_index)
        except:
            out = model(data)

        loss, pred_score = criterion(out, data.y)
        if metric == f1_score:
            _, pred = torch.max(F.softmax(out, dim=1), 1)
        elif metric == average_precision_score:
            pred = F.softmax(out, dim=1)
        loss.backward()
        optimizer.step()
        total_loss.append(float(loss))
        trainscores.append(metric(data.y.cpu().tolist(), pred.cpu().tolist(), average='macro'))

    for data in tqdm(val):
        model.eval()
        data.to(device)
        try:
            pred = model(data.x, data.edge_index)
            _, pred = torch.max(pred, 1)
        except:
            pred = model(data)
            if metric == f1_score:
                _, pred = torch.max(F.softmax(pred, dim=1), 1)
            elif metric == average_precision_score:
                pred = F.softmax(pred, dim=1)
        scores.append(metric(data.y.cpu().tolist(), pred.cpu().tolist(), average='macro'))
    return np.mean(total_loss), np.mean(trainscores), np.mean(scores)