"""Stage 3a: anomaly detection trained on NORMAL windows only.

Threshold choice uses Training motors (11-40) only:
- 5-fold GroupKFold over train motors. Each fold: fit on the in-fold normal windows, score the held-out
  windows (normal + abnormal), and express each score as its rank among the in-fold normal training scores
  (0..1, scale-free).
- cutoff q = the rank that maximises F1 on the pooled held-out windows.
Final model: fit on all train normal windows, threshold = q-quantile of their scores, evaluate on Test (1-10) once.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from features import build
from load import split_windows

SEED = 0
OUT = Path(__file__).resolve().parent.parent / 'results'
torch.manual_seed(SEED)
np.random.seed(SEED)


class Detector:
    """fit(normal X) / score(X) -> higher = more anomalous. Standardises with the normal data."""

    def __init__(self, kind, **kw):
        self.kind = kind
        self.kw = kw

    def fit(self, X):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        k, kw = self.kind, self.kw
        if k == 'OCSVM':
            self.m = OneClassSVM(nu=kw.get('nu', 0.05), gamma=kw.get('gamma', 'scale')).fit(Z)
        elif k == 'IsoForest':
            self.m = IsolationForest(n_estimators=kw.get('n_estimators', 300), max_samples=kw.get('max_samples', 'auto'),
                                     max_features=kw.get('max_features', 1.0), random_state=SEED, n_jobs=-1).fit(Z)
        elif k == 'Mahalanobis':
            self.m = LedoitWolf().fit(Z)
        elif k == 'PCA-recon':
            self.m = PCA(n_components=0.95, random_state=SEED).fit(Z)
        elif k == 'kNN-dist':
            self.m = NearestNeighbors(n_neighbors=kw.get('k', 10)).fit(Z)
        elif k == 'Autoencoder':
            self.m = self._train_ae(Z)
        return self

    def score(self, X):
        Z = self.sc.transform(X)
        k = self.kind
        if k == 'OCSVM':
            return -self.m.decision_function(Z)
        if k == 'IsoForest':
            return -self.m.score_samples(Z)
        if k == 'Mahalanobis':
            return self.m.mahalanobis(Z)
        if k == 'PCA-recon':
            return ((Z - self.m.inverse_transform(self.m.transform(Z))) ** 2).mean(1)
        if k == 'kNN-dist':
            return self.m.kneighbors(Z)[0].mean(1)
        with torch.no_grad():
            t = torch.tensor(Z, dtype=torch.float32)
            return ((self.m(t) - t) ** 2).mean(1).numpy()

    @staticmethod
    def _train_ae(Z):
        d = Z.shape[1]
        net = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, 16), torch.nn.ReLU(),
                                  torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, d))
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        t = torch.tensor(Z, dtype=torch.float32)
        for _ in range(40):
            for idx in torch.randperm(len(t)).split(256):
                opt.zero_grad()
                loss = ((net(t[idx]) - t[idx]) ** 2).mean()
                loss.backward()
                opt.step()
        return net.eval()


def rank_in(ref, s):
    """fraction of reference (normal training) scores below each s -> 0..1"""
    ref = np.sort(ref)
    return np.searchsorted(ref, s) / len(ref)


def best_cutoff(r, y):
    qs = np.linspace(0.5, 1.0, 501)
    f1s = [f1_score(y, r >= q) for q in qs]
    i = int(np.argmax(f1s))
    return qs[i], f1s[i]


KINDS = ['Mahalanobis', 'PCA-recon', 'kNN-dist', 'IsoForest', 'OCSVM', 'Autoencoder']
FEATS = ['fft', 'time+fft']


def main(ov):
    (Xtr, ytr, gtr), (Xte, yte, gte) = split_windows(ov)
    rows = []
    for fk in FEATS:
        Ftr, Fte = build(Xtr, fk), build(Xte, fk)
        for kind in KINDS:
            t0 = time.time()
            ranks, ys = [], []
            for tr, va in GroupKFold(5).split(Ftr, ytr, gtr):
                nrm = tr[ytr[tr] == 0]
                det = Detector(kind).fit(Ftr[nrm])
                ranks.append(rank_in(det.score(Ftr[nrm]), det.score(Ftr[va])))
                ys.append(ytr[va])
            q, cv_f1 = best_cutoff(np.concatenate(ranks), np.concatenate(ys))
            det = Detector(kind).fit(Ftr[ytr == 0])
            thr = np.quantile(det.score(Ftr[ytr == 0]), q)
            pred = det.score(Fte) >= thr
            tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
            per_motor = {int(g % 100): round(float(np.mean(pred[gte == g])), 2) for g in sorted(set(gte)) if g >= 100}
            r = dict(overlap=int(ov * 100), features=fk, model=kind, cutoff=round(q, 3), cv_f1=round(cv_f1, 4),
                     test_f1=round(f1_score(yte, pred), 4), tn=tn, fp=fp, fn=fn, tp=tp, sec=round(time.time() - t0, 1),
                     abnormal_motor_recall=per_motor)
            rows.append(r)
            print(r, flush=True)
    df = pd.DataFrame(rows).sort_values('cv_f1', ascending=False)
    df.to_csv(OUT / f'anomaly_{int(ov * 100)}.csv', index=False)
    print(df.drop(columns='abnormal_motor_recall').to_string(index=False))


if __name__ == '__main__':
    main(float(sys.argv[1]) / 100 if len(sys.argv) > 1 else 0.0)
