"""Stage 3d: 1D-CNN autoencoders trained on NORMAL windows only (overlap 0%).

Variants:
  raw  : input = 256-sample window, standardised with the train-normal mean/std
  spec : input = log FFT magnitude, bins 1..128 (DC dropped), standardised per bin
Anomaly score = reconstruction MSE. Same protocol as tune.py: 5-fold GroupKFold over Training motors for
out-of-fold ranks and the threshold; final model on all train normals; Test evaluated once.
Also tests the mean-rank ensemble with the stage-3c pick (IsoForest 500/mf0.5 + OCSVM nu0.01 on time+band).
"""
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from anomaly import Detector, rank_in
from features import build, fft_features
from load import split_windows

SEED = 0
OUT = Path(__file__).resolve().parent.parent / 'results'
torch.set_num_threads(8)
GRID = np.linspace(0.5, 1.0, 251)


class ConvAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        C = torch.nn.Conv1d
        T = torch.nn.ConvTranspose1d
        R = torch.nn.ReLU
        self.enc = torch.nn.Sequential(C(1, 16, 7, 2, 3), R(), C(16, 32, 5, 2, 2), R(), C(32, 32, 5, 2, 2), R())
        self.dec = torch.nn.Sequential(T(32, 32, 4, 2, 1), R(), T(32, 16, 4, 2, 1), R(), T(16, 1, 4, 2, 1))

    def forward(self, x):
        return self.dec(self.enc(x))


class CnnDetector:
    def __init__(self, kind):
        self.kind = kind

    def _prep(self, X):
        if self.kind == 'spec':
            X = fft_features(X)[:, 1:129]
        return X

    def fit(self, X):
        torch.manual_seed(SEED)
        Z = self._prep(X)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        t = torch.tensor((Z - self.mu) / self.sd, dtype=torch.float32)[:, None]
        self.net = ConvAE()
        opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        g = torch.Generator().manual_seed(SEED)
        for _ in range(25):
            for idx in torch.randperm(len(t), generator=g).split(256):
                opt.zero_grad()
                loss = ((self.net(t[idx]) - t[idx]) ** 2).mean()
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    def score(self, X):
        Z = self._prep(X)
        t = torch.tensor((Z - self.mu) / self.sd, dtype=torch.float32)[:, None]
        with torch.no_grad():
            return np.concatenate([((self.net(b) - b) ** 2).mean((1, 2)).numpy() for b in t.split(4096)])


def best_t(s, y):
    f = [f1_score(y, s >= t) for t in GRID]
    i = int(np.argmax(f))
    return GRID[i], f[i]


(Xtr, ytr, gtr), (Xte, yte, gte) = split_windows(0.0)
Btr, Bte = build(Xtr, 'time+band'), build(Xte, 'time+band')
dets = {
    'CNN-raw': (lambda: CnnDetector('raw'), Xtr, Xte),
    'CNN-spec': (lambda: CnnDetector('spec'), Xtr, Xte),
    'IsoF500_mf0.5': (lambda: Detector('IsoForest', n_estimators=500, max_features=0.5), Btr, Bte),
    'OCSVM_nu0.01': (lambda: Detector('OCSVM', nu=0.01), Btr, Bte),
}
oof, te = {}, {}
for name, (make, Ftr, Fte) in dets.items():
    t0 = time.time()
    o = np.zeros(len(ytr))
    for tr, va in GroupKFold(5).split(Ftr, ytr, gtr):
        nrm = tr[ytr[tr] == 0]
        d = make().fit(Ftr[nrm])
        o[va] = rank_in(d.score(Ftr[nrm]), d.score(Ftr[va]))
    d = make().fit(Ftr[ytr == 0])
    oof[name], te[name] = o, rank_in(d.score(Ftr[ytr == 0]), d.score(Fte))
    print(f'{name}: {time.time() - t0:.0f}s', flush=True)

rows = []
for r in (1, 2, 3, 4):
    for combo in itertools.combinations(dets, r):
        t, cv = best_t(np.mean([oof[c] for c in combo], 0), ytr)
        pred = np.mean([te[c] for c in combo], 0) >= t
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        miss = {int(g % 100): round(float(1 - pred[gte == g].mean()), 2) for g in sorted(set(gte)) if g >= 100}
        rows.append(dict(ensemble='+'.join(combo), t=round(t, 3), cv_f1=round(cv, 4), test_f1=round(f1_score(yte, pred), 4),
                         fp=fp, fn=fn, abnormal_miss_rate=miss))
df = pd.DataFrame(rows).sort_values('cv_f1', ascending=False)
df.to_csv(OUT / 'cnn.csv', index=False)
print(df.drop(columns='abnormal_miss_rate').to_string(index=False))
