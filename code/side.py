"""Stage 4d side experiments on the stage 4 final model (top row of results/tune4.csv). NOT the final model.

1. File-level smoothing: mean rank of the windows cut from one CSV (4 windows at 0% overlap), threshold from
   cv OOF. Shows the possible gain if decisions were made per file; the official result stays per window.
2. Seed variance: IsoForest seeds 0..4 for the final ensemble, Test F1 mean +- std.
Output: results/side4.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import anomaly
from anomaly import Detector, rank_in
from features import build
from load import make_windows, split_windows
from tune import CONFIGS, best_t

OUT = Path(__file__).resolve().parent.parent / 'results'


def ranks(Ftr, ytr, Fte, folds, combo):
    o, te = np.zeros(len(ytr)), np.zeros(len(Fte))
    for c in combo:
        kind, kw = CONFIGS[c]
        for tr, va in folds:
            nrm = tr[ytr[tr] == 0]
            d = Detector(kind, **kw).fit(Ftr[nrm])
            o[va] += rank_in(d.score(Ftr[nrm]), d.score(Ftr[va])) / len(combo)
        d = Detector(kind, **kw).fit(Ftr[ytr == 0])
        te += rank_in(d.score(Ftr[ytr == 0]), d.score(Fte)) / len(combo)
    return o, te


def file_mean(s, per):
    """replace each window score by the mean over its file (windows of one file are contiguous)."""
    return np.repeat(s.reshape(-1, per).mean(1), per)


def main():
    best = pd.read_csv(OUT / 'tune4.csv').iloc[0]
    ov, fk, combo = best.overlap / 100, best.features, best.ensemble.split('+')
    (Xtr, ytr, gtr), (Xte, yte, _) = split_windows(ov)
    per = len(make_windows(np.zeros(1024), ov))
    Ftr, Fte = build(Xtr, fk), build(Xte, fk)
    folds = list(GroupKFold(5).split(Ftr, ytr, gtr))
    rows = []
    seeds = []
    for seed in range(5):
        anomaly.SEED = seed
        o, te = ranks(Ftr, ytr, Fte, folds, combo)
        t, cv = best_t(o, ytr)
        seeds.append(f1_score(yte, te >= t))
        if seed == 0:
            rows.append(dict(exp='per window (official)', cv_f1=round(cv, 4), test_f1=round(seeds[0], 4)))
            ts, cvs = best_t(file_mean(o, per), ytr)
            rows.append(dict(exp='file-level mean (side)', cv_f1=round(cvs, 4),
                             test_f1=round(f1_score(yte, file_mean(te, per) >= ts), 4)))
    rows.append(dict(exp=f'seeds 0-4 test F1 mean+-std', cv_f1=np.nan,
                     test_f1=f'{np.mean(seeds):.4f} +- {np.std(seeds):.4f}'))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'side4.csv', index=False)
    print(f'model: overlap {ov}, {fk}, {"+".join(combo)}')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
