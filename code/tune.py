"""Stage 3c/4c: tune the anomaly-score ensemble with Training-motor CV only.

For each overlap and feature set:
  - out-of-fold score ranks for every detector config (fit on fold normals only), under two CV schemes:
      cv  : 5-fold GroupKFold over Training motors (selection score, as in stage 3)
      spd : leave-speed-group-out. Training normals are split by median rotation frequency f0 into
            slow (<30 Hz) / mid (30-45 Hz) / fast (>=45 Hz); each fold holds out one speed group plus a third
            of the abnormal motors. Simulates a Test motor running at a speed rarely seen in training.
  - test ranks for every config (fit on all train normals)
  - every ensemble = mean rank of 1..3 configs; threshold t chosen on OOF F1 of the `cv` scheme
      flt : leave-fault-type-out (abnormal motors clustered into 3 fault types, one type held out per fold).
Selection: highest cv F1, ties (within 0.002) broken by spd F1. Test F1 is recorded but never used to choose.
Usage: python tune.py [overlap% ...]   (default 0)
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from anomaly import Detector, rank_in
from features import build, estimate_f0
from load import split_windows

OUT = Path(__file__).resolve().parent.parent / 'results'
CONFIGS = {
    'Maha': ('Mahalanobis', {}),
    'IsoF300': ('IsoForest', {}),
    'IsoF500_mf0.5': ('IsoForest', {'n_estimators': 500, 'max_features': 0.5}),
    'OCSVM_nu0.01': ('OCSVM', {'nu': 0.01}),
    'OCSVM_nu0.05': ('OCSVM', {'nu': 0.05}),
    'OCSVM_nu0.1': ('OCSVM', {'nu': 0.1}),
    'kNN10': ('kNN-dist', {'k': 10}),
    'kNN30': ('kNN-dist', {'k': 30}),
}
FEATS = ['time+band', 'time+band+harm', 'time+band+fault', 'time+band+harm+fault', 'time+harm+fault',
         'time+order', 'time+order+fault']
GRID = np.linspace(0.5, 1.0, 251)
SPEED_EDGES = (30.0, 45.0)


def best_t(s, y):
    f = [f1_score(y, s >= t) for t in GRID]
    i = int(np.argmax(f))
    return GRID[i], f[i]


def speed_folds(X, y, g):
    """leave-speed-group-out folds: (train_idx, val_idx) list. Normal motors grouped by median f0."""
    f0 = estimate_f0(X)
    motors = sorted(set(g))
    abn = [m for m in motors if m >= 100]
    bins = {m: int(np.digitize(np.median(f0[g == m]), SPEED_EDGES)) for m in motors if m < 100}
    folds = []
    for k in range(3):
        held = {m for m, b in bins.items() if b == k} | set(abn[k::3])
        va = np.isin(g, list(held))
        folds.append((np.where(~va)[0], np.where(va)[0]))
    return folds, bins


def fault_type_folds(X, y, g):
    """leave-fault-type-out folds. Training abnormal motors are clustered (KMeans, k=3) by their mean log
    energy share in the fixed fault bands; each fold holds out one fault type plus a third of the normal
    motors. Simulates a Test fault type not seen in training (the stage 2 failure)."""
    from sklearn.cluster import KMeans
    from features import speed_features
    _, fault, _ = speed_features(X)
    abn = sorted(m for m in set(g) if m >= 100)
    prof = np.array([fault[g == m].mean(0) for m in abn])
    lab = KMeans(3, n_init=10, random_state=0).fit_predict(prof)
    nrm = sorted(m for m in set(g) if m < 100)
    folds = []
    for k in range(3):
        held = {m for m, c in zip(abn, lab) if c == k} | set(nrm[k::3])
        va = np.isin(g, list(held))
        folds.append((np.where(~va)[0], np.where(va)[0]))
    return folds, {int(m % 100): int(c) for m, c in zip(abn, lab)}


def oof(F, y, folds, kind, kw):
    o = np.zeros(len(y))
    for tr, va in folds:
        nrm = tr[y[tr] == 0]
        d = Detector(kind, **kw).fit(F[nrm])
        o[va] = rank_in(d.score(F[nrm]), d.score(F[va]))
    return o


def per_motor(pred, g, motors):
    return {int(m % 100) + (100 if m >= 100 else 0): round(float(np.mean(pred[g == m] == (m >= 100))), 2) for m in motors}


def select_order(df, tie=0.002):
    """Selection rule from the docstring: rows within `tie` of the best cv F1 are ranked by spd F1, then the rest
    by cv F1. Row 0 is the chosen model. (A first version rounded cv to 3 decimals, which did not match the rule.)"""
    near = df.cv_f1 >= df.cv_f1.max() - tie
    return pd.concat([df[near].sort_values(['spd_f1', 'cv_f1'], ascending=False),
                      df[~near].sort_values('cv_f1', ascending=False)])


def main():
    rows = []
    overlaps = [float(a) / 100 for a in sys.argv[1:]] or [0.0]
    for ov in overlaps:
        (Xtr, ytr, gtr), (Xte, yte, gte) = split_windows(ov)
        cv_folds = list(GroupKFold(5).split(Xtr, ytr, gtr))
        spd_folds, bins = speed_folds(Xtr, ytr, gtr)
        print('speed groups (normal motor: bin):', bins, flush=True)
        flt_folds, types = fault_type_folds(Xtr, ytr, gtr)
        print('fault types (abnormal motor: type):', types, flush=True)
        watch = [1, 3, 110]  # normal 1, normal 3, abnormal 10: recorded only
        for fk in FEATS:
            Ftr, Fte = build(Xtr, fk), build(Xte, fk)
            cv, spd, flt, te = {}, {}, {}, {}
            t0 = time.time()
            for name, (kind, kw) in CONFIGS.items():
                cv[name] = oof(Ftr, ytr, cv_folds, kind, kw)
                spd[name] = oof(Ftr, ytr, spd_folds, kind, kw)
                flt[name] = oof(Ftr, ytr, flt_folds, kind, kw)
                d = Detector(kind, **kw).fit(Ftr[ytr == 0])
                te[name] = rank_in(d.score(Ftr[ytr == 0]), d.score(Fte))
            print(f'overlap {ov} {fk}: detectors done in {time.time() - t0:.0f}s', flush=True)
            for r in (1, 2, 3):
                for combo in itertools.combinations(CONFIGS, r):
                    s_cv = np.mean([cv[c] for c in combo], 0)
                    t, cv_f1 = best_t(s_cv, ytr)
                    _, spd_f1 = best_t(np.mean([spd[c] for c in combo], 0), ytr)
                    _, flt_f1 = best_t(np.mean([flt[c] for c in combo], 0), ytr)
                    pred = np.mean([te[c] for c in combo], 0) >= t
                    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
                    pm = per_motor(pred, gte, watch)
                    rows.append(dict(overlap=int(ov * 100), features=fk, ensemble='+'.join(combo), t=round(t, 3),
                                     cv_f1=round(cv_f1, 4), spd_f1=round(spd_f1, 4), flt_f1=round(flt_f1, 4), test_f1=round(f1_score(yte, pred), 4),
                                     fp=fp, fn=fn, n1=pm[1], n3=pm[3], a10=pm[110]))
            pd.DataFrame(rows).to_csv(OUT / 'tune4.csv', index=False)

    df = select_order(pd.DataFrame(rows))
    df.to_csv(OUT / 'tune4.csv', index=False)
    print(df.head(20).to_string(index=False))
    print('\nbest per overlap/features:')
    print(df.groupby(['overlap', 'features']).head(1).to_string(index=False))


if __name__ == '__main__':
    main()
