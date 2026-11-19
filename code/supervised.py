"""Stage 4c: supervised models on speed-aware features.

Stage 2 supervised models on full spectra scored high CV but missed Test fault types. Here the inputs are only
harmonic/fault-band features (rotation speed removed), which may generalise across motors. Same two CV schemes
as tune.py; threshold from `cv` OOF probabilities. Test is recorded, never used to choose.
Output: results/supervised4.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build
from load import split_windows
from tune import fault_type_folds, per_motor, speed_folds

OUT = Path(__file__).resolve().parent.parent / 'results'
MODELS = {
    'LogReg_C0.1': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
    'LogReg_C1': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
    'HistGB_d3': lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0),
}
FEATS = ['fault', 'harm+fault', 'time+harm+fault', 'order', 'time+band']
GRID = np.linspace(0.0, 1.0, 201)


def oof(F, y, folds, make):
    o = np.zeros(len(y))
    for tr, va in folds:
        o[va] = make().fit(F[tr], y[tr]).predict_proba(F[va])[:, 1]
    return o


def best_p(s, y):
    f = [f1_score(y, s >= t) for t in GRID]
    i = int(np.argmax(f))
    return GRID[i], f[i]


def main():
    (Xtr, ytr, gtr), (Xte, yte, gte) = split_windows(0.0)
    cv_folds = list(GroupKFold(5).split(Xtr, ytr, gtr))
    spd_folds, _ = speed_folds(Xtr, ytr, gtr)
    flt_folds, _ = fault_type_folds(Xtr, ytr, gtr)
    rows = []
    for fk in FEATS:
        Ftr, Fte = build(Xtr, fk), build(Xte, fk)
        for name, make in MODELS.items():
            t, cv_f1 = best_p(oof(Ftr, ytr, cv_folds, make), ytr)
            _, spd_f1 = best_p(oof(Ftr, ytr, spd_folds, make), ytr)
            _, flt_f1 = best_p(oof(Ftr, ytr, flt_folds, make), ytr)
            pred = make().fit(Ftr, ytr).predict_proba(Fte)[:, 1] >= t
            tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
            pm = per_motor(pred, gte, [1, 3, 110])
            rows.append(dict(features=fk, model=name, t=round(t, 3), cv_f1=round(cv_f1, 4), spd_f1=round(spd_f1, 4), flt_f1=round(flt_f1, 4),
                             test_f1=round(f1_score(yte, pred), 4), fp=fp, fn=fn, n1=pm[1], n3=pm[3], a10=pm[110]))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows).sort_values('cv_f1', ascending=False)
    df.to_csv(OUT / 'supervised4.csv', index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
