"""Stage 4a: where does the stage 3 model fail on TRAINING motors? (Test is not used here.)

Out-of-fold predictions of the stage 3 final ensemble (time+band, IsoForest500 mf0.5 + OCSVM nu0.01, mean rank)
per Training motor, next to each motor's median rotation frequency f0 and its share of energy in fault bands.
Output: results/4a_diagnose.csv and results/4a_spectra.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from anomaly import Detector, rank_in
from features import build, estimate_f0, power_spectrum, speed_features
from load import split_windows

OUT = Path(__file__).resolve().parent.parent / 'results'
ENSEMBLE = [('IsoForest', {'n_estimators': 500, 'max_features': 0.5}), ('OCSVM', {'nu': 0.01})]


def oof_ranks(F, y, g, ensemble):
    """mean OOF rank of an ensemble (each detector fit on fold normals only)."""
    out = np.zeros(len(y))
    for kind, kw in ensemble:
        o = np.zeros(len(y))
        for tr, va in GroupKFold(5).split(F, y, g):
            nrm = tr[y[tr] == 0]
            d = Detector(kind, **kw).fit(F[nrm])
            o[va] = rank_in(d.score(F[nrm]), d.score(F[va]))
        out += o / len(ensemble)
    return out


def main():
    (Xtr, ytr, gtr), _ = split_windows(0.0)
    s = oof_ranks(build(Xtr, 'time+band'), ytr, gtr, ENSEMBLE)
    pred = s >= 0.942  # stage 3 threshold
    _, fault, f0 = speed_features(Xtr)
    rows = []
    for g in sorted(set(gtr)):
        i = gtr == g
        rows.append(dict(cls='abnormal' if g >= 100 else 'normal', motor=g % 100, n=int(i.sum()),
                         correct=round(float(np.mean(pred[i] == ytr[i])), 3), f0_median=round(float(np.median(f0[i])), 1),
                         fault_share_108=round(float(np.exp(np.median(fault[i, 2]))), 4)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / '4a_diagnose.csv', index=False)
    print(df.sort_values(['cls', 'correct']).to_string(index=False))

    # spectra: worst training normal vs best normal vs a fault motor
    nrm = df[df.cls == 'normal'].sort_values('correct')
    picks = [('normal', int(nrm.iloc[0].motor), 'worst normal'), ('normal', int(nrm.iloc[-1].motor), 'best normal'),
             ('abnormal', int(df[df.cls == 'abnormal'].iloc[0].motor), 'fault')]
    fig, ax = plt.subplots(len(picks), 1, figsize=(9, 7), sharex=True)
    for a, (cls, m, label) in zip(ax, picks):
        g = (100 if cls == 'abnormal' else 0) + m
        P, f = power_spectrum(Xtr[gtr == g])
        f0m = np.median(estimate_f0(None, P, f))
        a.semilogy(f, P.mean(0), lw=0.8)
        for k in range(1, 8):
            a.axvline(k * f0m, color='g', ls=':', lw=0.8)
        for lo, hi in [(80, 92), (100, 115), (180, 190)]:
            a.axvspan(lo, hi, color='r', alpha=0.1)
        a.set_title(f'{label}: {cls} {m}, f0={f0m:.1f} Hz (green = k*f0, red = fault bands)', fontsize=9)
    ax[-1].set_xlabel('Hz')
    fig.tight_layout()
    fig.savefig(OUT / '4a_spectra.png', dpi=120)


if __name__ == '__main__':
    main()
