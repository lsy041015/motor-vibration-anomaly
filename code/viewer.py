"""Builds ../viewer.html: a self-contained page to inspect the final model on the Test motors.

Final model = top row of results/tune4.csv (chosen by Training-motor CV only): overlap, feature set,
ensemble of detector configs and threshold. Fit on all Training normal windows; Test motors (1-10) only scored.
Also stores the estimated rotation frequency f0 per window so the FFT panel can draw k*f0 lines.
"""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from anomaly import Detector, rank_in
from features import FAULT_BANDS, FS, TIME_NAMES, build, estimate_f0, fft_features
from load import load_signals, make_windows, TEST_MOTORS
from tune import CONFIGS

ROOT = Path(__file__).resolve().parent.parent
tune = pd.read_csv(ROOT / 'results' / 'tune4.csv')
best = tune.iloc[0]
OV, FEAT, COMBO, THRESHOLD = best.overlap / 100, best.features, best.ensemble.split('+'), float(best.t)
PART_NAMES = {'time': TIME_NAMES,
              'harm': [f'고조파 {k}차' for k in range(1, 7)] + ['고조파 합', '비고조파'],
              'fault': [f'불량대역 {a}-{b}Hz' for a, b in FAULT_BANDS]}

x, y, motor, _ = load_signals()
is_test = np.isin(motor, list(TEST_MOTORS))

Wtr = np.concatenate([make_windows(s, OV) for s in x[(~is_test) & (y == 0)]])
Btr = build(Wtr, FEAT)
names = [n for p in FEAT.split('+') for n in PART_NAMES.get(p, [])]
if len(names) != Btr.shape[1]:  # band/order parts: plain indices
    names = [f'f{i}' for i in range(Btr.shape[1])]
mu, sd = Btr.mean(0), Btr.std(0) + 1e-12
dets = [Detector(CONFIGS[c][0], **CONFIGS[c][1]).fit(Btr) for c in COMBO]
ref = [d.score(Btr) for d in dets]


def window_scores(W):
    """mean rank (final score), per-detector ranks [n, k], feature z-scores vs Training normals [n, d]"""
    B = build(W, FEAT)
    ranks = np.column_stack([rank_in(r, d.score(B)) for d, r in zip(dets, ref)])
    return ranks.mean(1), ranks, (B - mu) / sd


normal_spec = fft_features(Wtr).mean(0)
abnormal_spec = fft_features(np.concatenate([make_windows(s, OV) for s in x[(~is_test) & (y == 1)]])).mean(0)

motors, all_y, all_p = [], [], []
for lab in (0, 1):
    for m in TEST_MOTORS:
        idx = np.where(is_test & (y == lab) & (motor == m))[0]
        files = []
        for i in idx:
            W = make_windows(x[i], OV)
            s, ranks, z = window_scores(W)
            q = np.clip(np.round(x[i] / 2 * 32767), -32767, 32767).astype('<i2')
            files.append({'sig': base64.b64encode(q.tobytes()).decode(), 'score': (np.floor(s * 1000 + 1e-9) / 1000).tolist(),  # floor, not round: threshold has 3 decimals, so s >= t is preserved
                          'f0': np.round(estimate_f0(W), 1).tolist(), 'ranks': np.round(ranks, 3).tolist(),
                          'z': np.round(z, 1).tolist()})
            all_y += [lab] * len(s)
            all_p += (s >= THRESHOLD).tolist()
        motors.append({'name': f"{'불량' if lab else '정상'} {m}번", 'label': lab, 'files': files})
    print('class', lab, 'done', flush=True)

all_y, all_p = np.array(all_y), np.array(all_p)
tn, fp, fn, tp = confusion_matrix(all_y, all_p).ravel()
step = int(256 * (1 - OV))
summary = {'f1': round(f1_score(all_y, all_p), 4), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
           'threshold': THRESHOLD, 'fs': FS, 'step': step, 'model': f"{FEAT} · {' + '.join(COMBO)} · 겹침 {int(OV * 100)}%",
           'cv_f1': float(best.cv_f1), 'spd_f1': float(best.spd_f1), 'flt_f1': float(best.flt_f1)}
cols = ['overlap', 'features', 'ensemble', 't', 'cv_f1', 'spd_f1', 'flt_f1', 'test_f1', 'fp', 'fn']
compare = tune.groupby(['overlap', 'features'], sort=False).head(1)[cols].head(12)  # tune4.csv is already in selection order
data = {'summary': summary, 'motors': motors, 'freqs': np.round(np.fft.rfftfreq(256, 1 / FS), 2).tolist(),
        'feat_names': names, 'detectors': COMBO, 'compare': compare.to_dict('records'),
        'normal_spec': np.round(normal_spec, 4).tolist(), 'abnormal_spec': np.round(abnormal_spec, 4).tolist(),
        'fault_bands': FAULT_BANDS}
html = (Path(__file__).with_name('viewer_template.html').read_text(encoding='utf-8')
        .replace('/*DATA*/null', json.dumps(data, ensure_ascii=False, default=lambda o: o.item())))
(ROOT / 'viewer.html').write_text(html, encoding='utf-8')
print(summary)
