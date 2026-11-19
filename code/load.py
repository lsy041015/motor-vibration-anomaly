"""IC-PBL data loading and windowing.

Rules from the lecture (p.14-15):
- split by motor folder: 1-10 = test, 11-40 = train, for both classes
- cut each signal into 256-sample windows, overlap 0 / 0.25 / 0.5, same for train and test
"""
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / 'data'
CACHE = Path(__file__).resolve().parent.parent / 'cache' / 'signals.npz'
CLASSES = {'normal': 0, 'abnormal': 1}
TEST_MOTORS = range(1, 11)
WIN = 256


def load_signals():
    """All CSV files as one array (n_files, 1024) plus label, motor id and dt per file. Cached as npz."""
    if CACHE.exists():
        z = np.load(CACHE)
        return z['x'], z['y'], z['motor'], z['dt']
    xs, ys, motors, dts = [], [], [], []
    for cls, label in CLASSES.items():
        for motor in range(1, 41):
            for f in sorted((DATA / cls / str(motor) / 'time_vib').glob('*.csv')):
                a = np.loadtxt(f, delimiter=',')
                xs.append(a[:, 1])
                dts.append(np.diff(a[:, 0]).mean())
                ys.append(label)
                motors.append(motor)
    x = np.stack(xs).astype(np.float32)
    y, motor, dt = np.array(ys), np.array(motors), np.array(dts)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, x=x, y=y, motor=motor, dt=dt)
    return x, y, motor, dt


def make_windows(sig, overlap=0.0, size=WIN):
    """Cut one 1-D signal into windows of `size`; overlap is 0, 0.25 or 0.5."""
    step = int(size * (1 - overlap))
    starts = range(0, len(sig) - size + 1, step)
    return np.stack([sig[s:s + size] for s in starts])


def split_windows(overlap=0.0):
    """Windowed train/test sets. Returns (Xtr, ytr, gtr), (Xte, yte, gte); g = motor group id (class*100 + motor)."""
    x, y, motor, _ = load_signals()
    parts = {'train': ([], [], []), 'test': ([], [], [])}
    for sig, label, m in zip(x, y, motor):
        w = make_windows(sig, overlap)
        key = 'test' if m in TEST_MOTORS else 'train'
        parts[key][0].append(w)
        parts[key][1].append(np.full(len(w), label))
        parts[key][2].append(np.full(len(w), label * 100 + m))
    out = []
    for key in ('train', 'test'):
        X, Y, G = (np.concatenate(v) for v in parts[key])
        out.append((X, Y, G))
    return tuple(out)


if __name__ == '__main__':
    sig = np.arange(1024)
    assert len(make_windows(sig, 0.0)) == 4
    assert len(make_windows(sig, 0.25)) == 5
    assert len(make_windows(sig, 0.5)) == 7
    assert make_windows(sig, 0.5)[1][0] == 128
    (Xtr, ytr, gtr), (Xte, yte, gte) = split_windows(0.0)
    assert not set(gtr) & set(gte), 'train/test motors overlap'
    print('ok', Xtr.shape, Xte.shape)
