"""Window features: time-domain statistics, FFT magnitudes and speed-aware (order-tracking) features.

Stage 4 finding: every normal motor spins at its own rotation frequency f0 (about 25-54 Hz), while fault
signatures sit at fixed frequencies (about 85-88, 108 and 185 Hz). A normal motor at 54 Hz has its 2nd harmonic
at 108 Hz and looks faulty to plain FFT features. The speed-aware features below estimate f0 per window and
separate harmonic energy (k*f0, normal rotation) from non-harmonic energy in the fault bands.
"""
import numpy as np
from scipy.stats import kurtosis, skew

FS = 401.9  # Hz, measured in eda.py
NFFT = 1024  # zero-padded FFT length for speed features: 0.39 Hz bins
F0_RANGE = (15.0, 60.0)  # Hz, search range for the rotation frequency
N_HARM = 12  # harmonics masked (12 * 15 Hz covers up to Nyquist for slow motors)
HARM_TOL = 2.5  # Hz, half-width of the band treated as one harmonic (Hann main lobe of a 256 window ~ 3 Hz)
FAULT_BANDS = [(55, 65), (80, 92), (100, 115), (180, 190)]  # Hz, from the stage 1 EDA
ORDERS = np.linspace(0.5, 8.0, 64)


def time_features(X):
    """9 time-domain statistics per 256-sample window."""
    rms = np.sqrt((X ** 2).mean(1))
    peak = np.abs(X).max(1)
    zc = (np.diff(np.sign(X - X.mean(1, keepdims=True)), axis=1) != 0).mean(1)
    return np.column_stack([X.mean(1), X.std(1), rms, peak, X.max(1) - X.min(1),
                            peak / (rms + 1e-12), kurtosis(X, axis=1), skew(X, axis=1), zc])


TIME_NAMES = ['mean', 'std', 'rms', 'peak', 'p2p', 'crest', 'kurtosis', 'skew', 'zero_cross']


def fft_features(X):
    """log FFT magnitude of the mean-removed window: 129 bins (0..~201 Hz)."""
    mag = np.abs(np.fft.rfft(X - X.mean(1, keepdims=True), axis=1)) / X.shape[1]
    return np.log1p(mag * 100)


def band_features(X, width=4):
    """log FFT magnitude averaged over `width` neighbouring bins (129 -> 32 bands, last bin dropped): less noisy than single bins."""
    mag = np.abs(np.fft.rfft(X - X.mean(1, keepdims=True), axis=1)) / X.shape[1]
    n = mag.shape[1] // width * width
    return np.log1p(mag[:, :n].reshape(len(X), -1, width).mean(2) * 100)


# ---------- speed-aware features ----------

def power_spectrum(X):
    """Hann-windowed, zero-padded power spectrum normalised to sum 1 per window (amplitude-free). Returns (P, freqs)."""
    Z = (X - X.mean(1, keepdims=True)) * np.hanning(X.shape[1])
    P = np.abs(np.fft.rfft(Z, n=NFFT, axis=1)) ** 2
    P[:, 0] = 0
    return P / (P.sum(1, keepdims=True) + 1e-20), np.fft.rfftfreq(NFFT, 1 / FS)


def estimate_f0(X, P=None, f=None):
    """Rotation frequency per window: strongest peak in F0_RANGE, refined by parabolic interpolation."""
    if P is None:
        P, f = power_spectrum(X)
    lo, hi = np.searchsorted(f, F0_RANGE)
    i = lo + P[:, lo:hi].argmax(1)
    a, b, c = (np.log(P[np.arange(len(P)), j] + 1e-20) for j in (i - 1, i, i + 1))
    shift = np.clip(0.5 * (a - c) / (a - 2 * b + c + 1e-20), -0.5, 0.5)
    return (i + shift) * (f[1] - f[0])


def harmonic_mask(f0, f):
    """bool (n, n_freq): True where a bin lies within HARM_TOL of k*f0, k = 1..N_HARM."""
    m = np.zeros((len(f0), len(f)), bool)
    for k in range(1, N_HARM + 1):  # loop keeps memory at one (n, n_freq) array
        m |= np.abs(f[None, :] - k * f0[:, None]) <= HARM_TOL
    return m


def speed_features(X):
    """harmonic + fault-band features, one pass. Returns (harm[n, 8], fault[n, 4], f0[n]).

    harm : log energy share of harmonics 1..6, total harmonic share, non-harmonic (residual) share
    fault: log share of NON-harmonic energy inside each fixed fault band
    f0 itself is not a feature: a rare speed must not look abnormal.
    """
    P, f = power_spectrum(X)
    f0 = estimate_f0(X, P, f)
    m = harmonic_mask(f0, f)
    harm = []
    for k in range(1, 7):
        near = np.abs(f[None, :] - k * f0[:, None]) <= HARM_TOL
        harm.append((P * near).sum(1))
    h_tot = (P * m).sum(1)
    harm = np.log(np.column_stack(harm + [h_tot, 1 - h_tot]) + 1e-6)
    resid = P * ~m
    fault = np.log(np.column_stack([resid[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in FAULT_BANDS]) + 1e-6)
    return harm, fault, f0


def order_spectrum(X):
    """log power share resampled onto the order axis f/f0 (0.5..8, 64 points). Orders above Nyquist get the floor."""
    P, f = power_spectrum(X)
    f0 = estimate_f0(X, P, f)
    out = np.empty((len(X), len(ORDERS)))
    for n in range(len(X)):
        out[n] = np.interp(ORDERS * f0[n], f, P[n], right=0.0)
    return np.log(out + 1e-6)


def build(X, kind):
    """Feature matrix for a feature-set name; parts joined with '+', e.g. 'time+band+harm'."""
    parts, cache = [], {}

    def speed():
        if 'speed' not in cache:
            cache['speed'] = speed_features(X)
        return cache['speed']

    for p in kind.split('+'):
        if p == 'time':
            parts.append(time_features(X))
        elif p == 'fft':
            parts.append(fft_features(X))
        elif p == 'band':
            parts.append(band_features(X))
        elif p == 'harm':
            parts.append(speed()[0])
        elif p == 'fault':
            parts.append(speed()[1])
        elif p == 'order':
            parts.append(order_spectrum(X))
        else:
            raise ValueError(f'unknown feature part {p!r}')
    return np.hstack(parts)


if __name__ == '__main__':
    t = np.arange(256) / FS
    # 40 Hz rotation + its 3rd harmonic: f0 found, 120 Hz energy counted as harmonic, fault bands empty
    clean = np.sin(2 * np.pi * 40 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    # same rotation + 88 Hz fault tone (not a multiple of 40)
    faulty = np.sin(2 * np.pi * 40 * t) + 0.5 * np.sin(2 * np.pi * 88 * t)
    # 54 Hz rotation: 2nd harmonic at 108 Hz lands inside a fault band but must stay harmonic
    fast = np.sin(2 * np.pi * 54 * t) + 0.5 * np.sin(2 * np.pi * 108 * t)
    harm, fault, f0 = speed_features(np.stack([clean, faulty, fast]))
    assert abs(f0[0] - 40) < 0.5 and abs(f0[2] - 54) < 0.5, f0
    assert harm[0, 6] > np.log(0.95), 'clean: almost all energy should be harmonic'
    assert fault[1, 1] > np.log(0.1) > fault[0, 1], '88 Hz must show in the 80-92 band only for the faulty signal'
    assert fault[2, 2] < np.log(0.01), '108 Hz harmonic of 54 Hz must not count as fault energy'
    o = order_spectrum(np.stack([clean, faulty]))
    i3, i22 = np.argmin(abs(ORDERS - 3)), np.argmin(abs(ORDERS - 2.2))
    assert o[0, i3] > o[0, i22] and o[1, i22] > o[0, i22], 'order axis: harmonic at 3, fault at 2.2'
    assert build(np.stack([clean, faulty]), 'time+band+harm+fault').shape == (2, 9 + 32 + 8 + 4)
    print('ok', np.round(f0, 2))
