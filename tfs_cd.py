"""
tfs_cd.py — Módulo 3: Extração de Features TFS — Blocos C e D
=============================================================
Versão 1.1

Depende exclusivamente de: tsallis_core.py

Blocos implementados:
    Bloco C — Tsallis sobre F0 e Perturbação (8 features)
    Bloco D — q-Cronoentropograma (12 features)

Total: 20 features por arquivo de áudio.

Uso rápido:
    from tfs_cd import extract_blocks_CD
    feats = extract_blocks_CD(signal=sig, fs=8000)
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt
import warnings
import os

from tsallis_core import (
    tsallis_entropy,
    tsallis_entropy_series,
    q_sweep,
    validate_signal,
)


# ─────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────

TARGET_FS        = 8000
Q_F0_VALUES      = np.array([0.7, 1.3, 2.0])
Q_CHRONO_VALUES  = np.array([1.3, 2.0])

WINDOW_MS  = 50
HOP_MS     = 10
ONSET_MS   = 250
STEADY_MS  = 500


# ─────────────────────────────────────────────────────────────────
# EXTRAÇÃO DE F0 CICLO-A-CICLO
# ─────────────────────────────────────────────────────────────────

def extract_f0_series(signal, fs, f0_min=60.0, f0_max=400.0, method='autocorr'):
    """
    Extrai a série de F0 ciclo-a-ciclo de um sinal de fonação sustentada.
    Tenta YIN via librosa; usa autocorrelação como fallback.
    """
    signal = np.asarray(signal, dtype=np.float64)

    if method in ('yin', 'auto'):
        try:
            import librosa
            frame_length = int(fs * 0.025)
            hop_length   = int(fs * 0.010)
            f0 = librosa.yin(
                signal.astype(np.float32),
                fmin=f0_min, fmax=f0_max, sr=fs,
                frame_length=frame_length, hop_length=hop_length
            )
            f0 = f0[(f0 >= f0_min) & (f0 <= f0_max)]
            if len(f0) >= 10:
                return f0.astype(np.float64)
        except ImportError:
            pass
        except Exception as e:
            warnings.warn("[tfs_cd] YIN falhou: " + str(e) + ". Usando autocorrelacao.")

    return _f0_autocorr(signal, fs, f0_min, f0_max)


def _f0_autocorr(signal, fs, f0_min, f0_max):
    """Autocorrelacao por janelas de 25 ms. Fallback robusto para F0."""
    frame_len = int(fs * 0.025)
    hop_len   = int(fs * 0.010)
    lag_max   = int(fs / f0_min)
    lag_min   = max(int(fs / f0_max), 1)

    f0_list = []
    for start in range(0, len(signal) - frame_len, hop_len):
        frame = signal[start:start + frame_len]
        frame = frame - frame.mean()
        if np.std(frame) < 1e-8:
            continue
        N = len(frame)
        fft_frame = np.fft.rfft(frame, n=2*N)
        acf = np.fft.irfft(fft_frame * np.conj(fft_frame))[:N]
        acf = acf / (acf[0] + 1e-10)
        lag_hi = min(lag_max, N-1)
        if lag_min >= lag_hi:
            continue
        acf_region = acf[lag_min:lag_hi]
        if len(acf_region) == 0:
            continue
        peak_lag = np.argmax(acf_region) + lag_min
        if acf[peak_lag] > 0.30:
            f0_est = fs / peak_lag
            if f0_min <= f0_est <= f0_max:
                f0_list.append(f0_est)

    if len(f0_list) < 10:
        raise ValueError(
            "[tfs_cd] Serie F0 insuficiente: apenas " + str(len(f0_list)) +
            " frames vocalizados."
        )
    return np.array(f0_list, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────
# AMPLITUDES E SHIMMER CICLO-A-CICLO
# ─────────────────────────────────────────────────────────────────

def extract_cycle_amplitudes(signal, fs, f0_median):
    """Amplitudes pico-a-pico de cada ciclo glótico."""
    cycle_len = max(1, int(fs / f0_median))
    n_cycles  = len(signal) // cycle_len
    if n_cycles < 10:
        raise ValueError(
            "[tfs_cd] Poucos ciclos glóticos (" + str(n_cycles) + ")."
        )
    amps = np.array([
        signal[i*cycle_len:(i+1)*cycle_len].max()
        - signal[i*cycle_len:(i+1)*cycle_len].min()
        for i in range(n_cycles)
    ])
    return amps


def extract_shimmer_series(amplitudes):
    """shimmer_i = |A_{i+1} - A_i| / ((A_i + A_{i+1}) / 2)"""
    A = np.maximum(np.asarray(amplitudes, dtype=np.float64), 1e-10)
    return np.abs(np.diff(A)) / ((A[:-1] + A[1:]) / 2.0)


# ─────────────────────────────────────────────────────────────────
# BLOCO C — Tsallis sobre F0 e Perturbação (8 features)
# ─────────────────────────────────────────────────────────────────

def _q_fit_grid(series, q_grid=None):
    """Grid search para q* que melhor ajusta q-Gaussiana à série."""
    if q_grid is None:
        q_grid = np.linspace(0.7, 2.3, 33)
    series = np.asarray(series, dtype=np.float64)
    series = series[np.isfinite(series)]
    mu   = float(np.mean(series))
    var  = float(np.var(series))
    beta = 1.0 / (var + 1e-12)
    best_q, best_ll = 1.0, np.inf
    for q in q_grid:
        arg = 1.0 - (1.0 - q) * beta * (series - mu) ** 2
        if abs(q - 1.0) < 1e-6:
            lp = -0.5 * beta * (series - mu) ** 2
        else:
            exp = 1.0 / (1.0 - q)
            with np.errstate(invalid='ignore', divide='ignore'):
                lp = np.where(arg > 0, exp * np.log(np.maximum(arg, 1e-300)), -1e10)
        finite = lp[np.isfinite(lp) & (lp > -1e9)]
        if len(finite) < 5:
            continue
        neg_ll = -np.mean(finite)
        if neg_ll < best_ll:
            best_ll = neg_ll
            best_q  = float(q)
    return best_q


def extract_block_C(signal, fs=TARGET_FS, n_bins=-1):
    """
    Bloco C: S_q da série F0, das diferenças inter-ciclo |dF0|,
    q* por ajuste de q-Gaussiana, S_q de amplitudes ciclo-a-ciclo,
    S_q do shimmer. Total: 8 features.
    """
    feats = {}
    C_KEYS = ['C_Sq_F0_q0.7', 'C_Sq_F0_q1.3', 'C_Sq_F0_q2.0',
              'C_Sq_dF0_q1.3', 'C_Sq_dF0_q2.0', 'C_q_fit_F0',
              'C_Sq_amp_cycle_q1.3', 'C_Sq_shimmer_q1.3']

    try:
        f0_series = extract_f0_series(signal, fs)
    except ValueError as e:
        warnings.warn("[tfs_cd] Bloco C: " + str(e))
        for k in C_KEYS:
            feats[k] = np.nan
        return feats

    # S_q da série F0
    for q_val in Q_F0_VALUES:
        key = "C_Sq_F0_q" + f"{q_val:.1f}"
        try:
            feats[key] = float(tsallis_entropy_series(f0_series, q=q_val, n_bins=n_bins))
        except Exception as e:
            warnings.warn("[tfs_cd] " + key + " falhou: " + str(e))
            feats[key] = np.nan

    # S_q de |dF0|
    delta_f0 = np.abs(np.diff(f0_series))
    for q_val in [1.3, 2.0]:
        key = "C_Sq_dF0_q" + f"{q_val:.1f}"
        if len(delta_f0) >= 5:
            try:
                feats[key] = float(tsallis_entropy_series(delta_f0, q=q_val, n_bins=n_bins))
            except Exception as e:
                warnings.warn("[tfs_cd] " + key + " falhou: " + str(e))
                feats[key] = np.nan
        else:
            feats[key] = np.nan

    # q* por ajuste de q-Gaussiana
    try:
        feats['C_q_fit_F0'] = float(_q_fit_grid(f0_series))
    except Exception as e:
        warnings.warn("[tfs_cd] C_q_fit_F0 falhou: " + str(e))
        feats['C_q_fit_F0'] = np.nan

    # Amplitudes ciclo-a-ciclo
    try:
        f0_med = float(np.median(f0_series))
        cycle_amps = extract_cycle_amplitudes(signal, fs, f0_med)
        feats['C_Sq_amp_cycle_q1.3'] = float(
            tsallis_entropy_series(cycle_amps, q=1.3, n_bins=n_bins)
        )
    except Exception as e:
        warnings.warn("[tfs_cd] C_Sq_amp_cycle_q1.3 falhou: " + str(e))
        feats['C_Sq_amp_cycle_q1.3'] = np.nan

    # Shimmer
    try:
        f0_med = float(np.median(f0_series))
        cycle_amps = extract_cycle_amplitudes(signal, fs, f0_med)
        shimmer = extract_shimmer_series(cycle_amps)
        feats['C_Sq_shimmer_q1.3'] = float(
            tsallis_entropy_series(shimmer, q=1.3, n_bins=n_bins)
        )
    except Exception as e:
        warnings.warn("[tfs_cd] C_Sq_shimmer_q1.3 falhou: " + str(e))
        feats['C_Sq_shimmer_q1.3'] = np.nan

    return feats


# ─────────────────────────────────────────────────────────────────
# BLOCO D — q-Cronoentropograma (12 features)
# ─────────────────────────────────────────────────────────────────

def compute_chronoentropogram(signal, fs, q, window_ms=WINDOW_MS, hop_ms=HOP_MS, n_bins=-1):
    """
    Calcula C_q(j) = S_q[p_{w,j}(x)] para cada janela j.
    Retorna array temporal de S_q ao longo do sinal.
    """
    signal    = np.asarray(signal, dtype=np.float64)
    w_samples = int(window_ms * fs / 1000)
    h_samples = max(1, int(hop_ms * fs / 1000))

    if w_samples < 20:
        raise ValueError("[tfs_cd] Janela muito pequena: " + str(w_samples) + " amostras.")

    profile = []
    for start in range(0, len(signal) - w_samples, h_samples):
        seg = signal[start:start + w_samples]
        try:
            sq = tsallis_entropy(seg, q=q, n_bins=n_bins)
            profile.append(sq)
        except Exception:
            profile.append(np.nan)

    profile = np.array(profile, dtype=np.float64)
    mask = np.isfinite(profile)
    if mask.sum() < 5:
        raise ValueError("[tfs_cd] Cronoentropograma: frames validos insuficientes.")
    if not mask.all():
        xp = np.where(mask)[0]
        fp = profile[mask]
        profile = np.interp(np.arange(len(profile)), xp, fp)

    return profile


def extract_block_D(signal, fs=TARGET_FS, window_ms=WINDOW_MS, hop_ms=HOP_MS, n_bins=-1):
    """
    Bloco D: 6 features x 2 valores de q = 12 features.
    Features: mean, std, onset, steady, delta, tau.
    """
    feats = {}
    onset_frames = max(1, int(ONSET_MS / hop_ms))

    for q_val in Q_CHRONO_VALUES:
        q_str = "q" + f"{q_val:.1f}"

        try:
            profile = compute_chronoentropogram(
                signal, fs, q=q_val,
                window_ms=window_ms, hop_ms=hop_ms, n_bins=n_bins
            )
        except Exception as e:
            warnings.warn("[tfs_cd] Cronoentropograma " + q_str + " falhou: " + str(e))
            for stat in ['mean', 'std', 'onset', 'steady', 'delta', 'tau']:
                feats["D_" + stat + "_" + q_str] = np.nan
            continue

        n = len(profile)

        feats["D_mean_"   + q_str] = float(np.mean(profile))
        feats["D_std_"    + q_str] = float(np.std(profile))

        n_onset = min(max(onset_frames, n // 5), n - 1)
        feats["D_onset_"  + q_str] = float(np.mean(profile[:n_onset]))

        c_start = n_onset
        c_end   = n - n_onset
        if c_end > c_start + 2:
            feats["D_steady_" + q_str] = float(np.mean(profile[c_start:c_end]))
        else:
            feats["D_steady_" + q_str] = feats["D_mean_" + q_str]

        n_final = min(max(onset_frames, n // 5), n - 1)
        s_final = float(np.mean(profile[-n_final:]))
        feats["D_delta_"  + q_str] = float(s_final - feats["D_onset_" + q_str])
        feats["D_tau_"    + q_str] = float(np.argmax(profile) / max(n - 1, 1))

    return feats


# ─────────────────────────────────────────────────────────────────
# LOAD + PREPROCESS (compatível com M2)
# ─────────────────────────────────────────────────────────────────

def _load_and_preprocess(filepath=None, signal=None, fs=TARGET_FS):
    """Carrega e pré-processa o sinal. Reutiliza tfs_ab se disponível."""
    if filepath is not None:
        try:
            from tfs_ab import load_audio, preprocess
            sig, fs = load_audio(filepath, target_fs=TARGET_FS)
            sig = preprocess(sig, fs, trim_silence=True, bandpass=True)
        except Exception:
            from scipy.io import wavfile
            fs_r, data = wavfile.read(filepath)
            if data.ndim > 1:
                data = data.mean(axis=1)
            sig = data.astype(np.float64)
            if np.issubdtype(data.dtype, np.integer):
                sig /= float(np.iinfo(data.dtype).max)
            fs = fs_r
        return sig[np.isfinite(sig)], int(fs)
    else:
        if signal is None:
            raise ValueError("[tfs_cd] Forneca 'filepath' ou 'signal'.")
        sig = np.asarray(signal, dtype=np.float64)
        try:
            from tfs_ab import preprocess
            sig = preprocess(sig, fs, trim_silence=True, bandpass=True)
        except Exception:
            sig = sig[np.isfinite(sig)]
        return sig, int(fs)


# ─────────────────────────────────────────────────────────────────
# FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def extract_blocks_CD(filepath=None, signal=None, fs=TARGET_FS,
                      n_bins=-1, window_ms=WINDOW_MS, hop_ms=HOP_MS,
                      verbose=False):
    """
    Extrai Blocos C e D. Aceita filepath ou signal array.
    Retorna dict com 20 features + metadados.
    """
    sig, fs = _load_and_preprocess(filepath, signal, fs)

    if verbose:
        src = os.path.basename(filepath) if filepath else "array"
        print("[tfs_cd] " + src + ": " + f"{len(sig)/fs:.2f}" + "s @ " + str(fs) + "Hz")
        print("[tfs_cd] Extraindo Bloco C...")

    feats_C = extract_block_C(sig, fs=fs, n_bins=n_bins)

    if verbose:
        print("[tfs_cd] Extraindo Bloco D...")

    feats_D = extract_block_D(sig, fs=fs, window_ms=window_ms,
                              hop_ms=hop_ms, n_bins=n_bins)

    feats = {}
    feats.update(feats_C)
    feats.update(feats_D)
    feats['n_samples']  = int(len(sig))
    feats['fs']         = int(fs)
    feats['duration_s'] = float(len(sig) / fs)

    if verbose:
        n_valid = sum(1 for k, v in feats.items()
                      if k.startswith(('C_', 'D_')) and
                      isinstance(v, float) and np.isfinite(v))
        print("[tfs_cd] Features validas (C+D): " + str(n_valid) + "/20")

    return feats


def batch_extract_CD(filepaths, labels=None, n_bins=-1, verbose=True):
    """Batch: retorna DataFrame com Blocos C+D para lista de arquivos."""
    import pandas as pd
    rows = []
    for i, fp in enumerate(filepaths):
        if verbose:
            print("[tfs_cd] [" + str(i+1) + "/" + str(len(filepaths)) + "] " +
                  os.path.basename(fp), end=" ")
        row = {'filename': os.path.basename(fp), 'error': None}
        try:
            row.update(extract_blocks_CD(filepath=fp, n_bins=n_bins, verbose=False))
            if verbose:
                print("OK")
        except Exception as e:
            row['error'] = str(e)
            if verbose:
                print("ERRO: " + str(e))
        if labels is not None and i < len(labels):
            row['label'] = labels[i]
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  tfs_cd.py v1.1 -- Self-Test (sinais sinteticos)")
    print("=" * 65)

    rng = np.random.default_rng(42)
    fs  = 8000
    t   = np.linspace(0, 3.0, 3 * fs)

    sig_hc = (
        0.8 * np.sin(2 * np.pi * 130 * t)
        + 0.25 * np.sin(2 * np.pi * 260 * t)
        + 0.04 * rng.normal(0, 1, len(t))
    )

    fade    = np.linspace(1.0, 0.6, len(t))
    f0_inst = 130 * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))
    phase   = 2 * np.pi * np.cumsum(f0_inst) / fs
    sig_dp  = (
        fade * (0.6 * np.sin(phase)
                + 0.2 * np.sin(2 * phase)
                + 0.08 * np.clip(rng.standard_cauchy(len(t)), -4, 4))
    )

    print("\n[1] Bloco C -- Tsallis sobre F0 e Perturbacao")
    C_hc = extract_block_C(sig_hc, fs=fs)
    C_dp = extract_block_C(sig_dp, fs=fs)

    keys_C = ['C_Sq_F0_q0.7', 'C_Sq_F0_q1.3', 'C_Sq_F0_q2.0',
              'C_Sq_dF0_q1.3', 'C_Sq_dF0_q2.0',
              'C_q_fit_F0', 'C_Sq_amp_cycle_q1.3', 'C_Sq_shimmer_q1.3']

    print("  {:<28} {:>9} {:>9} {:>12}".format("Feature", "HC", "DP", "delta(DP-HC)"))
    print("  " + "-"*62)
    for key in keys_C:
        v_hc = C_hc.get(key, np.nan)
        v_dp = C_dp.get(key, np.nan)
        if np.isnan(v_hc) or np.isnan(v_dp):
            print("  {:<28} {:>9} {:>9}".format(key, "NaN", "NaN"))
        else:
            print("  {:<28} {:>9.4f} {:>9.4f} {:>+12.4f}".format(key, v_hc, v_dp, v_dp - v_hc))

    n_C_valid = sum(1 for v in C_dp.values() if np.isfinite(v))
    print("\n  Features Bloco C validas: " + str(n_C_valid) + "/8")

    print("\n[2] Bloco D -- q-Cronoentropograma")
    D_hc = extract_block_D(sig_hc, fs=fs)
    D_dp = extract_block_D(sig_dp, fs=fs)

    keys_D = []
    for q_v in [1.3, 2.0]:
        for s in ['mean', 'std', 'onset', 'steady', 'delta', 'tau']:
            keys_D.append("D_" + s + "_q" + f"{q_v:.1f}")

    print("  {:<22} {:>9} {:>9} {:>12}".format("Feature", "HC", "DP", "delta(DP-HC)"))
    print("  " + "-"*56)
    for key in keys_D:
        v_hc = D_hc.get(key, np.nan)
        v_dp = D_dp.get(key, np.nan)
        if np.isnan(v_hc) or np.isnan(v_dp):
            print("  {:<22} {:>9} {:>9}".format(key, "NaN", "NaN"))
        else:
            print("  {:<22} {:>9.4f} {:>9.4f} {:>+12.4f}".format(key, v_hc, v_dp, v_dp - v_hc))

    n_D_valid = sum(1 for v in D_dp.values() if np.isfinite(v))
    print("\n  Features Bloco D validas: " + str(n_D_valid) + "/12")

    print("\n[3] extract_blocks_CD completo (signal mode)")
    feats_hc = extract_blocks_CD(signal=sig_hc, fs=fs, verbose=False)
    feats_dp = extract_blocks_CD(signal=sig_dp, fs=fs, verbose=False)
    n_C = sum(1 for k in feats_hc if k.startswith("C_"))
    n_D = sum(1 for k in feats_hc if k.startswith("D_"))
    n_valid = sum(1 for k, v in feats_hc.items()
                  if k.startswith(("C_", "D_")) and isinstance(v, float) and np.isfinite(v))
    print("  Features Bloco C: " + str(n_C) + " | Bloco D: " + str(n_D))
    print("  Features validas (C+D): " + str(n_valid) + "/20")
    print("  Duracao processada: " + f"{feats_hc['duration_s']:.2f}" + "s")

    print("\n[4] Verificacao hipotese H3 (D_delta)")
    for q_str in ["q1.3", "q2.0"]:
        d_hc = feats_hc.get("D_delta_" + q_str, np.nan)
        d_dp = feats_dp.get("D_delta_" + q_str, np.nan)
        direction = "H3 suportada" if (not np.isnan(d_dp) and not np.isnan(d_hc) and d_dp > d_hc) else "H3 nao suportada"
        print("  D_delta_" + q_str + ": HC=" + f"{d_hc:+.4f}" +
              ", DP=" + f"{d_dp:+.4f}" + "  -> " + direction)

    print("\n Modulo 3 pronto.")
    print("=" * 65)
