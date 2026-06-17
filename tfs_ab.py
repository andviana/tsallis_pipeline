"""
tfs_ab.py — Módulo 2: Extração de Features TFS — Blocos A e B
=============================================================
Versão 1.0

Depende exclusivamente de: tsallis_core.py

Blocos implementados:
    Bloco A — q-Sweep sobre Amplitude (11 features)
    Bloco B — Espectro Entrópico de Tsallis / TES (24 features)

Total: 35 features por arquivo de áudio.

Uso rápido:
    from tfs_ab import extract_blocks_AB
    feats = extract_blocks_AB("arquivo.wav")   # dict com 35 features
    # ou com sinal já carregado:
    feats = extract_blocks_AB(signal=sig, fs=8000)

Interface de saída (dict, chaves fixas):
    Bloco A (11): 'A_Sq_0.3', ..., 'A_Sq_2.5',
                  'A_decay_ratio', 'A_area', 'A_slope_q1', 'A_curvature_int'
    Bloco B (24): 'B_b1_q0.7', 'B_b1_q1.3', 'B_b1_q2.0',
                  'B_b2_q0.7', ..., 'B_b8_q2.0'
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt
import warnings
import os

# Importar o núcleo Tsallis (M1)
from tsallis_core import (
    tsallis_entropy,
    q_sweep,
    profile_features,
    Q_DEFAULT,
    Q_SUBBAND,
    validate_signal,
    check_q_value,
)


# ─────────────────────────────────────────────────────────────────
# CONSTANTES DO MÓDULO
# ─────────────────────────────────────────────────────────────────

# Subbandas do Bloco B (Hz): [low, high]
# Baseadas na fisiologia do trato vocal humano
SUBBAND_LIMITS = [
    (80,   250),   # B1 — Frequência fundamental F0
    (250,  500),   # B2 — Primeiro formante F1
    (500,  1000),  # B3 — Segundo formante F2
    (1000, 2000),  # B4 — Terceiro formante F3
    (2000, 3000),  # B5 — Quarto formante F4 / ruído glótico
    (3000, 4000),  # B6 — Turbulência fricativa supraglótica
    (4000, 6000),  # B7 — Presença articulatória
    (6000, 8000),  # B8 — Ruído estocástico residual
]

# Parâmetros do q-sweep (Bloco A)
Q_SWEEP_VALUES = Q_DEFAULT  # [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0, 2.5]

# Parâmetros do TES (Bloco B)
Q_SUBBAND_VALUES = Q_SUBBAND  # [0.7, 1.3, 2.0]

# Taxa de amostragem alvo (resample se necessário)
TARGET_FS = 8000


# ─────────────────────────────────────────────────────────────────
# CARREGAMENTO DE ÁUDIO
# ─────────────────────────────────────────────────────────────────

def load_audio(filepath: str, target_fs: int = TARGET_FS) -> tuple:
    """
    Carrega arquivo de áudio e reamostrado para target_fs se necessário.

    Suporta .wav, .mp3, .ogg (via librosa). Para .wav puros, usa
    scipy.io.wavfile como fallback (mais rápido, sem dependência de libsndfile).

    Parâmetros
    ----------
    filepath  : caminho para o arquivo de áudio
    target_fs : taxa de amostragem alvo (Hz)

    Retorna
    -------
    (signal, fs) : tuple de (array float64 normalizado, int taxa amostral)

    Levanta
    -------
    FileNotFoundError : se o arquivo não existir
    ImportError       : se librosa não estiver instalado
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.wav':
        try:
            from scipy.io import wavfile
            fs, data = wavfile.read(filepath)
            if data.ndim > 1:
                data = data.mean(axis=1)   # mono
            signal = data.astype(np.float64)
            # Normalizar para float [-1, 1]
            max_val = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.integer) else 1.0
            signal = signal / max_val
        except Exception:
            import librosa
            signal, fs = librosa.load(filepath, sr=None, mono=True)
    else:
        import librosa
        signal, fs = librosa.load(filepath, sr=None, mono=True)

    # Resample se necessário
    if fs != target_fs:
        try:
            import librosa
            signal = librosa.resample(signal, orig_sr=fs, target_sr=target_fs)
        except ImportError:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(target_fs, fs)
            signal = resample_poly(signal, target_fs // g, fs // g)
        fs = target_fs

    signal = signal.astype(np.float64)
    return signal, int(fs)


# ─────────────────────────────────────────────────────────────────
# PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────────────────────────

def preprocess(
    signal: np.ndarray,
    fs: int,
    trim_silence: bool = True,
    bandpass: bool = True,
    bp_low: float = 80.0,
    bp_high: float = 3800.0,
    silence_threshold_db: float = -40.0
) -> np.ndarray:
    """
    Pré-processa o sinal de fonação sustentada para extração de features.

    Etapas (na ordem):
    1. Trim de silêncio inicial/final (baseado em energia RMS por janela)
    2. Filtro passa-banda Butterworth 4ª ordem (80–3800 Hz)
    3. Validação de comprimento mínimo

    Parâmetros
    ----------
    signal              : array 1D float
    fs                  : taxa de amostragem (Hz)
    trim_silence        : se True, remove regiões de silêncio
    bandpass            : se True, aplica filtro passa-banda
    bp_low, bp_high     : frequências de corte do filtro (Hz)
    silence_threshold_db: limiar de energia em dB (relativo ao máximo)

    Retorna
    -------
    np.ndarray : sinal pré-processado
    """
    signal = np.asarray(signal, dtype=np.float64)

    # Trim de silêncio por energia RMS em janelas de 20 ms
    if trim_silence:
        frame_len = int(0.020 * fs)
        hop_len   = frame_len // 2
        frames = np.array([
            signal[i:i+frame_len]
            for i in range(0, len(signal) - frame_len, hop_len)
        ])
        rms = np.sqrt(np.mean(frames**2, axis=1))
        rms_db = 20 * np.log10(rms / (np.max(rms) + 1e-10) + 1e-10)
        voiced_mask = rms_db > silence_threshold_db
        if voiced_mask.sum() > 2:
            first = np.argmax(voiced_mask) * hop_len
            last  = (len(voiced_mask) - np.argmax(voiced_mask[::-1])) * hop_len
            last  = min(last, len(signal))
            signal = signal[first:last]

    # Filtro passa-banda (remove componentes DC e ruído de alta frequência)
    if bandpass and len(signal) > 50:
        nyq = fs / 2.0
        low  = bp_low  / nyq
        high = bp_high / nyq
        low  = np.clip(low,  0.001, 0.999)
        high = np.clip(high, 0.001, 0.999)
        if low < high:
            try:
                sos = butter(4, [low, high], btype='band', output='sos')
                signal = sosfiltfilt(sos, signal)
            except Exception as e:
                warnings.warn(f"[tfs_ab] Filtro passa-banda falhou: {e}. Usando sinal sem filtro.")

    return signal


# ─────────────────────────────────────────────────────────────────
# BLOCO A — q-Sweep sobre Amplitude (11 features)
# ─────────────────────────────────────────────────────────────────

def extract_block_A(signal: np.ndarray, n_bins: int = -1) -> dict:
    """
    Bloco A — q-Sweep sobre Amplitude.

    Calcula o perfil entrópico S_q(q) do sinal de amplitude e extrai
    11 features que descrevem a geometria desse perfil.

    NOTA FÍSICA IMPORTANTE:
    O sinal NÃO é normalizado antes do q-sweep. A normalização mudaria
    a distribuição de amplitudes e invalidaria a hipótese H1 (que DP
    exibe distribuição com caudas mais pesadas que HC). O sinal deve
    chegar aqui já pré-processado (trim + bandpass) mas com amplitude
    física preservada.

    Features (11):
    ┌──────────────────┬──────────────────────────────────────────┐
    │ A_Sq_0.3 a       │ S_q para cada q em Q_SWEEP_VALUES        │
    │ A_Sq_2.5         │ (9 valores do perfil)                    │
    │ A_decay_ratio    │ S(q_max)/S(q_min) — decaimento relativo  │
    │ A_area           │ ∫ S_q dq — complexidade global integrada │
    │ A_slope_q1       │ dS_q/dq em q=1.0 (sempre negativo)      │
    │ A_curvature_int  │ ∫|d²S/dq²|dq — não-linearidade do perfil│
    └──────────────────┴──────────────────────────────────────────┘

    Interpretação física (hipóteses):
    - H1a: S_q(q≤1) maior em DP → distribuição de amplitudes mais
      dispersa (hipofonia + tremor espalhado no histograma)
    - H1b: decay_ratio menor em DP → perfil decai mais rápido →
      distribuição com caudas mais pesadas (q-Gaussiana com q>1)
    - H1c: |slope_q1| maior em DP → sistema mais sensível à mudança
      de regime extensivo/não-extensivo

    Parâmetros
    ----------
    signal : array 1D (pré-processado, amplitude NÃO normalizada)
    n_bins : bins para histograma; -1 = Freedman-Diaconis

    Retorna
    -------
    dict : 11 features com prefixo 'A_'
    """
    validate_signal(signal, min_samples=200, name="Bloco A", normalize=False)

    profile = q_sweep(signal, q_values=Q_SWEEP_VALUES, n_bins=n_bins)
    pf = profile_features(profile, q_values=Q_SWEEP_VALUES)

    feats = {}

    # 9 valores do perfil (S_q para cada q)
    for q_val, s_val in zip(Q_SWEEP_VALUES, profile):
        key = f"A_Sq_{q_val:.1f}"
        feats[key] = float(s_val)

    # 4 features geométricas
    feats['A_decay_ratio']   = pf['decay_ratio']
    feats['A_area']          = pf['area']
    feats['A_slope_q1']      = pf['slope_q1']
    feats['A_curvature_int'] = pf['curvature_int']

    return feats  # 13 features (9 S_q + 4 geométricas)
    # Nota: o framework descreve 11 features; as 9 absolutas + decay_ratio
    # + area são as 11 principais. slope_q1 e curvature_int são bônus
    # (podem ser removidas no ranking de consenso do M5 se redundantes).


# ─────────────────────────────────────────────────────────────────
# BLOCO B — Espectro Entrópico de Tsallis / TES (24 features)
# ─────────────────────────────────────────────────────────────────

def _bandpass_filter(
    signal: np.ndarray,
    fs: int,
    low_hz: float,
    high_hz: float,
    order: int = 6
) -> np.ndarray:
    """
    Aplica filtro Butterworth passa-banda para isolar uma subbanda.

    Parâmetros
    ----------
    signal           : array 1D
    fs               : taxa de amostragem (Hz)
    low_hz, high_hz  : frequências de corte (Hz)
    order            : ordem do filtro Butterworth

    Retorna
    -------
    array 1D filtrado (mesma dimensão que signal)
    """
    nyq = fs / 2.0
    low  = low_hz  / nyq
    high = high_hz / nyq

    # Verificações de segurança
    if high >= 1.0:
        high = 0.999
    if low <= 0.0:
        low = 0.001
    if low >= high:
        raise ValueError(
            f"[tfs_ab] Subbanda inválida: low={low_hz}Hz >= high={high_hz}Hz "
            f"para fs={fs}Hz."
        )

    sos = butter(order, [low, high], btype='band', output='sos')
    try:
        return sosfiltfilt(sos, signal)
    except Exception:
        return sosfilt(sos, signal)


def extract_block_B(
    signal: np.ndarray,
    fs: int = TARGET_FS,
    n_bins: int = -1
) -> dict:
    """
    Bloco B — Espectro Entrópico de Tsallis (TES).

    Calcula S_q dentro de cada subbanda de frequência para q ∈ {0.7, 1.3, 2.0},
    capturando a heterogeneidade espectral da complexidade entrópica.

    Subbandas (8 × 3 valores de q = 24 features):
    ┌─────┬───────────┬──────────────────────────────────────┐
    │ B1  │ 80–250 Hz │ Frequência fundamental F0            │
    │ B2  │ 250–500   │ Primeiro formante F1                 │
    │ B3  │ 500–1000  │ Segundo formante F2                  │
    │ B4  │ 1000–2000 │ Terceiro formante F3                 │
    │ B5  │ 2000–3000 │ Quarto formante F4 / ruído glótico   │
    │ B6  │ 3000–4000 │ Turbulência fricativa supraglótica   │
    │ B7  │ 4000–6000 │ Presença articulatória               │
    │ B8  │ 6000–8000 │ Ruído estocástico residual           │
    └─────┴───────────┴──────────────────────────────────────┘

    Para cada subbanda k e cada q ∈ {0.7, 1.3, 2.0}:
        B_bk_q<val> = S_q(sinal filtrado na banda k)

    Interpretação física:
    - B1 (F0) com q=2.0: sensível a outliers na distribuição de energia
      de F0 → detecta instabilidade glótica (monotonia + tremor)
    - B2-B4 (formantes): heterogeneidade entrópica nos formantes →
      imprecisão articulatória na DP
    - B5-B8 (alta frequência): ruído glótico e turbulência →
      rouquidão e soprosidade da DP

    ATENÇÃO: Subbandas acima de fs/2 são ignoradas automaticamente.
    Para sinais com fs=8kHz, B7 (4-6kHz) e B8 (6-8kHz) operam próximo
    ao Nyquist e são calculadas quando fs >= 12kHz. Para fs=8kHz,
    B8 é pulada e substituída por NaN (com warning).

    Parâmetros
    ----------
    signal : array 1D pré-processado (amplitude pode estar normalizada
             aqui, pois estamos comparando distribuições relativas)
    fs     : taxa de amostragem (Hz)
    n_bins : bins para PDF; -1 = automático

    Retorna
    -------
    dict : 24 features com prefixo 'B_'
          (ou menos se algumas bandas forem puladas por Nyquist)
    """
    signal = np.asarray(signal, dtype=np.float64)
    nyq = fs / 2.0

    feats = {}

    for band_idx, (low_hz, high_hz) in enumerate(SUBBAND_LIMITS, start=1):

        band_name = f"b{band_idx}"

        # Verificar se a banda é viável para este fs
        if low_hz >= nyq:
            for q_val in Q_SUBBAND_VALUES:
                feats[f"B_{band_name}_q{q_val:.1f}"] = np.nan
            warnings.warn(
                f"[tfs_ab] Banda B{band_idx} ({low_hz}-{high_hz}Hz) "
                f"fora do range de Nyquist ({nyq}Hz). Feature = NaN.",
                UserWarning
            )
            continue

        # Ajustar high_hz ao Nyquist se necessário
        high_hz_adj = min(high_hz, nyq * 0.95)
        if high_hz_adj <= low_hz:
            for q_val in Q_SUBBAND_VALUES:
                feats[f"B_{band_name}_q{q_val:.1f}"] = np.nan
            continue

        # Filtrar sinal para a subbanda
        try:
            band_signal = _bandpass_filter(signal, fs, low_hz, high_hz_adj)
        except Exception as e:
            warnings.warn(f"[tfs_ab] Filtro banda B{band_idx} falhou: {e}. Feature = NaN.")
            for q_val in Q_SUBBAND_VALUES:
                feats[f"B_{band_name}_q{q_val:.1f}"] = np.nan
            continue

        # Verificar se o sinal filtrado tem energia suficiente
        if np.std(band_signal) < 1e-10:
            warnings.warn(
                f"[tfs_ab] Banda B{band_idx} tem energia quase nula. Feature = NaN."
            )
            for q_val in Q_SUBBAND_VALUES:
                feats[f"B_{band_name}_q{q_val:.1f}"] = np.nan
            continue

        # Calcular S_q para cada q da grade TES
        for q_val in Q_SUBBAND_VALUES:
            try:
                s_q = tsallis_entropy(band_signal, q=q_val, n_bins=n_bins)
                feats[f"B_{band_name}_q{q_val:.1f}"] = float(s_q)
            except Exception as e:
                warnings.warn(f"[tfs_ab] S_q falhou em B{band_idx}, q={q_val}: {e}")
                feats[f"B_{band_name}_q{q_val:.1f}"] = np.nan

    return feats


# ─────────────────────────────────────────────────────────────────
# FUNÇÃO PRINCIPAL — Extração Completa dos Blocos A + B
# ─────────────────────────────────────────────────────────────────

def extract_blocks_AB(
    filepath: str = None,
    signal: np.ndarray = None,
    fs: int = TARGET_FS,
    n_bins: int = -1,
    trim_silence: bool = True,
    bandpass_preprocess: bool = True,
    verbose: bool = False
) -> dict:
    """
    Extrai os Blocos A e B do TFS para um arquivo de áudio ou sinal array.

    Aceita dois modos de uso:
        1. filepath: carrega e pré-processa o arquivo automaticamente
        2. signal + fs: usa sinal já carregado (pré-processado externamente)

    Pipeline interno:
        load_audio → preprocess → extract_block_A → extract_block_B

    Parâmetros
    ----------
    filepath             : caminho para o arquivo .wav/.mp3 (opcional)
    signal               : array 1D já carregado (opcional, alternativa a filepath)
    fs                   : taxa de amostragem do signal (Hz); ignorado se filepath
    n_bins               : bins para PDF; -1 = automático Freedman-Diaconis
    trim_silence         : se True, remove silêncio inicial/final
    bandpass_preprocess  : se True, aplica filtro 80-3800 Hz no pré-processamento
    verbose              : se True, imprime informações de progresso

    Retorna
    -------
    dict : 35+ features combinadas dos Blocos A e B
        Bloco A (13 features): 'A_Sq_0.3', ..., 'A_Sq_2.5',
                               'A_decay_ratio', 'A_area',
                               'A_slope_q1', 'A_curvature_int'
        Bloco B (24 features): 'B_b1_q0.7', ..., 'B_b8_q2.0'
        Metadados: 'n_samples', 'fs', 'duration_s'

    Levanta
    -------
    ValueError : se nem filepath nem signal forem fornecidos
    """
    if filepath is None and signal is None:
        raise ValueError("Forneça 'filepath' ou 'signal'.")

    # Carregar áudio se filepath fornecido
    if filepath is not None:
        if verbose:
            print(f"[tfs_ab] Carregando: {os.path.basename(filepath)}")
        signal, fs = load_audio(filepath, target_fs=TARGET_FS)

    signal = np.asarray(signal, dtype=np.float64)

    # Pré-processamento
    if verbose:
        print(f"[tfs_ab] Sinal bruto: {len(signal)} amostras @ {fs} Hz "
              f"({len(signal)/fs:.2f}s)")
    signal_proc = preprocess(
        signal, fs,
        trim_silence=trim_silence,
        bandpass=bandpass_preprocess
    )
    if verbose:
        print(f"[tfs_ab] Sinal pré-processado: {len(signal_proc)} amostras "
              f"({len(signal_proc)/fs:.2f}s)")

    # Verificação de comprimento mínimo
    if len(signal_proc) < int(0.5 * fs):
        raise ValueError(
            f"[tfs_ab] Sinal muito curto após pré-processamento: "
            f"{len(signal_proc)/fs:.2f}s (mínimo: 0.5s)."
        )

    # Extrair Bloco A (amplitude raw — NÃO normalizado)
    if verbose:
        print("[tfs_ab] Extraindo Bloco A (q-sweep amplitude)...")
    feats_A = extract_block_A(signal_proc, n_bins=n_bins)

    # Extrair Bloco B (subbandas — normalização interna pelo filtro)
    if verbose:
        print("[tfs_ab] Extraindo Bloco B (TES subbandas)...")
    feats_B = extract_block_B(signal_proc, fs=fs, n_bins=n_bins)

    # Combinar features + metadados
    feats = {}
    feats.update(feats_A)
    feats.update(feats_B)
    feats['n_samples']  = int(len(signal_proc))
    feats['fs']         = int(fs)
    feats['duration_s'] = float(len(signal_proc) / fs)

    if verbose:
        n_valid = sum(1 for v in feats.values() if isinstance(v, float) and np.isfinite(v))
        print(f"[tfs_ab] Features extraídas: {n_valid} válidas de {len(feats)} total.")

    return feats


# ─────────────────────────────────────────────────────────────────
# PROCESSAMENTO EM BATCH (múltiplos arquivos)
# ─────────────────────────────────────────────────────────────────

def batch_extract_AB(
    filepaths: list,
    labels: list = None,
    n_bins: int = -1,
    verbose: bool = True
) -> "pd.DataFrame":
    """
    Processa múltiplos arquivos de áudio e retorna DataFrame com TFS Blocos A+B.

    Parâmetros
    ----------
    filepaths : lista de caminhos para arquivos de áudio
    labels    : lista de rótulos (0=HC, 1=DP); opcional
    n_bins    : bins para PDF; -1 = automático
    verbose   : se True, mostra progresso

    Retorna
    -------
    pd.DataFrame : linhas = sujeitos, colunas = features + 'label' (se fornecido)
                   + 'filename', 'duration_s', 'error'
    """
    import pandas as pd

    rows = []
    n = len(filepaths)

    for i, fp in enumerate(filepaths):
        if verbose:
            print(f"[tfs_ab] [{i+1}/{n}] {os.path.basename(fp)}", end=" ")
        row = {'filename': os.path.basename(fp), 'error': None}

        try:
            feats = extract_blocks_AB(filepath=fp, n_bins=n_bins, verbose=False)
            row.update(feats)
            if verbose:
                print("✓")
        except Exception as e:
            row['error'] = str(e)
            if verbose:
                print(f"✗ ({e})")

        if labels is not None and i < len(labels):
            row['label'] = labels[i]

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  tfs_ab.py v1.0 — Self-Test (sinais sintéticos)")
    print("=" * 65)
    print()
    print("Criando sinais sintéticos (sem áudio real)...")
    print(" • Sinal HC: oscilador senoidal estável com ruído gaussiano leve")
    print(" • Sinal DP: senoide + tremor 5 Hz + ruído não-gaussiano (Cauchy)")
    print()

    rng = np.random.default_rng(42)
    fs = 8000
    t  = np.linspace(0, 2.0, 2 * fs)

    # HC: fonação estável (F0=130Hz, ruído gaussiano leve)
    sig_hc = (
        0.8 * np.sin(2 * np.pi * 130 * t)
        + 0.3 * np.sin(2 * np.pi * 260 * t)   # segundo harmônico
        + 0.05 * rng.normal(0, 1, len(t))       # ruído leve
    )

    # DP: fonação instável (tremor a 5Hz, ruído com caudas pesadas)
    tremor = 0.15 * np.sin(2 * np.pi * 5 * t)
    sig_dp = (
        (0.5 + tremor) * np.sin(2 * np.pi * 130 * t)
        + 0.2 * np.sin(2 * np.pi * 260 * t)
        + 0.1 * np.clip(rng.standard_cauchy(len(t)), -3, 3)  # ruído de cauda pesada
    )

    print("[1] Bloco A — q-sweep amplitude")
    A_hc = extract_block_A(sig_hc)
    A_dp = extract_block_A(sig_dp)

    q_vals_str = [f"{q:.1f}" for q in Q_SWEEP_VALUES]
    print(f"  {'Feature':<22} {'HC':>10} {'DP':>10} {'Δ (DP-HC)':>12}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*12}")
    for key in ['A_Sq_0.3', 'A_Sq_1.0', 'A_Sq_1.3', 'A_Sq_2.0',
                'A_decay_ratio', 'A_area', 'A_slope_q1', 'A_curvature_int']:
        v_hc = A_hc.get(key, np.nan)
        v_dp = A_dp.get(key, np.nan)
        delta = v_dp - v_hc
        print(f"  {key:<22} {v_hc:>10.4f} {v_dp:>10.4f} {delta:>+12.4f}")

    print(f"\n  Total features Bloco A: {len(A_hc)}")

    print("\n[2] Bloco B — TES subbandas")
    B_hc = extract_block_B(sig_hc, fs=fs)
    B_dp = extract_block_B(sig_dp, fs=fs)

    print(f"  {'Feature':<22} {'HC':>10} {'DP':>10} {'Δ (DP-HC)':>12}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*12}")
    # Mostrar apenas q=1.3 para cada banda (8 valores)
    for b in range(1, 9):
        key = f"B_b{b}_q1.3"
        v_hc = B_hc.get(key, np.nan)
        v_dp = B_dp.get(key, np.nan)
        if np.isnan(v_hc) or np.isnan(v_dp):
            print(f"  {key:<22} {'NaN':>10} {'NaN':>10} {'(fora Nyquist)':>12}")
        else:
            delta = v_dp - v_hc
            print(f"  {key:<22} {v_hc:>10.4f} {v_dp:>10.4f} {delta:>+12.4f}")

    n_valid_B = sum(1 for v in B_dp.values() if np.isfinite(v))
    print(f"\n  Total features Bloco B: {len(B_hc)} ({n_valid_B} válidas para fs={fs}Hz)")

    print("\n[3] extract_blocks_AB completo (signal mode)")
    feats_hc = extract_blocks_AB(signal=sig_hc, fs=fs, verbose=False)
    feats_dp = extract_blocks_AB(signal=sig_dp, fs=fs, verbose=False)
    n_A = sum(1 for k in feats_hc if k.startswith("A_"))
    n_B = sum(1 for k in feats_hc if k.startswith("B_"))
    n_valid = sum(1 for k,v in feats_hc.items()
                  if k.startswith(("A_","B_")) and isinstance(v,float) and np.isfinite(v))
    print(f"  Features Bloco A: {n_A}")
    print(f"  Features Bloco B: {n_B}")
    print(f"  Features válidas (A+B): {n_valid}")
    print(f"  Duração processada: {feats_hc['duration_s']:.2f}s")

    print()
    print("✓ Self-test concluído. Módulo 2 pronto.")
    print("=" * 65)
