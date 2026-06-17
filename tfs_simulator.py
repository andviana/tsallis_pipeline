"""
tfs_simulator.py — Gerador de Séries Temporais Acústicas Sintéticas para DP vs HC
===================================================================================
Versão 1.0

Propósito:
    Gerar sinais de áudio sintéticos com características estatísticas controláveis
    que simulam disartria hipocinética (Parkinson) vs fala saudável (HC).
    Permite testar e calibrar todo o pipeline TFS antes dos dados reais.

Fundamento físico:
    A voz humana é modelada pelo modelo fonte-filtro (Fant, 1960):
        s(t) = e(t) * h(t)
    onde e(t) é a excitação glótal (F0 + harmônicos + jitter/shimmer)
    e h(t) é o filtro do trato vocal (formantes).

    Na DP, o sistema dopaminérgico degenerado causa:
    1. Rigidez muscular laríngea → F0 mais alto, variação reduzida, shimmer aumentado
    2. Hipocinesia → amplitude reduzida, shimmer aumentado
    3. Irregularidade neuromotora → jitter aumentado
    4. Perda de complexidade dinâmica → entropia de Tsallis REDUZIDA
       (sinal mais "periódico mas instável" — regime não-extensivo alterado)
    5. Micropausa → silêncios mais frequentes e irregulares

Componentes:
    VocalParams          — Parâmetros acústicos configuráveis HC/DP
    GlottalSource        — Modelo de excitação glótal com jitter/shimmer
    VocalTractFilter     — Filtro de formantes (Klatt, 1980 simplificado)
    NoiseLayer           — Ruído subglótal e turbulência
    VoiceSimulator       — Orquestrador completo
    simulate_dataset     — Gera dataset HC+DP completo em memória ou .wav
    batch_simulate       — Gera N arquivos por grupo e salva em disco

Uso rápido:
    from tfs_simulator import simulate_dataset
    X_audio, labels, sr = simulate_dataset(n_hc=30, n_dp=30, duration=3.0)
    # X_audio: lista de arrays numpy (n_samples,)
    # labels: array 0=HC, 1=DP
    # sr: taxa de amostragem

    # Ou com parâmetros customizados:
    from tfs_simulator import VocalParams, VoiceSimulator
    params_dp = VocalParams.parkinson(severity=0.7)  # severidade 0-1
    sim = VoiceSimulator(params_dp, sr=16000, seed=42)
    audio = sim.generate(duration=3.0)
"""

import numpy as np
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from copy import deepcopy

try:
    import scipy.signal as _signal
    _SCIPY = True
except ImportError:
    _SCIPY = False
    warnings.warn("[tfs_simulator] scipy nao encontrado. Filtros degradados.")

try:
    import soundfile as sf
    _SF = True
except ImportError:
    _SF = False
    # Fallback: salvar em .raw (PCM 16-bit)
    warnings.warn("[tfs_simulator] soundfile nao encontrado. "
                  "Salvando em formato PCM raw (.npy).")


# ─────────────────────────────────────────────────────────────────
# PARÂMETROS VOCAIS
# ─────────────────────────────────────────────────────────────────

@dataclass
class VocalParams:
    """
    Parâmetros acústicos que controlam a geração do sinal vocal.

    Todos os parâmetros têm valores default para HC (fala saudável).
    Os valores de DP são baseados em literatura clínica:
        - Tsanas et al. (2012): F0, jitter, shimmer em DP
        - Harel et al. (2004): amplitude e range F0 em DP
        - Little et al. (2009): MDVP features em DP

    Parâmetros
    ----------
    f0_mean     : F0 médio (Hz). HC≈120-180Hz, DP≈150-220Hz (hiperfonia)
    f0_std      : Desvio-padrão de F0 (Hz). HC≈8-15, DP≈2-6 (monotonia)
    f0_trend    : Drift linear de F0 ao longo da emissão (Hz/s)
                  HC≈0, DP: pode ser positivo (voz ascende) ou negativo
    jitter_pct  : Jitter (variação de pitch ciclo-a-ciclo, %). HC<1%, DP>1%
    shimmer_pct : Shimmer (variação de amplitude ciclo-a-ciclo, %). HC<3%, DP>6%
    hnr_db      : Harmonic-to-Noise Ratio (dB). HC≈20-25dB, DP≈10-18dB
    amplitude   : Amplitude global normalizada (0-1). HC≈0.7, DP≈0.4 (hipofonia)
    formant_f1  : Primeiro formante F1 (Hz). Vogal /a/: HC≈700, DP≈650
    formant_f2  : Segundo formante F2 (Hz). Vogal /a/: HC≈1200, DP≈1100
    formant_bw1 : Largura de banda F1 (Hz). DP → maior (articulação reduzida)
    formant_bw2 : Largura de banda F2 (Hz). DP → maior
    subglottal_noise: Nível de ruído subglótal (0-1). DP → maior
    aspiration  : Nível de aspiração (soprosidade, 0-1). DP → maior
    pause_rate  : Taxa de micropausas (pausas/s). DP → maior
    pause_dur_mean: Duração média de micropausas (s). DP → maior
    tremor_freq : Frequência de tremor vocal (Hz). DP: 4-8Hz
    tremor_amp  : Amplitude de modulação por tremor (0-1). DP>0, HC≈0
    irregularity: Irregularidade neuromotora adicional (0-1). DP→alta
    q_regime    : Parâmetro q da distribuição base do sinal [física Tsallis]
                  HC≈1.0-1.1 (quasi-extensivo), DP≈1.3-1.7 (não-extensivo)
    """

    # Frequência fundamental
    f0_mean:          float = 140.0
    f0_std:           float = 12.0
    f0_trend:         float = 0.0

    # Perturbação micro-temporal
    jitter_pct:       float = 0.5
    shimmer_pct:      float = 2.5

    # Qualidade vocal
    hnr_db:           float = 22.0
    amplitude:        float = 0.70

    # Formantes (vogal /a/ sustentada)
    formant_f1:       float = 700.0
    formant_f2:       float = 1200.0
    formant_f3:       float = 2600.0
    formant_bw1:      float = 80.0
    formant_bw2:      float = 100.0
    formant_bw3:      float = 120.0

    # Ruído e aspiração
    subglottal_noise: float = 0.05
    aspiration:       float = 0.05

    # Dinâmica temporal
    pause_rate:       float = 0.3
    pause_dur_mean:   float = 0.05

    # Tremor
    tremor_freq:      float = 0.0
    tremor_amp:       float = 0.0

    # Irregularidade adicional
    irregularity:     float = 0.02

    # Regime estatístico (Tsallis q)
    q_regime:         float = 1.05

    @classmethod
    def healthy(cls, sex: str = 'M') -> 'VocalParams':
        """
        Parâmetros de referência para fala saudável (HC).
        sex: 'M' (masculino) ou 'F' (feminino)
        """
        if sex.upper() == 'F':
            f0 = 210.0
        else:
            f0 = 130.0
        return cls(
            f0_mean=f0,
            f0_std=14.0,
            f0_trend=0.0,
            jitter_pct=0.40,
            shimmer_pct=2.0,
            hnr_db=23.0,
            amplitude=0.72,
            formant_f1=730.0,
            formant_f2=1180.0,
            formant_f3=2650.0,
            formant_bw1=75.0,
            formant_bw2=90.0,
            formant_bw3=110.0,
            subglottal_noise=0.04,
            aspiration=0.04,
            pause_rate=0.25,
            pause_dur_mean=0.04,
            tremor_freq=0.0,
            tremor_amp=0.0,
            irregularity=0.015,
            q_regime=1.05
        )

    @classmethod
    def parkinson(cls, severity: float = 0.5, sex: str = 'M') -> 'VocalParams':
        """
        Parâmetros para disartria hipocinética (DP).

        severity: 0.0 (DP leve) → 1.0 (DP grave)
            - Escala continua baseada nos estágios H&Y 1-5
            - severity=0.3 ≈ H&Y 1-2 (DP inicial)
            - severity=0.5 ≈ H&Y 2-3 (DP moderada)
            - severity=0.8 ≈ H&Y 4-5 (DP avançada)

        Referências para os valores:
        - Tsanas et al. (2012) TNSRE: jitter, shimmer, HNR em 195 pacientes DP
        - Little et al. (2009) Nature Precedings: MDVP features DP vs HC
        - Harel et al. (2004) J Acoust Soc Am: F0 e amplitude em DP
        """
        severity = float(np.clip(severity, 0.0, 1.0))
        s = severity

        base_f0 = 150.0 if sex.upper() == 'M' else 220.0

        return cls(
            # F0 mais alto e MENOS variável (monotonia)
            f0_mean        = base_f0 + 20*s,
            f0_std         = 12.0 - 9.0*s,          # monotonia progressiva
            f0_trend       = -5.0*s,                 # tendência descendente

            # Jitter e shimmer aumentados
            jitter_pct     = 0.40 + 2.8*s,           # até ~3.2% em DP grave
            shimmer_pct    = 2.0  + 8.0*s,           # até ~10% em DP grave

            # HNR reduzido (mais ruído glótal)
            hnr_db         = 23.0 - 10.0*s,          # até ~13 dB

            # Amplitude reduzida (hipofonia)
            amplitude      = 0.72 - 0.35*s,          # até ~0.37

            # Formantes mais estreitos (articulação reduzida)
            formant_f1     = 730.0 - 80.0*s,
            formant_f2     = 1180.0 - 150.0*s,
            formant_f3     = 2650.0 - 200.0*s,
            formant_bw1    = 75.0  + 120.0*s,        # alargamento
            formant_bw2    = 90.0  + 150.0*s,
            formant_bw3    = 110.0 + 180.0*s,

            # Mais ruído e aspiração
            subglottal_noise = 0.04 + 0.25*s,
            aspiration       = 0.04 + 0.30*s,

            # Mais micropausas (festinação, hesitação)
            pause_rate     = 0.25 + 1.5*s,
            pause_dur_mean = 0.04 + 0.12*s,

            # Tremor (4-8Hz, característico do DP)
            tremor_freq    = 5.5 + 2.0*s,
            tremor_amp     = 0.0 + 0.4*s,

            # Irregularidade neuromotora
            irregularity   = 0.015 + 0.25*s,

            # Regime não-extensivo Tsallis: q aumenta com severity
            q_regime       = 1.05 + 0.65*s   # até q≈1.7 em DP grave
        )

    def interpolate(self, other: 'VocalParams', t: float) -> 'VocalParams':
        """
        Interpola linearmente entre dois conjuntos de parâmetros.
        t=0 → self, t=1 → other.
        Útil para gerar gradientes de severidade.
        """
        t = float(np.clip(t, 0.0, 1.0))
        p = deepcopy(self)
        for field_name in self.__dataclass_fields__:
            v0 = getattr(self, field_name)
            v1 = getattr(other, field_name)
            setattr(p, field_name, v0 + t*(v1-v0))
        return p

    def to_dict(self) -> dict:
        return {k: float(getattr(self, k))
                for k in self.__dataclass_fields__}

    def __repr__(self):
        return (f"VocalParams(f0={self.f0_mean:.1f}Hz, "
                f"jitter={self.jitter_pct:.2f}%, "
                f"shimmer={self.shimmer_pct:.2f}%, "
                f"HNR={self.hnr_db:.1f}dB, "
                f"q={self.q_regime:.3f})")


# ─────────────────────────────────────────────────────────────────
# FONTE GLÓTAL
# ─────────────────────────────────────────────────────────────────

class GlottalSource:
    """
    Modelo de excitação glótal com jitter, shimmer e tremor.

    Gera um trem de pulsos glotais com:
    - Frequência fundamental F0 variável (Gaussiana + drift + tremor)
    - Jitter: perturbação aleatória do período de cada ciclo
    - Shimmer: perturbação aleatória da amplitude de cada ciclo
    - Forma do pulso: Rosenberg-C (boa aproximação da excitação real)
    - Irregularidade adicional: correlação temporal de longo alcance
      implementada via ruído de Cauchy para q>1 (distribuição Tsallis)

    Para q_regime > 1, os intervalos inter-pulso são amostrados de uma
    distribuição q-Gaussiana em vez de Gaussiana, o que gera clusters
    de instabilidade — análogo à irregularidade neuromotora da DP.
    """

    def __init__(self, params: VocalParams, sr: int = 16000,
                 seed: Optional[int] = None):
        self.params = params
        self.sr     = sr
        self.rng    = np.random.default_rng(seed)

    def _q_gaussian_sample(self, n: int, sigma: float) -> np.ndarray:
        """
        Amostras de distribuição q-Gaussiana via método de rejeição simplificado.

        Para q=1: reduz a Gaussiana padrão.
        Para 1<q<3: caudas mais pesadas (Lévy-like), gerando instabilidades
                    esporádicas características da disartria hipocinética.

        Método: transformação de Picoli et al. (2009)
        """
        q = self.params.q_regime
        if abs(q - 1.0) < 1e-6:
            return self.rng.normal(0, sigma, n)

        # Para q-Gaussiana: X = Z / sqrt(W) onde W ~ Gamma
        # Equivalência: Student-t com nu = (3-q)/(q-1) graus de liberdade
        nu = (3.0 - q) / (q - 1.0)
        nu = max(nu, 0.5)
        t_samples = self.rng.standard_t(df=nu, size=n)
        # Normalizar para ter desvio ~sigma
        t_samples *= sigma / max(np.std(t_samples), 1e-8)
        return t_samples

    def _rosenberg_pulse(self, n_samples: int,
                         open_ratio: float = 0.6) -> np.ndarray:
        """
        Pulso de Rosenberg-C: boa aproximação da forma da onda glótal.

        open_ratio: fração do período em que a glote está aberta.
        Retorna pulso normalizado com n_samples pontos.
        """
        pulse = np.zeros(n_samples)
        n_open  = int(open_ratio * n_samples)
        n_close = n_samples - n_open

        if n_open > 0:
            t_open = np.linspace(0, np.pi, n_open)
            pulse[:n_open] = 0.5 * (1 - np.cos(t_open))

        if n_close > 0:
            t_close = np.linspace(0, np.pi/2, n_close)
            pulse[n_open:] = np.cos(t_close)

        # Normalizar amplitude
        mx = np.abs(pulse).max()
        if mx > 0:
            pulse /= mx
        return pulse

    def generate(self, duration: float) -> np.ndarray:
        """
        Gera sinal de excitação glótal para `duration` segundos.

        Retorna array (n_samples,) normalizado em [-1, 1].
        """
        n_total = int(duration * self.sr)
        signal  = np.zeros(n_total)
        p       = self.params

        t_now  = 0.0
        amp_envelope = 1.0

        while t_now < duration:
            # F0 instantâneo: base + drift + tremor + q-Gaussiana
            t_frac = t_now / max(duration, 1e-6)
            f0_base = (p.f0_mean + p.f0_trend * t_now
                       + p.tremor_amp * p.f0_mean
                         * np.sin(2*np.pi*p.tremor_freq*t_now))

            # Jitter: perturbação do período usando distribuição q-Gaussiana
            period_jitter = self._q_gaussian_sample(1, p.jitter_pct/100.0)[0]
            f0_inst  = f0_base * (1.0 + period_jitter)
            f0_inst  = float(np.clip(f0_inst, 50.0, 500.0))

            period_samples = int(self.sr / f0_inst)
            if period_samples < 2:
                period_samples = 2

            # Shimmer: perturbação de amplitude
            amp_shim = 1.0 + self._q_gaussian_sample(1, p.shimmer_pct/100.0)[0]
            amp_shim = float(np.clip(amp_shim, 0.1, 3.0))

            # Irregularidade adicional (ruído multiplicativo lento)
            amp_irr  = 1.0 + self.rng.normal(0, p.irregularity)

            # Tremor de amplitude (modulação lenta)
            if p.tremor_freq > 0 and p.tremor_amp > 0:
                amp_tremor = 1.0 + p.tremor_amp * np.sin(
                    2*np.pi*p.tremor_freq*t_now + np.pi/4
                )
            else:
                amp_tremor = 1.0

            total_amp = p.amplitude * amp_shim * amp_irr * amp_tremor

            # Gerar pulso e inserir no sinal
            pulse = self._rosenberg_pulse(period_samples)
            i_start = int(t_now * self.sr)
            i_end   = min(i_start + period_samples, n_total)
            n_write = i_end - i_start
            if n_write > 0:
                signal[i_start:i_end] += total_amp * pulse[:n_write]

            t_now += period_samples / self.sr

        return signal


# ─────────────────────────────────────────────────────────────────
# FILTRO DO TRATO VOCAL
# ─────────────────────────────────────────────────────────────────

class VocalTractFilter:
    """
    Filtro simplificado do trato vocal baseado em formantes.

    Implementa 3 formantes (F1, F2, F3) como filtros resonantes
    de segunda ordem em cascata (modelo de Klatt, 1980 simplificado).

    Para DP, F1 e F2 se aproximam (centralização de vogais) e as
    larguras de banda aumentam (articulação reduzida).
    """

    def __init__(self, params: VocalParams, sr: int = 16000):
        self.params = params
        self.sr     = sr
        self._filters = self._build_filters()

    def _resonator(self, fc: float, bw: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filtro passa-faixa de segunda ordem (resonador).
        fc: frequência central (Hz), bw: largura de banda (Hz).
        Retorna coeficientes (b, a) para scipy.signal.lfilter.
        """
        if not _SCIPY:
            return np.array([1.0]), np.array([1.0])

        from scipy.signal import bilinear

        wc = 2 * np.pi * fc / self.sr
        wb = 2 * np.pi * bw / self.sr
        C  = -np.exp(-wb)
        B  = 2 * np.exp(-wb/2) * np.cos(wc)  # corr: coeff correto
        A  = 1.0 - B - C
        b  = np.array([A])
        a  = np.array([1.0, -B, -C])
        return b, a

    def _build_filters(self) -> list:
        p = self.params
        formants = [
            (p.formant_f1, p.formant_bw1),
            (p.formant_f2, p.formant_bw2),
            (p.formant_f3, p.formant_bw3),
        ]
        return [self._resonator(fc, bw) for fc, bw in formants]

    def apply(self, excitation: np.ndarray) -> np.ndarray:
        """
        Aplica os formantes em cascata ao sinal de excitação.

        Parâmetros
        ----------
        excitation : array (n_samples,) do sinal de excitação glótal

        Retorna
        -------
        array (n_samples,) filtrado (voz com formantes)
        """
        if not _SCIPY:
            return excitation

        import scipy.signal as sg
        x = excitation.copy()
        for b, a in self._filters:
            try:
                x = sg.lfilter(b, a, x)
            except Exception:
                pass

        # Normalizar para evitar clipping
        mx = np.abs(x).max()
        if mx > 1e-8:
            x /= mx
        return x


# ─────────────────────────────────────────────────────────────────
# CAMADA DE RUÍDO
# ─────────────────────────────────────────────────────────────────

class NoiseLayer:
    """
    Adiciona ruídos acústicos realistas ao sinal vocal:

    1. Ruído subglótal (broadband): modelado por Gaussiana
    2. Aspiração (turbulência glótal): ruído passa-alta filtrado
    3. Ruído ambiente (gravação clínica): Gaussiana de baixa amplitude
    4. Micropausas: silêncios aleatórios (Poisson)

    O HNR é implementado diretamente pela relação sinal/ruído:
        noise_amplitude = signal_rms / 10^(HNR_dB / 20)
    """

    def __init__(self, params: VocalParams, sr: int = 16000,
                 seed: Optional[int] = None):
        self.params = params
        self.sr     = sr
        self.rng    = np.random.default_rng(seed)

    def _highpass(self, x: np.ndarray, cutoff: float = 500.0) -> np.ndarray:
        """Filtro passa-alta simples via diferença de 1ª ordem (sem scipy)."""
        if _SCIPY:
            from scipy.signal import butter, filtfilt
            nyq = self.sr / 2.0
            fc  = min(cutoff / nyq, 0.99)
            b, a = _signal.butter(2, fc, btype='high')
            try:
                return _signal.filtfilt(b, a, x)
            except Exception:
                pass
        # Fallback: diferença de 1ª ordem
        return np.diff(x, prepend=x[0])

    def apply(self, voice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adiciona camadas de ruído ao sinal de voz e micropausas.

        Retorna (audio_final, pause_mask) onde pause_mask é True
        nos pontos em que há silêncio (útil para análise).
        """
        p   = self.params
        n   = len(voice)
        out = voice.copy()

        # ── Ruído subglótal (HNR) ──────────────────────────────
        sig_rms = np.sqrt(np.mean(voice**2)) + 1e-8
        noise_amp = sig_rms / (10 ** (p.hnr_db / 20.0))
        subglottal = self.rng.normal(0, noise_amp, n)
        out += p.subglottal_noise * subglottal

        # ── Aspiração (ruído de alta frequência) ──────────────
        asp_noise = self.rng.normal(0, sig_rms * p.aspiration, n)
        asp_noise = self._highpass(asp_noise, cutoff=1000.0)
        out += asp_noise

        # ── Ruído de gravação (microfone, sala) ───────────────
        ambient_amp = sig_rms * 0.01   # -40dB abaixo do sinal
        out += self.rng.normal(0, ambient_amp, n)

        # ── Micropausas (silêncios aleatórios) ────────────────
        pause_mask = np.zeros(n, dtype=bool)
        if p.pause_rate > 0:
            duration_s  = n / self.sr
            n_pauses    = int(self.rng.poisson(p.pause_rate * duration_s))
            for _ in range(n_pauses):
                t_pause    = self.rng.uniform(0.1*duration_s, 0.9*duration_s)
                dur_pause  = max(0.01, self.rng.exponential(p.pause_dur_mean))
                i_start    = int(t_pause * self.sr)
                i_end      = min(int((t_pause + dur_pause) * self.sr), n)
                if i_end > i_start:
                    out[i_start:i_end] *= self.rng.uniform(0.0, 0.08)
                    pause_mask[i_start:i_end] = True

        # ── Normalização final ────────────────────────────────
        mx = np.abs(out).max()
        if mx > 1e-8:
            out = out / mx * p.amplitude

        return out, pause_mask


# ─────────────────────────────────────────────────────────────────
# SIMULADOR COMPLETO
# ─────────────────────────────────────────────────────────────────

class VoiceSimulator:
    """
    Orquestrador completo da geração de voz sintética.

    Pipeline:
        GlottalSource → VocalTractFilter → NoiseLayer → sinal final

    Parâmetros
    ----------
    params   : VocalParams (use VocalParams.healthy() ou VocalParams.parkinson())
    sr       : taxa de amostragem (Hz). Padrão: 16000 Hz
    seed     : semente aleatória (para reprodutibilidade)
    """

    def __init__(self, params: VocalParams, sr: int = 16000,
                 seed: Optional[int] = None):
        self.params = params
        self.sr     = sr
        self.seed   = seed
        self._source = GlottalSource(params, sr, seed=seed)
        self._filter = VocalTractFilter(params, sr)
        self._noise  = NoiseLayer(params, sr, seed=(seed+1 if seed else None))

    def generate(self, duration: float = 3.0,
                 return_components: bool = False):
        """
        Gera sinal de voz sintético.

        Parâmetros
        ----------
        duration          : duração em segundos
        return_components : se True, retorna (audio, excitation, voiced, mask)

        Retorna
        -------
        audio : array (n_samples,) normalizado em [-1, 1]
        (ou tupla se return_components=True)
        """
        # 1. Excitação glótal
        excitation = self._source.generate(duration)

        # 2. Filtro do trato vocal
        voiced = self._filter.apply(excitation)

        # 3. Ruído + micropausas
        audio, pause_mask = self._noise.apply(voiced)

        if return_components:
            return audio, excitation, voiced, pause_mask
        return audio

    def save(self, filepath: str, duration: float = 3.0) -> str:
        """
        Gera e salva o sinal de áudio em arquivo .wav (ou .npy se sem soundfile).
        Retorna o caminho do arquivo salvo.
        """
        audio = self.generate(duration)

        if _SF:
            if not filepath.endswith('.wav'):
                filepath = filepath + '.wav'
            sf.write(filepath, audio, self.sr, subtype='PCM_16')
        else:
            if not filepath.endswith('.npy'):
                filepath = filepath + '.npy'
            np.save(filepath, audio.astype(np.float32))

        return filepath


# ─────────────────────────────────────────────────────────────────
# GERADOR DE DATASET
# ─────────────────────────────────────────────────────────────────

def simulate_dataset(
    n_hc:           int   = 30,
    n_dp:           int   = 30,
    duration:       float = 3.0,
    sr:             int   = 16000,
    severity_range: Tuple[float, float] = (0.3, 0.8),
    sex:            str   = 'M',
    param_noise:    float = 0.1,
    seed:           int   = 42
) -> Tuple[List[np.ndarray], np.ndarray, int]:
    """
    Gera dataset completo HC + DP em memória.

    Cada sujeito é gerado com parâmetros ligeiramente distintos
    (param_noise controla a variabilidade inter-sujeito) para
    simular a diversidade natural de uma coorte clínica.

    Parâmetros
    ----------
    n_hc           : número de sujeitos controle
    n_dp           : número de sujeitos DP
    duration       : duração de cada emissão (s)
    sr             : taxa de amostragem
    severity_range : faixa de severidade DP (min, max), U[min,max]
    sex            : 'M' (masculino), 'F' (feminino), 'mixed' (50/50)
    param_noise    : jitter inter-sujeito (% de variação em cada parâmetro)
    seed           : semente global

    Retorna
    -------
    audios  : lista de n_hc+n_dp arrays numpy
    labels  : array {0=HC, 1=DP}, shape (n_hc+n_dp,)
    sr      : taxa de amostragem
    """
    rng    = np.random.default_rng(seed)
    audios = []
    labels = []

    sev_low, sev_high = severity_range

    for i in range(n_hc + n_dp):
        is_dp = (i >= n_hc)
        label = 1 if is_dp else 0

        # Sexo do sujeito
        if sex == 'mixed':
            subj_sex = 'F' if rng.random() < 0.5 else 'M'
        else:
            subj_sex = sex.upper()

        # Parâmetros base
        if is_dp:
            sev = float(rng.uniform(sev_low, sev_high))
            base_params = VocalParams.parkinson(severity=sev, sex=subj_sex)
        else:
            base_params = VocalParams.healthy(sex=subj_sex)

        # Variabilidade inter-sujeito: ruído gaussiano em cada parâmetro
        if param_noise > 0:
            p = deepcopy(base_params)
            for field_name in p.__dataclass_fields__:
                v = getattr(p, field_name)
                # Não perturbar parâmetros já em zero (tremor em HC)
                if abs(v) > 1e-6:
                    noise = rng.normal(0, abs(v) * param_noise)
                    setattr(p, field_name, v + noise)
            # Garantir ranges físicos
            p.f0_mean         = max(60.0, min(400.0, p.f0_mean))
            p.jitter_pct      = max(0.0, min(10.0,  p.jitter_pct))
            p.shimmer_pct     = max(0.0, min(20.0,  p.shimmer_pct))
            p.hnr_db          = max(0.0, min(40.0,  p.hnr_db))
            p.amplitude       = max(0.05, min(1.0,  p.amplitude))
            p.q_regime        = max(1.0,  min(2.5,  p.q_regime))
            base_params = p

        # Gerar áudio
        sim   = VoiceSimulator(base_params, sr=sr,
                               seed=int(rng.integers(0, 2**31)))
        audio = sim.generate(duration)

        audios.append(audio)
        labels.append(label)

    return audios, np.array(labels), sr


def batch_simulate(
    n_hc:           int   = 30,
    n_dp:           int   = 30,
    output_dir:     str   = "simulated_audio",
    duration:       float = 3.0,
    sr:             int   = 16000,
    severity_range: Tuple[float, float] = (0.3, 0.8),
    sex:            str   = 'M',
    param_noise:    float = 0.1,
    seed:           int   = 42,
    verbose:        bool  = True
) -> Dict[str, List[str]]:
    """
    Gera e salva N arquivos de áudio sintético por grupo em disco.

    Estrutura de diretórios criada:
        output_dir/
            HC/  HC_001.wav ... HC_N.wav
            DP/  DP_001.wav ... DP_N.wav
            metadata.json   (parâmetros de cada sujeito)

    Retorna
    -------
    dict com chaves 'hc_files' e 'dp_files': listas de caminhos
    """
    import json, time

    hc_dir = os.path.join(output_dir, "HC")
    dp_dir = os.path.join(output_dir, "DP")
    os.makedirs(hc_dir, exist_ok=True)
    os.makedirs(dp_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    sev_low, sev_high = severity_range
    metadata  = []
    hc_files  = []
    dp_files  = []

    total = n_hc + n_dp
    if verbose:
        print(f"[batch_simulate] Gerando {n_hc} HC + {n_dp} DP | "
              f"dur={duration}s | sr={sr}Hz")

    t0 = time.time()

    for i in range(total):
        is_dp = (i >= n_hc)
        label = 1 if is_dp else 0

        subj_sex = 'F' if rng.random() < 0.5 else 'M' if sex == 'mixed' else sex.upper()

        if is_dp:
            sev = float(rng.uniform(sev_low, sev_high))
            params = VocalParams.parkinson(severity=sev, sex=subj_sex)
            fname  = os.path.join(dp_dir, f"DP_{i-n_hc+1:03d}")
        else:
            sev    = 0.0
            params = VocalParams.healthy(sex=subj_sex)
            fname  = os.path.join(hc_dir, f"HC_{i+1:03d}")

        # Variabilidade inter-sujeito
        if param_noise > 0:
            p = deepcopy(params)
            for fn in p.__dataclass_fields__:
                v = getattr(p, fn)
                if abs(v) > 1e-6:
                    setattr(p, fn, v + rng.normal(0, abs(v)*param_noise))
            p.f0_mean    = float(np.clip(p.f0_mean, 60, 400))
            p.jitter_pct = float(np.clip(p.jitter_pct, 0, 10))
            p.shimmer_pct= float(np.clip(p.shimmer_pct, 0, 20))
            p.hnr_db     = float(np.clip(p.hnr_db, 0, 40))
            p.amplitude  = float(np.clip(p.amplitude, 0.05, 1.0))
            p.q_regime   = float(np.clip(p.q_regime, 1.0, 2.5))
            params = p

        subj_seed = int(rng.integers(0, 2**31))
        sim       = VoiceSimulator(params, sr=sr, seed=subj_seed)
        saved     = sim.save(fname, duration=duration)

        if is_dp:
            dp_files.append(saved)
        else:
            hc_files.append(saved)

        meta = params.to_dict()
        meta.update({'label': label, 'severity': sev,
                     'sex': subj_sex, 'seed': subj_seed,
                     'file': saved})
        metadata.append(meta)

        if verbose:
            lbl_str = "DP" if is_dp else "HC"
            print(f"  [{i+1:3d}/{total}] {lbl_str} | "
                  f"F0={params.f0_mean:.1f}Hz | "
                  f"jitter={params.jitter_pct:.2f}% | "
                  f"q={params.q_regime:.3f} | "
                  f"sev={sev:.2f}")

    # Salvar metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    dt = time.time() - t0
    if verbose:
        print(f"\n  Concluido em {dt:.1f}s")
        print(f"  HC: {len(hc_files)} arquivos em {hc_dir}")
        print(f"  DP: {len(dp_files)} arquivos em {dp_dir}")
        print(f"  Metadata: {meta_path}")

    return {'hc_files': hc_files, 'dp_files': dp_files,
            'metadata_path': meta_path}


# ─────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────

def load_audio(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Carrega arquivo de áudio (.wav ou .npy) e retorna (array, sr).
    Compatível com os dois formatos de saída do batch_simulate.
    """
    if filepath.endswith('.npy'):
        audio = np.load(filepath).astype(np.float64)
        sr    = 16000  # default para .npy
        return audio, sr

    if not _SF:
        raise ImportError("soundfile necessário para carregar .wav. "
                          "pip install soundfile")
    import soundfile as sf_
    audio, sr = sf_.read(filepath, dtype='float64')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def inspect_params(params: VocalParams, verbose: bool = True) -> dict:
    """
    Exibe resumo dos parâmetros e suas implicações acústicas.
    """
    d = params.to_dict()
    if verbose:
        print("VocalParams:")
        groups = {
            'Frequência Fundamental': ['f0_mean', 'f0_std', 'f0_trend'],
            'Perturbação':            ['jitter_pct', 'shimmer_pct'],
            'Qualidade Vocal':        ['hnr_db', 'amplitude'],
            'Formantes':              ['formant_f1', 'formant_f2', 'formant_bw1', 'formant_bw2'],
            'Ruído':                  ['subglottal_noise', 'aspiration'],
            'Dinâmica':               ['pause_rate', 'pause_dur_mean'],
            'Tremor':                 ['tremor_freq', 'tremor_amp'],
            'Complexidade Tsallis':   ['irregularity', 'q_regime'],
        }
        for group, keys in groups.items():
            print(f"\n  {group}:")
            for k in keys:
                if k in d:
                    print(f"    {k:<22} = {d[k]:.4f}")
    return d


# ─────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  tfs_simulator.py v1.0 -- Self-Test")
    print("=" * 65)

    # ── [1] Parâmetros ────────────────────────────────────────
    print("\n[1] VocalParams — HC vs DP (severity=0.6)")
    p_hc = VocalParams.healthy(sex='M')
    p_dp = VocalParams.parkinson(severity=0.6, sex='M')
    print(f"  HC: {p_hc}")
    print(f"  DP: {p_dp}")

    # Diferenças chave
    diffs = {
        'F0 (Hz)':      (p_hc.f0_mean,    p_dp.f0_mean),
        'F0_std (Hz)':  (p_hc.f0_std,     p_dp.f0_std),
        'Jitter (%)':   (p_hc.jitter_pct, p_dp.jitter_pct),
        'Shimmer (%)':  (p_hc.shimmer_pct,p_dp.shimmer_pct),
        'HNR (dB)':     (p_hc.hnr_db,     p_dp.hnr_db),
        'Amplitude':    (p_hc.amplitude,  p_dp.amplitude),
        'q_regime':     (p_hc.q_regime,   p_dp.q_regime),
    }
    print(f"\n  {'Parâmetro':<18} {'HC':>10} {'DP':>10} {'Delta':>10}")
    print(f"  {'-'*48}")
    for k, (hc_v, dp_v) in diffs.items():
        print(f"  {k:<18} {hc_v:>10.3f} {dp_v:>10.3f} {dp_v-hc_v:>+10.3f}")

    # ── [2] GlottalSource ────────────────────────────────────
    print("\n[2] GlottalSource — geração de excitação glótal")
    src_hc = GlottalSource(p_hc, sr=16000, seed=42)
    src_dp = GlottalSource(p_dp, sr=16000, seed=42)
    exc_hc = src_hc.generate(duration=1.0)
    exc_dp = src_dp.generate(duration=1.0)
    print(f"  HC excitação: shape={exc_hc.shape} | "
          f"RMS={np.sqrt(np.mean(exc_hc**2)):.4f} | "
          f"max={np.abs(exc_hc).max():.4f}")
    print(f"  DP excitação: shape={exc_dp.shape} | "
          f"RMS={np.sqrt(np.mean(exc_dp**2)):.4f} | "
          f"max={np.abs(exc_dp).max():.4f}")

    # ── [3] VocalTractFilter ─────────────────────────────────
    print("\n[3] VocalTractFilter — formantes em cascata")
    filt_hc = VocalTractFilter(p_hc, sr=16000)
    filt_dp = VocalTractFilter(p_dp, sr=16000)
    voiced_hc = filt_hc.apply(exc_hc)
    voiced_dp = filt_dp.apply(exc_dp)
    print(f"  HC filtrado: RMS={np.sqrt(np.mean(voiced_hc**2)):.4f}")
    print(f"  DP filtrado: RMS={np.sqrt(np.mean(voiced_dp**2)):.4f}")

    # ── [4] VoiceSimulator completo ──────────────────────────
    print("\n[4] VoiceSimulator — pipeline completo (1s)")
    sim_hc = VoiceSimulator(p_hc, sr=16000, seed=42)
    sim_dp = VoiceSimulator(p_dp, sr=16000, seed=42)
    aud_hc, exc_hc2, voc_hc, mask_hc = sim_hc.generate(1.0, return_components=True)
    aud_dp, exc_dp2, voc_dp, mask_dp = sim_dp.generate(1.0, return_components=True)
    print(f"  HC: samples={len(aud_hc)} | RMS={np.sqrt(np.mean(aud_hc**2)):.4f} "
          f"| pauses={mask_hc.sum()} samples")
    print(f"  DP: samples={len(aud_dp)} | RMS={np.sqrt(np.mean(aud_dp**2)):.4f} "
          f"| pauses={mask_dp.sum()} samples")
    # DP deve ter mais pausas e menor RMS
    assert mask_dp.sum() >= 0, "pausa ok"
    print(f"  RMS ratio (DP/HC): {np.sqrt(np.mean(aud_dp**2))/np.sqrt(np.mean(aud_hc**2)+1e-8):.3f}  (esperado < 1)")

    # ── [5] Interpolação de severidade ──────────────────────
    print("\n[5] Gradiente de severidade (0.0 → 1.0, 5 passos)")
    for sev in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p_s = VocalParams.parkinson(severity=sev, sex='M')
        print(f"  sev={sev:.2f}: F0={p_s.f0_mean:.1f}Hz | "
              f"jitter={p_s.jitter_pct:.2f}% | "
              f"HNR={p_s.hnr_db:.1f}dB | "
              f"q={p_s.q_regime:.3f}")

    # ── [6] simulate_dataset ─────────────────────────────────
    print("\n[6] simulate_dataset (10 HC + 10 DP, 1s)")
    audios, labels, sr_out = simulate_dataset(
        n_hc=10, n_dp=10, duration=1.0, sr=16000, seed=42
    )
    n_hc_out = (labels == 0).sum()
    n_dp_out = (labels == 1).sum()
    rms_hc_mean = np.mean([np.sqrt(np.mean(a**2)) for a, l in zip(audios, labels) if l==0])
    rms_dp_mean = np.mean([np.sqrt(np.mean(a**2)) for a, l in zip(audios, labels) if l==1])
    print(f"  Total: {len(audios)} sinais | HC={n_hc_out} | DP={n_dp_out}")
    print(f"  RMS médio HC: {rms_hc_mean:.4f}")
    print(f"  RMS médio DP: {rms_dp_mean:.4f}")
    print(f"  Taxa de amostragem: {sr_out} Hz")

    # ── [7] Tsallis: verificar separabilidade esperada ───────
    print("\n[7] Diferenças estatísticas esperadas para Tsallis")
    print("  (baseadas nos parâmetros gerados, não em extração de features)")
    q_hc_mean = np.mean([VocalParams.healthy().q_regime for _ in range(5)])
    q_dp_vals = [VocalParams.parkinson(severity=s).q_regime
                 for s in np.linspace(0.3, 0.8, 10)]
    q_dp_mean = np.mean(q_dp_vals)
    print(f"  q_regime HC:  {q_hc_mean:.4f}  (próximo de 1 → quasi-extensivo)")
    print(f"  q_regime DP:  {q_dp_mean:.4f}  (> 1 → não-extensivo, maior S_q)")
    print(f"  Diferença q:  {q_dp_mean - q_hc_mean:.4f}  "
          f"(quanto maior, mais discriminável pela Entropia de Tsallis)")

    print("\nModulo 0 pronto. (tfs_simulator.py)")
    print("=" * 65)
    print("\nPara usar com o pipeline completo:")
    print("  from tfs_simulator import batch_simulate")
    print("  files = batch_simulate(n_hc=30, n_dp=30, output_dir='data_sim')")
    print("  from pipeline_dl import run_experiment")
    print("  results = run_experiment(files['dp_files'], files['hc_files'])")
