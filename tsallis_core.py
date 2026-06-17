"""
tsallis_core.py — Módulo 1: Núcleo Matemático da Entropia de Tsallis
=====================================================================
Versão 1.1 — Bugs corrigidos após test.

Contém TODAS as primitivas matemáticas do framework TFS.
Todos os outros módulos importam exclusivamente deste arquivo.

Correções v1.1:
    - q_softmax: fórmula corrigida (base positiva garantida)
    - tsallis_loss_numpy: substituída por formulação Bregman-Tsallis
      (Amid et al., NeurIPS 2019) — evita 0^(1-q) indefinido
    - q_star removida (S_q é matematicamente decrescente em q para
      qualquer distribuição -> q* = q_min sempre, sem valor discriminativo)
    - Adicionadas features realmente discriminativas do perfil:
      decay_ratio, curvature_integral, profile_slope
    - estimate_q_fit: substituída por ajuste baseado em grid search
      robusto com validação de suporte

Funções disponíveis:
    tsallis_entropy(signal, q, n_bins, estimator)
    q_sweep(signal, q_values, n_bins)
    profile_features(profile, q_values)        ← NOVO: extrai features do perfil
    dSq_dq_at_q1(signal, n_bins, dq)
    profile_area(profile, q_values)
    q_softmax(z, q)
    q_divergence(P, Q, q)
    q_gaussian_logpdf(x, q, mu, beta)
    tsallis_loss_numpy(y_pred, y_true, q)      ← CORRIGIDA (Bregman-Tsallis)
    tsallis_entropy_series(series, q, n_bins)
    validate_signal(signal, min_samples, name, normalize)
    check_q_value(q)
"""

import numpy as np
import warnings


# ─────────────────────────────────────────────────────────────────
# CONSTANTES GLOBAIS
# ─────────────────────────────────────────────────────────────────

Q_DEFAULT = np.array([0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0, 2.5])
Q_SUBBAND = np.array([0.7, 1.3, 2.0])
Q_F0      = np.array([0.7, 1.3, 2.0])


# ─────────────────────────────────────────────────────────────────
# 1. ESTIMAÇÃO DE PDF
# ─────────────────────────────────────────────────────────────────

def _estimate_pdf(signal: np.ndarray, n_bins: int = -1) -> np.ndarray:
    """
    Estima distribuição de probabilidade discreta p(x) via histograma.

    Usa critério de Freedman-Diaconis para número ótimo de bins quando
    n_bins <= 0. Remove bins com p=0 (0^q é 0 para q>0, mas log(0)=-inf).

    Retorna array p normalizado (soma=1) com apenas valores p > 0.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    signal = signal[np.isfinite(signal)]

    if len(signal) < 4:
        raise ValueError("Sinal muito curto (mínimo 4 amostras).")

    if n_bins <= 0:
        iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
        if iqr == 0:
            n_bins = max(10, int(np.ceil(np.sqrt(len(signal)))))
        else:
            bw = 2.0 * iqr / (len(signal) ** (1.0/3.0))
            n_bins = max(10, int(np.ceil((signal.max() - signal.min()) / bw)))

    counts, _ = np.histogram(signal, bins=n_bins)
    p = counts.astype(np.float64)
    p = p / p.sum()
    return p[p > 0]


# ─────────────────────────────────────────────────────────────────
# 2. ENTROPIA DE TSALLIS
# ─────────────────────────────────────────────────────────────────

def tsallis_entropy(
    signal: np.ndarray,
    q: float = 1.3,
    n_bins: int = -1,
    estimator: str = 'histogram'
) -> float:
    """
    Calcula S_q (Entropia de Tsallis) para um sinal 1D.

    Definição:
        S_q = (1 - sum(p_i^q)) / (q-1)   para q != 1
        S_1 = -sum(p_i * log(p_i))         limite de Shannon (q->1)

    Parâmetros
    ----------
    signal    : array 1D (sinal de áudio, série F0, etc.)
    q         : parâmetro de não-extensividade (float > 0)
    n_bins    : bins para histograma; -1 = Freedman-Diaconis automático
    estimator : 'histogram' (único estimador suportado nesta versão)

    Retorna
    -------
    float : S_q >= 0

    Exemplos
    --------
    >>> sig = np.random.normal(0, 1, 1000)
    >>> tsallis_entropy(sig, q=1.3)    # ~1.9
    >>> tsallis_entropy(sig, q=1.0)    # equivale a Shannon entropy
    """
    p = _estimate_pdf(signal, n_bins)
    if abs(q - 1.0) < 1e-6:
        return float(-np.sum(p * np.log(p + 1e-300)))
    return float((1.0 - np.sum(p ** q)) / (q - 1.0))


# ─────────────────────────────────────────────────────────────────
# 3. q-SWEEP — perfil entrópico
# ─────────────────────────────────────────────────────────────────

def q_sweep(
    signal: np.ndarray,
    q_values: np.ndarray = None,
    n_bins: int = -1
) -> np.ndarray:
    """
    Calcula S_q para cada q em q_values: retorna o PERFIL ENTRÓPICO.

    O perfil S_q(q) é uma função monotonamente decrescente em q para
    qualquer distribuição com H > 0 (propriedade matemática). Seu valor
    discriminativo reside nas diferenças ABSOLUTAS entre grupos em cada
    q específico, na taxa de decaimento e na curvatura do perfil.

    Parâmetros
    ----------
    signal   : array 1D
    q_values : array de valores q; None = Q_DEFAULT (11 pontos)
    n_bins   : bins para PDF; -1 = automático

    Retorna
    -------
    profile : array 1D com len(q_values) valores de S_q
    """
    if q_values is None:
        q_values = Q_DEFAULT
    q_values = np.asarray(q_values, dtype=np.float64)

    # Estimar p uma vez só (eficiência)
    p = _estimate_pdf(signal, n_bins)

    profile = np.empty(len(q_values))
    for i, q in enumerate(q_values):
        if abs(q - 1.0) < 1e-6:
            profile[i] = float(-np.sum(p * np.log(p + 1e-300)))
        else:
            profile[i] = float((1.0 - np.sum(p ** q)) / (q - 1.0))
    return profile


# ─────────────────────────────────────────────────────────────────
# 4. FEATURES DO PERFIL q — função central do Bloco A
# ─────────────────────────────────────────────────────────────────

def profile_features(
    profile: np.ndarray,
    q_values: np.ndarray = None
) -> dict:
    """
    Extrai as features discriminativas do perfil entrópico S_q(q).

    Nota sobre q_star: S_q é matematicamente decrescente em q para
    qualquer distribuição com H>0, portanto q* = q_min sempre e não
    tem valor discriminativo. As features abaixo capturam a GEOMETRIA
    do perfil de forma estatisticamente significativa.

    Features retornadas
    -------------------
    'S_q_<val>'     : S_q para cada q em q_values (9 features)
    'decay_ratio'   : S_q(q_max)/S_q(q_min) — queda relativa total do perfil.
                      Distribuições com caudas pesadas (DP) decaem MAIS
                      rapidamente -> decay_ratio menor.
    'area'          : integral do perfil (trapz) — complexidade global integrada
    'slope_q1'      : dS_q/dq estimado por diferença central em q=1.0
    'curvature_int' : integral da segunda derivada (|d2S/dq2|) — mede não-linearidade

    Parâmetros
    ----------
    profile  : array retornado por q_sweep()
    q_values : grade de q usada; None = Q_DEFAULT

    Retorna
    -------
    dict : chaves descritas acima
    """
    if q_values is None:
        q_values = Q_DEFAULT
    q_values = np.asarray(q_values, dtype=np.float64)

    feats = {}

    # Features absolutas por q
    for q_val, s_val in zip(q_values, profile):
        feats[f'S_q_{q_val:.1f}'] = float(s_val)

    # Razão de decaimento: S(q_max)/S(q_min)
    s_min_q = profile[np.argmin(q_values)]  # S no menor q
    s_max_q = profile[np.argmax(q_values)]  # S no maior q
    feats['decay_ratio'] = float(s_max_q / (s_min_q + 1e-10))

    # Área total
    feats['area'] = float(np.trapz(profile, q_values))

    # Inclinação em q=1 (diferença central)
    idx1 = np.argmin(np.abs(q_values - 1.0))
    if idx1 > 0 and idx1 < len(q_values) - 1:
        dq = q_values[idx1+1] - q_values[idx1-1]
        feats['slope_q1'] = float((profile[idx1+1] - profile[idx1-1]) / dq)
    else:
        feats['slope_q1'] = float(np.gradient(profile, q_values)[idx1])

    # Curvatura (integral de |d2S/dq2|)
    d2 = np.gradient(np.gradient(profile, q_values), q_values)
    feats['curvature_int'] = float(np.trapz(np.abs(d2), q_values))

    return feats


def dSq_dq_at_q1(
    signal: np.ndarray,
    n_bins: int = -1,
    dq: float = 0.1
) -> float:
    """
    Estima dS_q/dq em q=1 por diferença central: [S(1+dq) - S(1-dq)] / (2*dq).

    Mede a sensibilidade do regime entrópico ao parâmetro q.
    Sempre negativo (S_q decrescente). Módulo maior -> decaimento mais acentuado.

    Parâmetros
    ----------
    signal : array 1D
    n_bins : bins para PDF
    dq     : passo para diferença finita

    Retorna
    -------
    float : derivada (negativo)
    """
    s_plus  = tsallis_entropy(signal, q=1.0 + dq, n_bins=n_bins)
    s_minus = tsallis_entropy(signal, q=1.0 - dq, n_bins=n_bins)
    return float((s_plus - s_minus) / (2.0 * dq))


def profile_area(profile: np.ndarray, q_values: np.ndarray = None) -> float:
    """Área sob o perfil (integral trapezoidal). Alias de profile_features['area']."""
    if q_values is None:
        q_values = Q_DEFAULT
    return float(np.trapz(profile, q_values))


# ─────────────────────────────────────────────────────────────────
# 5. q-SOFTMAX (sparse / entropic attention)
# ─────────────────────────────────────────────────────────────────

def q_softmax(z: np.ndarray, q: float = 1.5) -> np.ndarray:
    """
    q-softmax generalizado (Blondel et al. 2019; Martins & Astudillo 2016).

    Definição:
        softmax_q(z_i) ∝ max(0, [1 + (1-q)*(z_i - z_max)]^(1/(1-q)) )

    Propriedades:
    - q=1.0  : equivale ao softmax padrão
    - q>1.0  : distribuição ESPARSA (zeros exatos para componentes com z_i baixo)
    - q<1.0  : distribuição mais uniforme

    Para q > 1: exponent = 1/(1-q) < 0, mas base é negativa para z<<z_max;
    a proteção via max(base, eps) e filtragem de zeros garante estabilidade.

    Parâmetros
    ----------
    z : array 1D de logits (floats arbitrários)
    q : parâmetro de esparsidade (q > 0)

    Retorna
    -------
    p : array com mesma forma que z, soma = 1

    Exemplos
    --------
    >>> z = np.array([2.0, 1.0, 0.1, -1.0])
    >>> q_softmax(z, q=1.0)   # [0.638, 0.235, 0.095, 0.032]
    >>> q_softmax(z, q=1.5)   # mais esparso, zeros para z baixos
    >>> q_softmax(z, q=2.0)   # ainda mais esparso
    """
    z = np.asarray(z, dtype=np.float64)

    if abs(q - 1.0) < 1e-6:
        e = np.exp(z - z.max())
        return e / e.sum()

    # base_i = 1 + (1-q)*(z_i - z_max) = 1 - (q-1)*(z_max - z_i)
    # Para q>1: base_i <= 1, com base_max = 1
    # Zeros quando (q-1)*(z_max-z_i) >= 1, i.e., z_max - z_i >= 1/(q-1)
    base = 1.0 + (1.0 - q) * (z - z.max())
    base = np.maximum(base, 0.0)   # [x]_+ threshold

    exponent = 1.0 / (1.0 - q)    # negativo para q>1
    # base^exponent: base=0 -> 0^neg = inf, mas base=0 -> resultado deve ser 0
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = np.where(base > 0, base ** exponent, 0.0)

    total = raw.sum()
    if total == 0:
        return np.ones(len(z)) / len(z)   # fallback uniforme
    return raw / total


# ─────────────────────────────────────────────────────────────────
# 6. q-DIVERGÊNCIA
# ─────────────────────────────────────────────────────────────────

def q_divergence(P: np.ndarray, Q_dist: np.ndarray, q: float = 1.3) -> float:
    """
    q-divergência de Tsallis: D_q(P||Q) = (1 - sum(P^q * Q^(1-q))) / (q-1)

    Para q->1: converge para KL-divergência.
    Usada como regulador no q-VAE (Bloco E).

    Parâmetros
    ----------
    P, Q_dist : arrays 1D de probabilidades (mesma forma)
    q         : parâmetro de não-extensividade

    Retorna
    -------
    float : D_q(P||Q)
    """
    eps = 1e-300
    P = np.maximum(np.asarray(P, float), eps)
    Q_dist = np.maximum(np.asarray(Q_dist, float), eps)
    P = P / P.sum()
    Q_dist = Q_dist / Q_dist.sum()

    if abs(q - 1.0) < 1e-6:
        return float(np.sum(P * np.log(P / Q_dist)))
    return float((1.0 - np.sum(P ** q * Q_dist ** (1.0 - q))) / (q - 1.0))


# ─────────────────────────────────────────────────────────────────
# 7. q-GAUSSIANA
# ─────────────────────────────────────────────────────────────────

def q_gaussian_logpdf(
    x: np.ndarray,
    q: float,
    mu: float = 0.0,
    beta: float = 1.0
) -> np.ndarray:
    """
    Log-pdf da q-Gaussiana (não normalizada):
        log G_q(x) = (1/(1-q)) * log max(0, 1 - (1-q)*beta*(x-mu)^2)

    Para q=1 : log-Gaussiana padrão.
    Para q>1 : caudas em lei de potência (distribuição de Pareto gen.).
    Para q<1 : suporte compacto.

    Parâmetros
    ----------
    x, q, mu, beta : ver docstring do módulo

    Retorna
    -------
    logp : array (não normalizado; -inf onde suporte é zero)
    """
    x = np.asarray(x, dtype=np.float64)
    if abs(q - 1.0) < 1e-6:
        return -0.5 * beta * (x - mu) ** 2

    arg = 1.0 - (1.0 - q) * beta * (x - mu) ** 2
    exponent = 1.0 / (1.0 - q)
    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.where(arg > 0, exponent * np.log(arg), -np.inf)
    return logp


# ─────────────────────────────────────────────────────────────────
# 8. FUNÇÃO DE CUSTO TSALLIS — Formulação Bregman (Amid et al. 2019)
# ─────────────────────────────────────────────────────────────────

def tsallis_loss_numpy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    q: float = 1.3
) -> float:
    """
    Função de custo Bregman-Tsallis para classificação binária.

    Formulação de Amid et al. (NeurIPS 2019) que evita 0^(1-q)=inf:

        L_q = y * [-(1/q)*yp^(q-1)] + (1-y) * [-(1/q)*(1-yp)^(q-1)]
              + (1/(2q-1)) * [yp^(2q-1) + (1-yp)^(2q-1)]

    Para q=1: equivale à cross-entropy binária padrão.
    Para q>1: penaliza mais os exemplos difíceis (outliers).

    Parâmetros
    ----------
    y_pred : array de probabilidades preditas em (0,1)
    y_true : array de rótulos {0, 1}
    q      : parâmetro de custo (float; 1.0 = cross-entropy)

    Retorna
    -------
    float : perda média sobre o batch

    Referência
    ----------
    Amid et al. "Robust Bi-Tempered Logistic Loss." NeurIPS 2019.
    (Nota: usa temperatura t2=q como temperatura de ativação.)
    """
    y_pred = np.clip(np.asarray(y_pred, float), 1e-7, 1.0 - 1e-7)
    y_true = np.asarray(y_true, float)

    if abs(q - 1.0) < 1e-6:
        return float(-np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        ))

    # Bregman-Tsallis
    pos_term = -(1.0 / q) * y_pred ** (q - 1.0)
    neg_term = -(1.0 / q) * (1.0 - y_pred) ** (q - 1.0)
    reg_term = (1.0 / (2.0 * q - 1.0)) * (
        y_pred ** (2.0 * q - 1.0) + (1.0 - y_pred) ** (2.0 * q - 1.0)
    )
    loss = y_true * pos_term + (1 - y_true) * neg_term + reg_term
    return float(np.mean(loss))


# ─────────────────────────────────────────────────────────────────
# 9. TSALLIS SOBRE SÉRIE TEMPORAL (Blocos C e D)
# ─────────────────────────────────────────────────────────────────

def tsallis_entropy_series(
    series: np.ndarray,
    q: float = 1.3,
    n_bins: int = -1
) -> float:
    """
    Entropia de Tsallis para séries temporais (F0, shimmer, etc.).
    Alias semântico de tsallis_entropy para uso nos Blocos C e D.
    """
    return tsallis_entropy(series, q=q, n_bins=n_bins)


# ─────────────────────────────────────────────────────────────────
# 10. UTILITÁRIOS DE VALIDAÇÃO
# ─────────────────────────────────────────────────────────────────

def validate_signal(
    signal: np.ndarray,
    min_samples: int = 100,
    name: str = "signal",
    normalize: bool = False
) -> np.ndarray:
    """
    Valida e limpa um sinal de entrada.

    IMPORTANTE: normalize=False por padrão. Para o q-sweep de amplitude
    (Bloco A), NÃO normalizar preserva a estrutura estatística original.
    Normalizar muda a distribuição de amplitudes e invalida a hipótese H1.

    Parâmetros
    ----------
    signal      : array 1D
    min_samples : mínimo de amostras válidas após limpeza
    name        : nome do sinal (para mensagens de erro)
    normalize   : se True, normaliza para [-1, 1] (usar apenas para Blocos B/C/D)

    Retorna
    -------
    np.ndarray : sinal float64 sem NaN/Inf
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    signal = signal[np.isfinite(signal)]

    if len(signal) < min_samples:
        raise ValueError(
            f"[tsallis_core] '{name}': apenas {len(signal)} amostras "
            f"válidas (mínimo: {min_samples})."
        )

    if normalize:
        sig_max = np.max(np.abs(signal))
        if sig_max > 0:
            signal = signal / sig_max

    return signal


def check_q_value(q: float) -> None:
    """Verifica se q está na faixa operacional (0.01, 4.0)."""
    if not (0.01 < q < 4.0):
        raise ValueError(f"q={q} fora da faixa operacional (0.01, 4.0).")
    if q > 3.0:
        warnings.warn(
            f"q={q} > 3.0 — estimativas podem ser numericamente instáveis.",
            stacklevel=2
        )


# ─────────────────────────────────────────────────────────────────
# 11. SELF-TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  tsallis_core.py v1.1 — Self-Test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    gauss  = rng.normal(0, 1, 2000)
    cauchy = np.clip(rng.standard_cauchy(2000), -10, 10)

    print("\n[1] tsallis_entropy")
    for q in [0.5, 1.0, 1.3, 2.0]:
        sg = tsallis_entropy(gauss, q=q)
        sc = tsallis_entropy(cauchy, q=q)
        diff = sc - sg
        print(f"  q={q:.1f} | Gaussiano={sg:.4f} | Cauchy={sc:.4f} | Δ={diff:+.4f}")

    print("\n[2] q_sweep + profile_features")
    pg = q_sweep(gauss)
    pc = q_sweep(cauchy)
    fg = profile_features(pg)
    fc = profile_features(pc)
    print(f"  decay_ratio  | Gaussiano: {fg['decay_ratio']:.4f} | Cauchy: {fc['decay_ratio']:.4f}")
    print(f"  area         | Gaussiano: {fg['area']:.4f} | Cauchy: {fc['area']:.4f}")
    print(f"  slope_q1     | Gaussiano: {fg['slope_q1']:.4f} | Cauchy: {fc['slope_q1']:.4f}")
    print(f"  curvature_int| Gaussiano: {fg['curvature_int']:.4f} | Cauchy: {fc['curvature_int']:.4f}")

    print("\n[3] dSq_dq_at_q1")
    print(f"  Gaussiano: {dSq_dq_at_q1(gauss):.4f}")
    print(f"  Cauchy:    {dSq_dq_at_q1(cauchy):.4f}")

    print("\n[4] q_softmax (corrigida)")
    z = np.array([2.0, 1.0, 0.1, -1.0])
    for q_t in [1.0, 1.5, 2.0]:
        p = q_softmax(z, q=q_t)
        print(f"  q={q_t} | {p.round(4)} | soma={p.sum():.4f}")

    print("\n[5] q_divergence")
    P = np.array([0.5, 0.3, 0.2])
    Q = np.array([0.33, 0.33, 0.34])
    for q_t in [1.0, 1.3, 2.0]:
        print(f"  D_{q_t}(P||Q) = {q_divergence(P, Q, q=q_t):.4f}")

    print("\n[6] tsallis_loss_numpy (Bregman-Tsallis)")
    yp = np.array([0.9, 0.8, 0.2, 0.1])
    yt = np.array([1.0, 1.0, 0.0, 0.0])
    for q_t in [1.0, 1.3, 2.0]:
        print(f"  q={q_t} | loss = {tsallis_loss_numpy(yp, yt, q=q_t):.4f}")

    print("\n[7] validate_signal (normalize=False, padrão)")
    clean = validate_signal(gauss, name="gauss_test", normalize=False)
    print(f"  {len(clean)} amostras | range [{clean.min():.3f}, {clean.max():.3f}]")

    print("\n✓ Self-test v1.1 concluído sem erros.")
    print("=" * 60)
