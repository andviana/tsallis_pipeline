# NonExtensive_Voices

## TFS Framework

**Tsallis Feature System · Parkinson Voice Analysis**

`v1.1 · Maio 2026 · Python 3.9+`

|                 |                                               |
|-----------------|-----------------------------------------------|
| **Versão**      | 1.1 (Maio 2026)                               |
| **Linguagem**   | Python 3.9+                                   |
| **Dependências core** | numpy · scipy                          |
| **Dependências opcionais** | librosa · torch · soundfile · pandas |
| **Módulos**     | 5 (tsallis_core · tfs_ab · tfs_cd · tfs_e_vae · tfs_simulator) |
| **Features totais** | 55 (A:13 + B:24 + C:8 + D:12) + representações q-VAE |
| **Benchmark alvo** | Iyer et al. 2023 — CNN/AUC=0.97 sobre espectrogramas |

---

> **Sobre:** O TFS é um pipeline científico para extração de biomarcadores acústicos da doença de Parkinson usando a Entropia de Tsallis S_q como medida de complexidade central. Opera sobre fonação sustentada (vogal /a/) e gera 55 features interpretáveis processadas por um q-VAE para aprendizado de representação não-extensiva.

---

## 1. Visão Geral da Arquitetura

O TFS é organizado em cinco módulos com responsabilidades únicas que formam um pipeline hierárquico. A premissa central é que a voz na DP mudou de regime dinâmico — transitando de um oscilador caótico controlado para um sistema de menor grau de liberdade. A Entropia de Tsallis S_q captura esta transição por ser sensível a correlações de longo alcance e distribuições de cauda pesada, que a Entropia de Shannon (q→1) é incapaz de detectar.

| Módulo | Arquivo          | Responsabilidade                                    | Output           |
|:------:|:-----------------|:----------------------------------------------------|:-----------------|
| **M1** | tsallis_core.py  | Primitivas matemáticas: S_q, q-sweep, q-softmax, q-VAE loss, q-divergência | Funções          |
| **M2** | tfs_ab.py        | Bloco A: q-Sweep amplitude (13 feat.) + Bloco B: TES subbandas (24 feat.) | 37 features      |
| **M3** | tfs_cd.py        | Bloco C: S_q de F0/perturbação (8 feat.) + Bloco D: q-Cronoentropograma (12 feat.) | 20 features      |
| **M4** | tfs_e_vae.py     | q-VAE com ELBO Tsallis: aprendizado de representação não-extensiva | Espaço latente Z |
| **M5** | tfs_simulator.py | Simulador biofísico (GlottalSource + VocalTract + NoiseLayer) para validação | Áudio sintético  |

**Fluxo:** `.wav` → [M2: load+preprocess] → [M2: A+B] + [M3: C+D] → vetor 55 feat. → [M4: q-VAE] → Z → classificador.

---

## 2. Instalação e Dependências

### 2.1 Dependências Core (obrigatórias)

```bash
pip install numpy scipy
```

### 2.2 Dependências Opcionais

| Pacote        | Necessário para                                                  |
|:--------------|:-----------------------------------------------------------------|
| librosa       | Carregamento .mp3/.ogg, YIN para F0 (M3), resampling de alta qualidade |
| torch         | TsallisVAE completo com GPU. Sem torch: NumpyQVAE (PCA+SVD) ativado automaticamente |
| soundfile     | Escrita de .wav no simulador (M5). Fallback: .npy                |
| pandas        | batch_extract_AB() e batch_extract_CD() retornam DataFrame       |
| scikit-learn  | Pipeline de classificação downstream (SVM, RF, XGBoost)          |

```bash
pip install librosa torch soundfile pandas scikit-learn
```

### 2.3 Estrutura de Arquivos

```
projeto/
├── tsallis_core.py    # M1 — DEVE estar no mesmo dir ou no sys.path
├── tfs_ab.py          # M2 — from tsallis_core import ...
├── tfs_cd.py          # M3 — from tsallis_core import ...
├── tfs_e_vae.py       # M4 — from tsallis_core import ...
├── tfs_simulator.py   # M5 — independente
└── dados/
    ├── HC/            # arquivos .wav controle saudável
    └── DP/            # arquivos .wav Parkinson
```

> **CRÍTICO:** todos os módulos importam tsallis_core com `from tsallis_core import ...`. O arquivo `tsallis_core.py` DEVE estar no diretório de trabalho ou em um caminho do sys.path.

---

## 3. Módulo 1 — tsallis_core.py

Primitivas matemáticas do framework:

- **entropia_tsallis(P, q):** calcula S_q = (1 - sum(P^q)) / (q - 1). Para q→1, recupera Shannon.
- **q_sweep(signal, q_min, q_max, n_q):** varre S_q ao longo dos níveis de amplitude do sinal (Bloco A).
- **q_softmax(logits, q):** generalização da softmax via q-exponencial (Blondel et al., 2019).
- **q_divergencia(P, Q, q):** divergência de Tsallis entre duas distribuições.
- **q_gaussiana(x, q, beta):** função de densidade q-gaussiana.

---

## 4. Módulo 2 — tfs_ab.py (Blocos A e B)

### Bloco A — q-Sweep de Amplitude (13 features)

Varre a entropia S_q ao longo dos níveis de amplitude do sinal normalizado.

- A1–A4: S_q em q = [0.5, 1.0, 1.5, 2.0]
- A5: Delta S_q entre q=0.5 e q=2.0 (sensibilidade a long-range correlations)
- A6: q_ótimo (valor de q que maximiza S_q)
- A7: S_q no q_ótimo
- A8: Taxa de decaimento de S_q com q
- A9–A13: Momentos estatísticos da curva S_q(q)

### Bloco B — TES (Tsallis Entropic Spectrum) (24 features)

Decompõe o sinal em subbandas e calcula S_q por subbanda.

- B1–B4: Entropia S_q nas subbandas [0–500Hz], [500–1500Hz], [1500–3000Hz], [3000–4000Hz]
- B5–B8: Entropia nas subbandas [0–200], [200–600], [600–1400], [1400–2800] (fs=8kHz)
- B9–B16: S_q em múltiplos q por subbanda (Detail subband entropy)
- B17–B24: Razões de entropia entre subbandas

> **Nota:** B8 retorna NaN se fs < 8kHz (violando Nyquist). Sempre verifique fs dos arquivos clínicos.

**Funções principais:**
- `extract_AB(wave, fs, q=1.3)` → vetor [A1..A13, B1..B24]
- `batch_extract_AB(filepaths, labels)` → DataFrame (requer pandas)
- `validate_signal(wave, fs, normalize=True)` → validação de sinal

---

## 5. Módulo 3 — tfs_cd.py (Blocos C e D)

### Bloco C — Frequência Fundamental e Perturbação (8 features)

- C1: S_q do F0 (complexidade da F0 ao longo do tempo)
- C2: q_S_q(F0) — variabilidade do S_q do F0 em múltiplos q
- C3: Jitter (via YIN/librosa) — perturbação cíclica de F0
- C4: Jitter local médio
- C5: Jitter RAP (Relative Average Perturbation)
- C6: Jitter PPQ (Pitch Period Perturbation Quotient)
- C7: Shimmer (perturbação cíclica de amplitude)
- C8: Shimmer local médio

**Extração de F0:** YIN (librosa) com fallback para autocorrelação.

### Bloco D — q-Cronoentropograma (12 features)

Aplica janelas deslizantes e calcula a evolução temporal de S_q.

- D1–D4: S_q nas fases: onset, ramp-up, steady-state, decay
- D5: Delta S_q (steady - onset) — dinâmica de transição
- D6: Taxa de variação de S_q no onset
- D7: Entropia média no plateau
- D8: Variabilidade no plateau
- D9: Tempo até atingir steady-state
- D10: Medida de não-estacionariedade
- D11: Assimetria temporal
- D12: Taxa de decaimento pós-steady

**Funções principais:**
- `extract_CD(wave, fs, q=1.3)` → vetor [C1..C8, D1..D12]
- `batch_extract_CD(filepaths, labels)` → DataFrame
- `detect_steady_state(signal, fs)` → segmentação automática da fase sustentada

---

## 6. Módulo 4 — tfs_e_vae.py (q-VAE)

Variational Autoencoder baseado em Entropia de Tsallis com ELBO não-extensiva.

### TsallisVAE (requer torch)

```python
from tfs_e_vae import TsallisVAE, qELBO_loss

# Instância
model = TsallisVAE(input_dim=61, latent_dim=8, q=1.3, beta=1.0)

# Treino
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(50):
    optimizer.zero_grad()
    x = torch.tensor(features, dtype=torch.float32)
    z, x_rec = model(x)
    loss = qELBO_loss(x, x_rec, model.mu, model.logvar, q=1.3)
    loss.backward()
    optimizer.step()
```

### NumpyQVAE (sem torch — fallback)

PCA + SVD para geração de representações latentes interpretáveis.

**Parâmetros:**
- `q`: entropic index (1.5 = caótico, 1.0 = Shannon, 1.3 = regime intermediário DP)
- `latent_dim`: dimensão do espaço latente Z (4–16)
- `beta`: peso do termo KL-generalizado no ELBO

---

## 7. Módulo 5 — tfs_simulator.py

Simulador biofísico para validação e geração de dados sintéticos.

```python
from tfs_simulator import VoiceSimulator, GlottalSource, VocalTract

sim = VoiceSimulator(fs=22050, duration=3.0, severity=0.7, q_regime=1.3)
audio, metadata = sim.generate()

# Parâmetros:
# severity: 0.0 (HC) → 1.0 (DP severo)
# q_regime: regime não-extensivo para jitter/shimmer
# fund_freq: F0 base (Hz)
# formants: [(F1, B1), (F2, B2), (F3, B3)]
```

**Componentes:**
- **GlottalSource:** modelo de fonte glotal (Rosenberg/Klatt)
- **VocalTract:** filtro formante para vogal /a/
- **NoiseLayer:** adição de ruído q-gaussiano

---

## 8. Pipeline Completo — Uso Rápido

### Exemplo completo

```python
import numpy as np
from tfs_ab import extract_AB, validate_signal
from tfs_cd import extract_CD, detect_steady_state
from tfs_e_vae import TsallisVAE
import librosa

# 1. Carregar áudio
wave, fs = librosa.load('dados/HC/subject01.wav', sr=None)

# 2. Validar e segmentar fase sustentada
wave = validate_signal(wave, fs, normalize=False)
wave_ss, onset = detect_steady_state(wave, fs)

# 3. Extrair features
A, B = extract_AB(wave_ss, fs, q=1.3)  # 13 + 24 = 37 feat.
C, D = extract_CD(wave_ss, fs, q=1.3)  # 8 + 12 = 20 feat.
features = np.concatenate([A, B, C, D])  # 55 feat.

# 4. q-VAE — representação latente
model = TsallisVAE(input_dim=55, latent_dim=8, q=1.3)
# (treinar com dados completos antes de usar)
z, x_rec = model(features)

# 5. Classificação
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

### Batch processing

```python
from tfs_ab import batch_extract_AB
from tfs_cd import batch_extract_CD
import pandas as pd
import glob

filepaths_HC = glob.glob('dados/HC/*.wav')
filepaths_DP = glob.glob('dados/DP/*.wav')

features_HC = batch_extract_AB(filepaths_HC, labels=0)
features_DP = batch_extract_AB(filepaths_DP, labels=1)

X = pd.concat([features_HC, features_DP])
y = np.array([0]*len(filepaths_HC) + [1]*len(filepaths_DP))
```

---

## 9. Dicionário de Features

### Bloco A (13 features)

| Código | Descrição                          | Fórmula/Origem           |
|:-------|:-----------------------------------|:------------------------|
| A1     | S_q em q=0.5                       | entropia_tsallis(P, 0.5)|
| A2     | S_q em q=1.0 (Shannon)             | -sum(P*ln(P))           |
| A3     | S_q em q=1.5                       | entropia_tsallis(P, 1.5)|
| A4     | S_q em q=2.0 (Simpson)             | 1 - sum(P^2)            |
| A5     | Delta S_q (q=0.5 → 2.0)            | A1 - A4                 |
| A6     | q_ótimo                            | argmax_q S_q(q)         |
| A7     | S_q no q_ótimo                     | max(S_q)               |
| A8     | Taxa de decaimento                 | dS_q/dq                 |
| A9–A13 | Momentos da curva S_q(q)           | mean, std, skew, kurt,熵 |

### Bloco B (24 features)

| Código | Descrição                      | Subbanda                  |
|:------:|:-------------------------------|:-------------------------  |
| B1     | S_q do espectro                | 0–500 Hz                  |
| B2     | S_q do espectro                | 500–1500 Hz               |
| B3     | S_q do espectro                | 1500–3000 Hz              |
| B4     | S_q do espectro                | 3000–4000 Hz              |
| B5–B8  | S_q em q=1.3                   | 4 subbandas baixas        |
| B9–B16 | S_q em múltiplos q             | subbandas detalhadas      |
| B17–B24| Razões B_i / B_j               | entre subbandas           |

### Bloco C (8 features)

| Código | Descrição                          |
|:-------|:-----------------------------------|
| C1     | S_q da série F0                    |
| C2     | q_S_q(F0) — variabilidade          |
| C3     | Jitter médio                       |
| C4     | Jitter local                       |
| C5     | Jitter RAP                         |
| C6     | Jitter PPQ                         |
| C7     | Shimmer médio                      |
| C8     | Shimmer local                      |

### Bloco D (12 features)

| Código | Descrição                          |
|:-------|:-----------------------------------|
| D1–D4  | S_q nas fases (onset, ramp, steady, decay) |
| D5     | Delta S_q (steady - onset)         |
| D6     | Taxa onset                         |
| D7     | Entropia plateau                   |
| D8     | Variabilidade plateau              |
| D9     | Tempo até steady                   |
| D10    | Não-estacionariedade               |
| D11    | Assimetria temporal                |
| D12    | Decaimento pós-steady              |

---

## 10. Fundamentos Físicos

### Premissa Teórica

A voz humana saudável é um sistema complexo com alto grau de liberdade, exibindo comportamento entre periódico e caótico. Na Doença de Parkinson, alterações neuromusculares reduzem o controle fino das pregas vocais, culminando em:

- **H1:** Redução da variabilidade de amplitude (menor rango de jitter/shimmer)
- **H2:** Perda de estrutura temporal (mais aleatório, menos caótico controlado)
- **H3:** Alteração no regime não-extensivo (mudança no índice q ótimo)

### Por que Tsallis?

A Entropia de Shannon (q→1) pressupõe:
- Independência entre eventos
- Distribuições de cauda leve (exponencial, gaussiana)

Sistemas complexos biológicos violam essas premissas:
- Correlações de longo alcance na voz
- Distribuições de cauda pesada
- Dependências não-lineares entre ciclos vocais

A Entropia de Tsallis S_q, com q ≠ 1, captura esses efeitos através de um parâmetro de entropic index q que quantifica o desvio do regime extensivo (Shannon).

---

## 11. Notas de Implementação

### Bugs corrigidos

- **normalize=False no Bloco A:** O sinal NÃO deve ser normalizado antes do Bloco A. A normalização altera a distribuição de amplitudes e invalida H1. Normalização é usada apenas nos Blocos B, C e D.
- **Nyquist no Bloco B:** Para fs=8kHz, B8 retorna NaN automaticamente. Sempre verifique fs dos arquivos clínicos antes de usar todas as 24 features.
- **YIN vs Autocorrelação:** O extrator de F0 usa YIN (librosa) quando disponível. Para fonação sustentada estável, autocorrelação é igualmente precisa. Para sinais de alta variabilidade (DP severo), YIN é preferível.
- **q-VAE com datasets pequenos:** Para n < 50 sujeitos, use batch_size=8–16 e latent_dim=4. Com n < 30, considere NumpyQVAE + PCA como baseline interpretável.
- **Imputação de NaN:** Features NaN devem ser imputadas por mediana antes do classificador: `SimpleImputer(strategy='median')`.
- **q_regime no Simulador:** Use severity_range=(0.5, 0.8) para simular DP moderada a severa com dados sintéticos.

### Reprodutibilidade

```python
import numpy as np
np.random.seed(42)
from tfs_simulator import set_seed
set_seed(42)
```

---

## Referências

- **Tsallis, C.** (1988). Possible generalization of Boltzmann-Gibbs statistics. *J. Stat. Phys.*, 52(1-2), 479–487.
- **Amid, E., et al.** (2019). Robust Bi-Tempered Logistic Loss. *NeurIPS 2019*.
- **Iyer, N., et al.** (2023). Detecting Parkinson's Disease from Speech. *Benchmark AUC = 0.97*.
- **Blondel, M., et al.** (2019). From Softmax to Sparsemax: A Flexible Model of Attention and Probabilistic Outputs. *ICML 2016*.
- **Klatt, D. H.** (1980). Software for a cascade/parallel formant synthesizer. *JASA*, 67(3), 971–995.

---

## Créditos

- **Autor:** Prof. Dr. Bruno Duarte Gomes
- **Organização:** LabNEOC - Lab de Computational Neuroscience, UFPA
- **Licença:** MIT License
