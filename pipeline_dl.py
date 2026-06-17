"""
pipeline_dl.py — Módulo 6: Pipeline Deep Learning para Classificação DP vs HC
==============================================================================
Versão 1.0

Depende de: tsallis_core.py, tfs_ab.py, tfs_cd.py, tfs_e_vae.py, pipeline_ml.py

Implementa o Bloco G do framework TFS:
    Redes neurais profundas com perda e atenção Tsallis

Componentes:
    TFSDataset          — Dataset PyTorch para features TFS
    TFSAttentionBlock   — Bloco de auto-atenção com q-softmax Tsallis
    TFSMHA              — Multi-Head Attention com q-softmax
    TFSTransformer      — Transformer compacto (features tabulares)
    TFS1DCNN            — CNN 1D sobre o TES (Bloco B ordenado por frequência)
    TFSHybridNet        — Rede híbrida: CNN(B) + MLP(A+C+D) + Transformer(todos)
    TFSDLTrainer        — Loop de treino com Tsallis Loss + avaliação completa
    run_experiment      — Função principal: extrai features + treina + avalia
"""

import numpy as np
import warnings
import os
import json
from copy import deepcopy

# ── Sklearn ──────────────────────────────────────────────────────
from sklearn.metrics          import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection  import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing    import StandardScaler
from sklearn.impute           import SimpleImputer

# ── PyTorch ──────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    _TORCH = True
except ImportError:
    _TORCH = False
    warnings.warn("[pipeline_dl] PyTorch nao encontrado. Somente modo sklearn disponivel.")

from tsallis_core import q_softmax as _q_softmax_np
from pipeline_ml  import (TFSFeatureBuilder, TFSPreprocessor, TFSEvaluator,
                           ALL_FEATURES, FEATURE_GROUPS)


# ─────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────

N_FEAT_A   = len(FEATURE_GROUPS['A'])   # 13
N_FEAT_B   = len(FEATURE_GROUPS['B'])   # 24
N_FEAT_C   = len(FEATURE_GROUPS['C'])   # 8
N_FEAT_D   = len(FEATURE_GROUPS['D'])   # 12
N_FEAT_ALL = len(ALL_FEATURES)          # 57

# Índices de cada bloco no vetor ALL_FEATURES
IDX_A = slice(0, N_FEAT_A)
IDX_B = slice(N_FEAT_A, N_FEAT_A + N_FEAT_B)
IDX_C = slice(N_FEAT_A + N_FEAT_B, N_FEAT_A + N_FEAT_B + N_FEAT_C)
IDX_D = slice(N_FEAT_A + N_FEAT_B + N_FEAT_C, N_FEAT_ALL)


# ─────────────────────────────────────────────────────────────────
# DATASET PYTORCH
# ─────────────────────────────────────────────────────────────────

if _TORCH:
    class TFSDataset(Dataset):
        """
        Dataset PyTorch para features TFS.

        Separa internamente os blocos A, B, C, D para alimentar
        os diferentes ramos da TFSHybridNet.

        Parâmetros
        ----------
        X      : array (n, 57) de features normalizadas
        y      : array (n,) de rótulos {0, 1}
        augment: se True, aplica augmentation Gaussiana leve (treino)
        noise_std: desvio do ruído de augmentation
        """
        def __init__(self, X: np.ndarray, y: np.ndarray,
                     augment: bool = False, noise_std: float = 0.02):
            self.X       = torch.tensor(X, dtype=torch.float32)
            self.y       = torch.tensor(y, dtype=torch.long)
            self.augment = augment
            self.noise_std = noise_std

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            x = self.X[idx]
            if self.augment:
                x = x + torch.randn_like(x) * self.noise_std
            return {
                'x_all': x,
                'x_A':   x[IDX_A],
                'x_B':   x[IDX_B],
                'x_C':   x[IDX_C],
                'x_D':   x[IDX_D],
            }, self.y[idx]


# ─────────────────────────────────────────────────────────────────
# BLOCO DE ATENÇÃO TSALLIS (q-softmax)
# ─────────────────────────────────────────────────────────────────

if _TORCH:
    class TsallisAttention(nn.Module):
        """
        Mecanismo de auto-atenção escalar com q-softmax Tsallis.

        Substitui o softmax padrão do Transformer por q-softmax,
        que para q > 1 produz pesos de atenção ESPARSOS — o modelo
        foca em um subconjunto de features, descartando as irrelevantes
        com peso exatamente zero.

        Para features TFS de DP vs HC, onde apenas ~15-20 das 57
        features são genuinamente discriminativas (Bloco C domina),
        a atenção esparsa é fisicamente motivada.

        Parâmetros
        ----------
        d_model : dimensão do embedding
        n_heads : número de cabeças de atenção
        q_attn  : parâmetro Tsallis para q-softmax (q=1 = softmax padrão)
        dropout : taxa de dropout
        """
        def __init__(self, d_model: int, n_heads: int = 4,
                     q_attn: float = 1.3, dropout: float = 0.1):
            super().__init__()
            assert d_model % n_heads == 0, "d_model deve ser divisivel por n_heads"
            self.d_model  = d_model
            self.n_heads  = n_heads
            self.d_k      = d_model // n_heads
            self.q_attn   = q_attn

            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model)
            self.drop = nn.Dropout(dropout)

            # Inicialização Xavier para estabilidade do gradiente
            for w in [self.W_q, self.W_k, self.W_v, self.W_o]:
                nn.init.xavier_uniform_(w.weight)

        def _q_softmax_torch(self, scores: torch.Tensor) -> torch.Tensor:
            """
            q-softmax aplicado aos scores de atenção.
            Para q=1: equivale a F.softmax.
            Para q>1: pesos esparsos (zeros exatos para scores baixos).
            """
            if abs(self.q_attn - 1.0) < 1e-6:
                return F.softmax(scores, dim=-1)

            # base_i = 1 + (1-q)*(scores_i - scores_max)
            s_max = scores.max(dim=-1, keepdim=True).values
            base  = 1.0 + (1.0 - self.q_attn) * (scores - s_max)
            base  = torch.clamp(base, min=0.0)

            exp   = 1.0 / (1.0 - self.q_attn)   # negativo para q > 1
            with torch.no_grad():
                mask = (base > 0).float()

            # base^exp: onde base=0 -> 0
            raw = torch.where(base > 0,
                              base.pow(exp),
                              torch.zeros_like(base))
            total = raw.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            return raw / total

        def forward(self, x: torch.Tensor,
                    mask: torch.Tensor = None) -> torch.Tensor:
            """
            x: (batch, seq_len, d_model)
            Retorna: (batch, seq_len, d_model)
            """
            B, S, D = x.shape

            # Projeções Q, K, V
            Q = self.W_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

            # Scores: (B, heads, S, S)
            scale  = self.d_k ** 0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attn   = self._q_softmax_torch(scores)
            attn   = self.drop(attn)

            # Contexto: (B, heads, S, d_k) -> (B, S, D)
            ctx = torch.matmul(attn, V)
            ctx = ctx.transpose(1, 2).contiguous().view(B, S, D)
            return self.W_o(ctx)


    # ─────────────────────────────────────────────────────────────
    # TFS TRANSFORMER (features tabulares)
    # ─────────────────────────────────────────────────────────────

    class TFSTransformerBlock(nn.Module):
        """
        Bloco Transformer compacto para features tabulares.
        Pre-LN (LayerNorm antes da atenção) — mais estável para treino curto.
        """
        def __init__(self, d_model: int, n_heads: int = 4,
                     q_attn: float = 1.3, dropout: float = 0.1,
                     ff_mult: int = 2):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.attn  = TsallisAttention(d_model, n_heads, q_attn, dropout)
            d_ff = d_model * ff_mult
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            # Pre-LN residual
            x = x + self.attn(self.norm1(x))
            x = x + self.ff(self.norm2(x))
            return x


    class TFSTransformer(nn.Module):
        """
        Transformer compacto para classificação de features TFS tabulares.

        Trata cada feature como um "token" no espaço de embedding.
        A q-atenção esparsa permite ao modelo selecionar automaticamente
        quais features (tokens) são relevantes para a decisão DP vs HC.

        Arquitetura:
            feature_embed → n_layers × TFSTransformerBlock → CLS token → Linear(2)

        Parâmetros
        ----------
        input_dim  : número de features (ex: 57)
        d_model    : dimensão do embedding por feature
        n_heads    : cabeças de atenção (d_model deve ser divisível por n_heads)
        n_layers   : número de blocos Transformer
        q_attn     : q do q-softmax (1.3 recomendado)
        dropout    : taxa de dropout
        """
        def __init__(self, input_dim: int = N_FEAT_ALL,
                     d_model: int = 32, n_heads: int = 4,
                     n_layers: int = 2, q_attn: float = 1.3,
                     dropout: float = 0.15):
            super().__init__()
            self.d_model   = d_model

            # Embedding: cada feature -> vetor d_model
            self.feat_embed = nn.Linear(1, d_model)
            # Token CLS aprendível
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

            self.blocks = nn.ModuleList([
                TFSTransformerBlock(d_model, n_heads, q_attn, dropout)
                for _ in range(n_layers)
            ])
            self.norm  = nn.LayerNorm(d_model)
            self.head  = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 2)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            x: (batch, input_dim)
            Retorna logits: (batch, 2)
            """
            B, D = x.shape
            # (B, D, 1) -> (B, D, d_model)
            tokens = self.feat_embed(x.unsqueeze(-1))
            # Adicionar CLS token: (B, D+1, d_model)
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

            for block in self.blocks:
                tokens = block(tokens)

            # Usar apenas o CLS token para classificação
            cls_out = self.norm(tokens[:, 0])
            return self.head(cls_out)


    # ─────────────────────────────────────────────────────────────
    # TFS 1D-CNN (sobre Bloco B: TES ordenado por frequência)
    # ─────────────────────────────────────────────────────────────

    class TFS1DCNN(nn.Module):
        """
        CNN 1D sobre o Espectro Entrópico de Tsallis (TES — Bloco B).

        O Bloco B tem estrutura natural de sequência: 8 subbandas
        ordenadas por frequência crescente (80→8000 Hz), cada uma com
        3 valores de q (q=0.7, 1.3, 2.0). Tratado como sinal 1D com
        3 canais e comprimento 8 (subbandas), a CNN captura padrões
        espectrais entrópicos que o MLP não consegue — como o gradiente
        de complexidade de F0 → ruído de alta frequência.

        Arquitetura:
            (B, 3, 8) → Conv1D → BN → ReLU → Conv1D → BN → ReLU →
            AdaptiveAvgPool → Linear(2)

        input: B_features reshaped como (batch, n_q=3, n_bandas=8)
        """
        def __init__(self, n_bandas: int = 8, n_q: int = 3,
                     dropout: float = 0.15):
            super().__init__()
            self.n_bandas = n_bandas
            self.n_q      = n_q

            self.conv1 = nn.Sequential(
                nn.Conv1d(n_q, 16, kernel_size=3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            )
            self.pool  = nn.AdaptiveAvgPool1d(1)
            self.head  = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 2)
            )

        def forward(self, x_B: torch.Tensor) -> torch.Tensor:
            """
            x_B: (batch, 24)  — features do Bloco B
            Retorna logits: (batch, 2)
            """
            B = x_B.shape[0]
            # Reshape: (batch, 24) -> (batch, n_q=3, n_bandas=8)
            x = x_B.view(B, self.n_q, self.n_bandas)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool(x)
            return self.head(x)


    # ─────────────────────────────────────────────────────────────
    # TFS HYBRID NET — Rede Híbrida Principal
    # ─────────────────────────────────────────────────────────────

    class TFSHybridNet(nn.Module):
        """
        Rede híbrida que processa os 4 blocos TFS com arquiteturas
        especializadas e funde os resultados.

        Ramos:
        ┌──────────────────────────────────────────────────────────┐
        │  Bloco A (13 feat) → MLP(32) → embedding_A (16)         │
        │  Bloco B (24 feat) → CNN1D  → embedding_B (16)          │
        │  Bloco C (8 feat)  → MLP(32) → embedding_C (16)         │
        │  Bloco D (12 feat) → MLP(32) → embedding_D (16)         │
        │  ALL (57 feat)     → Transformer(q-attn) → emb_T (16)   │
        └──────────────────────────────────────────────────────────┘
                   ↓ concatenação (80 dims)
              LayerNorm → Linear(40) → GELU → Dropout
                   → Linear(2) → logits

        A fusão tardia (late fusion) permite que cada ramo aprenda
        representações especializadas antes da decisão final.
        Pesos de fusão são aprendíveis (não fixos).

        Parâmetros
        ----------
        input_dim  : total de features (57)
        d_model    : dimensão do Transformer
        n_heads    : cabeças de atenção
        q_attn     : q do q-softmax para atenção
        dropout    : taxa de dropout
        """
        def __init__(self, input_dim: int = N_FEAT_ALL,
                     d_model: int = 32, n_heads: int = 4,
                     q_attn: float = 1.3, dropout: float = 0.2):
            super().__init__()

            emb_dim = 16  # dimensão de cada embedding de ramo

            # Ramo A: MLP
            self.branch_A = nn.Sequential(
                nn.Linear(N_FEAT_A, 32), nn.LayerNorm(32), nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(32, emb_dim), nn.ELU()
            )

            # Ramo B: CNN 1D
            self.branch_B = TFS1DCNN(n_bandas=8, n_q=3, dropout=dropout)
            # A CNN retorna logits(2) — precisamos do embedding, não dos logits
            # Substituímos o head da CNN por Linear(32, emb_dim)
            self.branch_B.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32, 16),
                nn.ELU(),
            )

            # Ramo C: MLP (features de F0 — mais importantes)
            self.branch_C = nn.Sequential(
                nn.Linear(N_FEAT_C, 32), nn.LayerNorm(32), nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(32, emb_dim), nn.ELU()
            )

            # Ramo D: MLP (Cronoentropograma)
            self.branch_D = nn.Sequential(
                nn.Linear(N_FEAT_D, 32), nn.LayerNorm(32), nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(32, emb_dim), nn.ELU()
            )

            # Ramo T: Transformer sobre todos
            self.branch_T = TFSTransformer(
                input_dim=input_dim,
                d_model=d_model, n_heads=n_heads,
                n_layers=2, q_attn=q_attn, dropout=dropout
            )
            # Reduzir saída do Transformer para emb_dim
            self.proj_T = nn.Linear(2, emb_dim)  # logits(2) -> emb_dim

            # Fusão: 5 ramos × emb_dim = 80
            fusion_dim = 5 * emb_dim
            self.fusion = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, 40),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(40, 2)
            )

        def forward(self, batch: dict) -> torch.Tensor:
            """
            batch: dict com chaves 'x_all', 'x_A', 'x_B', 'x_C', 'x_D'
            Retorna logits: (batch_size, 2)
            """
            emb_A = self.branch_A(batch['x_A'])
            emb_B = self.branch_B(batch['x_B'])
            emb_C = self.branch_C(batch['x_C'])
            emb_D = self.branch_D(batch['x_D'])
            emb_T = self.proj_T(self.branch_T(batch['x_all']))

            fused = torch.cat([emb_A, emb_B, emb_C, emb_D, emb_T], dim=1)
            return self.fusion(fused)


# ─────────────────────────────────────────────────────────────────
# TRAINER DL
# ─────────────────────────────────────────────────────────────────

if _TORCH:
    class TFSDLTrainer:
        """
        Loop de treino para modelos DL (TFSHybridNet, TFSTransformer).

        Funcionalidades:
        - Loss: CrossEntropy + Tsallis Loss (beta ponderado)
        - Otimizador: AdamW com weight decay
        - Scheduler: OneCycleLR
        - Early stopping baseado em AUC de validação
        - Avaliação completa (AUC, F1, sens, spec)

        Parâmetros
        ----------
        model      : TFSHybridNet ou TFSTransformer
        q_loss     : parâmetro q da Tsallis Loss
        lr         : learning rate máximo (OneCycleLR)
        weight_decay: L2 regularização
        beta_tsallis: peso da Tsallis Loss (0 = só CE, 1 = só Tsallis)
        device     : 'auto' | 'cpu' | 'cuda'
        """
        def __init__(self, model, q_loss: float = 1.3,
                     lr: float = 3e-4, weight_decay: float = 1e-3,
                     beta_tsallis: float = 0.3, device: str = 'auto'):
            if device == 'auto':
                device = ('cuda' if torch.cuda.is_available()
                          else 'mps' if (hasattr(torch.backends, 'mps')
                               and torch.backends.mps.is_available())
                          else 'cpu')
            self.device       = torch.device(device)
            self.model        = model.to(self.device)
            self.q_loss       = q_loss
            self.beta_tsallis = beta_tsallis
            self.lr           = lr

            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            self.history = {'train_loss': [], 'val_auc': []}
            self._best_state = None
            self._best_auc   = 0.0

        def _tsallis_ce_loss(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
        ) -> torch.Tensor:
            """
            Loss híbrida: (1-beta)*CrossEntropy + beta*TsallisLoss

            TsallisLoss:
                L_q(y, p) = (1/q) * [1 - sum_k p_k^q * y_k]
            Para q=1: equivale à CrossEntropy.
            Para q>1: mais robusta a rótulos ruidosos (outlier-resistant).
            """
            ce_loss = F.cross_entropy(logits, targets)

            if abs(self.beta_tsallis) < 1e-6:
                return ce_loss

            # Tsallis classificação multi-classe
            probs  = F.softmax(logits, dim=-1)
            n      = targets.shape[0]
            idx    = torch.arange(n, device=self.device)
            p_true = probs[idx, targets]   # p da classe correta

            if abs(self.q_loss - 1.0) < 1e-6:
                t_loss = -torch.log(p_true + 1e-8).mean()
            else:
                t_loss = ((1.0 - p_true.pow(self.q_loss - 1.0))
                          / (self.q_loss - 1.0)).mean()

            return (1.0 - self.beta_tsallis) * ce_loss + self.beta_tsallis * t_loss

        def _prepare_batch(self, batch_dict: dict, targets: torch.Tensor):
            """Move batch para device."""
            bd = {k: v.to(self.device) for k, v in batch_dict.items()}
            return bd, targets.to(self.device)

        def _forward(self, batch_dict: dict) -> torch.Tensor:
            """Forward pass: suporta TFSHybridNet (dict) e TFSTransformer (tensor)."""
            if isinstance(self.model, TFSTransformer):
                return self.model(batch_dict['x_all'])
            elif isinstance(self.model, TFSHybridNet):
                return self.model(batch_dict)
            else:
                return self.model(batch_dict['x_all'])

        def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
            n_epochs: int = 100,
            batch_size: int = 16,
            augment: bool = True,
            verbose: bool = True,
            log_every: int = 10,
            patience: int = 20
        ) -> 'TFSDLTrainer':
            """
            Treina o modelo DL.

            Se X_val/y_val forem fornecidos, faz early stopping por AUC.
            Caso contrário, usa apenas loss de treino.
            """
            train_ds = TFSDataset(X_train, y_train, augment=augment)
            train_dl = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, drop_last=False)

            # OneCycleLR: LR schedule agressivo, excelente para datasets pequenos
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.lr,
                steps_per_epoch=len(train_dl),
                epochs=n_epochs,
                pct_start=0.3,
                anneal_strategy='cos'
            )

            no_improve = 0

            for epoch in range(1, n_epochs + 1):
                self.model.train()
                ep_loss = 0.0
                n_batch = 0

                for batch_dict, targets in train_dl:
                    bd, tgt = self._prepare_batch(batch_dict, targets)
                    self.optimizer.zero_grad()
                    logits = self._forward(bd)
                    loss   = self._tsallis_ce_loss(logits, tgt)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()
                    scheduler.step()
                    ep_loss += loss.item()
                    n_batch += 1

                avg_loss = ep_loss / max(n_batch, 1)
                self.history['train_loss'].append(avg_loss)

                # Avaliação em validação
                val_auc = np.nan
                if X_val is not None and y_val is not None:
                    val_auc = self._eval_auc(X_val, y_val)
                    self.history['val_auc'].append(val_auc)

                    if val_auc > self._best_auc + 1e-4:
                        self._best_auc   = val_auc
                        self._best_state = deepcopy(self.model.state_dict())
                        no_improve = 0
                    else:
                        no_improve += 1

                if verbose and (epoch % log_every == 0 or epoch == 1):
                    auc_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else "N/A"
                    print(f"  Epoch {epoch:4d}/{n_epochs} | "
                          f"Loss={avg_loss:.4f} | ValAUC={auc_str}")

                if no_improve >= patience and X_val is not None:
                    if verbose:
                        print(f"  [Early stop] Epoch {epoch}. "
                              f"Melhor AUC={self._best_auc:.4f}")
                    break

            # Restaurar melhor modelo
            if self._best_state is not None:
                self.model.load_state_dict(self._best_state)

            return self

        def _eval_auc(self, X: np.ndarray, y: np.ndarray) -> float:
            """AUC de validação rápida."""
            probs = self.predict_proba(X)[:, 1]
            try:
                return float(roc_auc_score(y, probs))
            except Exception:
                return 0.5

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Probabilidades preditas (n, 2)."""
            ds = TFSDataset(X, np.zeros(len(X), dtype=int), augment=False)
            dl = DataLoader(ds, batch_size=64, shuffle=False)
            probs_list = []
            self.model.eval()
            with torch.no_grad():
                for batch_dict, _ in dl:
                    bd = {k: v.to(self.device) for k, v in batch_dict.items()}
                    logits = self._forward(bd)
                    probs_list.append(F.softmax(logits, dim=-1).cpu().numpy())
            return np.vstack(probs_list)

        def predict(self, X: np.ndarray) -> np.ndarray:
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
            """Métricas completas em dados de teste."""
            y_prob = self.predict_proba(X)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            cm = confusion_matrix(y, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            sens = tp / (tp + fn + 1e-10)
            spec = tn / (tn + fp + 1e-10)
            auc  = float(roc_auc_score(y, y_prob)) if len(np.unique(y)) > 1 else np.nan
            return {
                'auc_roc':      auc,
                'sensitivity':  float(sens),
                'specificity':  float(spec),
                'f1':           float(f1_score(y, y_pred, zero_division=0)),
                'balanced_acc': float((sens + spec) / 2.0),
                'confusion_matrix': cm.tolist(),
                'vs_benchmark': {
                    'iyer_auc': 0.97,
                    'tfs_auc':  round(auc, 4) if not np.isnan(auc) else None,
                    'delta':    round(auc - 0.97, 4) if not np.isnan(auc) else None,
                    'status':   ('SUPEROU' if (not np.isnan(auc) and auc >= 0.97)
                                 else 'ABAIXO')
                }
            }


# ─────────────────────────────────────────────────────────────────
# VALIDAÇÃO CRUZADA DL
# ─────────────────────────────────────────────────────────────────

def cross_validate_dl(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'hybrid',
    q_attn: float = 1.3,
    q_loss: float = 1.3,
    n_splits: int = 5,
    n_repeats: int = 3,
    n_epochs: int = 80,
    batch_size: int = 16,
    verbose: bool = True
) -> dict:
    """
    Validação cruzada estratificada para modelos DL.

    Parâmetros
    ----------
    X          : array pré-processado (n, n_feat)
    y          : rótulos binários
    model_type : 'hybrid' (TFSHybridNet) | 'transformer' (TFSTransformer)
    q_attn     : q para atenção esparsa
    q_loss     : q para Tsallis Loss
    n_splits   : folds por repetição
    n_repeats  : repetições de CV (default 3)
    n_epochs   : épocas por fold
    batch_size : batch
    verbose    : logs de treino

    Retorna
    -------
    dict com métricas médias e std sobre todos os folds
    """
    if not _TORCH:
        raise ImportError("PyTorch necessário para cross_validate_dl.")

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )

    auc_scores  = []
    f1_scores   = []
    sens_scores = []
    spec_scores = []

    total_folds = n_splits * n_repeats

    for fold_i, (tr_idx, te_idx) in enumerate(rskf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Evita que o imputer delete colunas 100% vazias e desalinhe a rede PyTorch
        nan_cols = np.isnan(X_tr).all(axis=0)
        X_tr[:, nan_cols] = 0.0
        X_te[:, nan_cols] = 0.0

        # Pré-processar (normalização dentro do fold para evitar data leakage)
        imp = SimpleImputer(strategy='median')
        sc  = StandardScaler()
        X_tr_pp = sc.fit_transform(imp.fit_transform(X_tr))
        X_te_pp = sc.transform(imp.transform(X_te))
        X_tr_pp = np.nan_to_num(X_tr_pp, nan=0.0)
        X_te_pp = np.nan_to_num(X_te_pp, nan=0.0)

        # Construir modelo
        n_feat = X_tr_pp.shape[1]
        if model_type == 'hybrid':
            # Ajustar input_dim dinamicamente
            model = TFSHybridNet(input_dim=n_feat, q_attn=q_attn)
        else:
            model = TFSTransformer(input_dim=n_feat, q_attn=q_attn)

        # Split treino/val interno (15% para early stopping)
        n_val = max(2, int(0.15 * len(X_tr_pp)))
        
        # Embaralhar para garantir que a validação tenha ambas as classes
        idx_shuf = np.random.permutation(len(X_tr_pp))
        X_tr_shuf = X_tr_pp[idx_shuf]
        y_tr_shuf = y_tr[idx_shuf]
        
        X_tr2, X_val = X_tr_shuf[:-n_val], X_tr_shuf[-n_val:]
        y_tr2, y_val = y_tr_shuf[:-n_val], y_tr_shuf[-n_val:]

        trainer = TFSDLTrainer(
            model, q_loss=q_loss, lr=3e-4, beta_tsallis=0.3
        )
        fold_verbose = verbose and (fold_i == 0)
        trainer.fit(
            X_tr2, y_tr2, X_val, y_val,
            n_epochs=n_epochs, batch_size=batch_size,
            verbose=fold_verbose, log_every=20, patience=15
        )

        metrics = trainer.evaluate(X_te_pp, y_te)
        auc_scores.append(metrics.get('auc_roc', np.nan))
        f1_scores.append(metrics['f1'])
        sens_scores.append(metrics['sensitivity'])
        spec_scores.append(metrics['specificity'])

        if verbose:
            auc_v = metrics.get('auc_roc', np.nan)
            print(f"  Fold {fold_i+1:2d}/{total_folds} | "
                  f"AUC={auc_v:.4f} | F1={metrics['f1']:.4f} | "
                  f"Sens={metrics['sensitivity']:.4f} | "
                  f"Spec={metrics['specificity']:.4f}")

    def _stats(arr):
        a = np.array(arr)
        return {'mean': float(np.nanmean(a)),
                'std':  float(np.nanstd(a)),
                'min':  float(np.nanmin(a)),
                'max':  float(np.nanmax(a))}

    results = {
        'model_type':  model_type,
        'q_attn':      q_attn,
        'q_loss':      q_loss,
        'auc_roc':     _stats(auc_scores),
        'f1':          _stats(f1_scores),
        'sensitivity': _stats(sens_scores),
        'specificity': _stats(spec_scores),
        'n_folds':     total_folds,
        'vs_benchmark': {
            'iyer_auc': 0.97,
            'tfs_auc':  round(float(np.nanmean(auc_scores)), 4),
            'delta':    round(float(np.nanmean(auc_scores)) - 0.97, 4),
            'status':   ('SUPEROU' if np.nanmean(auc_scores) >= 0.97
                         else 'ABAIXO')
        }
    }
    return results


# ─────────────────────────────────────────────────────────────────
# FUNÇÃO PRINCIPAL — run_experiment
# ─────────────────────────────────────────────────────────────────

def run_experiment(
    dp_files: list,
    hc_files: list,
    output_dir: str = "tfs_results",
    run_ml: bool = True,
    run_dl: bool = True,
    n_epochs_dl: int = 80,
    q_attn: float = 1.3,
    q_loss: float = 1.3,
    verbose: bool = True
) -> dict:
    """
    Pipeline completo: extração de features → ML → DL → resultados.

    1. Extrai features TFS (Blocos A+B+C+D) para todos os arquivos
    2. Treina e avalia classificadores ML (pipeline_ml.py)
    3. Treina e avalia TFSHybridNet e TFSTransformer
    4. Salva resultados em JSON

    Parâmetros
    ----------
    dp_files   : lista de caminhos para arquivos de áudio DP
    hc_files   : lista de caminhos para arquivos de áudio HC
    output_dir : diretório para salvar resultados
    run_ml     : se True, executa classificadores ML
    run_dl     : se True, executa modelos DL (requer PyTorch)
    n_epochs_dl: épocas de treino DL por fold
    q_attn     : q para atenção do Transformer
    q_loss     : q da Tsallis Loss
    verbose    : logs de progresso

    Retorna
    -------
    dict com todos os resultados de ML e DL
    """
    import time
    from tfs_ab import extract_blocks_AB, batch_extract_AB
    from tfs_cd import extract_blocks_CD, batch_extract_CD
    from pipeline_ml import TFSPipeline, ALL_FEATURES

    os.makedirs(output_dir, exist_ok=True)
    results = {}
    t0 = time.time()

    # ── Extração de features ─────────────────────────────────────
    if verbose:
        print("=" * 65)
        print("  TFS EXPERIMENT — run_experiment()")
        print("=" * 65)
        print(f"\n[STEP 1] Extraindo features TFS...")
        print(f"  DP: {len(dp_files)} arquivos | HC: {len(hc_files)} arquivos")

    all_files = dp_files + hc_files
    labels    = [1]*len(dp_files) + [0]*len(hc_files)

    feat_dicts = []
    for i, fp in enumerate(all_files):
        lbl_str = "DP" if labels[i] == 1 else "HC"
        if verbose:
            print(f"  [{i+1}/{len(all_files)}] {os.path.basename(fp)} ({lbl_str})",
                  end=" ")
        try:
            feats_AB = extract_blocks_AB(filepath=fp, verbose=False)
            feats_CD = extract_blocks_CD(filepath=fp, verbose=False)
            fd = {**feats_AB, **feats_CD}
            feat_dicts.append(fd)
            if verbose:
                print("OK")
        except Exception as e:
            feat_dicts.append({})
            if verbose:
                print(f"ERRO: {e}")

    # Construir matriz de features
    builder = TFSFeatureBuilder()
    X = builder.from_dicts(feat_dicts)
    y = np.array(labels)

    nan_pct = 100 * np.isnan(X).mean()
    if verbose:
        print(f"\n  Matrix: {X.shape} | NaN: {nan_pct:.1f}%")

    # Salvar features brutas
    feat_path = os.path.join(output_dir, "tfs_features.npz")
    np.savez(feat_path, X=X, y=y, feature_names=np.array(ALL_FEATURES))
    if verbose:
        print(f"  Features salvas em: {feat_path}")

    results['dataset'] = {
        'n_dp': int(len(dp_files)),
        'n_hc': int(len(hc_files)),
        'n_features': int(X.shape[1]),
        'nan_pct': float(nan_pct)
    }

    # ── Pipeline ML ──────────────────────────────────────────────
    if run_ml:
        if verbose:
            print(f"\n[STEP 2] Pipeline ML (CV 5-fold x 3 repeticoes)...")

        ml_results = {}
        for clf_name in ['svm', 'ensemble']:
            if verbose:
                print(f"  Classificador: {clf_name.upper()}")
            pipe = TFSPipeline(
                classifier=clf_name,
                k_features=25,
                q_vae=False,
                selection_method='anova'
            )
            cv_res = pipe.cross_validate(X, y, n_splits=5, n_repeats=3)
            ml_results[clf_name] = cv_res
            if verbose:
                auc_m = cv_res['auc_roc']['mean']
                auc_s = cv_res['auc_roc']['std']
                print(f"    AUC={auc_m:.4f}+/-{auc_s:.4f} | "
                      f"Benchmark: {cv_res['vs_benchmark']['status']}")

        results['ml'] = ml_results

    # ── Pipeline DL ──────────────────────────────────────────────
    if run_dl and _TORCH:
        if verbose:
            print(f"\n[STEP 3] Pipeline DL (CV 5-fold)...")

        dl_results = {}
        for mtype in ['transformer', 'hybrid']:
            if verbose:
                print(f"  Modelo: {mtype.upper()}")
            try:
                cv_res = cross_validate_dl(
                    X, y,
                    model_type=mtype,
                    q_attn=q_attn,
                    q_loss=q_loss,
                    n_splits=5,
                    n_repeats=1,   # 1 repetição por padrão (mais rápido)
                    n_epochs=n_epochs_dl,
                    batch_size=16,
                    verbose=verbose
                )
                dl_results[mtype] = cv_res
                if verbose:
                    auc_m = cv_res['auc_roc']['mean']
                    auc_s = cv_res['auc_roc']['std']
                    print(f"    AUC={auc_m:.4f}+/-{auc_s:.4f} | "
                          f"Benchmark: {cv_res['vs_benchmark']['status']}")
            except Exception as e:
                warnings.warn(f"[run_experiment] DL {mtype} falhou: {e}")
                dl_results[mtype] = {'error': str(e)}

        results['dl'] = dl_results
    elif run_dl and not _TORCH:
        warnings.warn("[run_experiment] PyTorch nao disponivel. DL pulado.")

    # ── Salvar resultados ─────────────────────────────────────────
    t_total = time.time() - t0
    results['runtime_s'] = round(t_total, 1)

    res_path = os.path.join(output_dir, "tfs_results.json")
    with open(res_path, 'w') as f:
        # Converter arrays para listas para serialização JSON
        def _serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64, np.float16)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            raise TypeError(f"Tipo nao serializavel: {type(obj)}")
        json.dump(results, f, indent=2, default=_serialize)

    if verbose:
        print(f"\n  Resultados salvos em: {res_path}")
        print(f"  Tempo total: {t_total:.1f}s")
        print("=" * 65)

    return results


# ─────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  pipeline_dl.py v1.0 -- Self-Test")
    print("=" * 65)

    rng = np.random.default_rng(42)
    n_hc, n_dp = 50, 50
    n_feat = N_FEAT_ALL  # 57

    # Dataset sintético com diferenças realistas baseadas nos módulos M2/M3
    X_hc = rng.normal(0, 1, (n_hc, n_feat))
    X_dp = rng.normal(0, 1, (n_dp, n_feat))
    X_dp[:, IDX_C] += 2.5   # Bloco C: S_q F0 muito maior em DP
    X_dp[:, IDX_A] -= 0.8   # Bloco A: decay_ratio menor em DP
    X_dp[:, IDX_D] += 0.6   # Bloco D: std e delta maiores em DP

    X = np.vstack([X_hc, X_dp])
    X[rng.random(X.shape) < 0.08] = np.nan  # 8% NaN
    y = np.array([0]*n_hc + [1]*n_dp)

    # Pré-processar
    imp = SimpleImputer(strategy='median')
    sc  = StandardScaler()
    X_pp = sc.fit_transform(imp.fit_transform(X))
    X_pp = np.nan_to_num(X_pp, nan=0.0)

    print(f"\nDataset: {X_pp.shape} | HC={n_hc} | DP={n_dp}")

    if not _TORCH:
        print("\nPyTorch nao disponivel. Self-test DL pulado.")
        print("Para usar os modelos DL, instale: pip install torch")
    else:
        print(f"\n[1] TFSDataset")
        ds = TFSDataset(X_pp, y, augment=True)
        batch, label = ds[0]
        print(f"  x_all: {batch['x_all'].shape}")
        print(f"  x_A:   {batch['x_A'].shape} (esperado: {N_FEAT_A})")
        print(f"  x_B:   {batch['x_B'].shape} (esperado: {N_FEAT_B})")
        print(f"  x_C:   {batch['x_C'].shape} (esperado: {N_FEAT_C})")
        print(f"  x_D:   {batch['x_D'].shape} (esperado: {N_FEAT_D})")

        print(f"\n[2] TsallisAttention (q-softmax)")
        attn = TsallisAttention(d_model=32, n_heads=4, q_attn=1.3)
        x_t  = torch.randn(4, 10, 32)
        out  = attn(x_t)
        print(f"  Input: {x_t.shape} -> Output: {out.shape}  OK")

        print(f"\n[3] TFS1DCNN")
        cnn  = TFS1DCNN(n_bandas=8, n_q=3)
        xb   = torch.randn(4, 24)
        out  = cnn(xb)
        print(f"  Input: {xb.shape} -> Output: {out.shape}  OK")

        print(f"\n[4] TFSTransformer")
        trans = TFSTransformer(input_dim=n_feat, d_model=32, n_heads=4)
        xt    = torch.randn(4, n_feat)
        out   = trans(xt)
        n_par = sum(p.numel() for p in trans.parameters())
        print(f"  Input: {xt.shape} -> Logits: {out.shape} | Params: {n_par:,}  OK")

        print(f"\n[5] TFSHybridNet")
        hybrid = TFSHybridNet(input_dim=n_feat, d_model=32, n_heads=4, q_attn=1.3)
        batch_t = {
            'x_all': torch.randn(4, n_feat),
            'x_A':   torch.randn(4, N_FEAT_A),
            'x_B':   torch.randn(4, N_FEAT_B),
            'x_C':   torch.randn(4, N_FEAT_C),
            'x_D':   torch.randn(4, N_FEAT_D),
        }
        out = hybrid(batch_t)
        n_par_h = sum(p.numel() for p in hybrid.parameters())
        print(f"  Output: {out.shape} | Params: {n_par_h:,}  OK")

        print(f"\n[6] TFSDLTrainer — Transformer (15 epocas, subset)")
        tr_idx = list(range(0, 40)) + list(range(50, 90))
        te_idx = list(range(40, 50)) + list(range(90, 100))
        X_tr, X_te = X_pp[tr_idx], X_pp[te_idx]
        y_tr, y_te = y[tr_idx],   y[te_idx]

        model_t   = TFSTransformer(input_dim=n_feat, d_model=32, n_heads=4, q_attn=1.3)
        trainer_t = TFSDLTrainer(model_t, q_loss=1.3, lr=3e-4, beta_tsallis=0.3)
        trainer_t.fit(X_tr, y_tr, X_te, y_te,
                      n_epochs=15, batch_size=16, verbose=True,
                      log_every=5, patience=10)
        m_t = trainer_t.evaluate(X_te, y_te)
        print(f"  Transformer -> AUC={m_t['auc_roc']:.4f} | "
              f"F1={m_t['f1']:.4f} | "
              f"vs Benchmark: {m_t['vs_benchmark']['status']}")

        print(f"\n[7] TFSDLTrainer — HybridNet (15 epocas, subset)")
        model_h   = TFSHybridNet(input_dim=n_feat, d_model=32, n_heads=4, q_attn=1.3)
        trainer_h = TFSDLTrainer(model_h, q_loss=1.3, lr=3e-4, beta_tsallis=0.3)
        trainer_h.fit(X_tr, y_tr, X_te, y_te,
                      n_epochs=15, batch_size=16, verbose=True,
                      log_every=5, patience=10)
        m_h = trainer_h.evaluate(X_te, y_te)
        print(f"  HybridNet   -> AUC={m_h['auc_roc']:.4f} | "
              f"F1={m_h['f1']:.4f} | "
              f"vs Benchmark: {m_h['vs_benchmark']['status']}")

    print("\nModulo 6 pronto.")
    print("=" * 65)
