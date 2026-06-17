"""
tfs_e_vae.py — Módulo 4: q-VAE para Aprendizado de Representação
=================================================================
Versão 1.0

Depende de: tsallis_core.py

Implementa o Bloco E do framework TFS:
    Variational Autoencoder com divergência de Tsallis (q-VAE)

Componentes:
    TsallisVAEEncoder   — Encoder com saída (mu, log_var)
    TsallisVAEDecoder   — Decoder com reconstrução q-gaussiana
    TsallisVAE          — Modelo completo (encoder + decoder + loss)
    QVAETrainer         — Loop de treino com elbo Tsallis

Loss:
    ELBO_q = -E[log p_q(x|z)] + beta * D_q(q(z|x) || p(z))
    onde D_q é a q-divergência de Tsallis (M1 / tsallis_core.py)

Uso rápido:
    from tfs_e_vae import TsallisVAE, QVAETrainer
    model   = TsallisVAE(input_dim=55, latent_dim=8, q=1.3)
    trainer = QVAETrainer(model, lr=1e-3, beta=1.0)
    trainer.fit(X_train, n_epochs=50)
    Z = trainer.encode(X_test)   # representações latentes (n, 8)
"""

import numpy as np
import warnings

# ── PyTorch (opcional — framework suporta fallback NumPy-only) ───
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    warnings.warn(
        "[tfs_e_vae] PyTorch nao encontrado. "
        "Apenas o modo NumPy (q-VAE simplificado linear) estara disponivel.",
        ImportWarning
    )

from tsallis_core import q_divergence, tsallis_loss_numpy


# ─────────────────────────────────────────────────────────────────
# UTILITÁRIOS MATEMÁTICOS (NumPy puro, independentes do PyTorch)
# ─────────────────────────────────────────────────────────────────

def q_log(x: np.ndarray, q: float) -> np.ndarray:
    """
    q-logaritmo de Tsallis:
        ln_q(x) = (x^(1-q) - 1) / (1-q)   para q != 1
        ln_1(x) = ln(x)                      para q -> 1

    Propriedades:
    - ln_q é côncavo e generaliza ln natural
    - Para q>1: crescimento mais lento que ln → penaliza menos outliers
    - Para q<1: diverge em 0 mais rápido → penaliza mais valores pequenos
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 1e-300)
    if abs(q - 1.0) < 1e-6:
        return np.log(x)
    return (x ** (1.0 - q) - 1.0) / (1.0 - q)


def q_exp(x: np.ndarray, q: float) -> np.ndarray:
    """
    q-exponencial de Tsallis (inversa de q_log):
        exp_q(x) = [1 + (1-q)*x]_+^(1/(1-q))   para q != 1
        exp_1(x) = exp(x)                         para q -> 1
    """
    x = np.asarray(x, dtype=np.float64)
    if abs(q - 1.0) < 1e-6:
        return np.exp(x)
    base = 1.0 + (1.0 - q) * x
    with np.errstate(invalid='ignore'):
        result = np.where(base > 0, base ** (1.0 / (1.0 - q)), 0.0)
    return result


def tsallis_kl_gaussian(
    mu: np.ndarray,
    log_var: np.ndarray,
    q: float = 1.3
) -> float:
    """
    q-divergência analítica entre N(mu, sigma^2) e N(0,1)
    usando a aproximação de primeira ordem válida para |q-1| << 1:

        D_q(N(mu,s^2) || N(0,1)) ≈ (1/2) * [s^2 + mu^2 - 1 - ln(s^2)]
                                    + (q-1)/2 * correction_term

    Para q=1: equivale exatamente à KL gaussiana do VAE padrão.
    Para q≠1: inclui termo de correção que penaliza distribuições
    com variância muito diferente de 1 (evita posterior collapse).

    Parâmetros
    ----------
    mu, log_var : arrays de parâmetros variacionais (batch)
    q           : parâmetro de não-extensividade

    Retorna
    -------
    float : D_q média sobre o batch
    """
    mu      = np.asarray(mu, dtype=np.float64)
    log_var = np.asarray(log_var, dtype=np.float64)
    var     = np.exp(np.clip(log_var, -10, 10))
    sigma   = np.sqrt(var + 1e-8)

    # KL padrão (q=1)
    kl_standard = 0.5 * (var + mu ** 2 - 1.0 - log_var)

    if abs(q - 1.0) < 1e-6:
        return float(np.mean(np.sum(kl_standard, axis=-1)))

    # Correção de primeira ordem para q != 1:
    # delta_kl = (q-1)/4 * (2*var^2 + 4*mu^2*var - 2*var + mu^4 - 2*mu^2 + 1)
    # Derivada de segunda ordem da KL gaussiana em q=1
    correction = (q - 1.0) / 4.0 * (
        2.0 * var**2
        + 4.0 * mu**2 * var
        - 2.0 * var
        + mu**4
        - 2.0 * mu**2
        + 1.0
    )
    kl_q = kl_standard + correction
    return float(np.mean(np.sum(kl_q, axis=-1)))


# ─────────────────────────────────────────────────────────────────
# IMPLEMENTAÇÃO PYTORCH (disponível se torch instalado)
# ─────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:

    def _q_log_torch(x: torch.Tensor, q: float) -> torch.Tensor:
        """q-logaritmo implementado em PyTorch."""
        x = torch.clamp(x, min=1e-7)
        if abs(q - 1.0) < 1e-6:
            return torch.log(x)
        return (x.pow(1.0 - q) - 1.0) / (1.0 - q)

    def _tsallis_recon_loss(
        x_recon: torch.Tensor,
        x_orig: torch.Tensor,
        q: float
    ) -> torch.Tensor:
        """
        Perda de reconstrução Tsallis:
            L_recon = -E[ln_q(p(x|z))]
        Aproximada por MSE pesado pelo q-logaritmo do erro:
            L = mean(|x_recon - x_orig|^(2-q) / (2-q))
        Para q=1: equivale a MSE padrão (up to const).
        Para q>1: menos sensível a outliers (erro elevado penalizado menos).
        """
        err = (x_recon - x_orig).pow(2).clamp(min=1e-7)
        if abs(q - 1.0) < 1e-6:
            return err.mean()
        # Generalização de MSE via q-log
        return (err.pow((2.0 - q) / 2.0) / (2.0 - q)).mean()

    def _kl_loss_torch(
        mu: torch.Tensor,
        log_var: torch.Tensor,
        q: float
    ) -> torch.Tensor:
        """KL / q-divergência entre N(mu,s^2) e N(0,1) em PyTorch."""
        var = torch.exp(log_var.clamp(-10, 10))
        kl  = 0.5 * (var + mu.pow(2) - 1.0 - log_var)
        if abs(q - 1.0) < 1e-6:
            return kl.sum(dim=-1).mean()
        correction = (q - 1.0) / 4.0 * (
            2.0 * var.pow(2)
            + 4.0 * mu.pow(2) * var
            - 2.0 * var
            + mu.pow(4)
            - 2.0 * mu.pow(2)
            + 1.0
        )
        return (kl + correction).sum(dim=-1).mean()


    class TsallisVAEEncoder(nn.Module):
        """
        Encoder q-VAE: input_dim → [mu, log_var] (latent_dim cada).

        Arquitetura:
            Linear(input_dim, hidden_dim) → LayerNorm → ELU
            Linear(hidden_dim, hidden_dim//2) → LayerNorm → ELU
            Linear(hidden_dim//2, latent_dim) × 2  [mu, log_var]

        Uso de LayerNorm (em vez de BatchNorm): estável com batch
        pequenos (comum em datasets de DP clínicos, n~100-200).
        """
        def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
            super().__init__()
            h2 = max(16, hidden_dim // 2)
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, h2),
                nn.LayerNorm(h2),
                nn.ELU(),
            )
            self.fc_mu      = nn.Linear(h2, latent_dim)
            self.fc_log_var = nn.Linear(h2, latent_dim)
            # Inicialização conservadora de log_var
            nn.init.constant_(self.fc_log_var.bias, -2.0)

        def forward(self, x):
            h = self.net(x)
            return self.fc_mu(h), self.fc_log_var(h)


    class TsallisVAEDecoder(nn.Module):
        """
        Decoder q-VAE: latent_dim → reconstrução (input_dim).

        Arquitetura espelhada ao Encoder.
        Sem ativação final: reconstrução no espaço de features
        normalizadas (z-score aplicado externamente).
        """
        def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 128):
            super().__init__()
            h2 = max(16, hidden_dim // 2)
            self.net = nn.Sequential(
                nn.Linear(latent_dim, h2),
                nn.LayerNorm(h2),
                nn.ELU(),
                nn.Linear(h2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, z):
            return self.net(z)


    class TsallisVAE(nn.Module):
        """
        q-VAE completo: Encoder + Decoder + ELBO Tsallis.

        Loss:
            ELBO_q = L_recon_q + beta * D_q(q(z|x) || p(z))

        onde:
        - L_recon_q  : perda de reconstrução Tsallis (generalização de MSE)
        - D_q        : q-divergência entre posterior variacional e prior N(0,I)
        - beta       : peso do termo de regularização (beta-VAE framework)
        - q          : parâmetro de não-extensividade (compartilhado)

        Parâmetros
        ----------
        input_dim  : dimensão do vetor de features (ex: 55 para A+B+C+D)
        latent_dim : dimensão do espaço latente (recomendado: 4-16)
        hidden_dim : neurônios nas camadas ocultas
        q          : parâmetro Tsallis (1.0 = VAE padrão)
        beta       : peso da regularização KL (1.0 = VAE padrão)
        """
        def __init__(
            self,
            input_dim: int,
            latent_dim: int = 8,
            hidden_dim: int = 128,
            q: float = 1.3,
            beta: float = 1.0
        ):
            super().__init__()
            self.input_dim  = input_dim
            self.latent_dim = latent_dim
            self.q          = q
            self.beta       = beta

            self.encoder = TsallisVAEEncoder(input_dim, latent_dim, hidden_dim)
            self.decoder = TsallisVAEDecoder(latent_dim, input_dim, hidden_dim)

        def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
            """
            Trick de reparametrização: z = mu + eps * sigma, eps ~ N(0,I).
            Permite gradiente fluir pelo encoder.
            """
            if self.training:
                std = torch.exp(0.5 * log_var.clamp(-10, 10))
                eps = torch.randn_like(std)
                return mu + eps * std
            return mu  # determinístico durante inferência

        def forward(self, x: torch.Tensor):
            """Retorna (x_recon, mu, log_var, z)."""
            mu, log_var = self.encoder(x)
            z           = self.reparameterize(mu, log_var)
            x_recon     = self.decoder(z)
            return x_recon, mu, log_var, z

        def loss(
            self,
            x: torch.Tensor,
            x_recon: torch.Tensor,
            mu: torch.Tensor,
            log_var: torch.Tensor
        ) -> tuple:
            """
            ELBO Tsallis:
                L = L_recon + beta * D_q(posterior || prior)

            Retorna: (loss_total, recon_loss, kl_loss)
            """
            recon = _tsallis_recon_loss(x_recon, x, self.q)
            kl    = _kl_loss_torch(mu, log_var, self.q)
            total = recon + self.beta * kl
            return total, recon, kl

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """Codifica x → mu (representação latente determinística)."""
            self.eval()
            with torch.no_grad():
                mu, _ = self.encoder(x)
            return mu


    class QVAETrainer:
        """
        Loop de treino para TsallisVAE com suporte a NumPy arrays.

        Funcionalidades:
        - Normalização automática (z-score) com persistência dos parâmetros
        - Detecção automática de GPU/CPU
        - Histórico de loss para diagnóstico
        - Early stopping baseado em variação de ELBO

        Parâmetros
        ----------
        model       : instância de TsallisVAE
        lr          : learning rate (Adam)
        beta        : override do beta do modelo (None = usa model.beta)
        batch_size  : tamanho do batch
        device      : 'auto' | 'cpu' | 'cuda' | 'mps'
        """

        def __init__(
            self,
            model: 'TsallisVAE',
            lr: float = 1e-3,
            beta: float = None,
            batch_size: int = 32,
            device: str = 'auto'
        ):
            if device == 'auto':
                if torch.cuda.is_available():
                    device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'

            self.device    = torch.device(device)
            self.model     = model.to(self.device)
            self.batch_size = batch_size

            if beta is not None:
                self.model.beta = beta

            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=10, factor=0.5, verbose=False
            )

            # Normalização z-score (fitado em fit())
            self._mu_norm  = None
            self._std_norm = None

            # Histórico de loss
            self.history = {'total': [], 'recon': [], 'kl': []}

        def _normalize(self, X: np.ndarray) -> np.ndarray:
            if self._mu_norm is None:
                self._mu_norm  = np.nanmean(X, axis=0)
                self._std_norm = np.nanstd(X, axis=0)
                self._std_norm[self._std_norm < 1e-8] = 1.0
            return (X - self._mu_norm) / self._std_norm

        def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return torch.tensor(X, dtype=torch.float32).to(self.device)

        def fit(
            self,
            X: np.ndarray,
            n_epochs: int = 100,
            verbose: bool = True,
            log_every: int = 10,
            early_stop_patience: int = 20
        ) -> 'QVAETrainer':
            """
            Treina o q-VAE nos dados X.

            Parâmetros
            ----------
            X                  : array (n_samples, n_features)
            n_epochs           : épocas de treinamento
            verbose            : se True, exibe loss
            log_every          : frequência de log (épocas)
            early_stop_patience: épocas sem melhora para parar

            Retorna
            -------
            self (para encadeamento)
            """
            X_norm = self._normalize(X)
            X_t    = self._to_tensor(X_norm)
            dataset = TensorDataset(X_t)
            loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            best_loss = np.inf
            no_improve = 0

            for epoch in range(1, n_epochs + 1):
                self.model.train()
                ep_total = ep_recon = ep_kl = 0.0
                n_batches = 0

                for (batch_x,) in loader:
                    self.optimizer.zero_grad()
                    x_recon, mu, log_var, _ = self.model(batch_x)
                    loss, recon, kl = self.model.loss(batch_x, x_recon, mu, log_var)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    ep_total += loss.item()
                    ep_recon += recon.item()
                    ep_kl    += kl.item()
                    n_batches += 1

                avg_total = ep_total / max(n_batches, 1)
                avg_recon = ep_recon / max(n_batches, 1)
                avg_kl    = ep_kl    / max(n_batches, 1)

                self.history['total'].append(avg_total)
                self.history['recon'].append(avg_recon)
                self.history['kl'].append(avg_kl)
                self.scheduler.step(avg_total)

                if verbose and (epoch % log_every == 0 or epoch == 1):
                    print(
                        f"  Epoch {epoch:4d}/{n_epochs} | "
                        f"ELBO={avg_total:.4f} | "
                        f"Recon={avg_recon:.4f} | "
                        f"KL={avg_kl:.4f}"
                    )

                # Early stopping
                if avg_total < best_loss - 1e-5:
                    best_loss  = avg_total
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= early_stop_patience:
                    if verbose:
                        print(f"  [Early stop] Epoca {epoch}, sem melhora em "
                              f"{early_stop_patience} epocas.")
                    break

            return self

        def encode(self, X: np.ndarray) -> np.ndarray:
            """
            Codifica X → representações latentes (mu).

            Parâmetros
            ----------
            X : array (n_samples, n_features) — mesma escala do fit()

            Retorna
            -------
            Z : array (n_samples, latent_dim)
            """
            X_norm = self._normalize(X) if self._mu_norm is not None else X
            X_t    = self._to_tensor(X_norm)
            self.model.eval()
            with torch.no_grad():
                mu, _ = self.model.encoder(X_t)
            return mu.cpu().numpy()

        def reconstruct(self, X: np.ndarray) -> np.ndarray:
            """Reconstrói X no espaço original (desnormalizado)."""
            X_norm = self._normalize(X) if self._mu_norm is not None else X
            X_t    = self._to_tensor(X_norm)
            self.model.eval()
            with torch.no_grad():
                x_recon, _, _, _ = self.model(X_t)
            X_rec = x_recon.cpu().numpy()
            if self._std_norm is not None:
                X_rec = X_rec * self._std_norm + self._mu_norm
            return X_rec

        def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
            """
            Calcula erro de reconstrução por amostra (norma L2 no espaço normalizado).
            Útil como score de anomalia / discriminação DP vs HC.

            Retorna
            -------
            errors : array (n_samples,) de erros de reconstrução
            """
            X_norm = self._normalize(X) if self._mu_norm is not None else X
            X_t    = self._to_tensor(X_norm)
            self.model.eval()
            with torch.no_grad():
                x_recon, _, _, _ = self.model(X_t)
            err = (x_recon.cpu().numpy() - X_norm) ** 2
            return np.sqrt(err.mean(axis=1))


# ─────────────────────────────────────────────────────────────────
# FALLBACK NumPy — q-VAE LINEAR (sem PyTorch)
# ─────────────────────────────────────────────────────────────────

class NumpyQVAE:
    """
    q-VAE linear simplificado (PCA + q-regularização) para uso
    sem PyTorch. Usa SVD truncado para o encoder e reconstrução
    por projeção inversa. A q-divergência é calculada analiticamente.

    NÃO equivalente ao TsallisVAE completo — use apenas como fallback
    ou para prototipagem rápida sem GPU.

    Parâmetros
    ----------
    input_dim  : dimensão do input
    latent_dim : dimensão do espaço latente
    q          : parâmetro Tsallis (afeta apenas o cálculo da KL)
    """

    def __init__(self, input_dim: int, latent_dim: int = 8, q: float = 1.3):
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.q          = q
        self._W         = None   # matriz de projeção (input_dim, latent_dim)
        self._mu_norm   = None
        self._std_norm  = None

    def fit(self, X: np.ndarray, verbose: bool = False) -> 'NumpyQVAE':
        """Ajusta o modelo usando SVD truncado."""
        X = np.nan_to_num(np.asarray(X, dtype=np.float64))
        self._mu_norm  = X.mean(axis=0)
        self._std_norm = X.std(axis=0)
        self._std_norm[self._std_norm < 1e-8] = 1.0
        X_norm = (X - self._mu_norm) / self._std_norm

        U, s, Vt = np.linalg.svd(X_norm, full_matrices=False)
        k = min(self.latent_dim, len(s))
        self._W  = Vt[:k].T   # (input_dim, k)
        self._s  = s[:k]

        if verbose:
            var_explained = (s[:k]**2).sum() / (s**2).sum()
            print(f"  [NumpyQVAE] Variancia explicada: {var_explained:.3f} "
                  f"com {k} componentes")
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=np.float64))
        X_norm = (X - self._mu_norm) / self._std_norm
        return X_norm @ self._W

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        Z     = self.encode(X)
        X_rec = Z @ self._W.T
        return X_rec * self._std_norm + self._mu_norm

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        X     = np.nan_to_num(np.asarray(X, dtype=np.float64))
        X_rec = self.reconstruct(X)
        return np.sqrt(((X - X_rec)**2).mean(axis=1))


# ─────────────────────────────────────────────────────────────────
# FUNÇÃO DE FÁBRICA — cria o modelo correto baseado na disponibilidade
# ─────────────────────────────────────────────────────────────────

def build_qvae(
    input_dim: int,
    latent_dim: int = 8,
    hidden_dim: int = 128,
    q: float = 1.3,
    beta: float = 1.0,
    force_numpy: bool = False
):
    """
    Cria o q-VAE apropriado (TsallisVAE se PyTorch disponível, else NumpyQVAE).

    Parâmetros
    ----------
    input_dim   : dimensão das features (ex: 55)
    latent_dim  : dimensão do espaço latente
    hidden_dim  : neurônios ocultos (ignorado no fallback NumPy)
    q           : parâmetro de não-extensividade
    beta        : peso da regularização KL
    force_numpy : se True, usa NumpyQVAE mesmo com PyTorch disponível

    Retorna
    -------
    model : TsallisVAE ou NumpyQVAE
    """
    if _TORCH_AVAILABLE and not force_numpy:
        return TsallisVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            q=q,
            beta=beta
        )
    else:
        warnings.warn(
            "[tfs_e_vae] Usando NumpyQVAE (fallback linear). "
            "Instale PyTorch para o modelo completo.",
            UserWarning
        )
        return NumpyQVAE(input_dim=input_dim, latent_dim=latent_dim, q=q)


# ─────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  tfs_e_vae.py v1.0 -- Self-Test")
    print("=" * 65)

    rng = np.random.default_rng(42)

    # Simular dataset com 55 features (A+B+C+D do TFS)
    # HC: features centradas, baixa variância
    # DP: features com maior dispersão e valores mais extremos
    n_hc, n_dp = 60, 60
    n_feat = 55

    X_hc = rng.normal(0, 1, (n_hc, n_feat))
    X_dp = rng.normal(0.5, 1.4, (n_dp, n_feat))
    X_dp[:, :13] += 2.0  # Bloco A: S_q maior em DP (hipótese H1)
    X_dp[:, 13:37] += 0.5  # Bloco B: diferença moderada
    X_dp[:, 37:45] += 3.0  # Bloco C: S_q de F0 muito maior (hipótese H2)
    X_dp[:, 45:57] += 1.0  # Bloco D: std maior (hipótese H3)

    X = np.vstack([X_hc, X_dp])
    y = np.array([0]*n_hc + [1]*n_dp)

    print("\n[1] q_log e q_exp")
    x_test = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    for q_v in [0.7, 1.0, 1.3]:
        ql  = q_log(x_test, q_v)
        qe  = q_exp(ql, q_v)
        max_err = np.max(np.abs(qe - x_test))
        status = "OK" if max_err < 1e-6 else "FALHOU"
        print(f"  q={q_v}: q_exp(q_log(x)) == x  max_err={max_err:.2e}  [{status}]")

    print("\n[2] tsallis_kl_gaussian")
    mu_test      = rng.normal(0, 1, (10, 4))
    logvar_test  = rng.normal(-1, 0.5, (10, 4))
    for q_v in [1.0, 1.3, 2.0]:
        kl = tsallis_kl_gaussian(mu_test, logvar_test, q=q_v)
        print(f"  q={q_v}: KL_q = {kl:.4f}  ({'> 0 OK' if kl > 0 else 'ATENCAO: <= 0'})")

    if not _TORCH_AVAILABLE:
        print("\n[3] PyTorch nao disponivel. Testando NumpyQVAE (fallback).")
        model = NumpyQVAE(input_dim=n_feat, latent_dim=8, q=1.3)
        model.fit(X, verbose=True)
        Z    = model.encode(X)
        errs = model.reconstruction_error(X)
        print(f"  Codificacao: X({X.shape}) -> Z({Z.shape})")
        print(f"  Erro recon HC: {errs[:n_hc].mean():.4f} +/- {errs[:n_hc].std():.4f}")
        print(f"  Erro recon DP: {errs[n_hc:].mean():.4f} +/- {errs[n_hc:].std():.4f}")
        # Verificar se erro de DP > HC (DP mais "anômalo" no espaço HC)
        status = "H_anomalia suportada" if errs[n_hc:].mean() > errs[:n_hc].mean() else "sem diferenca"
        print(f"  -> {status}")

    else:
        print("\n[3] TsallisVAE (PyTorch)")
        model = TsallisVAE(
            input_dim=n_feat, latent_dim=8, hidden_dim=64, q=1.3, beta=1.0
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parametros: {n_params:,}")
        print(f"  q={model.q}, beta={model.beta}")

        trainer = QVAETrainer(model, lr=1e-3, batch_size=16, device='cpu')
        print(f"  Device: {trainer.device}")
        print("  Treinando 30 epocas (subset de self-test)...")
        trainer.fit(X, n_epochs=30, verbose=True, log_every=10, early_stop_patience=10)

        Z    = trainer.encode(X)
        errs = trainer.reconstruction_error(X)
        print(f"\n  Codificacao: X{X.shape} -> Z{Z.shape}")
        print(f"  Erro recon HC: {errs[:n_hc].mean():.4f} +/- {errs[:n_hc].std():.4f}")
        print(f"  Erro recon DP: {errs[n_hc:].mean():.4f} +/- {errs[n_hc:].std():.4f}")

        # Separabilidade no espaço latente
        Z_hc = Z[:n_hc]
        Z_dp = Z[n_hc:]
        centroid_dist = np.linalg.norm(Z_hc.mean(axis=0) - Z_dp.mean(axis=0))
        print(f"  Distancia entre centroides (HC vs DP) no espaco Z: {centroid_dist:.4f}")

        # Verificar hipótese de anomalia
        status = "H_anomalia suportada" if errs[n_hc:].mean() > errs[:n_hc].mean() else "sem diferenca"
        print(f"  Erro DP > HC: {status}")

    print("\n[4] build_qvae (fabrica)")
    m = build_qvae(input_dim=55, latent_dim=8, q=1.3)
    print(f"  Modelo criado: {type(m).__name__}")

    print("\nModulo 4 pronto.")
    print("=" * 65)
