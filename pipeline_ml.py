"""
pipeline_ml.py — Módulo 5: Pipeline de ML para Classificação DP vs HC
======================================================================
Versão 1.0

Depende de: tsallis_core.py, tfs_ab.py, tfs_cd.py, tfs_e_vae.py

Implementa o Bloco F do framework TFS:
    Pré-processamento, seleção de features, classificação e avaliação

Componentes:
    TFSFeatureBuilder   — Consolida features A+B+C+D (+Z do q-VAE)
    TFSPreprocessor     — Normalização, imputação, seleção de features
    TFSClassifier       — SVM-RBF, XGBoost, MLP, Ensemble (votação soft)
    TFSEvaluator        — AUC-ROC, F1, sensibilidade, especificidade,
                          curva ROC, matriz de confusão, permutation test
    TFSPipeline         — Orquestrador completo (fit + predict + evaluate)

Uso rápido:
    from pipeline_ml import TFSPipeline
    pipe = TFSPipeline(classifier='ensemble', q_vae=True, latent_dim=8)
    pipe.fit(X_train, y_train)
    metrics = pipe.evaluate(X_test, y_test)
    print(metrics['auc_roc'])
"""

import numpy as np
import warnings
import os
import json
from copy import deepcopy

# ── Dependências científicas ─────────────────────────────────────
from sklearn.svm              import SVC
from sklearn.neural_network   import MLPClassifier
from sklearn.preprocessing    import StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.pipeline         import Pipeline as SKPipeline
from sklearn.model_selection  import (StratifiedKFold, cross_validate,
                                       permutation_test_score)
from sklearn.metrics          import (roc_auc_score, f1_score,
                                       confusion_matrix, roc_curve,
                                       classification_report)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.inspection        import permutation_importance
from sklearn.calibration       import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    warnings.warn("[pipeline_ml] XGBoost nao encontrado. "
                  "Usando GradientBoosting da sklearn como substituto.", ImportWarning)
    from sklearn.ensemble import GradientBoostingClassifier

from tsallis_core import validate_signal


# ─────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────

# Nomes canônicos das features (ordem fixa para reprodutibilidade)
Q_SWEEP_VALUES = np.array([0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0, 2.5])
Q_SUBBAND_VALUES = np.array([0.7, 1.3, 2.0])
SUBBAND_COUNT = 8
Q_CHRONO_VALUES = np.array([1.3, 2.0])

FEATURE_GROUPS = {
    'A': (  # q-sweep amplitude: 9 S_q + 4 geom = 13
        [f"A_Sq_{q:.1f}" for q in Q_SWEEP_VALUES] +
        ['A_decay_ratio', 'A_area', 'A_slope_q1', 'A_curvature_int']
    ),
    'B': (  # TES subbandas: 8 bandas x 3 q = 24
        [f"B_b{b}_q{q:.1f}" for b in range(1, SUBBAND_COUNT+1)
         for q in Q_SUBBAND_VALUES]
    ),
    'C': (  # Tsallis F0 e perturbação: 8
        ['C_Sq_F0_q0.7', 'C_Sq_F0_q1.3', 'C_Sq_F0_q2.0',
         'C_Sq_dF0_q1.3', 'C_Sq_dF0_q2.0', 'C_q_fit_F0',
         'C_Sq_amp_cycle_q1.3', 'C_Sq_shimmer_q1.3']
    ),
    'D': (  # q-Cronoentropograma: 6 stats x 2 q = 12
        [f"D_{s}_q{q:.1f}"
         for q in Q_CHRONO_VALUES
         for s in ['mean', 'std', 'onset', 'steady', 'delta', 'tau']]
    ),
}

ALL_FEATURES = (
    FEATURE_GROUPS['A'] +
    FEATURE_GROUPS['B'] +
    FEATURE_GROUPS['C'] +
    FEATURE_GROUPS['D']
)   # 57 features canônicas (13+24+8+12)


# ─────────────────────────────────────────────────────────────────
# BLOCO F.1 — BUILDER DE FEATURES
# ─────────────────────────────────────────────────────────────────

class TFSFeatureBuilder:
    """
    Consolida features dos Blocos A, B, C, D em matriz numpy (n, n_feat).

    Aceita lista de dicts (saída de extract_blocks_AB + extract_blocks_CD)
    ou DataFrame pandas. Garante ordem canônica das colunas e preenche
    com NaN features ausentes.

    Parâmetros
    ----------
    use_vae      : se True, inclui representações latentes do q-VAE
    latent_dim   : dimensão do espaço latente do q-VAE
    """

    def __init__(self, use_vae: bool = False, latent_dim: int = 8):
        self.use_vae    = use_vae
        self.latent_dim = latent_dim
        self.feature_names_ = list(ALL_FEATURES)

    def from_dicts(self, feat_list: list) -> np.ndarray:
        """
        Constrói matriz X a partir de lista de dicts de features.

        Parâmetros
        ----------
        feat_list : lista de dicts (cada dict = um sujeito)

        Retorna
        -------
        X : array (n_subjects, n_features)
        """
        n = len(feat_list)
        X = np.full((n, len(self.feature_names_)), np.nan)

        for i, fd in enumerate(feat_list):
            for j, fname in enumerate(self.feature_names_):
                val = fd.get(fname, np.nan)
                if val is not None and np.isfinite(float(val)):
                    X[i, j] = float(val)

        return X

    def from_dataframe(self, df) -> np.ndarray:
        """
        Constrói matriz X a partir de DataFrame pandas.
        Colunas ausentes são preenchidas com NaN.
        """
        import pandas as pd
        X = np.full((len(df), len(self.feature_names_)), np.nan)
        for j, fname in enumerate(self.feature_names_):
            if fname in df.columns:
                X[:, j] = pd.to_numeric(df[fname], errors='coerce').values
        return X

    def feature_group_mask(self, group: str) -> np.ndarray:
        """Retorna máscara booleana para o grupo de features (A, B, C ou D)."""
        names = FEATURE_GROUPS.get(group.upper(), [])
        return np.array([f in names for f in self.feature_names_])


# ─────────────────────────────────────────────────────────────────
# BLOCO F.2 — PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────────────────────────

class TFSPreprocessor:
    """
    Pré-processamento das features TFS:
    1. Imputação de NaN (mediana)
    2. Z-score por feature
    3. Seleção de features (ANOVA F-test ou mutual info)
    4. (Opcional) Remoção de features com variância zero

    Parâmetros
    ----------
    k_features    : número de features a selecionar (None = todas)
    selection_method : 'anova' | 'mutual_info'
    remove_zero_var  : se True, remove features com var=0
    """

    def __init__(
        self,
        k_features: int = None,
        selection_method: str = 'anova',
        remove_zero_var: bool = True
    ):
        self.k_features       = k_features
        self.selection_method = selection_method
        self.remove_zero_var  = remove_zero_var

        self._imputer   = SimpleImputer(strategy='median')
        self._scaler    = StandardScaler()
        self._selector  = None
        self._zero_mask = None
        self.selected_features_ = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: list = None) -> 'TFSPreprocessor':
        """
        Ajusta imputador, scaler e seletor de features nos dados de treino.

        Parâmetros
        ----------
        X            : array (n, n_feat)
        y            : array (n,) de rótulos binários {0, 1}
        feature_names: lista de nomes (para rastreabilidade)
        """
        X = np.asarray(X, dtype=np.float64)

        # Imputação
        X_imp = self._imputer.fit_transform(X)

        # Remover variância zero
        if self.remove_zero_var:
            var = X_imp.var(axis=0)
            self._zero_mask = var > 1e-10
            X_imp = X_imp[:, self._zero_mask]
            if feature_names is not None:
                feature_names = [f for f, m in zip(feature_names, self._zero_mask) if m]

        # Z-score
        X_sc = self._scaler.fit_transform(X_imp)

        # Seleção de features
        if self.k_features is not None and self.k_features < X_sc.shape[1]:
            score_fn = (f_classif if self.selection_method == 'anova'
                        else mutual_info_classif)
            self._selector = SelectKBest(score_fn, k=self.k_features)
            self._selector.fit(X_sc, y)
            selected_idx = self._selector.get_support(indices=True)
        else:
            selected_idx = np.arange(X_sc.shape[1])

        if feature_names is not None:
            self.selected_features_ = [feature_names[i] for i in selected_idx]

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica transformação (imputação + scaling + seleção) em novos dados."""
        X = np.asarray(X, dtype=np.float64)
        X_imp = self._imputer.transform(X)
        if self._zero_mask is not None:
            X_imp = X_imp[:, self._zero_mask]
        X_sc = self._scaler.transform(X_imp)
        if self._selector is not None:
            X_sc = self._selector.transform(X_sc)
        return X_sc

    def fit_transform(self, X: np.ndarray, y: np.ndarray,
                      feature_names: list = None) -> np.ndarray:
        self.fit(X, y, feature_names)
        return self.transform(X)

    def feature_importances_anova(
        self, X: np.ndarray, y: np.ndarray
    ) -> dict:
        """
        Retorna scores ANOVA F para todas as features (após imputação + scaling).
        Útil para ranking de importância das features TFS.
        """
        X_imp = self._imputer.transform(X)
        if self._zero_mask is not None:
            X_imp = X_imp[:, self._zero_mask]
        X_sc = self._scaler.transform(X_imp)
        from sklearn.feature_selection import f_classif as fc
        f_scores, p_values = fc(X_sc, y)
        return {'f_scores': f_scores, 'p_values': p_values}


# ─────────────────────────────────────────────────────────────────
# BLOCO F.3 — CLASSIFICADORES
# ─────────────────────────────────────────────────────────────────

def _build_svm(C: float = 1.0, gamma: str = 'scale') -> SKPipeline:
    """SVM-RBF calibrado com Platt scaling (probabilidades confiáveis)."""
    svm_base = SVC(C=C, kernel='rbf', gamma=gamma, 
                   class_weight='balanced', random_state=42)
    calibrated = CalibratedClassifierCV(svm_base, cv=5, method='sigmoid', ensemble=False)
    return calibrated


def _build_xgb(n_estimators: int = 200, max_depth: int = 4,
               lr: float = 0.05) -> object:
    """XGBoost com class_weight automático para datasets desbalanceados."""
    if _XGB_AVAILABLE:
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=1.0,  # ajustado no fit()
            random_state=42,
            verbosity=0
        )
    else:
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            random_state=42
        )


def _build_mlp(hidden_layer_sizes: tuple = (64, 32),
               alpha: float = 1e-3) -> MLPClassifier:
    """MLP com dropout implícito via regularização L2."""
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=alpha,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        tol=1e-4
    )


from sklearn.base import BaseEstimator, ClassifierMixin as _CM

class SoftVotingEnsemble(BaseEstimator, _CM):
    """
    Ensemble por votação soft (média de probabilidades).
    Combina SVM-RBF + XGB/GBM + MLP.

    O ensemble é mais robusto que qualquer modelo individual em
    datasets de DP pequenos (n<200), pois os erros individuais
    tendem a ser ortogonais nos três espaços de hipóteses.
    """

    def __init__(self, weights: tuple = (1.0, 1.0, 1.0)):
        self.weights = np.array(weights) / np.sum(weights)
        self.models_ = {
            'svm': _build_svm(),
            'xgb': _build_xgb(),
            'mlp': _build_mlp()
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SoftVotingEnsemble':
        """Treina os três classificadores."""
        self.classes_ = np.unique(y)  # requerido pelo sklearn
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        if _XGB_AVAILABLE and n_pos > 0:
            self.models_['xgb'].scale_pos_weight = n_neg / n_pos

        for name, clf in self.models_.items():
            clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades médias ponderadas."""
        proba_list = []
        for (name, clf), w in zip(self.models_.items(), self.weights):
            if hasattr(clf, 'predict_proba'):
                p = clf.predict_proba(X)
            else:
                # fallback: decisão binária
                d = clf.decision_function(X)
                p = np.column_stack([1/(1+np.exp(d)), 1/(1+np.exp(-d))])
            proba_list.append(w * p)
        return np.sum(proba_list, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TFSClassifier:
    """
    Interface unificada para os classificadores do Bloco F.

    Parâmetros
    ----------
    model_type : 'svm' | 'xgb' | 'mlp' | 'ensemble'
    **kwargs   : parâmetros do modelo específico
    """

    VALID_MODELS = ('svm', 'xgb', 'mlp', 'ensemble')

    def __init__(self, model_type: str = 'ensemble', **kwargs):
        if model_type not in self.VALID_MODELS:
            raise ValueError(
                f"model_type deve ser um de: {self.VALID_MODELS}. "
                f"Recebido: '{model_type}'"
            )
        self.model_type = model_type
        self.kwargs     = kwargs
        self.model_     = None

    def _build(self) -> object:
        if self.model_type == 'svm':
            return _build_svm(
                C=self.kwargs.get('C', 1.0),
                gamma=self.kwargs.get('gamma', 'scale')
            )
        elif self.model_type == 'xgb':
            return _build_xgb(
                n_estimators=self.kwargs.get('n_estimators', 200),
                max_depth=self.kwargs.get('max_depth', 4),
                lr=self.kwargs.get('lr', 0.05)
            )
        elif self.model_type == 'mlp':
            return _build_mlp(
                hidden_layer_sizes=self.kwargs.get('hidden_layer_sizes', (64, 32)),
                alpha=self.kwargs.get('alpha', 1e-3)
            )
        elif self.model_type == 'ensemble':
            return SoftVotingEnsemble(
                weights=self.kwargs.get('weights', (1.0, 1.0, 1.0))
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TFSClassifier':
        self.model_ = self._build()
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X)
        raise AttributeError(
            f"Modelo {self.model_type} nao suporta predict_proba."
        )


# ─────────────────────────────────────────────────────────────────
# BLOCO F.4 — AVALIAÇÃO
# ─────────────────────────────────────────────────────────────────

class TFSEvaluator:
    """
    Avaliação completa de performance do classificador TFS.

    Métricas calculadas:
    - AUC-ROC
    - Sensibilidade (recall DP)
    - Especificidade (recall HC)
    - F1-score
    - Acurácia balanceada
    - Curva ROC (fpr, tpr, thresholds)
    - Matriz de confusão
    - p-valor via permutation test (n_permutations=200)

    Comparação automática com Iyer et al. (AUC=0.97) como benchmark.
    """

    BENCHMARK_AUC = 0.97  # Iyer et al. 2023

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None
    ) -> dict:
        """
        Calcula métricas de avaliação.

        Parâmetros
        ----------
        y_true : rótulos verdadeiros {0=HC, 1=DP}
        y_pred : rótulos preditos
        y_prob : probabilidades preditas da classe positiva (DP)

        Retorna
        -------
        dict : métricas completas
        """
        y_true = np.asarray(y_true, int)
        y_pred = np.asarray(y_pred, int)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        sens = tp / (tp + fn + 1e-10)
        spec = tn / (tn + fp + 1e-10)

        metrics = {
            'sensitivity':    float(sens),
            'specificity':    float(spec),
            'f1':             float(f1_score(y_true, y_pred, zero_division=0)),
            'balanced_acc':   float((sens + spec) / 2.0),
            'confusion_matrix': cm.tolist(),
        }

        if y_prob is not None:
            auc = float(roc_auc_score(y_true, y_prob))
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            metrics['auc_roc']    = auc
            metrics['roc_fpr']    = fpr.tolist()
            metrics['roc_tpr']    = tpr.tolist()
            metrics['vs_benchmark'] = {
                'iyer_auc':     self.BENCHMARK_AUC,
                'tfs_auc':      auc,
                'delta':        round(auc - self.BENCHMARK_AUC, 4),
                'status':       'SUPEROU' if auc >= self.BENCHMARK_AUC else 'ABAIXO'
            }

        return metrics

    def cross_validate(
        self,
        classifier: 'TFSClassifier',
        preprocessor: 'TFSPreprocessor',
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        n_repeats: int = 3
    ) -> dict:
        """
        Validação cruzada estratificada k-fold (k=5, repetida 3x).

        Retorna métricas médias e desvios-padrão sobre os folds.
        """
        from sklearn.model_selection import RepeatedStratifiedKFold
        from sklearn.base import clone

        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=42
        )

        auc_scores  = []
        f1_scores   = []
        sens_scores = []
        spec_scores = []

        for fold, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Pré-processar apenas no treino, transformar teste
            prep = deepcopy(preprocessor)
            X_tr_pp = prep.fit_transform(X_tr, y_tr)
            X_te_pp = prep.transform(X_te)

            # Treinar classificador
            clf = deepcopy(classifier)
            clf.fit(X_tr_pp, y_tr)

            # Avaliar
            y_prob = clf.predict_proba(X_te_pp)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            metrics = self.evaluate(y_te, y_pred, y_prob)
            auc_scores.append(metrics.get('auc_roc', np.nan))
            f1_scores.append(metrics['f1'])
            sens_scores.append(metrics['sensitivity'])
            spec_scores.append(metrics['specificity'])

        def _stats(arr):
            a = np.array(arr)
            return {'mean': float(np.nanmean(a)),
                    'std':  float(np.nanstd(a)),
                    'min':  float(np.nanmin(a)),
                    'max':  float(np.nanmax(a))}

        results = {
            'auc_roc':     _stats(auc_scores),
            'f1':          _stats(f1_scores),
            'sensitivity': _stats(sens_scores),
            'specificity': _stats(spec_scores),
            'n_folds':     n_splits * n_repeats,
        }

        # Comparação com benchmark
        mean_auc = results['auc_roc']['mean']
        results['vs_benchmark'] = {
            'iyer_auc': self.BENCHMARK_AUC,
            'tfs_auc':  round(mean_auc, 4),
            'delta':    round(mean_auc - self.BENCHMARK_AUC, 4),
            'status':   'SUPEROU' if mean_auc >= self.BENCHMARK_AUC else 'ABAIXO'
        }

        return results

    def permutation_test(
        self,
        classifier: 'TFSClassifier',
        preprocessor: 'TFSPreprocessor',
        X: np.ndarray,
        y: np.ndarray,
        n_permutations: int = 200,
        cv: int = 5
    ) -> dict:
        """
        Teste de permutação para validar significância estatística.
        Estima o p-valor da hipótese nula: o classificador performa
        aleatoriamente (sem relação entre features e rótulos).

        p < 0.05 é necessário para validação científica do resultado.
        """
        from sklearn.pipeline import Pipeline as SKPipeline
        from sklearn.base import BaseEstimator, ClassifierMixin

        class _WrappedCLF(BaseEstimator, ClassifierMixin):
            def __init__(self, clf_type, prep):
                self.clf_type = clf_type
                self.prep = prep
            def fit(self, X, y):
                self._prep = deepcopy(self.prep)
                X_pp = self._prep.fit_transform(X, y)
                self._clf = deepcopy(classifier)
                self._clf.fit(X_pp, y)
                return self
            def score(self, X, y):
                X_pp = self._prep.transform(X)
                y_prob = self._clf.predict_proba(X_pp)[:, 1]
                return float(roc_auc_score(y, y_prob))

        wrapped = _WrappedCLF(classifier.model_type, preprocessor)
        score, perm_scores, p_value = permutation_test_score(
            wrapped, X, y,
            scoring='roc_auc',
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            n_permutations=n_permutations,
            n_jobs=1,
            random_state=42
        )

        return {
            'observed_auc':   float(score),
            'p_value':        float(p_value),
            'n_permutations': n_permutations,
            'significant':    bool(p_value < 0.05),
            'perm_scores_mean': float(np.mean(perm_scores)),
        }

    def feature_importance_report(
        self,
        classifier: 'TFSClassifier',
        preprocessor: 'TFSPreprocessor',
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list = None,
        n_repeats: int = 10
    ) -> dict:
        """
        Importância de features via permutation importance (manual, model-agnostic).
        Baralha cada feature individualmente e mede queda no AUC-ROC.
        Compatível com qualquer modelo (SVM, MLP, Ensemble).
        """
        X_pp = preprocessor.transform(X)
        names = preprocessor.selected_features_ or feature_names or \
                [f"feat_{i}" for i in range(X_pp.shape[1])]

        clf = classifier.model_

        # Score base
        y_prob_base = clf.predict_proba(X_pp)[:, 1]
        base_auc = float(roc_auc_score(y, y_prob_base))

        n_feats = X_pp.shape[1]
        imp_mean = np.zeros(n_feats)
        imp_std  = np.zeros(n_feats)

        rng_imp = np.random.default_rng(42)
        for j in range(n_feats):
            aucs_j = []
            for _ in range(n_repeats):
                X_perm = X_pp.copy()
                X_perm[:, j] = rng_imp.permutation(X_perm[:, j])
                y_prob_j = clf.predict_proba(X_perm)[:, 1]
                try:
                    aucs_j.append(float(roc_auc_score(y, y_prob_j)))
                except Exception:
                    aucs_j.append(base_auc)
            imp_mean[j] = base_auc - np.mean(aucs_j)
            imp_std[j]  = np.std(aucs_j)

        idx_sorted = np.argsort(imp_mean)[::-1]
        return {
            'base_auc': base_auc,
            'feature_names': [names[i] for i in idx_sorted if i < len(names)],
            'importances_mean': imp_mean[idx_sorted].tolist(),
            'importances_std':  imp_std[idx_sorted].tolist(),
        }


# ─────────────────────────────────────────────────────────────────
# ORQUESTRADOR PRINCIPAL — TFSPipeline
# ─────────────────────────────────────────────────────────────────

class TFSPipeline:
    """
    Orquestrador completo do framework TFS.

    Gerencia: pré-processamento → (q-VAE opcional) → classificação → avaliação

    Parâmetros
    ----------
    classifier      : 'svm' | 'xgb' | 'mlp' | 'ensemble'
    k_features      : features a selecionar (None = todas)
    q_vae           : se True, adiciona representações latentes do q-VAE
    latent_dim      : dimensão do espaço latente do q-VAE
    q_vae_epochs    : épocas de treino do q-VAE
    q_tsallis       : parâmetro q para o q-VAE
    selection_method: 'anova' | 'mutual_info'
    **clf_kwargs    : kwargs do classificador
    """

    def __init__(
        self,
        classifier: str = 'ensemble',
        k_features: int = None,
        q_vae: bool = False,
        latent_dim: int = 8,
        q_vae_epochs: int = 80,
        q_tsallis: float = 1.3,
        selection_method: str = 'anova',
        **clf_kwargs
    ):
        self.classifier      = classifier
        self.k_features      = k_features
        self.q_vae           = q_vae
        self.latent_dim      = latent_dim
        self.q_vae_epochs    = q_vae_epochs
        self.q_tsallis       = q_tsallis
        self.selection_method = selection_method
        self.clf_kwargs      = clf_kwargs

        self.builder_      = TFSFeatureBuilder(use_vae=q_vae, latent_dim=latent_dim)
        self.preprocessor_ = TFSPreprocessor(
            k_features=k_features, selection_method=selection_method
        )
        self.clf_          = TFSClassifier(model_type=classifier, **clf_kwargs)
        self.evaluator_    = TFSEvaluator()
        self._vae_trainer  = None
        self._is_fitted    = False

    def _append_vae_features(
        self, X: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        """Opcional: adiciona latent features do q-VAE ao vetor de features."""
        from tfs_e_vae import build_qvae, QVAETrainer
        if fit:
            model = build_qvae(
                input_dim=X.shape[1],
                latent_dim=self.latent_dim,
                q=self.q_tsallis
            )
            try:
                self._vae_trainer = QVAETrainer(model, lr=1e-3, batch_size=16)
                self._vae_trainer.fit(X, n_epochs=self.q_vae_epochs, verbose=False)
            except Exception:
                # Fallback NumpyQVAE
                model.fit(X, verbose=False)
                self._vae_trainer = model

        try:
            if hasattr(self._vae_trainer, 'encode'):
                Z = self._vae_trainer.encode(X)
            else:
                Z = self._vae_trainer.encode(X)
        except Exception as e:
            warnings.warn(f"[pipeline_ml] q-VAE encode falhou: {e}. Sem latent features.")
            return X

        return np.hstack([X, Z])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list = None
    ) -> 'TFSPipeline':
        """
        Treina o pipeline completo.

        Parâmetros
        ----------
        X            : array (n, n_features) de features TFS
        y            : array (n,) rótulos {0=HC, 1=DP}
        feature_names: nomes das features (para rastreabilidade)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, int)

        names = feature_names or list(ALL_FEATURES)[:X.shape[1]]

        # Pré-processar (imputação + scaling + seleção)
        X_pp = self.preprocessor_.fit_transform(X, y, feature_names=names)

        # Adicionar features latentes do q-VAE (opcional)
        if self.q_vae:
            X_pp = self._append_vae_features(X_pp, fit=True)

        # Treinar classificador
        self.clf_.fit(X_pp, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz rótulos para novos dados."""
        X_pp = self.preprocessor_.transform(np.asarray(X, dtype=np.float64))
        if self.q_vae and self._vae_trainer is not None:
            X_pp = self._append_vae_features(X_pp, fit=False)
        return self.clf_.predict(X_pp)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Prediz probabilidades para novos dados."""
        X_pp = self.preprocessor_.transform(np.asarray(X, dtype=np.float64))
        if self.q_vae and self._vae_trainer is not None:
            X_pp = self._append_vae_features(X_pp, fit=False)
        return self.clf_.predict_proba(X_pp)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> dict:
        """Avalia o pipeline em dados de teste."""
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)[:, 1]
        return self.evaluator_.evaluate(y, y_pred, y_prob)

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        n_repeats: int = 3
    ) -> dict:
        """Validação cruzada estratificada repetida."""
        return self.evaluator_.cross_validate(
            self.clf_, self.preprocessor_, X, y,
            n_splits=n_splits, n_repeats=n_repeats
        )

    def permutation_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_permutations: int = 200
    ) -> dict:
        """Teste de permutação para validação estatística."""
        return self.evaluator_.permutation_test(
            self.clf_, self.preprocessor_, X, y,
            n_permutations=n_permutations
        )

    def feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list = None
    ) -> dict:
        """Ranking de importância das features."""
        return self.evaluator_.feature_importance_report(
            self.clf_, self.preprocessor_, X, y, feature_names
        )

    def save(self, filepath: str) -> None:
        """Salva o pipeline treinado em formato pickle."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"[pipeline_ml] Pipeline salvo em: {filepath}")

    @staticmethod
    def load(filepath: str) -> 'TFSPipeline':
        """Carrega pipeline salvo."""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  pipeline_ml.py v1.0 -- Self-Test")
    print("=" * 65)

    rng = np.random.default_rng(42)

    # Simular matriz de features TFS (n=120, 57 features)
    n_hc, n_dp = 60, 60
    n_feat = len(ALL_FEATURES)

    X_hc = rng.normal(0, 1, (n_hc, n_feat))
    X_dp = rng.normal(0, 1, (n_dp, n_feat))
    # Introduzir diferenças realistas (baseadas nos resultados M2+M3)
    # Bloco A: decay_ratio menor em DP (índice 11)
    X_dp[:, 11] -= 1.5
    # Bloco C: Sq_F0 muito maior em DP (índices 37-42)
    X_dp[:, 37:43] += 3.0
    # Bloco D: std maior em DP, delta maior (índices 45-57)
    X_dp[:, 45:57] += 1.2
    # Adicionar 10% NaN realista
    nan_mask = rng.random((n_hc + n_dp, n_feat)) < 0.10
    X = np.vstack([X_hc, X_dp])
    X[nan_mask] = np.nan
    y = np.array([0]*n_hc + [1]*n_dp)

    print(f"\nDataset sintetico: {X.shape[0]} sujeitos x {X.shape[1]} features")
    print(f"NaN: {np.isnan(X).sum()} ({100*np.isnan(X).mean():.1f}%)")
    print(f"HC={n_hc}, DP={n_dp}")

    print("\n[1] TFSFeatureBuilder")
    builder = TFSFeatureBuilder()
    feat_dicts = [dict(zip(ALL_FEATURES, row)) for row in X[:5]]
    X_from_dicts = builder.from_dicts(feat_dicts)
    print(f"  from_dicts: {X_from_dicts.shape}  OK")
    mask_A = builder.feature_group_mask('A')
    mask_C = builder.feature_group_mask('C')
    print(f"  Grupo A: {mask_A.sum()} features | Grupo C: {mask_C.sum()} features")

    print("\n[2] TFSPreprocessor")
    prep = TFSPreprocessor(k_features=30, selection_method='anova')
    X_pp = prep.fit_transform(X, y, feature_names=list(ALL_FEATURES))
    print(f"  Input: {X.shape} -> Output: {X_pp.shape}")
    print(f"  NaN apos processamento: {np.isnan(X_pp).sum()}")
    print(f"  Features selecionadas (top 5): {prep.selected_features_[:5]}")

    print("\n[3] TFSClassifier (modelos individuais)")
    for clf_type in ['svm', 'mlp']:
        clf = TFSClassifier(model_type=clf_type)
        clf.fit(X_pp, y)
        y_prob = clf.predict_proba(X_pp)[:, 1]
        auc_train = roc_auc_score(y, y_prob)
        print(f"  {clf_type.upper()} AUC (treino): {auc_train:.4f}")

    print("\n[4] SoftVotingEnsemble")
    ens = SoftVotingEnsemble()
    ens.fit(X_pp, y)
    y_prob_ens = ens.predict_proba(X_pp)[:, 1]
    auc_ens = roc_auc_score(y, y_prob_ens)
    print(f"  Ensemble AUC (treino): {auc_ens:.4f}")

    print("\n[5] TFSEvaluator")
    evaluator = TFSEvaluator()
    y_pred = ens.predict(X_pp)
    metrics = evaluator.evaluate(y, y_pred, y_prob_ens)
    print(f"  AUC-ROC:      {metrics['auc_roc']:.4f}")
    print(f"  Sensibilidade:{metrics['sensitivity']:.4f}")
    print(f"  Especificidade:{metrics['specificity']:.4f}")
    print(f"  F1:           {metrics['f1']:.4f}")
    print(f"  vs Benchmark (Iyer AUC=0.97): {metrics['vs_benchmark']['status']}")

    print("\n[6] TFSPipeline end-to-end (CV 5-fold)")
    pipe = TFSPipeline(
        classifier='ensemble',
        k_features=8,
        q_vae=False,
        selection_method='anova'
    )
    # Treinar em 80% e avaliar em 20%
    n = len(y)
    idx = rng.permutation(n)
    tr_idx, te_idx = idx[:int(0.8*n)], idx[int(0.8*n):]
    pipe.fit(X[tr_idx], y[tr_idx], feature_names=list(ALL_FEATURES))
    hold_metrics = pipe.evaluate(X[te_idx], y[te_idx])
    print(f"  Hold-out AUC: {hold_metrics['auc_roc']:.4f}")
    print(f"  Hold-out F1:  {hold_metrics['f1']:.4f}")
    print(f"  vs Benchmark: {hold_metrics['vs_benchmark']['status']} "
          f"(delta={hold_metrics['vs_benchmark']['delta']:+.4f})")

    print("\n[7] Importancia de features (permutation, subset rapido)")
    feat_imp = pipe.feature_importance(X[tr_idx], y[tr_idx])
    print(f"  Top 5 features mais discriminativas:")
    for i in range(min(5, len(feat_imp['feature_names']))):
        print(f"    {i+1}. {feat_imp['feature_names'][i]:<30} "
              f"imp={feat_imp['importances_mean'][i]:.4f} "
              f"+/-{feat_imp['importances_std'][i]:.4f}")

    print("\nModulo 5 pronto.")
    print("=" * 65)
