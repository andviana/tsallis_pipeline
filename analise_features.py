import numpy as np
import pandas as pd
from pipeline_ml import TFSPipeline

import numpy as np
import pandas as pd
import json
import os
from pipeline_ml import TFSPipeline

def analisar_resultados_tfs():
    # Caminhos dos arquivos gerados pelo pipeline
    pasta_resultados = "dados/features/resultados_dissertacao"
    caminho_npz = os.path.join(pasta_resultados, "tfs_features.npz")
    caminho_csv = os.path.join(pasta_resultados, "tfs_features_tabela.csv")
    caminho_json = os.path.join(pasta_resultados, "tfs_results.json")
    
    print("-" * 60)
    print("1. CONVERTENDO .NPZ PARA .CSV")
    print("-" * 60)
    try:
        data = np.load(caminho_npz)
        X = data['X']
        y = data['y']
        feature_names = data['feature_names']
        
        # Cria o DataFrame e insere o rótulo (Target) no final
        df = pd.DataFrame(X, columns=feature_names)
        df['TARGET_DP'] = y 
        
        # Salva o CSV
        df.to_csv(caminho_csv, index=False, sep=';', decimal=',')
        print(f"✅ Matriz de features salva com sucesso em:\n   {caminho_csv}")
        print(f"   Dimensões: {df.shape[0]} amostras x {df.shape[1]} colunas.")
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {caminho_npz}. Rode o main.py primeiro.")
        return

    print("\n" + "-" * 60)
    print("2. SELEÇÃO DE FEATURES (ANOVA) PARA ML")
    print("-" * 60)
    
    # Instancia o pipeline idêntico ao usado no run_experiment
    pipe = TFSPipeline(classifier='ensemble', k_features=25, selection_method='anova')
    
    # O fit calcula a imputação, z-score, variância e o SelectKBest
    pipe.fit(X, y, feature_names=list(feature_names))
    
    features_selecionadas = pipe.preprocessor_.selected_features_
    print(f"✅ {len(features_selecionadas)} features selecionadas pelo ANOVA para SVM/Ensemble:\n")
    print(", ".join(features_selecionadas))

    print("\n" + "-" * 60)
    print("3. RANKING DE IMPORTÂNCIA DAS FEATURES")
    print("-" * 60)
    
    # O TFSEvaluator faz o Permutation Importance baralhando cada feature
    print("Calculando o impacto de cada feature no AUC-ROC (Permutation Test)... aguarde.")
    ranking = pipe.feature_importance(X, y, feature_names=list(feature_names))
    
    print("\n🏆 Top 10 Features Mais Discriminativas:")
    for i in range(10):
        nome = ranking['feature_names'][i]
        imp_mean = ranking['importances_mean'][i]
        imp_std = ranking['importances_std'][i]
        print(f"   {i+1}º | {nome:<25} | Queda no AUC: {imp_mean:.4f} ± {imp_std:.4f}")

    print("\n" + "-" * 60)
    print("4. TABELA COMPARATIVA DE DESEMPENHO DOS MODELOS")
    print("-" * 60)
    
    try:
        with open(caminho_json, 'r') as f:
            resultados_json = json.load(f)
            
        linhas_tabela = []
        
        # Extraindo resultados de Machine Learning Clássico
        if 'ml' in resultados_json:
            for nome_modelo, metricas in resultados_json['ml'].items():
                linhas_tabela.append({
                    'Categoria': 'ML Clássico',
                    'Modelo': nome_modelo.upper(),
                    'AUC-ROC (Média ± Std)': f"{metricas['auc_roc']['mean']:.4f} ± {metricas['auc_roc']['std']:.4f}",
                    'F1-Score': f"{metricas['f1']['mean']:.4f}",
                    'Sensibilidade': f"{metricas['sensitivity']['mean']:.4f}",
                    'Especificidade': f"{metricas['specificity']['mean']:.4f}",
                    'vs Benchmark': metricas.get('vs_benchmark', {}).get('status', 'N/A')
                })
                
        # Extraindo resultados de Deep Learning
        if 'dl' in resultados_json:
            for nome_modelo, metricas in resultados_json['dl'].items():
                if 'error' in metricas:
                    linhas_tabela.append({
                        'Categoria': 'Deep Learning',
                        'Modelo': nome_modelo.capitalize(),
                        'AUC-ROC (Média ± Std)': 'ERRO',
                        'F1-Score': '-',
                        'Sensibilidade': '-',
                        'Especificidade': '-',
                        'vs Benchmark': '-'
                    })
                else:
                    linhas_tabela.append({
                        'Categoria': 'Deep Learning',
                        'Modelo': nome_modelo.capitalize(),
                        'AUC-ROC (Média ± Std)': f"{metricas['auc_roc']['mean']:.4f} ± {metricas['auc_roc']['std']:.4f}",
                        'F1-Score': f"{metricas['f1']['mean']:.4f}",
                        'Sensibilidade': f"{metricas['sensitivity']['mean']:.4f}",
                        'Especificidade': f"{metricas['specificity']['mean']:.4f}",
                        'vs Benchmark': metricas.get('vs_benchmark', {}).get('status', 'N/A')
                    })
        
        # Criando e exibindo a tabela formatada
        df_modelos = pd.DataFrame(linhas_tabela)
        
        # Ajusta a exibição do pandas para não cortar colunas no terminal
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        
        print(df_modelos.to_string(index=False))
        
        # Opcional: Salvar a tabela comparativa em CSV também
        caminho_tabela_modelos = os.path.join(pasta_resultados, "tabela_comparativa_modelos.csv")
        df_modelos.to_csv(caminho_tabela_modelos, index=False, sep=';')
        print(f"\n✅ Tabela salva em: {caminho_tabela_modelos}")

    except FileNotFoundError:
        print(f"❌ Arquivo {caminho_json} não encontrado. Execute o pipeline principal primeiro.")
    except Exception as e:
        print(f"❌ Erro ao ler os resultados JSON: {e}")

if __name__ == "__main__":
    analisar_resultados_tfs()

    


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def gerar_visualizacoes_top3():
    # Caminho do arquivo gerado no passo anterior
    caminho_csv = "dados/features/resultados_dissertacao/tfs_features_tabela.csv"
    pasta_saida = "dados/features/resultados_dissertacao/"

    print("-" * 50)
    print("GERANDO GRÁFICOS DAS TOP 3 FEATURES")
    print("-" * 50)

    try:
        # Lendo o CSV. Atenção ao separador e decimal definidos anteriormente
        df = pd.read_csv(caminho_csv, sep=';', decimal=',')
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {caminho_csv}. Rode o analise_features.py primeiro.")
        return

    # Definindo as Top 3 features do ranking
    top3_features = ['B_b1_q0.7', 'B_b3_q2.0', 'C_Sq_dF0_q1.3']
    target_col = 'TARGET_DP'

    # Verifica se as colunas estão no dataset para evitar erros
    for col in top3_features + [target_col]:
        if col not in df.columns:
            print(f"❌ Erro: Coluna {col} não encontrada no dataset.")
            return

    # Mapeando os rótulos para legendas mais bonitas nos gráficos
    df['Classe'] = df[target_col].map({0: 'Controle Saudável (HC)', 1: 'Parkinson (DP)'})
    
    # Definindo cores (Verde para HC, Vermelho/Bordô para DP)
    cores = {'Controle Saudável (HC)': '#2ca02c', 'Parkinson (DP)': '#d62728'}

    # Configuração de estilo do Seaborn
    sns.set_theme(style="whitegrid", context="talk")

    # ---------------------------------------------------------
    # 1. GERANDO BOXPLOTS
    # ---------------------------------------------------------
    print("Gerando Boxplots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Distribuição das Top 3 Features: DP vs HC', fontsize=18, fontweight='bold', y=1.05)

    for i, feature in enumerate(top3_features):
        # Boxplot principal
        sns.boxplot(x='Classe', y=feature, data=df, ax=axes[i], hue='Classe', palette=cores, legend=False, width=0.5, boxprops=dict(alpha=0.8))
        # Adicionando os pontos reais (stripplot) para ver a densidade dos dados
        sns.stripplot(x='Classe', y=feature, data=df, ax=axes[i], color='black', alpha=0.4, jitter=True, size=5)
        
        axes[i].set_title(f'{feature}', fontsize=14)
        axes[i].set_ylabel('Valor da Feature' if i == 0 else '')
        axes[i].set_xlabel('')

    plt.tight_layout()
    caminho_boxplot = os.path.join(pasta_saida, 'top3_boxplots.png')
    plt.savefig(caminho_boxplot, dpi=300, bbox_inches='tight')
    print(f"✅ Boxplots salvos em: {caminho_boxplot}")

    # ---------------------------------------------------------
    # 2. GERANDO GRÁFICO DE DISPERSÃO (PAIRPLOT)
    # ---------------------------------------------------------
    print("Gerando Gráfico de Dispersão (Pairplot)... aguarde.")
    
    # Criando o pairplot apenas com as features selecionadas e a classe
    pairplot = sns.pairplot(
        df[top3_features + ['Classe']], 
        hue='Classe', 
        palette=cores, 
        diag_kind='kde', # Mostra a curva de densidade na diagonal
        markers=["o", "s"], 
        plot_kws={'alpha': 0.7, 's': 60, 'edgecolor': 'k'}
    )
    
    pairplot.fig.suptitle('Dispersão e Densidade das Top 3 Features', y=1.02, fontsize=18, fontweight='bold')
    
    caminho_dispersao = os.path.join(pasta_saida, 'top3_dispersao.png')
    pairplot.savefig(caminho_dispersao, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico de dispersão salvo em: {caminho_dispersao}")







if __name__ == "__main__":
    analisar_resultados_tfs()
    gerar_visualizacoes_top3()