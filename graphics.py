import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def encontrar_tabela_consolidada():
    """Busca o arquivo tabela_comparativa_modelos.csv de forma fail-safe."""
    candidatos = [
        "dados/features/resultados_dissertacao/tabela_comparativa_modelos.csv",
        "tabela_comparativa_modelos.csv"
    ]
    for c in candidatos:
        if os.path.exists(c):
            return c
            
    # Procura recursiva em último caso
    for raiz, dirs, arquivos in os.walk('.'):
        if '.venv' in raiz or '__pycache__' in raiz:
            continue
        if "tabela_comparativa_modelos.csv" in arquivos:
            return os.path.join(raiz, "tabela_comparativa_modelos.csv")
    return None

def parse_metrica(valor_bruto):
    """
    Extrai média e desvio padrão de strings complexas como '0.6964 ± 0.1289'.
    Retorna uma tupla (média, desvio).
    """
    if pd.isna(valor_bruto):
        return 0.0, 0.0
    val_str = str(valor_bruto).strip()
    if '±' in val_str:
        parts = val_str.split('±')
        try:
            mean = float(parts[0].strip())
            std = float(parts[1].strip())
            return mean, std
        except ValueError:
            return 0.0, 0.0
    try:
        return float(val_str), 0.0
    except ValueError:
        return 0.0, 0.0

def gerar_4_graficos_distintos():
    pasta_resultados = "dados/features/resultados_dissertacao"
    caminho_csv = encontrar_tabela_consolidada()
    
    print("-" * 60)
    print("GERANDO GRÁFICOS INDEPENDENTES POR CARACTERÍSTICA")
    print("-" * 60)
    
    if not caminho_csv:
        print("❌ Erro: Arquivo unificado 'tabela_comparativa_modelos.csv' não encontrado.")
        print("Garanta que você rodou o script 'consolidar_tabelas.py' primeiro.")
        return

    print(f"📖 Lendo dados a partir de: {caminho_csv}")
    df = pd.read_csv(caminho_csv, sep=';')
    df['Modelo'] = df['Modelo'].str.upper().str.strip()
    
    # Mapeamento de colunas para títulos formais e nomes de arquivos
    metricas_mapeamento = {
        'AUC-ROC (Média ± Std)': ('AUC-ROC', 'Plot_AUC', True),
        'F1-Score': ('F1-Score', 'Plot_F1', False),
        'Sensibilidade': ('Sensibilidade (Recall DP)', 'Plot_Sensibilidade', False),
        'Especificidade': ('Especificidade (Recall HC)', 'Plot_Especificidade', False)
    }
    
    modelos_ordem = ['SVM', 'ENSEMBLE', 'TRANSFORMER', 'HYBRID']
    k_ordem = [8, 12, 16, 25]
    
    cores_modelos = {
        'SVM': '#1f77b4',
        'ENSEMBLE': '#d62728',
        'TRANSFORMER': '#2ca02c',
        'HYBRID': '#9467bd'
    }
    
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    })
    
    for col_csv, (nome_formal, sufixo_arquivo, usa_std) in metricas_mapeamento.items():
        print(f"Gerando gráfico para: {nome_formal}...")
        
        fig, ax = plt.subplots(figsize=(11, 6))
        
        n_modelos = len(modelos_ordem)
        largura_barra = 0.18
        indices_k = np.arange(len(k_ordem))
        
        for i, modelo in enumerate(modelos_ordem):
            valores_media = []
            valores_std = []
            
            for k in k_ordem:
                sub_df = df[(df['k (Features)'] == k) & (df['Modelo'] == modelo)]
                if not sub_df.empty:
                    med, std = parse_metrica(sub_df[col_csv].values[0])
                    valores_media.append(med)
                    valores_std.append(std)
                else:
                    valores_media.append(0.0)
                    valores_std.append(0.0)
            
            deslocamento = (i - (n_modelos - 1) / 2) * largura_barra
            
            if usa_std:
                barras = ax.bar(indices_k + deslocamento, valores_media, largura_barra, 
                                yerr=valores_std, capsize=4, error_kw={'elinewidth':1.2, 'ecolor':'#404040'},
                                label=modelo if modelo != 'HYBRID' else 'Hybrid (DL)', 
                                color=cores_modelos[modelo], edgecolor='black', alpha=0.85)
            else:
                barras = ax.bar(indices_k + deslocamento, valores_media, largura_barra, 
                                label=modelo if modelo != 'HYBRID' else 'Hybrid (DL)', 
                                color=cores_modelos[modelo], edgecolor='black', alpha=0.85)
            
            # Anotação inteligente: posiciona o texto acima da barra de erro se ela existir
            for j, barra in enumerate(barras):
                height = barra.get_height()
                if height > 0.01:
                    offset_y = valores_std[j] if usa_std else 0.0
                    ax.annotate(f'{height:.3f}',
                                xy=(barra.get_x() + barra.get_width() / 2, height + offset_y),
                                xytext=(0, 4 if not usa_std else 8),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
        # Configurações estéticas
        ax.set_ylabel('Score (0.0 a 1.0)')
        ax.set_xlabel('Quantidade de Features Selecionadas (k)')
        ax.set_title(f'Análise Comparativa de Desempenho: {nome_formal}', pad=15, fontweight='bold')
        ax.set_xticks(indices_k)
        ax.set_xticklabels([f'k = {k}' for k in k_ordem])
        ax.set_ylim(0, 1.15)  # Espaço para o texto acima da barra de erro
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Legenda posicionada estritamente na parte exterior direita do gráfico
        ax.legend(title='Modelos', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        
        # Salva o gráfico
        os.makedirs(pasta_resultados, exist_ok=True)
        nome_arquivo_oficial = os.path.join(pasta_resultados, f"{sufixo_arquivo}.png")
        nome_arquivo_local = f"{sufixo_arquivo}.png"
        
        plt.savefig(nome_arquivo_oficial, dpi=300, bbox_inches='tight')
        plt.savefig(nome_arquivo_local, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Gráfico exportado com sucesso em: {nome_arquivo_oficial}")

if __name__ == "__main__":
    gerar_4_graficos_distintos()