import os
import json
import pandas as pd

def consolidar_todos_os_resultados():
    pasta_resultados = "dados/features/resultados_dissertacao"
    
    # 1. Dados de k=25 (Extraídos diretamente do JSON unificado)
    caminho_json = "dados/features/resultados_dissertacao/all_features_result_k=25/tfs_results_all_features.json"
    
    dados_consolidados = []
    
    print("-" * 60)
    print("CONSOLIDANDO RESULTADOS DE PIPELINES (k=25, 16, 12, 08)")
    print("-" * 60)
    
    if os.path.exists(caminho_json):
        print(f"✅ Lendo dados de k=25 a partir de: {caminho_json}")
        with open(caminho_json, 'r') as f:
            js = json.load(f)
            
        # ML k=25
        if 'ml' in js:
            for mod, met in js['ml'].items():
                dados_consolidados.append({
                    'k (Features)': 25, 'Categoria': 'ML Clássico', 'Modelo': mod.upper(),
                    'AUC-ROC (Média ± Std)': f"{met['auc_roc']['mean']:.4f} ± {met['auc_roc']['std']:.4f}",
                    'F1-Score': f"{met['f1']['mean']:.4f}", 'Sensibilidade': f"{met['sensitivity']['mean']:.4f}",
                    'Especificidade': f"{met['specificity']['mean']:.4f}", 'vs Benchmark': met.get('vs_benchmark', {}).get('status', 'ABAIXO')
                })
        # DL k=25
        if 'dl' in js:
            for mod, met in js['dl'].items():
                dados_consolidados.append({
                    'k (Features)': 25, 'Categoria': 'Deep Learning', 'Modelo': mod.upper(),
                    'AUC-ROC (Média ± Std)': f"{met['auc_roc']['mean']:.4f} ± {met['auc_roc']['std']:.4f}",
                    'F1-Score': f"{met['f1']['mean']:.4f}", 'Sensibilidade': f"{met['sensitivity']['mean']:.4f}",
                    'Especificidade': f"{met['specificity']['mean']:.4f}", 'vs Benchmark': met.get('vs_benchmark', {}).get('status', 'ABAIXO')
                })
    else:
        print(f"⚠️ Aviso: Arquivo {caminho_json} não encontrado. k=25 ficará ausente.")

    # 2. Mapeamento das tabelas parciais de k de outros experimentos
    tabelas_parciais = [
        ('dados/features/resultados_dissertacao/resultado k=16/tabela_comparativa_modelos_k=16.csv', 16),
        ('dados/features/resultados_dissertacao/resultado k=12/tabela_comparativa_modelos_k=12.csv', 12),
        ('dados/features/resultados_dissertacao/tabela_comparativa_modelos_k=08.csv', 8)
    ]
    
    for nome_arq, k_val in tabelas_parciais:
        if os.path.exists(nome_arq):
            print(f"✅ Lendo dados de k={k_val:02d} a partir de: {nome_arq}")
            try:
                # Lê o arquivo parcial (usando separador ponto e vírgula padrão do pipeline)
                df_parcial = pd.read_csv(nome_arq, sep=';')
                for _, row in df_parcial.iterrows():
                    dados_consolidados.append({
                        'k (Features)': k_val,
                        'Categoria': row['Categoria'],
                        'Modelo': str(row['Modelo']).upper(),
                        'AUC-ROC (Média ± Std)': row['AUC-ROC (Média ± Std)'],
                        'F1-Score': f"{float(row['F1-Score']):.4f}",
                        'Sensibilidade': f"{float(row['Sensibilidade']):.4f}",
                        'Especificidade': f"{float(row['Especificidade']):.4f}",
                        'vs Benchmark': row['vs_benchmark'] if 'vs_benchmark' in row else row.get('vs Benchmark', 'ABAIXO')
                    })
            except Exception as e:
                print(f"❌ Erro ao processar o arquivo {nome_arq}: {e}")
        else:
            print(f"⚠️ Aviso: Arquivo {nome_arq} não encontrado no diretório atual.")

    if not dados_consolidados:
        print("❌ Nenhum dado foi encontrado para consolidação.")
        return

    # 3. Criando o DataFrame final ordenado por k decrescente
    df_final = pd.DataFrame(dados_consolidados)
    df_final = df_final.sort_values(by=['k (Features)', 'Categoria'], ascending=[False, False])
    
    # 4. Salvando na pasta correta esperada pelos gráficos
    os.makedirs(pasta_resultados, exist_ok=True)
    caminho_saida = os.path.join(pasta_resultados, "tabela_comparativa_modelos.csv")
    df_final.to_csv(caminho_saida, index=False, sep=';')
    
    print("\n" + "=" * 60)
    print(f"🏆 TABELA UNIFICADA GERADA COM SUCESSO!")
    print(f"💾 Salva em: {caminho_saida}")
    print("=" * 60)
    print(df_final.to_string(index=False))

if __name__ == "__main__":
    consolidar_todos_os_resultados()