import glob
import os
from pipeline_dl import run_experiment

# 1. Mapeamento dos caminhos absolutos dos áudios
caminho_hc = glob.glob("dados/processed_data/HC_AH/*.wav")
caminho_dp = glob.glob("dados/processed_data/PD_AH/*.wav")

# 2. Configure e rode o pipeline unificado
# Isso fará o extract_blocks AB/CD para todos os arquivos, 
# salvará a matriz numpy "tfs_features.npz" e executará o ML e DL.
resultados = run_experiment(
    dp_files=caminho_dp,
    hc_files=caminho_hc,
    output_dir="dados/features/resultados_dissertacao", 
    run_ml=True,            # Ativa os classificadores clássicos e o SoftVotingEnsemble
    run_dl=True,            # Ativa a TFSHybridNet e TFSTransformer (Requer PyTorch)
    n_epochs_dl=80,
    q_attn=1.3,             # Parâmetro q recomendado para atenção esparsa
    q_loss=1.3,             # Parâmetro q para a Tsallis Loss
    verbose=True
)

print("Finalizado! Verifique a pasta 'resultados_dissertacao' para o JSON e as features extraídas.")