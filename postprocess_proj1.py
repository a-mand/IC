
import argparse
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

def load_features(csv_path):
    """Carrega características e rótulos de um arquivo CSV."""
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ajusta o limiar de decisão usando o conjunto de validação.")
    parser.add_argument('--val_features', required=True, help='Caminho para o CSV com as características de validação.')
    parser.add_argument('--model_path', required=True, help='Caminho para o modelo treinado.')
    parser.add_argument('--scaler_path', required=True, help='Caminho para o scaler treinado.')
    parser.add_argument('--output_dir', default='results', help="Diretório para salvar o gráfico de limiares.")
    args = parser.parse_args()

    # Carrega modelo, scaler e dados de validação
    print("Carregando modelo, scaler e dados de validação...")
    model = joblib.load(args.model_path)
    scaler = joblib.load(args.scaler_path)
    X_val, y_val = load_features(args.val_features)

    # Normaliza as características de validação
    X_val_scaled = scaler.transform(X_val)

    # Obtém as probabilidades para a classe positiva (MEL)
    probabilities = model.predict_proba(X_val_scaled)[:, 1]

    # Testa múltiplos limiares
    thresholds = np.arange(0.05, 0.51, 0.01)
    scores = []

    print("Testando diferentes limiares...")
    for thresh in thresholds:
        # Aplica o limiar para obter as predições
        preds = (probabilities >= thresh).astype(int)

        # Calcula precisão, recall e f1-score para a classe MEL (label=1)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, preds, average=None, labels=[0, 1])

        # Armazena os scores da classe MEL (índice 1)
        scores.append([thresh, precision[1], recall[1], f1[1]])

    scores_df = pd.DataFrame(scores, columns=['Threshold', 'Precision', 'Recall', 'F1-Score'])

    # Encontra o melhor limiar com base no maior F1-Score
    best_score = scores_df.loc[scores_df['F1-Score'].idxmax()]
    print("\n--- Resultados do Ajuste de Limiar ---")
    print(f"Melhor Limiar (baseado no F1-Score): {best_score['Threshold']:.2f}")
    print(f"  - Precisão neste limiar: {best_score['Precision']:.2f}")
    print(f"  - Recall neste limiar: {best_score['Recall']:.2f}")
    print(f"  - F1-Score neste limiar: {best_score['F1-Score']:.2f}")

    # Gera o gráfico
    os.makedirs(args.output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(scores_df['Threshold'], scores_df['Precision'], 'b--', label='Precision')
    plt.plot(scores_df['Threshold'], scores_df['Recall'], 'g-', label='Recall')
    plt.plot(scores_df['Threshold'], scores_df['F1-Score'], 'r-', lw=2, label='F1-Score')
    plt.axvline(x=best_score['Threshold'], color='purple', linestyle='--', label=f'Melhor Limiar ({best_score["Threshold"]:.2f})')
    plt.title('Precision, Recall e F1-Score vs. Limiar de Decisão')
    plt.xlabel('Limiar de Decisão')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(args.output_dir, 'threshold_tuning_plot.png')
    plt.savefig(output_path)
    print(f"\nGráfico do ajuste de limiar salvo em: {output_path}")
