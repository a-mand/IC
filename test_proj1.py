
import argparse
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_features(csv_path):
    """Carrega características e rótulos de um arquivo CSV."""
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y

def evaluate_model(model, X_test, y_test, scaler, results_dir, threshold=0.5):
    """Avalia o modelo usando um limiar de decisão específico."""
    print(f"\nAvaliando modelo no conjunto de teste com limiar = {threshold:.2f}...")
    X_test_scaled = scaler.transform(X_test)

    # Usa o limiar para fazer as predições
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    report = classification_report(y_test, y_pred, target_names=['Não-MEL', 'MEL'])
    auc_score = roc_auc_score(y_test, y_proba)

    print(f"AUC Score: {auc_score:.4f}")
    print("Relatório de Classificação:\n", report)

    os.makedirs(results_dir, exist_ok=True)

    # Salva relatório
    with open(os.path.join(results_dir, 'test_report.txt'), 'w') as f:
        f.write(f"Limiar de Decisão Usado: {threshold:.2f}\n")
        f.write(f"AUC Score: {auc_score:.4f}\n\n")
        f.write(report)

    # Salva Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não-MEL', 'MEL'], yticklabels=['Não-MEL', 'MEL'])
    plt.title(f'Matriz de Confusão (Teste, Limiar={threshold:.2f})')
    plt.savefig(os.path.join(results_dir, 'test_confusion_matrix.png'))
    plt.close()

    print(f"Resultados salvos em '{results_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testa um modelo com características pré-extraídas.")
    parser.add_argument('--test_features', required=True, help='Caminho para o CSV com as características de teste.')
    parser.add_argument('--model_path', required=True, help='Caminho para o modelo treinado.')
    parser.add_argument('--scaler_path', required=True, help='Caminho para o scaler treinado.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Limiar de decisão para a classificação.')
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    scaler = joblib.load(args.scaler_path)

    X_test, y_test = load_features(args.test_features)

    evaluate_model(model, X_test, y_test, scaler, results_dir=args.results_dir, threshold=args.threshold)
