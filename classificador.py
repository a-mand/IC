import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, auc, precision_recall_curve)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image
import glob

def load_data(data_dir):
    """Carrega imagens e rótulos do dataset"""
    # Carrega os CSVs assumindo que MEL é a segunda coluna (índice 1)
    train_df = pd.read_csv(os.path.join(data_dir, 'treino.csv'), header=None)
    val_df = pd.read_csv(os.path.join(data_dir, 'validacao.csv'), header=None)
    test_df = pd.read_csv(os.path.join(data_dir, 'teste.csv'), header=None)

    # Função para carregar imagens
    def load_images(folder, df):
        images = []
        labels = []
        for img_path in glob.glob(os.path.join(data_dir, folder, '*.jpg')):
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            if img_id in df[0].values:  # Primeira coluna tem os IDs
                img = Image.open(img_path).resize((128, 128)).convert('RGB')
                images.append(np.array(img).flatten())
                # Pega o valor MEL (coluna 1) como label
                labels.append(df[df[0] == img_id][1].values[0])  
        return np.array(images), np.array(labels)

    # Carrega cada conjunto
    X_train, y_train = load_images('treino_images', train_df)
    X_val, y_val = load_images('validacao_images', val_df)
    X_test, y_test = load_images('teste_images', test_df)

    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_data(X_train, X_test):
    """Normaliza os dados"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train, model_type='rf', cv_folds=5):
    """Treina o modelo com seleção de hiperparâmetros"""
    if model_type == 'rf':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None]
        }
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'svm':
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
            'class_weight': ['balanced', None]
        }
        model = SVC(probability=True, random_state=42)
    
    # Usa GridSearchCV para seleção de hiperparâmetros
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Melhores parâmetros para {model_type}: {grid_search.best_params_}")
    print(f"Melhor score AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Avalia o modelo e gera métricas e gráficos"""
    # Previsões
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Não-MEL', 'MEL'], 
                yticklabels=['Não-MEL', 'MEL'])
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig('pr_curve.png')
    plt.close()
    
    # Relatório de classificação
    report = classification_report(y_test, y_pred, target_names=['Não-MEL', 'MEL'])
    print(report)
    
    return {
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'classification_report': report
    }

def save_results(model, scaler, results, model_type, run_id):
    """Salva os resultados e modelos"""
    # Cria diretório para a execução
    os.makedirs(f'run_{run_id}', exist_ok=True)
    
    # Salva o modelo e scaler
    joblib.dump(model, f'run_{run_id}/{model_type}_model.joblib')
    joblib.dump(scaler, f'run_{run_id}/scaler.joblib')
    
    # Salva os resultados
    with open(f'run_{run_id}/results.txt', 'w') as f:
        f.write(f"Modelo: {model_type}\n")
        f.write(f"Run ID: {run_id}\n\n")
        f.write("Matriz de Confusão:\n")
        f.write(np.array2string(results['confusion_matrix']) + "\n\n")
        f.write(f"AUC-ROC: {results['roc_auc']:.4f}\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(results['classification_report'])

def main(data_dir, model_type='rf', run_id='001'):
    """Função principal"""
    # 1. Carregar dados
    print("Carregando dados...")
    X, y, labels_df = load_data(data_dir)
    
    # Verifica balanceamento de classes
    class_counts = labels_df['target'].value_counts()
    print("\nDistribuição de classes:")
    print(class_counts)
    print(f"\nProporção MEL: {class_counts[1]/sum(class_counts):.2%}")
    
    # 2. Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Pré-processamento
    print("\nPré-processando dados...")
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
    
    # 4. Treinar modelo
    print("\nTreinando modelo...")
    model = train_model(X_train_scaled, y_train, model_type)
    
    # 5. Avaliar modelo
    print("\nAvaliando modelo...")
    results = evaluate_model(model, X_test_scaled, y_test)
    
    # 6. Salvar resultados
    print("\nSalvando resultados...")
    save_results(model, scaler, results, model_type, run_id)
    
    print("\nProcesso concluído!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Caminho para a pasta dataset')
    parser.add_argument('--model_type', default='rf', choices=['rf', 'svm'])
    parser.add_argument('--run_id', default='001')
    args = parser.parse_args()
    
    # Carrega os dados (agora recebendo 6 valores)
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.data_dir)
    
    # Continua com o processamento
    print("\nPré-processando dados...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print("\nTreinando modelo...")
    model = train_model(X_train, y_train, args.model_type)
    
    print("\nAvaliando modelo...")
    print("\nValidação:")
    val_results = evaluate_model(model, X_val, y_val)
    print("\nTeste:")
    test_results = evaluate_model(model, X_test, y_test)
    
    print("\nSalvando resultados...")
    os.makedirs(f'resultados_{args.run_id}', exist_ok=True)
    joblib.dump(model, f'resultados_{args.run_id}/modelo.joblib')
    joblib.dump(scaler, f'resultados_{args.run_id}/scaler.joblib')
    with open(f'resultados_{args.run_id}/relatorio.txt', 'w') as f:
        f.write(f"Modelo: {args.model_type}\n\nValidação:\n{val_results}\n\nTeste:\n{test_results}")
    
    print("\nProcesso concluído!")