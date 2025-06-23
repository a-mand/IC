
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import os

def load_features(csv_path):
    """Carrega características e rótulos de um arquivo CSV."""
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y

def train_model(X_train, y_train, model_type='rf'):
    """
    Treina o modelo com uma busca de hiperparâmetros mais ampla (GridSearchCV).
    """
    print(f"\nTreinando modelo {model_type.upper()} com OTIMIZAÇÃO DE HIPERPARÂMETROS...")

    # --- MUDANÇA APLICADA AQUI ---
    # Expandimos a grade de busca para encontrar uma combinação melhor de parâmetros.
    if model_type == 'rf':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced']
        }
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'svm':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf'],
            'class_weight': ['balanced']
        }
        model = SVC(probability=True, random_state=42)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print("\n--- Otimização Concluída ---")
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    print(f"Melhor score AUC na validação cruzada: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina um modelo com características pré-extraídas.")
    parser.add_argument('--train_features', required=True, help='Caminho para o CSV com as características de treino.')
    parser.add_argument('--model_type', default='rf', choices=['rf', 'svm'])
    parser.add_argument('--output_dir', default='artifacts')
    args = parser.parse_args()

    X_train, y_train = load_features(args.train_features)

    print(f"Formato original do treino: {X_train.shape}")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Formato do treino após SMOTE: {X_train_res.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)

    model = train_model(X_train_scaled, y_train_res, model_type=args.model_type)

    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.output_dir, f'{args.model_type}_model.joblib'))
    joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.joblib'))

    print(f"\nModelo e scaler salvos em '{args.output_dir}'.")
