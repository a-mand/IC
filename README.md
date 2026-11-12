# Classificação de Melanoma com Random Forest e SVM

Sistema de classificação automática em Python que detecta **melanoma** (rótulo `MEL`) em imagens de lesões de pele. O problema é tratado como **classificação binária**: melanoma ou não melanoma.

Projeto da disciplina de Inteligência Computacional da UFPA.

## Dataset
[ISIC Challenge 2018, Task 3](https://challenge.isic-archive.com/landing/2018/):
- **Treino:** 10.015 imagens com CSV de rótulos.
- **Validação:** usada para ajustar o limiar de decisão.
- **Teste:** 1.512 imagens para a avaliação final.

## Pipeline
1. **Pré-processamento:** extração de características das imagens (CSV em `features/`).
2. **Treino:** Random Forest ou SVM, com `GridSearchCV` (validação cruzada estratificada de 5 folds) e balanceamento de classes com **SMOTE**.
3. **Pós-processamento:** ajuste do limiar de decisão para maximizar o F1-Score da classe `MEL`.
4. **Teste:** avaliação final com relatório de classificação e matriz de confusão.

## Como executar
```bash
git clone https://github.com/a-mand/IC.git
cd IC
pip install -r requirements.txt
```

```bash
# 1. Extração de características
python preprocess_proj1.py --input_dir sample_data --output_dir features

# 2. Treino (use "rf" ou "svm")
python train_proj1.py --train_features features/train_features.csv --model_type "rf" --output_dir artifacts

# 3. Ajuste do limiar na validação
python postprocess_proj1.py --val_features features/validation_features.csv --model_path artifacts/rf_model.joblib --scaler_path artifacts/scaler.joblib --output_dir results

# 4. Teste final
python test_proj1.py --test_features features/test_features.csv --model_path artifacts/rf_model.joblib --scaler_path artifacts/scaler.joblib --results_dir results --threshold 0.45
```
Para o SVM, troque `rf` por `svm` e use `--threshold 0.28` (diretórios `results_svm*`).

## Resultados
Como as classes são desbalanceadas, a acurácia sozinha engana. Avaliamos com AUC, precisão, recall e F1 da classe `MEL`.

| Modelo | Melhores hiperparâmetros | AUC (CV 5-fold) | AUC (teste) | F1 `MEL` | Recall `MEL` | Precisão `MEL` | Acurácia |
|---|---|---|---|---|---|---|---|
| Random Forest | `class_weight=balanced`, `max_depth=20`, `min_samples_leaf=5`, `min_samples_split=5`, `n_estimators=200` | 0,9706 | 0,8599 | 0,47 | 0,67 | 0,36 | 0,83 |
| SVM | `C=10`, `kernel=rbf`, `gamma=0.1`, `class_weight=balanced` | 0,9717 | 0,7560 | 0,33 | 0,33 | 0,32 | 0,83 |

<!-- TODO: confirmar o limiar do RF no teste (README antigo cita 0.35 no resultado e 0.45 no comando). -->

O Random Forest generaliza melhor: a AUC cai de 0,97 (validação cruzada) para 0,86 no teste, o que indica diferença entre os dados de treino e de teste.

## Tecnologias
Python, scikit-learn, pandas, NumPy, OpenCV, scikit-image, imbalanced-learn (SMOTE), matplotlib, seaborn, joblib.

## Equipe
Amanda Lopes, Filipe Corrêa, Giovanna Cunha e Giovana Nascimento.
