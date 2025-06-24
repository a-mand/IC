# Projeto de Classificação de Melanoma

## 1. Objetivo

Este projeto tem como objetivo desenvolver um sistema de classificação automática em Python para detectar câncer de pele do tipo melanoma (rótulo "MEL"). O problema é tratado como uma classificação binária: é melanoma ou não é melanoma.

## 2. Dataset

Utilizamos a Task 3 do ISIC Challenge 2018, que consiste em:
* **Dados de Treino:** 10015 imagens e um arquivo CSV correspondente com os rótulos.
* **Dados de Validação:** Conjunto de dados para ajuste do limiar de decisão e avaliação do modelo.
* **Dados de Teste:** 1512 imagens para avaliação final do modelo.

## 3. Estrutura do Projeto

O projeto é dividido em quatro etapas principais, executadas por scripts Python "puros" (não Jupyter Notebooks):

* `Pré-processamento`: Extração de características das imagens.
* `Treino`: Treinamento do modelo, incluindo otimização de hiperparâmetros.
* `Pós-processamento`: Ajuste do limiar de decisão e geração de gráficos de avaliação.
* `Teste`: Avaliação final do modelo no conjunto de teste.

## 4. Instalação do Ambiente

Para configurar o ambiente, siga os passos abaixo:

1.  **Clonar o repositório do dataset:**
    ```bash
    git clone -b main [https://github.com/a-mand/IC.git](https://github.com/a-mand/IC.git)
    ```
    Este comando irá baixar os dados necessários para o projeto na pasta `IC/dataset`.

2.  **Instalar as dependências:**
    Recomendamos usar `pipreqs` para gerar um arquivo `requirements.txt` minimalista.
    ```bash
    pip install pipreqs
    pipreqs --force --encoding=utf-8  . # Execute na raiz do seu projeto para gerar o requirements.txt
    pip install -r requirements.txt
    ```

## 5. Partes Principais do Código
Em ordem de execução do modelo:

1.  **Pré-processamento (Extração de Características):**
    Este passo extrai características das imagens e as salva em arquivos CSV na pasta `features/`. Basta executar o comando abaixo:
    ``` bash
    python .\preprocess_proj1.py --input_dir IC/sample_data--output_dir features
    ```

2.  **Treinamento do Modelo (Random Forest):**
    Este comando treina um modelo Random Forest, incluindo a otimização de hiperparâmetros com GridSearchCV e balanceamento de classes com SMOTE. O modelo treinado e o scaler são salvos na pasta `artifacts/`. Em model_type, você pode escolher entre "rf" (Random Forest) ou "svm" (SVM). O comando para treinar o modelo Random Forest é:
    ``` bash
    python .\train_proj1.py --train_features features/train_features.csv --model_type "rf" --output_dir artifacts
    ```

3.  **Treinamento do Modelo (SVM):**
    Este comando treina um modelo SVM, seguindo os mesmos passos do Random Forest.
    ``` bash
    python .\train_proj1.py  --train_features features/train_features.csv --model_type "svm" --output_dir artifacts
    ```

4.  **Pós-processamento (Ajuste do Limiar de Decisão para Random Forest):**
    Este passo ajusta o limiar de decisão para o modelo Random Forest usando o conjunto de validação e gera um gráfico.
    ``` bash
    python .\postprocess_proj1.py --val_features features/validation_features.csv --model_path artifacts/rf_model.joblib --scaler_path artifacts/scaler.joblib --output_dir results
    ```
    

5.  **Pós-processamento (Ajuste do Limiar de Decisão para SVM):**
    Este passo ajusta o limiar de decisão para o modelo SVM usando o conjunto de validação e gera um gráfico.
    ``` bash
    python postprocess_proj1.py --val_features features/validation_features.csv --model_path artifacts/svm_model.joblib --scaler_path artifacts/scaler.joblib --output_dir results
    ```

6.  **Avaliação na Validação (Random Forest):**
    Este comando avalia o modelo Random Forest no conjunto de validação com o limiar otimizado, gerando um relatório de classificação e uma matriz de confusão.
    ``` bash
    python test_proj1.py --test_features features/validation_features.csv --model_path artifacts/rf_model.joblib --scaler_path artifacts/scaler.joblib --results_dir results_validation --threshold 0.35
    ```

7.  **Avaliação na Validação (SVM):**
    Este comando avalia o modelo SVM no conjunto de validação com o limiar otimizado, gerando um relatório de classificação e uma matriz de confusão.
    ``` bash
    python test_proj1.py --test_features features/validation_features.csv --model_path artifacts/svm_model.joblib --scaler_path artifacts/scaler.joblib --results_dir results_svm_validation --threshold 0.11
    ```

    
8.  **Teste Final (Random Forest):**
    Este comando executa a avaliação final do modelo Random Forest no conjunto de teste.
    ``` bash
    python test_proj1.py --test_features features/test_features.csv --model_path artifacts/rf_model.joblib --scaler_path artifacts/scaler.joblib --results_dir results --threshold 0.35
    ```

9.  **Teste Final (SVM):**
    Este comando executa a avaliação final do modelo SVM no conjunto de teste.
    ``` bash
    python test_proj1.py --test_features features/test_features.csv --model_path artifacts/svm_model.joblib --scaler_path artifacts/scaler.joblib --results_dir results_svm --threshold 0.11
    ```

## 6. Avaliação e Resultados

A equipe buscou o melhor desempenho possível, considerando que o problema pode envolver dados desbalanceados. Em vez de se basear apenas na acurácia — métrica que pode ser inadequada em cenários com classes desproporcionais — foram utilizadas métricas mais robustas, tais como 'Precision', 'Recall', 'F1-Score', 'MEL' e 'AUC'. A metodologia de seleção de hiperparâmetros foi realizada de forma automática por meio da técnica de GridSearchCV com validação cruzada estratificada (5-fold), o que assegura uma avaliação mais estável e representativa da performance do modelo em diferentes partições dos dados, com atenção para evitar "data leakage". Complementarmente, foi conduzida uma calibração do limiar de decisão do modelo, testando múltiplos valores de threshold com o objetivo de identificar aquele que maximizasse o F1-Score da classe MEL. O melhor limiar encontrado foi então utilizado para avaliar o desempenho final do modelo no conjunto de teste.

Após a etapa de otimização de hiperparâmetros via validação cruzada com GridSearchCV, foram identificadas as seguintes configurações como ideais para cada modelo:
**Random Forest**  
  Melhores parâmetros encontrados:  
  `{'class_weight': 'balanced', 'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}`  
  Melhor AUC na validação cruzada: **0.9848**

**Support Vector Machine**  
  Melhores parâmetros encontrados:  
  `{'C': 10, 'class_weight': 'balanced', 'gamma': 1, 'kernel': 'rbf'}`  
  Melhor AUC na validação cruzada: **0.9926**

A avaliação em testes foi feita utilizando diferentes limiares de decisão para observar seus impactos nas métricas de desempenho:

**AVALIAÇÃO FINAL do modelo RF - Limiar 0.35**  
AUC: **0.8548**  
F1-score para MEL: **0.49**  
Recall para MEL: **0.81**  
Precisão para MEL: **0.35**

**AVALIAÇÃO FINAL do modelo SVM - Limiar 0.11**  
AUC: **0.6885**  
F1-score para MEL: **0.33**  
Recall para MEL: **0.33**  
Precisão para MEL: **0.32**

## 7. Participantes

* [Amanda Lopes] - [202207040043]
* [Filipe Correa] - [202006840020]
* [Giovanna Cunha] - [202206840039]
* [Giovana Nascimento] - [202206840015]

## 8. Referências

Este projeto utiliza os seguintes softwares e bibliotecas:
* scikit-learn (para classificadores e pré-processamento) 
* pandas (para manipulação de dados)
* numpy (para operações numéricas)
* opencv-python (para processamento de imagens)
* scikit-image (para extração de características de imagem)
* matplotlib (para plotagem de gráficos)
* seaborn (para visualização de dados)
* imblearn (para lidar com o desbalanceamento de classes, ex: SMOTE)
* joblib (para salvar e carregar modelos)
