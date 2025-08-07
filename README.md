# 📞 Análise de Churn em Telecomunicações

![Ciência de Dados](https://img.shields.io/badge/-Ciência%20de%20Dados-blueviolet)
![Machine Learning](https://img.shields.io/badge/-Machine%20Learning-orange)
![Python](https://img.shields.io/badge/-Python-yellowgreen)

# Informação do projeto (Missão e objetivo)

Este projeto tem como objetivo desenvolver modelos preditivos para identificar clientes da Telecom X com maior probabilidade de cancelar seus serviços. Através da análise de dados, pré-processamento, modelagem e interpretação, buscamos fornecer insights acionáveis para a implementação de estratégias de retenção eficazes.

### 🎯 Missão
Desenvolver modelos preditivos capazes de prever quais clientes têm maior chance de cancelar seus serviços.

A empresa quer antecipar o problema da evasão, e cabe a você construir um pipeline robusto para essa etapa inicial de modelagem.

### 🧠 Objetivos do Desafio
Preparar os dados para a modelagem (tratamento, encoding, normalização).

Realizar análise de correlação e seleção de variáveis.

Treinar dois ou mais modelos de classificação.

Avaliar o desempenho dos modelos com métricas.

Interpretar os resultados, incluindo a importância das variáveis.

Criar uma conclusão estratégica apontando os principais fatores que influenciam a evasão.
"""

# Metodologia Sumário 
## Metodologia

Este projeto seguiu um pipeline estruturado de análise e modelagem de dados para prever a evasão de clientes. As principais etapas incluíram:

1.  **Carregamento e Análise Inicial dos Dados:** Os dados foram carregados a partir do arquivo 'telecomX_dados_tratados.csv' em um DataFrame pandas. Foi realizada uma inspeção inicial dos dados, incluindo a visualização de amostras, estatísticas descritivas (`describe()`), informações sobre tipos de dados e valores não nulos (`info()`), e a contagem de valores nulos por coluna (`isnull().sum()`).

2.  **Exploração de Dados (EDA):** A análise exploratória focou em entender a distribuição da variável alvo (`churn`) e a relação entre as demais variáveis e o churn.
    *   A distribuição da variável alvo foi visualizada usando um countplot, e a proporção de churn foi calculada.
    *   A distribuição das variáveis numéricas foi examinada através de histogramas.
    *   Boxplots foram utilizados para visualizar a relação entre as variáveis numéricas e o churn.
    *   Countplots foram gerados para explorar a relação entre as variáveis categóricas e o churn.
    *   Uma matriz de correlação foi plotada para entender a relação entre as variáveis numéricas.

3.  **Pré-processamento e Tratamento dos Dados:** Nesta etapa, os dados foram preparados para a modelagem.
    *   A coluna 'customerid', considerada irrelevante para a modelagem, foi removida.
    *   Variáveis categóricas foram codificadas usando One-Hot Encoding (`pd.get_dummies`).
    *   As colunas resultantes do One-Hot Encoding com tipo 'bool' foram convertidas para 'int'.
    *   Outliers foram visualizados usando boxplots para as colunas numéricas. A detecção de outliers foi realizada utilizando o método do Intervalo Interquartil (IQR), mas não houve tratamento explícito no código.

4.  **Balanceamento das Classes:** Para lidar com o desbalanceamento da variável alvo, diferentes técnicas de balanceamento foram aplicadas no conjunto de treino.
    *   Os dados foram divididos em conjuntos de treino e teste, mantendo a proporção da classe alvo (estratificação).
    *   As estratégias de balanceamento testadas foram: Original (sem balanceamento), SMOTE (oversampling), RandomUnderSampler (undersampling) e SMOTEENN (combinação de over e undersampling).
    *   As proporções das classes após a aplicação de cada estratégia foram visualizadas.

5.  **Modelagem Preditiva:** Foram treinados dois modelos de classificação: Regressão Logística e Random Forest.
    *   Um pipeline de pré-processamento foi definido utilizando `ColumnTransformer` para aplicar `StandardScaler` às colunas numéricas. As colunas categóricas (já em formato dummy) foram mantidas.
    *   Pipelines completos foram criados combinando o pré-processador e cada modelo.
    *   O desempenho de cada modelo com cada estratégia de balanceamento foi avaliado usando validação cruzada (cv=5) no conjunto de treino balanceado, utilizando o F1-score como métrica.

6.  **Avaliação de Modelos:** O modelo Random Forest treinado com SMOTE foi escolhido como exemplo para uma avaliação mais detalhada no conjunto de teste não balanceado.
    *   O modelo final foi treinado no conjunto de treino balanceado com SMOTE.
    *   As previsões e probabilidades de churn foram geradas para o conjunto de teste.
    *   Um relatório de classificação (`classification_report`) foi impresso, fornecendo métricas como precisão, recall e F1-score para cada classe.
    *   A matriz de confusão foi visualizada usando um heatmap.
    *   A curva ROC (Receiver Operating Characteristic) e o valor AUC (Area Under the Curve) foram calculados e plotados para avaliar a capacidade do modelo de distinguir entre as classes.
    *   A curva Precision-Recall foi plotada para avaliar o trade-off entre precisão e recall em diferentes thresholds.
    *   O desempenho do modelo foi avaliado com diferentes thresholds de classificação para analisar o impacto na precisão e recall.

7.  **Interpretação e Explicabilidade:** Métodos para entender a importância das features foram aplicados.
    *   A importância das features para o modelo Random Forest foi extraída e visualizada usando um barplot.
    *   Os coeficientes das features para o modelo de Regressão Logística (treinado com SMOTE) foram extraídos e visualizados para entender a direção e magnitude do impacto de cada feature.
    *   SHAP (SHapley Additive exPlanations) foi utilizado para fornecer interpretabilidade local e global do modelo Random Forest, visualizando o impacto de cada feature nas previsões individuais e no modelo como um todo através de um summary plot.

# Resumo das principais descobertas

## Principais Fatores de Evasão

Os fatores mais influentes na evasão de clientes incluem:

- Tempo de Contrato (Tenure): Clientes com menor tempo de contrato apresentam maior propensão à evasão.
- Encargos Mensais (Charges.monthly) e Encargos Totais (Charges.total): Valores mais altos estão associados a uma maior probabilidade de churn.
- Tipo de Contrato: Clientes com contratos de dois anos ou um ano demonstram menor probabilidade de evasão em comparação com aqueles em contratos mensais.
- Serviço de Internet (Fiber Optic): A presença deste serviço foi identificada como um fator que aumenta a probabilidade de churn.
- Serviços de Segurança e Suporte (TechSupport, OnlineSecurity, OnlineBackup): A adesão a esses serviços está associada a uma menor probabilidade de evasão.
- Método de Pagamento (Electronic check e Mailed check): Cheque Eletrônico demonstrou aumentar a probabilidade de churn, enquanto Cheque Enviado parece estar associado a uma menor probabilidade.
- Contas Diárias (contas_diarias): A frequência e o valor dos encargos impactam a decisão de evasão.

# Resumo do desempenho dos modelos

## Desempenho dos Modelos

Foram avaliados modelos de Regressão Logística e Random Forest com diferentes estratégias de balanceamento. A tabela abaixo resume o desempenho no conjunto de teste, focando no F1-score para a classe de churn (evasão) e na métrica AUC.

| Estratégia de Balanceamento | Modelo              | F1 Test (Churn) | AUC Test |
|-----------------------------|---------------------|-----------------|----------|
| Original                    | Logistic Regression | 0.621           | 0.843    |
| Original                    | Random Forest       | 0.555           | 0.812    |
| SMOTE                       | Logistic Regression | 0.550           | 0.831    |
| SMOTE                       | Random Forest       | 0.555           | 0.812    |
| RandomUnderSampler          | Logistic Regression | 0.619           | 0.843    |
| RandomUnderSampler          | Random Forest       | 0.604           | 0.814    |
| SMOTEENN                    | Logistic Regression | 0.603           | 0.836    |
| SMOTEENN                    | Random Forest       | 0.621           | 0.833    |

*Nota: O F1 Test (Churn) e o AUC Test foram calculados no conjunto de teste não balanceado.*

Os resultados indicam que as estratégias de balanceamento melhoraram o desempenho na validação cruzada no treino. Ao avaliar no conjunto de teste, o F1-score para a classe de churn variou, com a Regressão Logística (Original e RandomUnderSampler) e o Random Forest (SMOTEENN) apresentando os melhores resultados para esta métrica. O AUC foi relativamente consistente entre as estratégias de balanceamento para cada modelo.

# Esboço da Estrutura do Projeto

## Estrutura do Projeto

O repositório contém os seguintes arquivos principais:

-   `telecomX_dados_tratados.ipynb`: O notebook Jupyter que documenta todo o processo de análise, pré-processamento, modelagem e avaliação.
-   `telecomX_dados_tratados.csv`: O conjunto de dados utilizado para a análise e treinamento dos modelos.
-   `previsoes_churn.csv`: Arquivo CSV contendo as previsões de churn (real, predito e probabilidade) para o conjunto de teste.
-   `modelo_random_forest.pkl`: Arquivo pickle contendo o modelo final de Random Forest treinado, pronto para ser carregado e utilizado para novas previsões.
