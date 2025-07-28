# 📞 Análise de Churn em Telecomunicações

![Ciência de Dados](https://img.shields.io/badge/-Ciência%20de%20Dados-blueviolet)
![Machine Learning](https://img.shields.io/badge/-Machine%20Learning-orange)
![Python](https://img.shields.io/badge/-Python-yellowgreen)

## 📌 Visão Geral

Este projeto analisa o churn (cancelamento) de clientes em uma empresa de telecomunicações utilizando técnicas de machine learning. Desenvolvemos modelos preditivos para identificar clientes com risco de cancelamento e extraímos insights acionáveis para reduzir as taxas de evasão.

## 📊 Principais Resultados

### Características dos Dados
- **Taxa de churn**: 26.5% (desequilíbrio tratado com SMOTE)
- **Fatores mais influentes**:
  - Tipo de contrato (mensal vs anual)
  - Tempo como cliente (tenure)
  - Valor total gasto
  - Serviços adicionais (segurança, backup, suporte técnico)
  - Método de pagamento

### Desempenho dos Modelos
| Modelo               | Acurácia | Precisão | Recall | F1-Score |
|----------------------|----------|----------|--------|----------|
| Random Forest        | 0.82     | 0.75     | 0.78   | 0.73     |
| Regressão Logística  | 0.79     | 0.68     | 0.81   | 0.70     |

## 🛠️ Implementação Técnica

### Pré-requisitos
- Python 3.7+
- Bibliotecas: pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn

### Instalação
```bash
git clone https://github.com/seuusuario/analise-churn-telecom.git
cd analise-churn-telecom
pip install -r requirements.txt
```
## Estrutura do Projeto
├── dados/
    
    │   
      └── telecomX_dados_tratados.csv    # Dataset processado

├── notebooks/
    
    │   
      └── analise_churn.ipynb            # Notebook com a análise completa

├── modelos/
    
    │   
        ├── random_forest.pkl              # Modelo Random Forest salvo

    │   └── regressao_logistica.pkl        # Modelo Regressão Logística salvo

├── relatorios/
    
    │   
         └── conclusoes.md                  # Principais achados e recomendações

└── README.md

## 🚀 Como Usar
1. Execute o notebook Jupyter para reproduzir a análise:

```bash
jupyter notebook notebooks/analise_churn.ipynb
```

2. Para fazer previsões com os modelos treinados:
- import joblib
- import pandas as pd

### Carregar modelo
modelo = joblib.load('modelos/random_forest.pkl')

### Preparar novos dados (mesmo formato dos dados de treino)
novos_dados = pd.DataFrame(...)

### Fazer previsões
previsoes = modelo.predict(novos_dados)

### Carregar modelo
modelo = joblib.load('modelos/random_forest.pkl')

### Preparar novos dados (mesmo formato dos dados de treino)
novos_dados = pd.DataFrame(...)

### Fazer previsões
previsoes = modelo.predict(novos_dados)

## 💡 Recomendações para Negócios
### 1. Programas de Retenção

- Converter contratos mensais para anuais com incentivos

- Criar programas de fidelidade para clientes antigos

### 2. Melhorias de Serviço

- Pacotes com serviços agregados (segurança, backup)

- Melhorar tempo de resposta do suporte técnico

### 3. Otimização de Pagamentos

- Promover métodos de pagamento automático

- Simplificar processo para pagamentos por cheque eletrônico

## 🤝 Como Contribuir
Contribuições são bem-vindas! Por favor abra uma issue primeiro para discutir as mudanças propostas.

## 📜 Licença
MIT License - veja [LICENSE](https://github.com/licenses/MIT) para detalhes.

## 📧 Contato
Para dúvidas ou sugestões: linomarprimat@gmail.com
