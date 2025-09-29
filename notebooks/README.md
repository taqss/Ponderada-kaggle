# 🚀 Startup Success Prediction - Notebooks

## Sobre os Dados

⚠️ **IMPORTANTE**: Para que os notebooks funcionem, você precisa ter os seguintes arquivos na mesma pasta:

- `train.csv` - Dados de treinamento com a variável alvo 'labels'
- `test.csv` - Dados de teste (sem a variável alvo)
- `sample_submission.csv` - Exemplo do formato de submissão

## Como Usar os Notebooks

1. **Ordem de Execução:**
   - Execute os notebooks na ordem numérica (01, 02, 03, etc.)
   - Cada notebook salva arquivos que são usados pelos próximos

2. **Notebooks Incluídos:**
   - `01_data_exploration_preprocessing.ipynb` - Exploração e pré-processamento
   - `02_hypothesis_formulation.ipynb` - Formulação de hipóteses
   - `03_feature_selection.ipynb` - Seleção de características
   - `04_model_training.ipynb` - Treinamento de modelos
   - `05_hyperparameter_tuning.ipynb` - Otimização de hiperparâmetros
   - `06_final_predictions.ipynb` - Predições finais e submissão

## Bibliotecas Necessárias

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

## Estrutura do Projeto

```
projeto/
│
├── train.csv                              # Dados de treino
├── test.csv                               # Dados de teste  
├── sample_submission.csv                  # Formato de submissão
│
├── 01_data_exploration_preprocessing.ipynb
├── 02_hypothesis_formulation.ipynb
├── 03_feature_selection.ipynb
├── 04_model_training.ipynb
├── 05_hyperparameter_tuning.ipynb
└── 06_final_predictions.ipynb
```

## Arquivos Gerados

Os notebooks criarão os seguintes arquivos intermediários:
- `train_processed.csv` - Dados de treino processados
- `test_processed.csv` - Dados de teste processados  
- `train_selected_features.csv` - Dados com características selecionadas
- `selected_features.txt` - Lista de características selecionadas
- `submission.csv` - Arquivo final de submissão

## Meta de Acurácia

🎯 **Objetivo**: Atingir pelo menos **80% de acurácia** no conjunto de teste

## Hipóteses Testadas

1. **Hipótese de Financiamento**: Startups com maior financiamento têm maior taxa de sucesso
2. **Hipótese de Rede**: Startups com mais relacionamentos têm maior taxa de sucesso  
3. **Hipótese de Experiência**: Startups que atingem marcos mais rapidamente têm maior taxa de sucesso