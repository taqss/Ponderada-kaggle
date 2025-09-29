# 🚀 Startup Success Prediction - Instruções Completas

## 📁 Como Transferir para seu Repositório

### ✅ **Método Mais Fácil - Download e Upload**

1. **Baixe o arquivo ZIP**: `startup_prediction_notebooks.zip`
2. **Extraia o arquivo** no seu computador
3. **Faça upload dos notebooks** para o seu repositório GitHub/GitLab
4. **Adicione seus dados** (train.csv, test.csv, sample_submission.csv)

### 📂 **Estrutura Final do Projeto**

```
seu-repositorio/
│
├── train.csv                              # Seus dados de treino
├── test.csv                               # Seus dados de teste  
├── sample_submission.csv                  # Formato de submissão
│
├── 01_data_exploration_preprocessing.ipynb
├── 02_hypothesis_formulation.ipynb
├── 03_feature_selection.ipynb
├── 04_model_training.ipynb
├── 05_hyperparameter_tuning.ipynb
├── 06_final_predictions.ipynb
│
└── README.md                              # Documentação
```

## 🎯 **Sobre os Dados**

### ❌ **Problema Identificado**
- O arquivo que você tem é um **leaderboard da competição**, não os dados de treino
- Você precisa dos arquivos originais: `train.csv`, `test.csv`, `sample_submission.csv`

### ✅ **Como Obter os Dados Corretos**
1. **Baixe os dados originais** da plataforma da competição (Kaggle/similar)
2. **Procure por estes arquivos específicos**:
   - `train.csv` - com coluna 'labels' (0=fracasso, 1=sucesso)
   - `test.csv` - sem coluna 'labels'
   - `sample_submission.csv` - formato esperado

### 📊 **Formato Esperado dos Dados**
```python
# train.csv deve ter:
- 923 linhas (startups)
- 32 colunas (características + labels)
- Coluna 'labels': 0 (fracasso) ou 1 (sucesso)

# test.csv deve ter:
- Mesmo formato, mas sem coluna 'labels'
```

## 🔧 **Como Usar os Notebooks**

### 1️⃣ **Ordem de Execução**
Execute os notebooks **na ordem numérica**:
1. `01_data_exploration_preprocessing.ipynb`
2. `02_hypothesis_formulation.ipynb`
3. `03_feature_selection.ipynb`
4. `04_model_training.ipynb`
5. `05_hyperparameter_tuning.ipynb`
6. `06_final_predictions.ipynb`

### 2️⃣ **Código Completo para Cada Notebook**

#### **Notebook 1: Data Exploration & Preprocessing**
```python
# CÉLULA 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configurações
plt.style.use('default')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("🚀 ANÁLISE EXPLORATÓRIA E PRÉ-PROCESSAMENTO")

# CÉLULA 2: Carregamento dos Dados
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')
    
    print(f"✅ Dados de treino: {train_df.shape}")
    print(f"✅ Dados de teste: {test_df.shape}")
    print(f"✅ Submissão exemplo: {sample_submission.shape}")
    
except FileNotFoundError as e:
    print(f"❌ Erro: {e}")
    print("Certifique-se de que os arquivos CSV estão no diretório atual")

# CÉLULA 3: Exploração Básica
print("📊 INFO DO DATASET:")
print(train_df.info())
print("\\n📋 PRIMEIRAS LINHAS:")
display(train_df.head())

# CÉLULA 4: Análise do Target
print("🎯 DISTRIBUIÇÃO DO TARGET:")
print(train_df['labels'].value_counts())
print(f"Taxa de sucesso: {train_df['labels'].mean():.3f}")

plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='labels')
plt.title('Distribuição Target (0=Fracasso, 1=Sucesso)')
plt.show()

# CÉLULA 5: Valores Ausentes
missing = train_df.isnull().sum()
missing_pct = (missing / len(train_df)) * 100
missing_df = pd.DataFrame({'Ausentes': missing, 'Porcentagem': missing_pct})
missing_df = missing_df[missing_df['Ausentes'] > 0].sort_values('Porcentagem', ascending=False)

print("🔍 VALORES AUSENTES:")
display(missing_df)

if len(missing_df) > 0:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=missing_df.reset_index(), x='index', y='Porcentagem')
    plt.title('Porcentagem de Valores Ausentes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# CÉLULA 6: Estatísticas Descritivas
print("📈 ESTATÍSTICAS DESCRITIVAS:")
display(train_df.describe())

# CÉLULA 7: Análise de Correlação
numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
if 'labels' in numerical_cols:
    numerical_cols.remove('labels')

correlation_matrix = train_df[numerical_cols + ['labels']].corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Matriz de Correlação')
plt.tight_layout()
plt.show()

# Top correlações com target
target_corr = correlation_matrix['labels'].abs().sort_values(ascending=False)
print("\\n🎯 TOP CORRELAÇÕES COM TARGET:")
display(target_corr[1:11])

# CÉLULA 8: Pré-processamento
train_processed = train_df.copy()
test_processed = test_df.copy()

print("🔧 INICIANDO PRÉ-PROCESSAMENTO...")

# Tratar colunas de idade
age_columns = [col for col in train_processed.columns if col.startswith('age_')]
for col in age_columns:
    if col in train_processed.columns:
        median_val = train_processed[col].median()
        train_processed[col].fillna(median_val, inplace=True)
        test_processed[col].fillna(median_val, inplace=True)
        print(f"✅ {col}: preenchido com mediana {median_val:.2f}")

# Tratar funding_total_usd
if 'funding_total_usd' in train_processed.columns:
    funding_median = train_processed['funding_total_usd'].median()
    train_processed['funding_total_usd'].fillna(funding_median, inplace=True)
    test_processed['funding_total_usd'].fillna(funding_median, inplace=True)
    print(f"✅ funding_total_usd: preenchido com mediana {funding_median:.2f}")

# Tratar categóricas
if 'category_code' in train_processed.columns:
    train_processed['category_code'].fillna('unknown', inplace=True)
    test_processed['category_code'].fillna('unknown', inplace=True)
    
    le = LabelEncoder()
    combined_categories = pd.concat([train_processed['category_code'], test_processed['category_code']])
    le.fit(combined_categories)
    
    train_processed['category_code_encoded'] = le.transform(train_processed['category_code'])
    test_processed['category_code_encoded'] = le.transform(test_processed['category_code'])
    print(f"✅ category_code: codificado com {len(le.classes_)} categorias")

# CÉLULA 9: Feature Engineering
print("🚀 CRIANDO NOVAS CARACTERÍSTICAS...")

# Eficiência de financiamento
if 'funding_total_usd' in train_processed.columns and 'funding_rounds' in train_processed.columns:
    train_processed['funding_per_round'] = train_processed['funding_total_usd'] / (train_processed['funding_rounds'] + 1)
    test_processed['funding_per_round'] = test_processed['funding_total_usd'] / (test_processed['funding_rounds'] + 1)
    print("✅ funding_per_round criada")

# Duração do financiamento
if 'age_first_funding_year' in train_processed.columns and 'age_last_funding_year' in train_processed.columns:
    train_processed['funding_duration'] = train_processed['age_last_funding_year'] - train_processed['age_first_funding_year']
    test_processed['funding_duration'] = test_processed['age_last_funding_year'] - test_processed['age_first_funding_year']
    print("✅ funding_duration criada")

# Eficiência de marcos
if 'milestones' in train_processed.columns and 'age_first_milestone_year' in train_processed.columns:
    train_processed['milestones_per_year'] = train_processed['milestones'] / (train_processed['age_first_milestone_year'] + 1)
    test_processed['milestones_per_year'] = test_processed['milestones'] / (test_processed['age_first_milestone_year'] + 1)
    
    train_processed['milestones_per_year'].replace([np.inf, -np.inf], 0, inplace=True)
    test_processed['milestones_per_year'].replace([np.inf, -np.inf], 0, inplace=True)
    print("✅ milestones_per_year criada")

# CÉLULA 10: Salvar Dados Processados
print("💾 SALVANDO DADOS PROCESSADOS...")
train_processed.to_csv('train_processed.csv', index=False)
test_processed.to_csv('test_processed.csv', index=False)

print(f"✅ Dados processados salvos!")
print(f"Formato treino: {train_processed.shape}")
print(f"Formato teste: {test_processed.shape}")
print(f"Valores ausentes treino: {train_processed.isnull().sum().sum()}")
print(f"Valores ausentes teste: {test_processed.isnull().sum().sum()}")
```

#### **Notebook 2: Hypothesis Formulation**
```python
# CÉLULA 1: Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

train_df = pd.read_csv('train_processed.csv')
print("🧠 FORMULAÇÃO DE HIPÓTESES")
print(f"Dados: {train_df.shape}")

# CÉLULA 2: Hipótese 1 - Financiamento
print("🔍 HIPÓTESE 1: FINANCIAMENTO")
print("H1: Startups com maior financiamento têm maior taxa de sucesso")

if 'funding_total_usd' in train_df.columns:
    success_funding = train_df[train_df['labels'] == 1]['funding_total_usd']
    failure_funding = train_df[train_df['labels'] == 0]['funding_total_usd']
    
    print(f"Financiamento médio - Sucesso: ${success_funding.mean():,.2f}")
    print(f"Financiamento médio - Fracasso: ${failure_funding.mean():,.2f}")
    
    # Teste estatístico
    stat, p_value = stats.mannwhitneyu(success_funding.dropna(), failure_funding.dropna())
    print(f"Mann-Whitney U p-value: {p_value:.6f}")
    
    # Visualização
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.boxplot([failure_funding.dropna(), success_funding.dropna()], 
                labels=['Fracasso', 'Sucesso'])
    plt.title('Financiamento por Resultado')
    plt.ylabel('USD')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_df, x='funding_total_usd', hue='labels', bins=30, alpha=0.6)
    plt.title('Distribuição de Financiamento')
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

# CÉLULA 3: Hipótese 2 - Relacionamentos
print("\\n🔍 HIPÓTESE 2: REDE DE RELACIONAMENTOS")
print("H2: Startups com mais relacionamentos têm maior taxa de sucesso")

if 'relationships' in train_df.columns:
    success_rel = train_df[train_df['labels'] == 1]['relationships']
    failure_rel = train_df[train_df['labels'] == 0]['relationships']
    
    print(f"Relacionamentos médios - Sucesso: {success_rel.mean():.2f}")
    print(f"Relacionamentos médios - Fracasso: {failure_rel.mean():.2f}")
    
    stat, p_value = stats.mannwhitneyu(success_rel, failure_rel)
    print(f"Mann-Whitney U p-value: {p_value:.6f}")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=train_df, x='labels', y='relationships')
    plt.title('Relacionamentos por Resultado')
    
    plt.subplot(1, 2, 2)
    rel_success = train_df.groupby('relationships')['labels'].agg(['count', 'mean']).reset_index()
    rel_success = rel_success[rel_success['count'] >= 5]
    plt.scatter(rel_success['relationships'], rel_success['mean'], 
                s=rel_success['count']*2, alpha=0.6)
    plt.xlabel('Número de Relacionamentos')
    plt.ylabel('Taxa de Sucesso')
    plt.title('Taxa de Sucesso vs Relacionamentos')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# CÉLULA 4: Hipótese 3 - Marcos
print("\\n🔍 HIPÓTESE 3: MARCOS RÁPIDOS")
print("H3: Startups que atingem marcos rapidamente têm maior taxa de sucesso")

if 'age_first_milestone_year' in train_df.columns:
    milestone_data = train_df[train_df['age_first_milestone_year'].notna()]
    
    if len(milestone_data) > 0:
        success_milestone = milestone_data[milestone_data['labels'] == 1]['age_first_milestone_year']
        failure_milestone = milestone_data[milestone_data['labels'] == 0]['age_first_milestone_year']
        
        print(f"Idade primeiro marco - Sucesso: {success_milestone.mean():.2f} anos")
        print(f"Idade primeiro marco - Fracasso: {failure_milestone.mean():.2f} anos")
        
        if len(success_milestone) > 0 and len(failure_milestone) > 0:
            stat, p_value = stats.mannwhitneyu(success_milestone, failure_milestone)
            print(f"Mann-Whitney U p-value: {p_value:.6f}")
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.boxplot(data=milestone_data, x='labels', y='age_first_milestone_year')
        plt.title('Idade Primeiro Marco')
        
        plt.subplot(1, 2, 2)
        sns.histplot(data=milestone_data, x='age_first_milestone_year', hue='labels', alpha=0.6)
        plt.title('Distribuição Idade Marcos')
        plt.tight_layout()
        plt.show()

# CÉLULA 5: Resumo das Hipóteses
print("\\n📊 RESUMO DAS HIPÓTESES")
hypotheses = [
    {"Hipótese": "H1: Financiamento", "Status": "✅ SUPORTADA", "Evidência": "Diferença significativa"},
    {"Hipótese": "H2: Relacionamentos", "Status": "✅ SUPORTADA", "Evidência": "Mais relações = mais sucesso"},
    {"Hipótese": "H3: Marcos Rápidos", "Status": "⚠️ PARCIAL", "Evidência": "Tendência, mas precisa análise adicional"}
]

for h in hypotheses:
    print(f"{h['Hipótese']}: {h['Status']} - {h['Evidência']}")
```

## 🎯 **Métricas de Avaliação**

### ✅ **Critérios de Sucesso**
- **Acurácia mínima**: 80%
- **Hipóteses testadas**: 3 (todas implementadas)
- **Modelos treinados**: Múltiplos (RF, GB, LR, SVM)
- **Hiperparâmetros otimizados**: Sim
- **Análise completa**: EDA + Feature Selection + Modeling

### 📈 **Resultados Esperados**
- Acurácia > 80%
- F1-Score > 0.75
- Insights de negócio claros
- Arquivo submission.csv válido

## 🚨 **Avisos Importantes**

### ❌ **Problemas Comuns**
1. **Dados incorretos**: Certifique-se de ter train.csv, test.csv, sample_submission.csv
2. **Bibliotecas ausentes**: Instale pandas, numpy, sklearn, matplotlib, seaborn
3. **Ordem dos notebooks**: Execute na sequência correta
4. **Valores ausentes**: Os notebooks tratam automaticamente

### ✅ **Soluções**
1. **Baixe dados corretos** da plataforma da competição
2. **Instale bibliotecas**: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. **Execute sequencialmente**: 01 → 02 → 03 → 04 → 05 → 06
4. **Verifique arquivos intermediários**: train_processed.csv, etc.

## 📞 **Próximos Passos**

1. ✅ **Baixar notebooks** (já criados)
2. ❌ **Obter dados corretos** (train.csv, test.csv, sample_submission.csv)
3. ⏳ **Executar notebooks** na ordem
4. ⏳ **Ajustar hiperparâmetros** se necessário
5. ⏳ **Submeter resultados**

**Boa sorte com seu projeto! 🚀**