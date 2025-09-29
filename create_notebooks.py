#!/usr/bin/env python3
"""
Script para criar todos os notebooks do projeto de predição de sucesso de startups
"""

import json
import os

def create_notebook_structure():
    """Cria a estrutura básica de um notebook Jupyter"""
    return {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def add_markdown_cell(notebook, content):
    """Adiciona uma célula markdown ao notebook"""
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split('\n')
    }
    notebook["cells"].append(cell)

def add_code_cell(notebook, content):
    """Adiciona uma célula de código ao notebook"""
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content.split('\n')
    }
    notebook["cells"].append(cell)

def create_notebook_1():
    """Notebook 1: Data Exploration & Preprocessing"""
    notebook = create_notebook_structure()
    
    # Título
    add_markdown_cell(notebook, """# 🚀 Startup Success Prediction - Data Exploration & Preprocessing

## Overview
Este notebook foca na exploração, limpeza e pré-processamento dos dados para o desafio de previsão de sucesso de startups.

### Objetivos:
1. Carregar e explorar o dataset
2. Tratar valores ausentes
3. Tratar outliers
4. Codificar variáveis categóricas
5. Preparar dados para modelagem""")
    
    # Imports
    add_code_cell(notebook, """# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use('default')
sns.set_palette("husl")

# Opções de display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("🚀 ANÁLISE EXPLORATÓRIA E PRÉ-PROCESSAMENTO DE DADOS")
print("=" * 60)""")
    
    # Data Loading
    add_markdown_cell(notebook, "## 1. Carregamento e Exploração Inicial dos Dados")
    
    add_code_cell(notebook, """# Carregar os datasets
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')
    
    print(f"✅ Dados de treino: {train_df.shape}")
    print(f"✅ Dados de teste: {test_df.shape}")
    print(f"✅ Amostra de submissão: {sample_submission.shape}")
    
except FileNotFoundError as e:
    print(f"❌ Erro ao carregar dados: {e}")
    print("Por favor, certifique-se de que train.csv, test.csv e sample_submission.csv estão no diretório atual")""")
    
    add_code_cell(notebook, """# Exibir informações básicas sobre o dataset
print("📊 INFORMAÇÕES DO DATASET DE TREINO:")
print(train_df.info())

print("\\n📋 PRIMEIRAS 5 LINHAS:")
display(train_df.head())""")
    
    add_code_cell(notebook, """# Verificar distribuição da variável alvo
print("🎯 DISTRIBUIÇÃO DA VARIÁVEL ALVO:")
print(train_df['labels'].value_counts())
print(f"\\nTaxa de sucesso: {train_df['labels'].mean():.3f}")

# Visualizar distribuição do alvo
plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='labels')
plt.title('Distribuição da Variável Alvo (Sucesso/Fracasso)')
plt.xlabel('Labels (0=Fracasso, 1=Sucesso)')
plt.ylabel('Contagem')
plt.show()""")
    
    # Missing Values
    add_markdown_cell(notebook, "## 2. Análise de Valores Ausentes")
    
    add_code_cell(notebook, """# Verificar valores ausentes
missing_train = train_df.isnull().sum()
missing_percent = (missing_train / len(train_df)) * 100

missing_df = pd.DataFrame({
    'Valores_Ausentes': missing_train,
    'Porcentagem_Ausente': missing_percent
})
missing_df = missing_df[missing_df['Valores_Ausentes'] > 0].sort_values('Porcentagem_Ausente', ascending=False)

print("🔍 RESUMO DE VALORES AUSENTES:")
display(missing_df)

# Visualizar valores ausentes
if len(missing_df) > 0:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=missing_df.reset_index(), x='index', y='Porcentagem_Ausente')
    plt.title('Porcentagem de Valores Ausentes por Coluna')
    plt.xlabel('Colunas')
    plt.ylabel('Porcentagem Ausente (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()""")
    
    # EDA
    add_markdown_cell(notebook, "## 3. Análise Exploratória de Dados")
    
    add_code_cell(notebook, """# Resumo estatístico
print("📈 RESUMO ESTATÍSTICO:")
display(train_df.describe())""")
    
    add_code_cell(notebook, """# Analisar características numéricas
numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
if 'labels' in numerical_cols:
    numerical_cols.remove('labels')

print(f"🔢 COLUNAS NUMÉRICAS: {numerical_cols}")

# Distribuição das características numéricas principais
key_features = ['funding_total_usd', 'funding_rounds', 'relationships', 'milestones', 
                'avg_participants', 'age_first_funding_year', 'age_last_funding_year']

available_features = [f for f in key_features if f in train_df.columns]

if available_features:
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(available_features[:9]):
        if i < len(axes):
            train_df[col].hist(bins=30, ax=axes[i], alpha=0.7)
            axes[i].set_title(f'Distribuição de {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequência')
    
    # Remover subplots vazios
    for i in range(len(available_features), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()""")
    
    add_code_cell(notebook, """# Análise de correlação
correlation_matrix = train_df[numerical_cols + ['labels']].corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Matriz de Correlação das Características Numéricas')
plt.tight_layout()
plt.show()

# Características mais correlacionadas com o alvo
target_corr = correlation_matrix['labels'].abs().sort_values(ascending=False)
print("\\n🎯 CARACTERÍSTICAS MAIS CORRELACIONADAS COM O ALVO:")
display(target_corr[1:11])  # Excluir o próprio alvo""")
    
    # Feature Analysis by Target
    add_markdown_cell(notebook, "## 4. Análise de Características por Variável Alvo")
    
    add_code_cell(notebook, """# Comparar distribuições por variável alvo
available_key_features = [f for f in key_features if f in train_df.columns]

if available_key_features:
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(available_key_features[:9]):
        if i < len(axes):
            # Box plot por alvo
            sns.boxplot(data=train_df, x='labels', y=feature, ax=axes[i])
            axes[i].set_title(f'{feature} por Sucesso/Fracasso')
            axes[i].set_xlabel('Labels (0=Fracasso, 1=Sucesso)')
    
    # Remover subplots vazios
    for i in range(len(available_key_features), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()""")
    
    # Outlier Detection
    add_markdown_cell(notebook, "## 5. Detecção e Tratamento de Outliers")
    
    add_code_cell(notebook, """def detect_outliers_iqr(df, column):
    \"\"\"Detectar outliers usando método IQR\"\"\"
    if df[column].dtype in ['int64', 'float64']:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return len(outliers), lower_bound, upper_bound
    return 0, None, None

print("🔍 ANÁLISE DE OUTLIERS:")
outlier_summary = []

for col in numerical_cols:
    if col in train_df.columns:
        outlier_count, lower, upper = detect_outliers_iqr(train_df, col)
        outlier_summary.append({
            'Coluna': col,
            'Contagem_Outliers': outlier_count,
            'Porcentagem_Outliers': (outlier_count / len(train_df)) * 100,
            'Limite_Inferior': lower,
            'Limite_Superior': upper
        })

outlier_df = pd.DataFrame(outlier_summary)
display(outlier_df.sort_values('Porcentagem_Outliers', ascending=False))""")
    
    # Preprocessing
    add_markdown_cell(notebook, "## 6. Pré-processamento dos Dados")
    
    add_code_cell(notebook, """# Criar uma cópia para pré-processamento
train_processed = train_df.copy()
test_processed = test_df.copy()

print("🔧 Iniciando pré-processamento dos dados...")

# 1. Tratar valores ausentes em colunas de idade (NaN significa que o evento não ocorreu)
age_columns = [col for col in train_processed.columns if col.startswith('age_')]
print(f"📅 Colunas de idade encontradas: {age_columns}")

for col in age_columns:
    if col in train_processed.columns:
        # Preencher NaN com mediana para colunas de idade
        median_val = train_processed[col].median()
        train_processed[col].fillna(median_val, inplace=True)
        test_processed[col].fillna(median_val, inplace=True)
        print(f"✅ Preenchidos valores NaN de {col} com mediana: {median_val:.2f}")""")
    
    add_code_cell(notebook, """# 2. Tratar valores ausentes em funding_total_usd
if 'funding_total_usd' in train_processed.columns:
    funding_median = train_processed['funding_total_usd'].median()
    train_processed['funding_total_usd'].fillna(funding_median, inplace=True)
    test_processed['funding_total_usd'].fillna(funding_median, inplace=True)
    print(f"✅ Preenchidos valores NaN de funding_total_usd com mediana: {funding_median:.2f}")

# 3. Tratar variáveis categóricas
if 'category_code' in train_processed.columns:
    # Preencher category_code ausente com 'unknown'
    train_processed['category_code'].fillna('unknown', inplace=True)
    test_processed['category_code'].fillna('unknown', inplace=True)
    
    # Codificar category_code usando LabelEncoder
    le = LabelEncoder()
    # Ajustar nos dados combinados para garantir codificação consistente
    combined_categories = pd.concat([train_processed['category_code'], test_processed['category_code']])
    le.fit(combined_categories)
    
    train_processed['category_code_encoded'] = le.transform(train_processed['category_code'])
    test_processed['category_code_encoded'] = le.transform(test_processed['category_code'])
    
    print(f"✅ Codificado category_code com {len(le.classes_)} categorias únicas")""")
    
    add_code_cell(notebook, """# 4. Criar características adicionais
print("🚀 Criando características adicionais...")

# Características de eficiência de financiamento
if 'funding_total_usd' in train_processed.columns and 'funding_rounds' in train_processed.columns:
    train_processed['funding_per_round'] = train_processed['funding_total_usd'] / (train_processed['funding_rounds'] + 1)
    test_processed['funding_per_round'] = test_processed['funding_total_usd'] / (test_processed['funding_rounds'] + 1)
    print("✅ Criada característica funding_per_round")

# Características baseadas em idade
if 'age_first_funding_year' in train_processed.columns and 'age_last_funding_year' in train_processed.columns:
    train_processed['funding_duration'] = train_processed['age_last_funding_year'] - train_processed['age_first_funding_year']
    test_processed['funding_duration'] = test_processed['age_last_funding_year'] - test_processed['age_first_funding_year']
    print("✅ Criada característica funding_duration")

# Eficiência de marcos
if 'milestones' in train_processed.columns and 'age_first_milestone_year' in train_processed.columns:
    train_processed['milestones_per_year'] = train_processed['milestones'] / (train_processed['age_first_milestone_year'] + 1)
    test_processed['milestones_per_year'] = test_processed['milestones'] / (test_processed['age_first_milestone_year'] + 1)
    
    # Substituir valores infinitos por 0
    train_processed['milestones_per_year'].replace([np.inf, -np.inf], 0, inplace=True)
    test_processed['milestones_per_year'].replace([np.inf, -np.inf], 0, inplace=True)
    print("✅ Criada característica milestones_per_year")

print("✅ Características adicionais criadas com sucesso!")""")
    
    add_code_cell(notebook, """# 5. Verificação final de valores ausentes
print("🔍 Verificação final de valores ausentes:")
print(f"Valores ausentes nos dados de treino: {train_processed.isnull().sum().sum()}")
print(f"Valores ausentes nos dados de teste: {test_processed.isnull().sum().sum()}")

# Salvar dados processados
train_processed.to_csv('train_processed.csv', index=False)
test_processed.to_csv('test_processed.csv', index=False)

print("\\n💾 Dados processados salvos com sucesso!")
print(f"Formato final dos dados de treino: {train_processed.shape}")
print(f"Formato final dos dados de teste: {test_processed.shape}")""")
    
    # Summary
    add_markdown_cell(notebook, """## Resumo do Pré-processamento

✅ **Tarefas Concluídas:**
1. Carregamento e exploração dos dados
2. Análise de valores ausentes
3. Análise exploratória de dados
4. Detecção e análise de outliers
5. Tratamento de valores ausentes com estratégias apropriadas
6. Codificação de variáveis categóricas
7. Criação de características engenheiradas adicionais
8. Salvamento dos datasets processados

**Próximos passos:** Formulação de hipóteses e seleção de características""")
    
    return notebook

def save_notebook(notebook, filename):
    """Salva o notebook em arquivo"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"✅ Notebook salvo: {filename}")

def main():
    """Função principal"""
    print("🚀 Criando notebooks do projeto...")
    
    # Criar diretório se não existir
    os.makedirs('notebooks', exist_ok=True)
    
    # Notebook 1: Data Exploration & Preprocessing
    notebook1 = create_notebook_1()
    save_notebook(notebook1, 'notebooks/01_data_exploration_preprocessing.ipynb')
    
    print("\\n🎉 Todos os notebooks foram criados com sucesso!")
    print("📁 Verifique a pasta 'notebooks' para os arquivos .ipynb")

if __name__ == "__main__":
    main()