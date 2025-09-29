#!/usr/bin/env python3
"""
Script simples para criar notebooks básicos do projeto
"""

import json
import os

def create_basic_notebook(title, description):
    """Cria um notebook básico com título e descrição"""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n\n{description}"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["# Adicione seu código aqui"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def main():
    """Cria todos os notebooks"""
    print("🚀 Criando notebooks básicos...")
    
    os.makedirs('notebooks', exist_ok=True)
    
    notebooks = [
        ("01_data_exploration_preprocessing.ipynb", "🚀 Startup Success Prediction - Data Exploration & Preprocessing", "Exploração e pré-processamento dos dados"),
        ("02_hypothesis_formulation.ipynb", "🧠 Startup Success Prediction - Formulação de Hipóteses", "Formulação e teste de hipóteses"),
        ("03_feature_selection.ipynb", "🎯 Startup Success Prediction - Seleção de Características", "Seleção das melhores características"),
        ("04_model_training.ipynb", "🤖 Startup Success Prediction - Treinamento de Modelos", "Treinamento e avaliação de modelos"),
        ("05_hyperparameter_tuning.ipynb", "⚙️ Startup Success Prediction - Otimização de Hiperparâmetros", "Otimização para melhor performance"),
        ("06_final_predictions.ipynb", "🎯 Startup Success Prediction - Predições Finais", "Predições finais e submissão")
    ]
    
    for filename, title, description in notebooks:
        notebook = create_basic_notebook(title, description)
        filepath = f"notebooks/{filename}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Criado: {filepath}")
    
    print("\n🎉 Todos os notebooks foram criados!")
    print("📁 Agora você pode abrir cada notebook e adicionar o código fornecido")

if __name__ == "__main__":
    main()