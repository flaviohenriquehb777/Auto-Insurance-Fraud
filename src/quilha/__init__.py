"""Módulos do projeto Quilha Seguros — Detecção de fraude em sinistros de automóvel.

Estrutura:
    dados       — leitura da base crua e divisão temporal treino/teste.
    preparacao  — tratamento determinístico dos sinistros (função `preparar`).
    pipeline    — Pipeline do scikit-learn: preparo -> imputação -> modelo.
    explicacao  — explicação SHAP agregada por feature original.
    mlops       — tracking no MLflow com degradação limpa (sem credencial, roda local).
    modelagem   — modelos, baselines e métricas de negócio da Fase 4.
"""

__version__ = "0.1.0"
