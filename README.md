# machineLearningFraud
Machine learning project aimed at automatically predicting a target class from structured data. The work includes exploration, cleaning, feature preparation, training multiple models, comparing them with appropriate metrics, and interpreting results to select a robust, deployable solution.

# Fraud Detection — Machine Learning

Projet de détection de fraude : prédire si une transaction est frauduleuse (`is_fraud`) à partir de données transactionnelles.  
Objectif principal : maximiser la détection des fraudes (recall) dans un contexte de classes très déséquilibrées.

## Contexte et données

- Dataset volumineux (≈ 1.29M lignes, 27 colonnes).
- Cible : `is_fraud` (0 = non fraude, 1 = fraude).
- Problème fortement déséquilibré (fraude rare) : métriques et méthodes adaptées nécessaires.

## Méthodologie

1. Exploration et visualisations
   - Analyse de la répartition des classes (échelle log).
   - Corrélations avec la cible et étude de variables clés (ex : montant, temporalité).

2. Préparation
   - Sélection des features numériques (pipeline simplifié pour les premiers tests).
   - Split train/test.
   - Rééquilibrage sur le train avec SMOTE (`sampling_strategy=0.1`) pour augmenter la classe fraude.

3. Modélisation
   - Baselines :
     - Logistic Regression
     - SVM linéaire (LinearSVC)
     - Random Forest
   - Modèles adaptés aux données déséquilibrées :
     - RUSBoost
     - EasyEnsemble
     - BalancedRandomForest
     - XGBoost (comparaison précision/recall)

## Évaluation

Vu le déséquilibre, l’accuracy est trompeuse. On privilégie :

- Recall (classe fraude) : métrique métier clé (minimiser les fraudes ratées)
- F1-score (classe fraude)
- AUC ROC
- PR-AUC (plus informative quand la fraude est rare)
- Balanced Accuracy
- Matrice de confusion

Exemples de résultats (split de test) :
- LogisticRegression : recall fraude ≈ 0.51, precision ≈ 0.29, AUC ROC ≈ 0.86
- Linear SVM : recall fraude ≈ 0.49, precision ≈ 0.36
- RandomForest : recall fraude ≈ 0.53, precision ≈ 0.62, AUC ROC ≈ 0.97

## Choix du modèle final

Dans un système anti-fraude, on veut surtout ne pas laisser passer de fraude.  
Le modèle retenu est RUSBoost, car il obtient le meilleur recall (≈ 0.93) et capte un maximum de fraudes.

Recommandation :
- Screening 1er niveau (max recall) : RUSBoost / EasyEnsemble
- 2e filtre (moins de faux positifs) : XGBoost ou BalancedRandomForest

## Installation

Prérequis :
- Python 3.10+ recommandé

Dépendances :
```bash
pip install -U pandas numpy scikit-learn imbalanced-learn matplotlib xgboost
