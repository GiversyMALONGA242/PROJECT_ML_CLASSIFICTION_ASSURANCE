# 🚗 Prédiction d'Appétence à l'Assurance Automobile (Cross-Selling)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

## 📋 Contexte & Problématique Métier

Dans le secteur de l'assurance, la vente croisée (*Cross-Selling*) consiste à proposer un produit complémentaire à des clients déjà acquis. L'objectif de ce projet est de prédire quels clients possédant une assurance **Santé** sont susceptibles de souscrire à une **assurance Automobile**[cite: 1].

* **Enjeu commercial :** Éviter les campagnes d'appels aléatoires coûteuses en ciblant prioritairement les clients à forte probabilité de conversion[cite: 1].
* **Problème Machine Learning :** Classification binaire supervisée sur un jeu de données fortement déséquilibré (~87,7 % de non-intéressés contre ~12,3 % d'intéressés)[cite: 1].

---

## 📊 Source des Données

* **Nom du dataset :** *Health Insurance Cross Sell Prediction Data*[cite: 1]
* **Source :** [Kaggle - Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)[cite: 1]
* **Dimensions :** **381 109 lignes** et **12 colonnes** (`id`, `Gender`, `Age`, `Driving_License`, `Region_Code`, `Previously_Insured`, `Vehicle_Age`, `Vehicle_Damage`, `Annual_Premium`, `Policy_Sales_Channel`, `Vintage`, `Response`)[cite: 1].

---

## 🔍 Enseignements Clés de l'Analyse Exploratoire (EDA)

L'audit statistique a mis en évidence trois facteurs discriminants majeurs :

1. **Assurance automobile préalable (`Previously_Insured`) :**
   * Clients **non assurés** (`0`) : **22,55 %** d'intérêt[cite: 1].
   * Clients **déjà assurés** (`1`) : **0,09 %** d'intérêt[cite: 1].
2. **Antécédents de sinistres (`Vehicle_Damage`) :**
   * Véhicule **déjà endommagé** (`Yes`) : **23,77 %** d'intérêt[cite: 1].
   * Véhicule **jamais endommagé** (`No`) : **0,52 %** d'intérêt[cite: 1].
3. **Ancienneté du véhicule (`Vehicle_Age`) :**
   * Plus de 2 ans (`> 2 Years`) : **29,37 %** d'intérêt[cite: 1].
   * Entre 1 et 2 ans (`1-2 Year`) : **17,38 %** d'intérêt[cite: 1].
   * Moins d'1 an (`< 1 Year`) : **4,37 %** d'intérêt[cite: 1].

---

## ⚙️ Méthodologie & Architecture Technique

1. **Feature Engineering :**
   * `Prime_par_jour` : Ratio entre la prime annuelle (`Annual_Premium`) et l'ancienneté client (`Vintage + 1`)[cite: 1].
   * `Categorie_Age` : Discrétisation de l'âge en 4 segments tarifaires (`Jeune`, `Actif`, `Senior`, `Retraite`)[cite: 1].
2. **Prétraitement & Pipeline (`Scikit-Learn`) :**
   * Séparation stratifiée **80 % Apprentissage / 20 % Évaluation**[cite: 1].
   * `ColumnTransformer` associant `StandardScaler` (variables numériques) et `OneHotEncoder(handle_unknown='ignore')` (variables catégorielles)[cite: 1].
3. **Gestion du déséquilibre & Optimisation :**
   * Pondération des classes via `class_weight='balanced'` et `scale_pos_weight`[cite: 1].
   * Ajustement du seuil de décision via la courbe *Precision-Recall* pour maximiser le **F1-Score** opérationnel (seuil optimal retenu : **`0,65`**)[cite: 1, 2].

---

## 📈 Benchmark & Résultats des Modèles

### 1. Comparaison des algorithmes (Seuil par défaut = 0,50)

| Modèle | ROC-AUC
