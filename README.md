# 🚗 Prédiction d'Appétence à l'Assurance Automobile (Cross-Selling)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

## 📋 Contexte & Problématique Métier

Dans le secteur de l'assurance, la vente croisée (*Cross-Selling*) consiste à proposer un produit complémentaire à des clients déjà acquis. L'objectif de ce projet est de prédire quels assurés **Santé** d'une compagnie d'assurance sont susceptibles de souscrire à une **assurance Automobile**.

* **Enjeu commercial :** Éviter les campagnes d'appels aléatoires coûteuses et peu efficaces en ciblant prioritairement les clients à forte probabilité de conversion.
* **Problème Machine Learning :** Classification binaire supervisée sur un jeu de données fortement déséquilibré (~87,7 % de non-intéressés contre ~12,3 % d'intéressés).

---

## 📊 Source des Données

* **Nom du dataset :** *Health Insurance Cross Sell Prediction Data*
* **Source :** [Kaggle - Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)
* **Dimensions :** **381 109 lignes** (clients) et **12 colonnes** (`id`, `Gender`, `Age`, `Driving_License`, `Region_Code`, `Previously_Insured`, `Vehicle_Age`, `Vehicle_Damage`, `Annual_Premium`, `Policy_Sales_Channel`, `Vintage`, `Response`).

---

## 🔍 Enseignements Clés de l'Analyse Exploratoire (EDA)

L'audit statistique a mis en évidence trois facteurs discriminants majeurs sur le taux de souscription (`Response = 1`) :

1. **Assurance automobile préalable (`Previously_Insured`) :**
   * Clients **non assurés** (`0`) : **22,55 %** d'intérêt.
   * Clients **déjà assurés** (`1`) : **0,09 %** d'intérêt.
2. **Antécédents de sinistres (`Vehicle_Damage`) :**
   * Véhicule **déjà endommagé** (`Yes`) : **23,77 %** d'intérêt.
   * Véhicule **jamais endommagé** (`No`) : **0,52 %** d'intérêt.
3. **Ancienneté du véhicule (`Vehicle_Age`) :**
   * Plus de 2 ans (`> 2 Years`) : **29,37 %** d'intérêt.
   * Entre 1 et 2 ans (`1-2 Year`) : **17,38 %** d'intérêt.
   * Moins d'1 an (`< 1 Year`) : **4,37 %** d'intérêt.

---

## ⚙️ Méthodologie & Architecture Technique

1. **Feature Engineering :**
   * `Prime_par_jour` : Ratio entre la prime annuelle (`Annual_Premium`) et l'ancienneté client (`Vintage + 1`).
   * `Categorie_Age` : Discrétisation de l'âge en 4 segments tarifaires (`Jeune`, `Actif`, `Senior`, `Retraite`).
2. **Découpage & Prétraitement (`Scikit-Learn Pipeline`) :**
   * Séparation stratifiée **80 % Apprentissage** (304 887 clients) / **20 % Évaluation** (76 222 clients).
   * `ColumnTransformer` combinant `StandardScaler` (variables numériques) et `OneHotEncoder(handle_unknown='ignore')` (variables catégorielles) pour prévenir toute fuite de données (*data leakage*).
3. **Gestion du déséquilibre des classes :**
   * Pondération explicite des classes via `class_weight='balanced'` et `scale_pos_weight`.
4. **Optimisation du seuil de décision :**
   * Ajustement du seuil de probabilité via la courbe *Precision-Recall* afin de maximiser le **F1-Score** opérationnel.

---

## 📈 Benchmark & Résultats des Modèles

### 1. Comparaison des algorithmes (Seuil par défaut = 0,50)

| Modèle | ROC-AUC | Recall (Classe 1) | Precision (Classe 1) | F1-Score (Classe 1) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **0,8571** | 0,9168 | **0,2866** | **0,4367** |
| **Régression Logistique (Baseline)** | 0,8536 | 0,9244 | 0,2830 | 0,4334 |
| **Random Forest Classifier** | 0,8536 | **0,9269** | 0,2793 | 0,4292 |

### 2. Performance finale de XGBoost après optimisation du seuil (`0,6488`)

* **Seuil optimal :** `0,6488`[cite: 1]
* **F1-Score (Classe 1) :** `0,4581`[cite: 1]
* **Recall (Classe 1) :** `79 %` (7 384 acheteurs potentiels détectés sur 9 342)[cite: 1, 2]
* **Precision (Classe 1) :** `32 %`[cite: 1]

**Impact opérationnel :** En ciblant uniquement les **30 %** de clients recommandés par le modèle au lieu d'appeler toute la base au hasard, l'équipe commerciale capte **79 % des souscriptions potentielles** tout en économisant **70 % des appels inutiles**[cite: 1, 2].

---

## 🛠️ Structure du Projet

```text
├── data/
│   └── train.csv                              # Dataset Kaggle (non versionné si > 100 Mo)
├── notebooks/
│   └── insurance_cross_sell_pipeline.ipynb    # Analyse exploratoire, modélisation et évaluation
├── models/
│   └── modele_cross_sell_assurance.joblib     # Pipeline Scikit-Learn/XGBoost + seuil optimal
├── README.md
└── requirements.txt
