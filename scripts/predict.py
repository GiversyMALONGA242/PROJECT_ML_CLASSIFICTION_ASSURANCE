import joblib
import pandas as pd

# 1. Chargement du pipeline et du seuil optimal depuis le dossier models/
artefact = joblib.load("models/modele_cross_sell_assurance.joblib")
pipeline = artefact["pipeline"]
seuil = artefact["seuil_optimal"]

# 2. Définition d'un nouveau client
client = pd.DataFrame([{
    "Gender": "Male",
    "Age": 38,
    "Driving_License": 1,
    "Region_Code": 28.0,
    "Previously_Insured": 0,
    "Vehicle_Age": "1-2 Year",
    "Vehicle_Damage": "Yes",
    "Annual_Premium": 35000.0,
    "Policy_Sales_Channel": 26.0,
    "Vintage": 150
}])

# 3. Application du Feature Engineering
client['Prime_par_jour'] = client['Annual_Premium'] / (client['Vintage'] + 1)
client['Categorie_Age'] = pd.cut(
    client['Age'], bins=[17, 25, 45, 60, 100], labels=['Jeune', 'Actif', 'Senior', 'Retraite']
)

# 4. Prédiction de la probabilité et décision métier
probabilite = pipeline.predict_proba(client)[:, 1][0]
decision = "OUI (Prioritaire)" if probabilite >= seuil else "NON"

print(f"Probabilité de souscription : {probabilite:.2%} | Appeler le client : {decision}")
