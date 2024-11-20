import time  # Assurez-vous que time est importé
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
def train_random_forest_model(df):
    # Mesurer le temps d'entraînement
    start_time = time.time()

    # Prétraitement des données
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['date_numeric'] = df['Order Date'].map(lambda x: x.toordinal() if pd.notnull(x) else None)
    df['Avis clients'] = df['Avis clients'].astype(str)  # Convertir en chaîne de caractères
    df['Avis clients'] = df['Avis clients'].str.replace(',', '.', regex=False).astype(float)  # Remplacer les virgules par des points et convertir en float

    df = df.dropna(subset=['date_numeric', 'Purchase Price Per Unit', 'Avis clients'])

    # Séparer les caractéristiques (X) et la cible (y)
    X = df[['date_numeric', 'Quantity', 'Avis clients']]
    y = df['Purchase Price Per Unit']

    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Appliquer le modèle Random Forest
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    # Prédictions et calcul du MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Calculer le temps d'entraînement
    training_time = time.time() - start_time

    # Sauvegarder les résultats dans un fichier CSV dans le dossier 'uploads'
    output_folder = 'uploads'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Définir le chemin du fichier de sortie
    output_file = os.path.join(output_folder, 'predicted_prices_random_forest.csv')

    # Sauvegarder les prédictions dans un fichier CSV
    predicted_data = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    predicted_data.to_csv(output_file, index=False)

    # Retourner les résultats, y compris le fichier de sortie
    return {"mse": mse, "training_time": training_time, "output_file": output_file}
