import time  # Assurez-vous que time est importé
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from datetime import datetime
import numpy as np

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


def train_random_forest_Gene(X, Y):
    #print("Je suis dans la fonction random")
    # Mesurer le temps d'entraînement
    start_time = time.time()
    
    # Initialiser des listes pour stocker les prédictions, les vraies valeurs et les données pour le fichier CSV
    predicted_values = []
    true_values = []
    csv_predictions = []
    
    # Boucle pour chaque produit
    for product_code, product_data in X.groupby('Product'):  # Utiliser 'product_code' comme identifiant
        # Trier les données par date pour garantir l'ordre chronologique
        product_data = product_data.sort_values('Date')

        # Extraire X et y pour ce produit
        X_prod = product_data[['Date', 'Price', 'Product']].copy()

        # Vérification et ajout des colonnes supplémentaires si elles sont présentes
        if 'Quantity' in product_data.columns and product_data['Quantity'].notna().any():
            X_prod.loc[:, 'Quantity'] = product_data['Quantity']
        
        if 'Category' in product_data.columns and product_data['Category'].notna().any():
            X_prod.loc[:, 'Category'] = product_data['Category']
        
        if 'Customer_Review' in product_data.columns and product_data['Customer_Review'].notna().any():
            X_prod.loc[:, 'Customer_Review'] = product_data['Customer_Review']
        
        if 'Competing_Price' in product_data.columns and product_data['Competing_Price'].notna().any():
            X_prod.loc[:, 'Competing_Price'] = product_data['Competing_Price']

        # Convertir 'Date' en format numérique
        X_prod.loc[:, 'Date'] = pd.to_datetime(X_prod['Date'], errors='coerce')
        X_prod.loc[:, 'date_numeric'] = X_prod['Date'].map(datetime.toordinal)


        # Ajouter 'date_numeric' et 'Price' aux colonnes numériques pour la normalisation
        numeric_columns = X_prod.select_dtypes(include=[np.number]).columns
        #print("Colonnes numériques après transformation:", numeric_columns)

        # Préparer X et y
        X_final = X_prod[['date_numeric', 'Price']]  # Nous utilisons 'date_numeric' et 'Price' pour la normalisation
        
        if 'Quantity' in X_prod.columns:
            X_final['Quantity'] = X_prod['Quantity']
        if 'Customer_Review' in X_prod.columns:
            X_final['Customer_Review'] = X_prod['Customer_Review']
        if 'Category' in X_prod.columns:
            X_final['Category'] = X_prod['Category']
        if 'Competing_Price' in X_prod.columns:
            X_final['Competing_Price'] = X_prod['Competing_Price']
        
        y_final = X_prod['Price']

        n_occurrences = len(product_data)

        # Diviser les données en train et test selon le nombre d'occurrences
        if n_occurrences > 1:
            if n_occurrences <= 5:
                # Utiliser n-1 pour le train et 1 pour le test
                X_train = X_final.iloc[:-1]
                y_train = y_final.iloc[:-1]
                X_test = X_final.iloc[-1:]
                y_test = y_final.iloc[-1]  # Dernière valeur pour le test
            else:
                # Utiliser 80% pour le train et 20% pour le test
                split_index = int(n_occurrences * 0.8)
                X_train = X_final.iloc[:split_index]
                y_train = y_final.iloc[:split_index]
                X_test = X_final.iloc[split_index:]
                y_test = y_final.iloc[split_index:]

            # Vérifier les colonnes avant la normalisation
            #print("Colonnes avant normalisation:", X_train.columns)

            # Normaliser les données
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train[numeric_columns])  # Utilisez numeric_columns ici
            X_test_scaled = scaler.transform(X_test[numeric_columns])  # Assurez-vous que 'numeric_columns' est bien défini

            # Créer le modèle Random Forest
            model = RandomForestRegressor(n_estimators=300, random_state=42)
            model.fit(X_train_scaled, y_train)

            # Faire des prédictions
            predicted_prices = model.predict(X_test_scaled)

            # Stocker les prédictions et les valeurs réelles pour le calcul du MSE
            if isinstance(y_test, pd.Series):
                predicted_values.extend(predicted_prices)
                true_values.extend(y_test.values)
            else:
                predicted_values.append(predicted_prices[0])
                true_values.append(y_test)
            
            # Ajouter les prédictions dans la liste pour l'export CSV
            for i, pred_price in enumerate(predicted_prices):
                # Récupérer la valeur actuelle correspondante
                actual_price = y_test.iloc[i] if n_occurrences > 5 else y_test
                csv_predictions.append({
                    'date': product_data['Date'].iloc[split_index + i] if n_occurrences > 5 else product_data['Date'].iloc[-1],
                    'Prix Actuel': actual_price,
                    'prix Predit': pred_price,
                    'product Code': product_code
                })
    
    # Calculer le temps d'entraînement
    training_time = time.time() - start_time
    # Calculer le MSE global
    mse = mean_squared_error(true_values, predicted_values)
    print(f"Mean Squared Error (MSE) global: {mse}")

    # Enregistrer les prédictions dans un fichier CSV
    predicted_data = pd.DataFrame(csv_predictions)
    predicted_data.to_csv('static/predicted_prices_Forest.csv', index=False)
    print("Les prédictions ont été sauvegardées dans 'predicted_prices_Forest.csv'")

    return model, predicted_values, mse, true_values, training_time
