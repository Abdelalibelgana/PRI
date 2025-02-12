import time  # Assurez-vous que time est importé
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


def visualize_limited_predictions(output_df, output_path, title="Prix Réels vs Prédits (Limité) for RF"):
    """
    Affiche les prix réels et prédits pour un sous-ensemble des données.

    Args:
        output_df (pd.DataFrame): DataFrame contenant 'Prix Actuel' et 'Prix Optimisé'.
        output_path (str): Chemin pour sauvegarder le graphique.
        title (str): Titre du graphique.
    """
    if 'Prix Actuel' not in output_df.columns or 'Prix Optimisé' not in output_df.columns:
        raise ValueError("Les colonnes 'Prix Actuel' et 'Prix Optimisé' sont nécessaires dans le DataFrame.")
    
    subset_df = output_df.head(200)  # Limiter à 200 premières lignes
    plt.figure(figsize=(10, 6))
    plt.plot(subset_df.index, subset_df['Prix Actuel'], label="Prix Réels", marker='o')
    plt.plot(subset_df.index, subset_df['Prix Optimisé'], label="Prix Prédits", marker='x')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique limité sauvegardé : {output_path}")


def plot_error_distribution(output_df, output_path, title="Distribution des Erreurs for RF"):
    """
    Trace un histogramme pour la distribution des écarts entre prix réels et prédits.

    Args:
        output_df (pd.DataFrame): DataFrame contenant 'Prix Actuel' et 'Prix Optimisé'.
        output_path (str): Chemin pour sauvegarder le graphique.
        title (str): Titre du graphique.
    """
    if 'Prix Actuel' not in output_df.columns or 'Prix Optimisé' not in output_df.columns:
        raise ValueError("Les colonnes 'Prix Actuel' et 'Prix Optimisé' sont nécessaires dans le DataFrame.")
    
    erreurs = output_df['Prix Actuel'] - output_df['Prix Optimisé']
    plt.figure(figsize=(10, 6))
    plt.hist(erreurs, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.title(title)
    plt.xlabel("Erreur (Prix Réel - Prix Prédit)")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Histogramme des erreurs sauvegardé : {output_path}")


def scatter_predictions_with_ideal(output_df, output_path, title="Ensemble Entraînement - Réel vs Prédit  for RF"):
    """
    Trace un graphique de dispersion avec une ligne idéale.

    Args:
        output_df (pd.DataFrame): DataFrame contenant 'Prix Actuel' et 'Prix Optimisé'.
        output_path (str): Chemin pour sauvegarder le graphique.
        title (str): Titre du graphique.
    """
    if 'Prix Actuel' not in output_df.columns or 'Prix Optimisé' not in output_df.columns:
        raise ValueError("Les colonnes 'Prix Actuel' et 'Prix Optimisé' sont nécessaires dans le DataFrame.")
    
    plt.figure(figsize=(8, 8))
    plt.scatter(output_df['Prix Actuel'], output_df['Prix Optimisé'], alpha=0.7, color='green', label="Prédictions")
    plt.plot(output_df['Prix Actuel'], output_df['Prix Actuel'], color='red', linestyle='--', label="Valeurs Réelles")
    plt.title(title)
    plt.xlabel("Valeurs Réelles")
    plt.ylabel("Valeurs Prédites")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique sauvegardé : {output_path}")


def visualize_predictions(output_df, output_path, title="Prix Réels vs Prix Prédits  for RF"):
    """
    Sauvegarde un graphique comparant les prix réels et prédits.

    Args:
        output_df (pd.DataFrame): DataFrame contenant les colonnes 'Prix Actuel' et 'Prix Optimisé'.
        output_path (str): Chemin du fichier de sortie.
        title (str): Titre du graphique.
    """
    if 'Prix Actuel' not in output_df.columns or 'Prix Optimisé' not in output_df.columns:
        raise ValueError("Les colonnes 'Prix Actuel' et 'Prix Optimisé' sont nécessaires dans le DataFrame.")
    
    plt.figure(figsize=(10, 6))
    plt.plot(output_df.index, output_df['Prix Actuel'], label="Prix Réels", marker='o')
    plt.plot(output_df.index, output_df['Prix Optimisé'], label="Prix Prédits", marker='x')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)  # Sauvegarder le graphique
    plt.close()  # Fermer la figure pour éviter les erreurs
    print(f"Graphique sauvegardé : {output_path}")
   

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

        X_prod = product_data[['Date', 'Price', 'Product']].copy()

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
        

        # Ajouter 'Quantity' si présent
        if 'Quantity' in X_prod.columns:
            X_final = X_final.copy()  # Crée une copie explicite pour éviter les avertissements
            X_final['Quantity'] = X_prod['Quantity']

        # Ajouter 'Customer_Review' si présent
        if 'Customer_Review' in X_prod.columns:
            X_final = X_final.copy()
            X_final['Customer_Review'] = X_prod['Customer_Review']

        # Ajouter 'Category' si présent
        if 'Category' in X_prod.columns:
            X_final = X_final.copy()
            X_final['Category'] = X_prod['Category']

        # Ajouter 'Competing_Price' si présent
        if 'Competing_Price' in X_prod.columns:
            X_final = X_final.copy()
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
            model = RandomForestRegressor(n_estimators=1000, random_state=42)
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
                    'Prix Optimisé': pred_price,
                    'product Code': product_code
                })
    
    # Calculer le temps d'entraînement
    training_time = time.time() - start_time
    
     # Calculer les métriques globales
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    # Afficher les métriques
    print(f"Mean Squared Error (MSE) global: {mse}")
    print(f"Mean Absolute Error (MAE) global: {mae}")
    print(f"Coefficient de Détermination (R^2) global: {r2}")

    # Enregistrer les prédictions dans un fichier CSV
    predicted_data = pd.DataFrame(csv_predictions)
    print(predicted_data.columns)
    predicted_data.to_csv('static/predicted_prices_Forest.csv', index=False)
    print("Les prédictions ont été sauvegardées dans 'predicted_prices_Forest.csv'")
    visualize_predictions( predicted_data, "static/prix_comparaison_RF.png" )
    scatter_predictions_with_ideal( predicted_data, "static/scatter_comparaison_ideal_RF.png" )
    plot_error_distribution(predicted_data, "static/histogram_erreurs_RF.png")
    visualize_limited_predictions( predicted_data, "static/limite_comparaison_RF.png" )

    return model, predicted_values, mse, mae, r2, true_values, training_time
