from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


def visualize_limited_predictions(output_df, output_path, title="Prix Réels vs Prédits (Limité) for LR"):
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


def plot_error_distribution(output_df, output_path, title="Distribution des Erreurs for LR"):
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


def scatter_predictions_with_ideal(output_df, output_path, title="Ensemble Entraînement - Réel vs Prédit  for LR"):
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


def visualize_predictions(output_df, output_path, title="Prix Réels vs Prix Prédits  for LR"):
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
   


def train_linear_regression_Gene(X, Y):
   
    start_time = time.time()
    # Créer le modèle de régression linéaire
    model = LinearRegression()
   
    # Initialiser les listes pour les prédictions et les vraies valeurs
    predicted_values = []
    true_values = []
    csv_predictions = []

    # Encodage des colonnes catégorielles (exemple: 'Category', 'Product Name')
    label_encoder = LabelEncoder()

    # Sélectionner seulement les colonnes numériques pour la normalisation
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    #X_numeric = X[numeric_columns]

    # Vérifier si la colonne 'ASIN/ISBN (Product Code)' existe bien dans X
    if "Product" not in X.columns:
        #print("LR product not in X")
        raise ValueError("Column 'Product ' not found in DataFrame.")

    # Groupement par produit pour effectuer les prédictions de chaque produit indépendamment
    for product_code, product_data in X.groupby("Product"):  # Utilisation de 'product_name_column' comme identifiant
        product_data = product_data.sort_values("Date")  # Trier les données par date (colonne 'Order Date')

        # Ajoutez cette ligne pour travailler sur une copie explicite :
        X_prod = product_data[['Date', 'Price', 'Product']].copy()
        
        # Vérification et ajout des colonnes si elles existent
        if 'Quantity' in product_data.columns and product_data['Quantity'].notna().any():
            X_prod.loc[:, 'Quantity'] = product_data['Quantity']

        if 'Category' in product_data.columns and product_data['Category'].notna().any():
            X_prod.loc[:, 'Category'] = product_data['Category']

        if 'Customer_Review' in product_data.columns and product_data['Customer_Review'].notna().any():
            X_prod.loc[:, 'Customer_Review'] = product_data['Customer_Review']

        if 'Competing_Price' in product_data.columns and product_data['Competing_Price'].notna().any():
            X_prod.loc[:, 'Competing_Price'] = product_data['Competing_Price']

        
        y_prod = Y[product_data.index]
        n_occurrences = len(product_data)

        # Diviser les données en train et test selon la logique de votre exemple
        if n_occurrences > 1:
            if n_occurrences <= 5:
                # Utiliser n-1 pour le train et 1 pour le test
                X_train = X_prod.iloc[:-1]
                y_train = y_prod.iloc[:-1]
                X_test = X_prod.iloc[-1:]
                y_test = y_prod.iloc[-1]  # Dernière valeur pour le test
            else:
                # Utiliser 80% pour le train et 20% pour le test
                split_index = int(n_occurrences * 0.8)
                X_train = X_prod.iloc[:split_index]
                y_train = y_prod.iloc[:split_index]
                X_test = X_prod.iloc[split_index:]
                y_test = y_prod.iloc[split_index:]
            
            # Normaliser les données uniquement pour les colonnes numériques
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
            X_test_scaled = scaler.transform(X_test[numeric_columns])

            # Entraîner le modèle de régression linéaire
            model.fit(X_train_scaled, y_train)

            # Prédire les prix pour les données de test
            predicted_prices = model.predict(X_test_scaled)
            predicted_prices = np.maximum(predicted_prices, 0)

            # Stocker les prédictions et les valeurs réelles pour le calcul du MSE
            predicted_values.extend(predicted_prices)

            # Si y_test est une série ou un tableau avec plusieurs éléments
            if hasattr(y_test, 'values'):
                true_values.extend(y_test.values)
            else:
                true_values.append(y_test)  # Sinon, si c'est une seule valeur

            # Ajouter les prédictions dans la liste pour l'export CSV
            for i, pred_price in enumerate(predicted_prices):
                # Déterminer la date correspondante à la prédiction
                date_value = product_data['Date'].iloc[split_index + i] if n_occurrences > 5 else product_data['Date'].iloc[-1]
                
                # Si la valeur de date est un entier ordinal, la convertir en date lisible
                if isinstance(date_value, int):
                    date_value = datetime.fromordinal(date_value)
                elif isinstance(date_value, str):
                    try:
                        date_value = pd.to_datetime(date_value, errors='coerce')
                    except:
                        date_value = None  # Si la conversion échoue, assigner None

                # Récupérer la valeur actuelle correspondante
                actual_price = y_test.iloc[i] if n_occurrences > 5 else y_test

                # Ajouter les informations dans le dictionnaire
                csv_predictions.append({
                    'Date': date_value,
                    'Prix Actuel': actual_price,
                    'Prix Optimisé': pred_price,
                    'Product Code': product_code
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


    # Sauvegarder les prédictions dans un fichier CSV
    predicted_data = pd.DataFrame(csv_predictions)
    #predicted_data['dateDePridiction'] = pd.to_datetime(predicted_data['dateDePridiction'], errors='coerce')
    predicted_data.to_csv('static/predicted_prices_Linear.csv', index=False)
    print("Les prédictions ont été sauvegardées dans 'predicted_prices_Linear.csv'")
    # Appeler les fonctions de visualisation
    visualize_predictions( predicted_data, "static/prix_comparaison_LR.png" )
    scatter_predictions_with_ideal( predicted_data, "static/scatter_comparaison_ideal_LR.png" )
    plot_error_distribution(predicted_data, "static/histogram_erreurs_LR.png")
    visualize_limited_predictions( predicted_data, "static/limite_comparaison_LR.png" )
    return model, predicted_values, mse, mae, r2, true_values, training_time

    #return model, predicted_values, mse, true_values,training_time
