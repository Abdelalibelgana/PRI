import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Backend non interactif pour éviter les erreurs GUI
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Utiliser un backend non interactif pour éviter les erreurs GUI
import matplotlib.pyplot as plt

def visualize_predictions(output_df, output_path, title="Prix Réels vs Prix Prédits for RF_QL"):
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
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique sauvegardé : {output_path}")

def scatter_predictions_with_ideal(output_df, output_path, title="Ensemble Entraînement - Réel vs Prédit"):
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

def plot_error_distribution(output_df, output_path, title="Distribution des Erreurs for RF_QL"):
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

def visualize_limited_predictions(output_df, output_path, title="Prix Réels vs Prix Prédits (Limité) for RF_QL"):
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


def random_forest_q_learning(X, Y, alpha=0.5, beta=0.3, gamma=0.2, epochs=50):
    """
    Combine Random Forest et Q-Learning pour prédire et optimiser les prix.

    Args:
        X (pd.DataFrame): Données d'entrée contenant les colonnes nécessaires.
        Y (pd.Series): Cible contenant les prix réels.
        alpha, beta, gamma (float): Coefficients pour la demande estimée.
        epochs (int): Nombre d'époques pour le Q-learning.

    Returns:
        tuple: Modèle, prédictions, MSE global, vraies valeurs, temps d'exécution
    """
    start_time = time.time()
    print("Démarrage de Random Forest + Q-Learning")

    # Vérification et préparation des colonnes nécessaires
    required_columns = ['Date', 'Price', 'Product']
    for col in required_columns:
        if col not in X.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    X['Date'] = pd.to_datetime(X['Date'], errors='coerce')
    X['date_numeric'] = X['Date'].map(lambda x: x.toordinal() if pd.notnull(x) else None)

    # Nettoyer et convertir les colonnes supplémentaires
    optional_columns = ['Quantity', 'Category', 'Customer_Review', 'Competing_Price']
    for col in optional_columns:
        if col in X.columns:
            X[col] = X[col].astype(str).str.replace(',', '.')
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Supprimer les lignes avec des valeurs manquantes dans les colonnes critiques
    X = X.dropna(subset=['date_numeric', 'Price'])

    # Initialisation du modèle Random Forest
    model_rf = RandomForestRegressor(n_estimators=300, random_state=42)

    # Classe Q-Learning Agent
    class QLearningAgent:
        def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
            self.q_table = {}
            self.learning_rate = learning_rate
            self.discount_factor = discount_factor
            self.exploration_rate = exploration_rate
            self.exploration_decay = exploration_decay
            self.actions = [-0.05, 0, 0.05]

        def _get_state(self, produit):
            return tuple(round(produit.get(col, 0), 2) for col in ['Price', 'Quantity', 'Customer_Review'])

        def choisir_action(self, state):
            if np.random.rand() < self.exploration_rate:
                return np.random.choice(self.actions)
            if state in self.q_table:
                return max(self.q_table[state], key=self.q_table[state].get)
            return np.random.choice(self.actions)

        def decroissance_exploration(self):
            self.exploration_rate *= self.exploration_decay

        def mettre_a_jour_q_table(self, state, action, reward, next_state):
            if state not in self.q_table:
                self.q_table[state] = {a: 0 for a in self.actions}
            if next_state not in self.q_table:
                self.q_table[next_state] = {a: 0 for a in self.actions}
            current_q_value = self.q_table[state][action]
            max_future_q = max(self.q_table[next_state].values())
            new_q_value = (1 - self.learning_rate) * current_q_value + \
                          self.learning_rate * (reward + self.discount_factor * max_future_q)
            self.q_table[state][action] = new_q_value

        def appliquer_action(self, produit, action):
            produit['Price'] *= (1 + action)
            produit['Demande estimée'] = alpha * produit.get('Customer_Review', 0) + \
                                         beta * produit.get('Quantity', 0) + \
                                         gamma * (1 / (1 + np.exp(-produit['Price'])))
            produit['Revenu estimé'] = produit['Price'] * produit['Demande estimée']
            return produit

    agent = QLearningAgent()

    predicted_values = []
    true_values = []
    csv_predictions = []

    # Entraînement par produit
    for product_code, product_data in X.groupby('Product'):
        product_data = product_data.sort_values('date_numeric')
        y_product = Y[product_data.index]

        # Préparer les colonnes dynamiquement
        X_prod = product_data[['date_numeric', 'Price']].copy()
        for col in optional_columns:
            if col in product_data.columns and product_data[col].notna().any():
                X_prod[col] = product_data[col]

        # Séparer les données en train et test
        split_index = int(len(product_data) * 0.8)
        X_train = X_prod.iloc[:split_index]
        y_train = y_product.iloc[:split_index]
        X_test = X_prod.iloc[split_index:]
        y_test = y_product.iloc[split_index:]

        # Vérifier si les données de formation ne sont pas vides
        if X_train.empty or y_train.empty:
            print(f"Produit {product_code} ignoré en raison de données insuffisantes.")
            continue

        # Normaliser les données
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Entraîner le modèle Random Forest
        model_rf.fit(X_train_scaled, y_train)

        # Prédictions
        predicted_prices = model_rf.predict(X_test_scaled)
        predicted_values.extend(predicted_prices)
        true_values.extend(y_test)

        # Phase d'entraînement Q-learning
        for epoch in range(epochs):
            for _, produit in product_data.iterrows():
                state = agent._get_state(produit)
                action = agent.choisir_action(state)
                produit = agent.appliquer_action(produit, action)
                reward = produit['Revenu estimé']
                next_state = agent._get_state(produit)
                agent.mettre_a_jour_q_table(state, action, reward, next_state)
            agent.decroissance_exploration()

        # Collecte des résultats
        for i, pred_price in enumerate(predicted_prices):
            actual_price = y_test.iloc[i]
            csv_predictions.append({
                'Produit': product_code,
                'Prix Actuel': actual_price,
                'Prix Optimisé': pred_price,
            })

    # Calculer le temps d'exécution
    training_time = time.time() - start_time

    # Calculer le MSE global
    mse = mean_squared_error(true_values, predicted_values)

    # Enregistrer les résultats dans un fichier CSV
    output_df = pd.DataFrame(csv_predictions)
    output_df.to_csv('static/RF_QL.csv', index=False)
    visualize_predictions(output_df, output_path="static/prix_comparaison_RF_QL.png")
    scatter_predictions_with_ideal(output_df, output_path="static/scatter_comparaison_ideal_RF_QL.png")
    plot_error_distribution(output_df, output_path="static/histogram_erreurs_RF_QL.png")
    visualize_limited_predictions(output_df, output_path="static/limite_comparaison_RF_QL.png")

    print(f"MSE global: {mse}")
    
    return model_rf, predicted_values, mse, training_time
