import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non interactif pour éviter les erreurs GUI

from datetime import datetime, timedelta
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

def visualize_limited_predictions(output_df, output_path, title="Prix Réels vs Prédits (Limité) for LR_QL"):
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


def plot_error_distribution(output_df, output_path, title="Distribution des Erreurs"):
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


def scatter_predictions_with_ideal(output_df, output_path, title="Ensemble Entraînement - Réel vs Prédit for LR_QL"):
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


def visualize_predictions(output_df, output_path, title="Prix Réels vs Prix Prédits for LR_QL"):
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


def linear_regression_q_learning_test(X, Y, alpha=0.5, gamma=0.2, epochs=50):
    start_time = time.time()
    print("Démarrage de Linear Regression + Q-Learning")
    required_columns = ['Date', 'Price', 'Quantity', 'Customer_Review']
    for col in required_columns:
        if col not in X.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    X['Date'] = pd.to_datetime(X['Date'], errors='coerce')
    X['date_numeric'] = X['Date'].map(lambda x: x.toordinal() if pd.notnull(x) else None)
    X = X.dropna(subset=['date_numeric', 'Price'])
    
    # Convertir les colonnes numériques en float et remplacer les virgules par des points
    numeric_columns = ['Price', 'Quantity', 'Customer_Review']
    for col in numeric_columns:
        X[col] = X[col].astype(str).str.replace(',', '.')  # Remplacement des virgules
        X[col] = pd.to_numeric(X[col], errors='coerce')  # Conversion en float
    
    model_lr = LinearRegression()
    
    class QLearningAgent:
        def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
            self.q_table = {}
            self.learning_rate = learning_rate
            self.discount_factor = discount_factor
            self.exploration_rate = exploration_rate
            self.exploration_decay = exploration_decay
            self.actions = [-1, 0, 1]
        
        def get_state(self, produit):
            return tuple(round(produit[col], 2) for col in ['Price', 'Quantity', 'Customer_Review'])
        
        def choose_action(self, state):
            if np.random.rand() < self.exploration_rate:
                return np.random.choice(self.actions)
            
            if state not in self.q_table:
                self.q_table[state] = {a: 0 for a in self.actions}  # Initialisation de l'état
            
            return max(self.q_table[state], key=lambda a: self.q_table[state][a])
        
        def update_q_table(self, state, action, reward, next_state):
            if state not in self.q_table:
                self.q_table[state] = {a: 0 for a in self.actions}
            if next_state not in self.q_table:
                self.q_table[next_state] = {a: 0 for a in self.actions}
            current_q = self.q_table[state][action]
            max_future_q = max(self.q_table[next_state].values())
            self.q_table[state][action] = (1 - self.learning_rate) * current_q + \
                                          self.learning_rate * (reward + self.discount_factor * max_future_q)
        
        def decay_exploration(self):
            self.exploration_rate *= self.exploration_decay
    
    agent = QLearningAgent()
    predicted_values, true_values = [], []
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    Y_train, Y_test = Y.iloc[:split_index], Y.iloc[split_index:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[['date_numeric', 'Quantity', 'Customer_Review']])
    X_test_scaled = scaler.transform(X_test[['date_numeric', 'Quantity', 'Customer_Review']])
    
    # Vérifier et gérer les NaN après la normalisation
    if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
        print("⚠️ Attention : Des NaN ont été détectés après la normalisation.")
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=np.nanmean(X_train_scaled))
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=np.nanmean(X_test_scaled))
    
    model_lr.fit(X_train_scaled, Y_train)
    predicted_prices = model_lr.predict(X_test_scaled)
    
    for epoch in range(epochs):
        state = X.sample(n=1).iloc[0]
        state_tuple = agent.get_state(state)
        for _ in range(len(X)):
            action = agent.choose_action(state_tuple)
            new_price = state['Price'] * (1 + action * 0.05)
            new_state = state.copy()
            new_state['Price'] = new_price
            new_state['Quantity'] = alpha * new_state['Quantity'] + gamma * new_price
            new_state['Revenue'] = new_price * new_state['Quantity']
            next_state_tuple = agent.get_state(new_state)
            reward = new_state['Revenue']
            agent.update_q_table(state_tuple, action, reward, next_state_tuple)
            state_tuple = next_state_tuple
        agent.decay_exploration()
    
    predicted_prices = model_lr.predict(X_test_scaled)
    output_df = pd.DataFrame({'Prix Actuel': Y_test, 'Prix Optimisé': predicted_prices})
    output_df.to_csv('static/LR_QL_test.csv', index=False)
    mse = mean_squared_error(Y_test, predicted_prices)
    mae = mean_absolute_error(Y_test, predicted_prices)
    r2 = r2_score(Y_test, predicted_prices)
    training_time = time.time() - start_time
    print(f"MSE global: {mse}")
    print(f"MAE global: {mae}")
    print(f"R² global: {r2}")
    visualize_predictions(output_df, "static/prix_comparaison_LR_QL_test.png")
    scatter_predictions_with_ideal(output_df, "static/scatter_comparaison_ideal_LR_QL_test.png")
    plot_error_distribution(output_df, "static/histogram_erreurs_LR_QL_test.png")
    visualize_limited_predictions(output_df, "static/limite_comparaison_LR_QL_test.png")
    return model_lr, predicted_prices, mse, mae, r2, training_time
