import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Backend non interactif pour éviter les erreurs GUI
import matplotlib.pyplot as plt

# 📌 Fonctions de Visualisation
def visualize_predictions(output_df, output_path, title="Prix Réels vs Prix Prédits for LR_QL"):
    if 'Prix Actuel' not in output_df.columns or 'Prix Optimisé' not in output_df.columns:
        raise ValueError("Les colonnes 'Prix Actuel' et 'Prix Optimisé' sont nécessaires.")
    
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

def scatter_predictions_with_ideal(output_df, output_path, title="Réel vs Prédit for LR_QL"):
    if 'Prix Actuel' not in output_df.columns or 'Prix Optimisé' not in output_df.columns:
        raise ValueError("Les colonnes 'Prix Actuel' et 'Prix Optimisé' sont nécessaires.")
    
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

def plot_error_distribution(output_df, output_path, title="Distribution des Erreurs for LR_QL"):
    if 'Prix Actuel' not in output_df.columns or 'Prix Optimisé' not in output_df.columns:
        raise ValueError("Les colonnes 'Prix Actuel' et 'Prix Optimisé' sont nécessaires.")
    
    erreurs = output_df['Prix Actuel'] - output_df['Prix Optimisé']
    plt.figure(figsize=(10, 6))
    plt.hist(erreurs, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.title(title)
    plt.xlabel("Erreur (Prix Réel - Prix Prédit)")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def visualize_limited_predictions(output_df, output_path, title="Prix Réels vs Prix Prédits (Limité) for LR_QL"):
    if 'Prix Actuel' not in output_df.columns or 'Prix Optimisé' not in output_df.columns:
        raise ValueError("Les colonnes 'Prix Actuel' et 'Prix Optimisé' sont nécessaires.")
    
    subset_df = output_df.head(200)
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

# 📌 Classe Q-Learning Agent
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.actions = [-0.05, 0, 0.05]  

    def choisir_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.actions)
        if state in self.q_table:
            return max(self.q_table[state], key=self.q_table[state].get)
        return np.random.choice(self.actions)

    def appliquer_action(self, produit, action):
        produit['Price'] *= (1 + action)
        if produit['Price'] < 1:  # Empêcher un prix négatif ou nul
            produit['Price'] = max(1, produit['Price'])
        return produit

# 📌 Entraînement Régression Linéaire + Q-Learning
def regression_linear_q_learning(X, Y, epochs=50):
    start_time = time.time()

    # Conversion des dates en valeurs numériques
    X['Date'] = pd.to_datetime(X['Date'], errors='coerce')
    X['date_numeric'] = X['Date'].map(lambda x: x.toordinal() if pd.notnull(x) else None)
    X = X.dropna(subset=['date_numeric', 'Price'])

    model = LinearRegression()
    agent = QLearningAgent()

    true_values = []
    optimized_values = []
    csv_predictions = []

    for product_code, product_data in X.groupby('Product'):
        product_data = product_data.sort_values('date_numeric')
        y_product = Y.loc[product_data.index]

        X_prod = product_data[['date_numeric', 'Price']].copy()
        split_index = int(len(product_data) * 0.8)

        X_train, y_train = X_prod.iloc[:split_index], y_product.iloc[:split_index]
        X_test, y_test = X_prod.iloc[split_index:], y_product.iloc[split_index:]

        if X_train.empty or y_train.empty or X_test.empty or y_test.empty:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        predicted_prices = model.predict(X_test_scaled)

        true_values.extend(y_test)
        optimized_values.extend(predicted_prices)

        for i in range(len(y_test)):
            produit_test = product_data.iloc[split_index + i].copy()
            action = agent.choisir_action(produit_test)
            produit_optimisé = agent.appliquer_action(produit_test, action)
            optimized_values[i] = produit_optimisé['Price']

        csv_predictions.extend(zip(true_values, optimized_values))

    # Création du DataFrame des résultats
    output_df = pd.DataFrame(csv_predictions, columns=["Prix Actuel", "Prix Optimisé"])
    output_df.to_csv('static/LR_QL.csv', index=False)

    # Vérification des tailles avant calcul des métriques
    if len(true_values) != len(optimized_values):
        raise ValueError(f"Incohérence de taille : true_values={len(true_values)}, optimized_values={len(optimized_values)}")

    # Calcul des métriques
    mse = mean_squared_error(true_values, optimized_values)
    mae = mean_absolute_error(true_values, optimized_values)
    r2 = r2_score(true_values, optimized_values)

    # 📊 Génération des visualisations
    visualize_predictions(output_df, "static/prix_comparaison_LR_QL.png")
    scatter_predictions_with_ideal(output_df, "static/scatter_comparaison_ideal_LR_QL.png")
    plot_error_distribution(output_df, "static/histogram_erreurs_LR_QL.png")
    visualize_limited_predictions(output_df, "static/limite_comparaison_LR_QL.png")

    return model, optimized_values, mse, mae, r2, time.time() - start_time
