import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Backend non interactif pour éviter les erreurs GUI
import matplotlib.pyplot as plt


# 📌 Fonctions de Visualisation (Identiques à ton code)
def visualize_predictions(output_df, output_path, title="Prix Réels vs Prix Prédits for RF_QL"):
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


def scatter_predictions_with_ideal(output_df, output_path, title="Ensemble Entraînement - Réel vs Prédit"):
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


def plot_error_distribution(output_df, output_path, title="Distribution des Erreurs for RF_QL"):
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


def visualize_limited_predictions(output_df, output_path, title="Prix Réels vs Prix Prédits (Limité) for RF_QL"):
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

    def mettre_a_jour_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 10 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}
        current_q_value = self.q_table[state][action]
        max_future_q = max(self.q_table[next_state].values())
        new_q_value = (1 - self.learning_rate) * current_q_value + \
                      self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state][action] = new_q_value

    def appliquer_action(self, produit, action):
        produit['Price'] *= (1 + action)
        if produit['Price'] < 1:  # Empêcher un prix ≤ 0
            produit['Price'] = max(1, produit['Price'])  
        return produit


# 📌 Entraînement Random Forest + Q-Learning
def random_forest_q_learning(X, Y, epochs=50):
    start_time = time.time()

    X['Date'] = pd.to_datetime(X['Date'], errors='coerce')
    X['date_numeric'] = X['Date'].map(lambda x: x.toordinal() if pd.notnull(x) else None)

    X = X.dropna(subset=['date_numeric', 'Price'])

    model_rf = RandomForestRegressor(n_estimators=300, random_state=42)
    agent = QLearningAgent()

    predicted_values = []
    optimized_values = []
    true_values = []
    csv_predictions = []

    for product_code, product_data in X.groupby('Product'):
        product_data = product_data.sort_values('date_numeric')
        y_product = Y[product_data.index]

        X_prod = product_data[['date_numeric', 'Price']].copy()

        split_index = int(len(product_data) * 0.8)
        X_train, y_train = X_prod.iloc[:split_index], y_product.iloc[:split_index]
        X_test, y_test = X_prod.iloc[split_index:], y_product.iloc[split_index:]

        if X_train.empty or y_train.empty:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_rf.fit(X_train_scaled, y_train)
        predicted_prices = model_rf.predict(X_test_scaled)
        
        # Ajouter uniquement les valeurs de test dans true_values
        true_values.extend(y_test)
        predicted_values.extend(predicted_prices)

        # Appliquer Q-learning sur les données de test uniquement
        for i in range(len(y_test)):
            produit_test = product_data.iloc[split_index + i].copy()
            action = agent.choisir_action(produit_test)
            produit_optimisé = agent.appliquer_action(produit_test, action)

            optimized_values.append(produit_optimisé['Price'])

            csv_predictions.append({
                'Produit': product_code,
                'Prix Actuel': y_test.iloc[i],
                'Prix Optimisé': produit_optimisé['Price'],
            })


    mse = mean_squared_error(true_values, optimized_values)
    mae = mean_absolute_error(true_values, optimized_values)
    r2 = r2_score(true_values, optimized_values)

    output_df = pd.DataFrame(csv_predictions)
    output_df.to_csv('static/RF_QL.csv', index=False)

    visualize_predictions(output_df, "static/prix_comparaison_RF_QL.png")
    scatter_predictions_with_ideal(output_df, "static/scatter_comparaison_ideal_RF_QL.png")
    plot_error_distribution(output_df, "static/histogram_erreurs_RF_QL.png")
    visualize_limited_predictions(output_df, "static/limite_comparaison_RF_QL.png")

    return model_rf, predicted_values, mse, mae, r2, time.time() - start_time
