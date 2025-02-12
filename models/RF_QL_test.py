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
from sklearn.metrics import mean_absolute_error, r2_score


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


import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def random_forest_q_learning_test(X, Y, alpha=0.5, beta=0.3, gamma=0.2, epochs=25, epsilon=1.0, gamma_q=0.9, learning_rate=0.1, exploration_decay=0.995):
    start_time = time.time()
    print("Démarrage de Q-Learning avec mise à jour continue du Random Forest")

    # 1️⃣ Vérification et prétraitement des données
    required_columns = ['Date', 'Price', 'Competing_Price', 'Quantity', 'Customer_Review']
    for col in required_columns:
        if col not in X.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    X['Date'] = pd.to_datetime(X['Date'], errors='coerce')
    X['date_numeric'] = X['Date'].map(lambda x: x.toordinal() if pd.notnull(x) else None)
    X = X.dropna(subset=['date_numeric', 'Price'])

    # Conversion des colonnes numériques
    numeric_columns = ['Price', 'Competing_Price', 'Quantity', 'Customer_Review']
    for col in numeric_columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # 2️⃣ Initialisation de Q-Learning avec les données brutes
    class QLearningAgent:
        def __init__(self, learning_rate, gamma, epsilon, exploration_decay):
            self.q_table = {}
            self.learning_rate = learning_rate
            self.gamma = gamma
            self.epsilon = epsilon
            self.exploration_decay = exploration_decay
            self.actions = [-0.05, 0, 0.05]  # -5%, 0, +5% 

        def get_state(self, produit):
            return tuple(round(produit[col], 2) for col in ['Price', 'Competing_Price', 'Quantity', 'Customer_Review'])
        
        def choose_action(self, state):
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.actions)  # Exploration aléatoire
            
            if state not in self.q_table:
                self.q_table[state] = {a: 0 for a in self.actions}  # Initialisation Q-Table
            
            return max(self.q_table[state], key=lambda a: self.q_table[state][a])  # Exploitation

        def update_q_table(self, state, action, reward, next_state):
            if state not in self.q_table:
                self.q_table[state] = {a: 0 for a in self.actions}
            if next_state not in self.q_table:
                self.q_table[next_state] = {a: 0 for a in self.actions}
            current_q = self.q_table[state][action]
            max_future_q = max(self.q_table[next_state].values())
            self.q_table[state][action] = (1 - self.learning_rate) * current_q + \
                                          self.learning_rate * (reward + self.gamma * max_future_q)

        def decay_exploration(self):
            self.epsilon *= self.exploration_decay  # Diminution progressive d'exploration

    agent = QLearningAgent(learning_rate, gamma_q, epsilon, exploration_decay)

    # 3️⃣ Initialisation de Random Forest avec un petit jeu de données
    print("Initialisation du Random Forest")
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[['date_numeric', 'Competing_Price', 'Quantity', 'Customer_Review']])
    model_rf.fit(X_scaled, Y)  # Entraînement initial

    # Stocker l'ensemble d'entraînement mis à jour au fil du temps
    training_data = X.copy()
    training_labels = Y.copy()

    # 4️⃣ Boucle d'apprentissage Q-Learning avec mise à jour continue du RF
    for episode in range(epochs):

        state = X.sample(n=1).iloc[0]  # Sélection aléatoire d’un état initial
        state_tuple = agent.get_state(state)

        max_iterations = min(len(X), 100)  # Limite à 100 itérations par epoch
        for _ in range(max_iterations):  
            
            # Exploration à travers les données
            action = agent.choose_action(state_tuple)
            print(f"Épisode {episode}, itération {_}, état actuel : {state_tuple}, action choisie : {action}")
            # Simuler la nouvelle situation après l'action
            new_price = state['Price'] * (1 + action)  # Ajustement du prix (-5%, 0, +5%)
            new_state = state.copy()
            new_state['Price'] = new_price
            
            # Mise à jour de la quantité en fonction du prix et de la concurrence
            new_state['Quantity'] = alpha * new_state['Quantity'] + beta * new_state['Competing_Price'] + gamma * new_price
            
            # **Prédiction avec Random Forest après l'action**
            # Convertir new_state en DataFrame avant de le passer au StandardScaler
            new_state_df = pd.DataFrame([new_state], columns=['date_numeric', 'Competing_Price', 'Quantity', 'Customer_Review'])

            # Appliquer le scaler sur les données formatées
            new_state_scaled = scaler.transform(new_state_df)

            # Prédire le prix avec le modèle Random Forest
            predicted_price = model_rf.predict(new_state_scaled)[0]
            new_state['Price'] = predicted_price

            
            next_state_tuple = agent.get_state(new_state)

            # Calcul de la récompense basée sur le revenu
            reward = new_price * new_state['Quantity']

            # Mise à jour de la Q-Table
            agent.update_q_table(state_tuple, action, reward, next_state_tuple)

            # **Ajout de la nouvelle donnée à l'ensemble d'entraînement**
            training_data = pd.concat([training_data, pd.DataFrame([new_state])], ignore_index=True)
            training_labels = pd.concat([training_labels, pd.Series([predicted_price])], ignore_index=True)


            # **Réentraînement de Random Forest avec la nouvelle donnée**
            if _ % 10 == 0:  # Met à jour le modèle toutes les 10 itérations au lieu de chaque itération
                training_data_scaled = scaler.fit_transform(training_data[['date_numeric', 'Competing_Price', 'Quantity', 'Customer_Review']])
                model_rf.fit(training_data_scaled, training_labels)

            # Passage au nouvel état
            state_tuple = next_state_tuple

        agent.decay_exploration()

    # 5️⃣ Évaluation des performances finales
    X_test_scaled = scaler.transform(X[['date_numeric', 'Competing_Price', 'Quantity', 'Customer_Review']])
    predicted_prices = model_rf.predict(X_test_scaled)
    output_df = pd.DataFrame({'Prix Actuel': Y.values, 'Prix Optimisé': predicted_prices})
    visualize_predictions(output_df, "static/prix_comparaison_RF_QL_test.png")
    scatter_predictions_with_ideal(output_df, "static/scatter_comparaison_ideal_RF_QL_test.png")
    plot_error_distribution(output_df, "static/histogram_erreurs_RF_QL_test.png")
    visualize_limited_predictions(output_df, "static/limite_comparaison_RF_QL_test.png")

    # 📌 Calcul des métriques
    mse = mean_squared_error(Y, predicted_prices)
    mae = mean_absolute_error(Y, predicted_prices)
    r2 = r2_score(Y, predicted_prices)
    training_time = time.time() - start_time

    print(f"MSE global: {mse}")
    print(f"MAE global: {mae}")
    print(f"R² global: {r2}")
    
    return model_rf, predicted_prices, mse, mae, r2, training_time
