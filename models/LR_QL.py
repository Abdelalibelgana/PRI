import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time

def regression_linear_q_learning(X, Y, alpha=0.5, beta=0.3, gamma=0.2, epochs=50):
    """
    Combine la régression linéaire et le Q-learning pour prédire et optimiser les prix.

    Args:
        X (pd.DataFrame): Données d'entrée contenant les colonnes nécessaires.
        Y (pd.Series): Cible contenant les prix réels.
        alpha, beta, gamma (float): Coefficients pour la demande estimée.
        epochs (int): Nombre d'époques pour le Q-learning.

    Returns:
        pd.DataFrame: Résultats finaux avec décisions optimisées.
        float: Temps d'exécution.
    """
    start_time = time.time()
    # Créer le modèle de régression linéaire
    model = LinearRegression()

    # Initialiser les listes pour les prédictions et les vraies valeurs
    predicted_values = []
    true_values = []
    csv_predictions = []

    # Vérifier si la colonne 'Product' existe bien dans X
    if "Product" not in X.columns:
        raise ValueError("Column 'Product' not found in DataFrame.")
    # Nettoyer les valeurs de la colonne 'Customer_Review'
    if 'Customer_Review' in X.columns:
        X['Customer_Review'] = X['Customer_Review'].astype(str).str.replace(',', '.')  # Remplacer les virgules par des points
        X['Customer_Review'] = pd.to_numeric(X['Customer_Review'], errors='coerce')  # Convertir en float
        X = X.dropna(subset=['Customer_Review'])  # Supprimer les lignes avec des valeurs non valides


    print(X.dtypes)
    print(X['Customer_Review'].head())

    # Classe Q-Learning Agent
    class QLearningAgent:
        def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
            self.q_table = {}
            self.learning_rate = learning_rate
            self.discount_factor = discount_factor
            self.exploration_rate = exploration_rate
            self.exploration_decay = exploration_decay
            self.actions = [-0.05, 0, 0.05]  # Réduction, maintien ou augmentation du prix

        def _get_state(self, produit):
            return (
                round(float(produit.get('Price', 0)), 2),
                round(float(produit.get('Quantity', 0)), 1),
                round(float(produit.get('Customer_Review', 0)), 1)
            )

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

    # Initialisation de l'agent
    agent = QLearningAgent()

    # Groupement par produit pour effectuer les prédictions de chaque produit indépendamment
    for product_code, product_data in X.groupby("Product"):
        product_data = product_data.sort_values("Date")

        # Ajouter les colonnes essentielles si elles existent
        X_prod = product_data[['Date', 'Price', 'Product']].copy()

        if 'Quantity' in product_data.columns and product_data['Quantity'].notna().any():
            X_prod['Quantity'] = product_data['Quantity']

        if 'Category' in product_data.columns and product_data['Category'].notna().any():
            X_prod['Category'] = product_data['Category']

        if 'Customer_Review' in product_data.columns and product_data['Customer_Review'].notna().any():
            X_prod['Customer_Review'] = product_data['Customer_Review']

        if 'Competing_Price' in product_data.columns and product_data['Competing_Price'].notna().any():
            X_prod['Competing_Price'] = product_data['Competing_Price']

        # Prétraitement des colonnes
        X_prod['Date'] = pd.to_datetime(X_prod['Date']).map(datetime.toordinal)
        numeric_columns = X_prod.select_dtypes(include=[np.number]).columns
        y_prod = Y[product_data.index]
       
        n_occurrences = len(product_data)

        if n_occurrences > 1:
            if n_occurrences <= 5:
                X_train = X_prod.iloc[:-1]
                y_train = y_prod.iloc[:-1]
                X_test = X_prod.iloc[-1:]
                y_test = y_prod.iloc[-1]
            else:
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
            predicted_prices = model.predict(X_train_scaled)
            product_data['Predicted_Price'] = np.concatenate(
                (predicted_prices, model.predict(X_test_scaled)), axis=0
            )

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

            # Décision finale après Q-learning
            final_state = agent._get_state(product_data.iloc[-1])
            best_action = max(agent.q_table.get(final_state, {0: 0}), key=agent.q_table.get(final_state, {0: 0}).get)
            explanation = (
                "Prix augmenté pour maximiser le revenu estimé." if best_action > 0 else
                "Prix réduit pour stimuler la demande." if best_action < 0 else
                "Prix maintenu pour conserver une demande stable."
            )

            # Stocker les résultats finaux
            csv_predictions.append({
                'Produit': product_code,
                'Prix Actuel': product_data.iloc[-1]['Price'],
                'Prix Initial': product_data.iloc[-1]['Predicted_Price'],
                'Décision Finale': best_action,
                'Prix Optimisé': product_data.iloc[-1]['Price'] * (1 + best_action),
                'Explication': explanation
            })
    

    output_df = pd.DataFrame(csv_predictions)
    # Calculer le temps d'exécution
    execution_time = time.time() - start_time
    print ("voici les première colone = ", output_df['Prix Actuel'].head())
    mse = mean_squared_error(output_df['Prix Actuel'], output_df['Prix Optimisé'])
    # Sauvegarder les prédictions
   
    output_df.to_csv('static/LR_QL.csv', index=False)
    print(f"MSE global: {mse}")

    return model, output_df, mse, execution_time
