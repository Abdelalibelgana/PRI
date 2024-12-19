import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler

def train_q_learning(X, Y, alpha=0.5, beta=0.3, gamma=0.2, epochs=50):
    """
    Entraîne un modèle de Q-Learning pour optimiser les décisions de tarification.

    Args:
        X (pd.DataFrame): Données d'entrée contenant les colonnes nécessaires (Date, Price, Product, etc.).
        Y (pd.Series): Cible contenant les quantités ou revenus associés.
        alpha (float): Poids pour les avis clients.
        beta (float): Poids pour la quantité achetée.
        gamma (float): Poids pour l'effet du prix.
        epochs (int): Nombre d'époques pour l'entraînement.

    Returns:
        pd.DataFrame: Résultats finaux contenant les décisions de tarification.
        dict: Q-table finale.
        float: Temps d'exécution.
    """
 
    start_time = time.time()
    print("Je suis dans le Q-learning")

    # Vérifier si les colonnes nécessaires sont présentes
    required_columns = ['Date', 'Price', 'Product', 'Quantity', 'Customer_Review']
    for col in required_columns:
        if col not in X.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Ajouter la conversion de la colonne 'Date' en ordinal
    X['Date'] = pd.to_datetime(X['Date'], errors='coerce')
    X['Customer_Review'] = pd.to_numeric(X['Customer_Review'].str.replace(',', '.'), errors='coerce')
    X['date_numeric'] = X['Date'].map(lambda x: x.toordinal() if pd.notnull(x) else None)

    # Conversion des avis clients en valeurs numériques
    #X['Customer_Review'] = pd.to_numeric(X['Customer_Review'], errors='coerce')

    # Supprimer les lignes avec des valeurs manquantes dans les colonnes critiques
    X = X.dropna(subset=['date_numeric', 'Price', 'Quantity', 'Customer_Review'])

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
                round(produit['Price'], 2),
                round(produit['Quantity'], 1),
                round(produit['Customer_Review'], 1)
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
            produit['Demande estimée'] = alpha * produit['Customer_Review'] + \
                                         beta * produit['Quantity'] + \
                                         gamma * (1 / (1 + np.exp(-produit['Price'])))
            produit['Revenu estimé'] = produit['Price'] * produit['Demande estimée']
            return produit

    # Initialisation de l'agent
    agent = QLearningAgent()

    # Résultats finaux
    output_data = []

    # Groupement par produit
    for product_code, product_data in X.groupby('Product'):
        product_data = product_data.sort_values('date_numeric')

        # Phase d'entraînement
        for epoch in range(epochs):
            for _, produit in product_data.iterrows():
                state = agent._get_state(produit)
                action = agent.choisir_action(state)
                produit = agent.appliquer_action(produit, action)
                reward = produit['Revenu estimé']
                next_state = agent._get_state(produit)
                agent.mettre_a_jour_q_table(state, action, reward, next_state)
            agent.decroissance_exploration()

        # Décision finale après entraînement
        final_state = agent._get_state(product_data.iloc[-1])
        best_action = max(agent.q_table.get(final_state, {0: 0}), key=agent.q_table.get(final_state, {0: 0}).get)

        explanation = (
            "Prix augmenté pour maximiser le revenu estimé." if best_action > 0 else
            "Prix réduit pour stimuler la demande." if best_action < 0 else
            "Prix maintenu pour conserver une demande stable."
        )

        output_data.append({
            'Produit': product_code,
            'Prix Actuel': product_data.iloc[-1]['Price'],
            'Décision Finale': best_action,
            'Prix Final': product_data.iloc[-1]['Price'] * (1 + best_action),
            'Explication': explanation
        })

    # Créer le DataFrame des résultats
    output_df = pd.DataFrame(output_data)

    # Temps d'exécution
    execution_time = time.time() - start_time
    return output_df, agent.q_table, execution_time
