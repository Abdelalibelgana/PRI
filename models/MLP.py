import time
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_training(history, output_path_loss, output_path_mae):
    """
    Enregistre les graphiques de l'historique d'entraînement.

    Paramètres :
        history : History - Historique de l'entraînement.
        output_path_loss : str - Chemin pour sauvegarder le graphique de perte.
        output_path_mae : str - Chemin pour sauvegarder le graphique de MAE.
    """
    # Loss
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(output_path_loss)
    plt.close()
    print(f"Graphique des pertes sauvegardé : {output_path_loss}")

    # MAE
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.savefig(output_path_mae)
    plt.close()
    print(f"Graphique de MAE sauvegardé : {output_path_mae}")


def scatter_plot(actual, predicted, output_path, title, color='blue'):
    """
    Enregistre un graphique de dispersion des valeurs réelles vs prédites.

    Paramètres :
        actual : array-like - Les valeurs réelles.
        predicted : array-like - Les valeurs prédites.
        output_path : str - Chemin pour sauvegarder le graphique.
        title : str - Le titre du graphique.
        color : str - La couleur des points.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, color=color, label='Prédictions')
    plt.plot(actual, actual, color='red', linestyle='--', label='Valeurs Réelles')
    plt.title(title)
    plt.xlabel('Valeurs Réelles')
    plt.ylabel('Valeurs Prédites')
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique de dispersion sauvegardé : {output_path}")


def mlp_model_q_learning(X, Y):
    """
    Implémente un modèle MLP combiné avec Q-Learning pour prédire les prix des produits.

    Paramètres :
        X : pandas.DataFrame - Les caractéristiques d'entrée (y compris les dates si disponibles).
        Y : pandas.Series - La cible (prix).

    Retourne :
        results : pandas.DataFrame - Les prédictions du modèle.
        q_table : dict - La table Q utilisée pour l'apprentissage par renforcement.
        execution_time : float - Temps d'exécution du modèle.
        history : History object - Historique de l'entraînement.
        Y_train, Y_train_pred, Y_val, Y_val_pred, Y_test, Y_test_pred : array-like - Données et prédictions.
    """
    from datetime import datetime

    # Mesurer le temps d'exécution
    start_time = time.time()

    # Vérifiez si une colonne de date est présente
    if 'Date' in X.columns:
        dates = X['Date']
        X = X.drop(columns=['Date'])  # Retirez temporairement la colonne de date pour l'entraînement
    else:
        raise ValueError("La colonne 'Date' doit être incluse dans X.")

    # Sélectionner uniquement les colonnes numériques
    X_numeric = X.select_dtypes(include=[np.number])

    # Prétraitement des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    # Séparation des données en ensembles d'entraînement, de validation et de test
    X_train, X_temp, Y_train, Y_temp, dates_train, dates_temp = train_test_split(
        X_scaled, Y, dates, test_size=0.4, random_state=50
    )
    X_val, X_test, Y_val, Y_test, dates_val, dates_test = train_test_split(
        X_temp, Y_temp, dates_temp, test_size=0.5, random_state=50
    )

    # Construction du modèle MLP
    model = Sequential([
        Dense(512, input_dim=X_train.shape[1]),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128),
        LeakyReLU(alpha=0.1),
        Dense(1, activation='linear')  # Sortie pour la régression
    ])

    optimizer = "adam"
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Ajout de l'arrêt anticipé et du réducteur de taux d'apprentissage
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Entraînement du modèle avec callbacks
    history = model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=64,
        verbose=1,
        validation_data=(X_val, Y_val),
        callbacks=[early_stopping, lr_scheduler]
    )

    # Prédictions
    Y_train_pred = model.predict(X_train)
    Y_val_pred = model.predict(X_val)
    Y_test_pred = model.predict(X_test)

    # Calcul de la MSE
    mse = mean_squared_error(Y_test, Y_test_pred)
    print(f"Mean Squared Error (MSE): {mse}")

    # Simuler une table Q (fictive pour ce modèle)
    q_table = {}
    for i in range(len(X_test)):
        state = tuple(X_test[i])  # Représentation de l'état
        action = Y_test_pred[i]  # Action prédite
        q_table[state] = action

    # Temps d'exécution
    execution_time = time.time() - start_time

    #date_str = datetime.now().strftime('%Y-%m-%d')
    # Résultats avec les dates ajoutées
    results = pd.DataFrame({
        'Date': dates_test.values,  # Ajouter les dates associées
        'Actual': Y_test.values,
        #'Predection date ' : date_str,
        'Predicted': Y_test_pred.flatten()
    })

    
    results_file = f'static/MLP.csv'

    results.to_csv(results_file, index=False)
    print(f"Fichier des prédictions sauvegardé : {results_file}")

    # Visualisations
    plot_training(history, output_path_loss='static/training_loss_mlp.png', output_path_mae='static/training_mae_mlp.png')

    scatter_plot(Y_train, Y_train_pred, output_path='static/scatter_train_mlp.png', title='Ensemble Entraînement - Réel vs Prédit', color='green')

    scatter_plot(Y_val, Y_val_pred, output_path='static/scatter_validation_mlp.png', title='Ensemble Validation - Réel vs Prédit', color='orange')

    scatter_plot(Y_test, Y_test_pred, output_path='static/scatter_test_mlp.png', title='Ensemble Test - Réel vs Prédit', color='purple')

    return results, q_table, mse, execution_time

