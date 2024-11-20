# test_import.py
import pandas as pd
from models.RandomForest import train_random_forest_model
from models.LinearRegression import train_linear_model
import os 

# Changez le répertoire courant pour faciliter le chargement des fichiers
os.chdir('/Users/abdelalibelgana/Desktop/DI5/Projet_PRI/Harverd')
# Charger le fichier fusionné
merged_df = pd.read_csv('merged_amazon_data.csv')

df = pd.DataFrame(merged_df)

# Appeler la fonction avec ce DataFrame
#train_random_forest_model(df)  # Appel avec un DataFrame valide
train_linear_model(df)
