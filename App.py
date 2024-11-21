from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory
import os
import pandas as pd
import time

from models.RandomForest import train_random_forest_model
from models.LinearRegression import train_linear_model

app = Flask(__name__)

# Définir un dossier pour enregistrer les fichiers téléchargés
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/view-data', methods=['GET', 'POST'])  # Permet à la route de gérer GET et POST
def view_results():
    if request.method == 'POST':
        # Vérifier si un fichier a été téléchargé
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            # Lire le fichier CSV dans un DataFrame pandas
            df = pd.read_csv(file)
            # Afficher les 5 premières lignes du DataFrame
            head_data = df.head()
            # Convertir les données en un format qui peut être utilisé dans HTML
            columns = df.columns.tolist()
            rows = head_data.values.tolist()
            return render_template('data.html', columns=columns, rows=rows)
        else:
            return "Invalid file format. Please upload a CSV file."

    # Si c'est une requête GET, afficher simplement le formulaire pour télécharger un fichier
    return render_template('upload_csv.html')

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/process-columns', methods=['POST'])
def process_columns():
    # Récupérer les colonnes sélectionnées depuis le formulaire
    prix_column = request.form['prix']
    date_column = request.form['date']
    product_name_column = request.form['product_name']
    quantity_column = request.form.get('quantity')
    category_column = request.form.get('category')
    model_choice = request.form['model']

    # Afficher ou traiter les colonnes et le modèle choisis
    print(f"Model: {model_choice}")
    print(f"Columns chosen: {prix_column}, {date_column}, {product_name_column}, {quantity_column}, {category_column}")

    # Selon le modèle choisi, vous pouvez entraîner un modèle spécifique
    if model_choice == "linear_regression":
        # Entraîner le modèle de régression linéaire ici
        pass
    elif model_choice == "random_forest":
        # Entraîner le modèle Random Forest ici
        pass

    # Après l'entraînement, rediriger ou afficher une page de résultats
    return render_template('confirmation.html', model=model_choice, 
                           prix=prix_column, date=date_column, 
                           product_name=product_name_column)

# Route pour afficher la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')

# Route pour la comparaison des modèles
@app.route('/compare', methods=['GET', 'POST'])
def compare_models():
    if request.method == 'POST':
        # Gérer l'importation du fichier CSV
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Sauvegarder le fichier téléchargé
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Charger le fichier CSV dans un DataFrame
            df = pd.read_csv(file_path)

            # Entraîner les modèles
            linear_model_results = train_linear_model(df)  # Résultats du modèle de régression linéaire
            random_forest_results = train_random_forest_model(df)  # Résultats du modèle Random Forest

            # Comparer les MSE
            if linear_model_results['mse'] < random_forest_results['mse']:
                best_model = "Linear Regression"
                best_mse = linear_model_results['mse']
                prediction_file = linear_model_results['output_file']
            else:
                best_model = "Random Forest"
                best_mse = random_forest_results['mse']
                prediction_file = random_forest_results['output_file']

            # Afficher les résultats sur une nouvelle page
            return render_template('compare_results.html', 
                                   best_model=best_model, 
                                   best_mse=best_mse, 
                                   prediction_file=prediction_file)

    return render_template('compare_models.html')  # Si GET, afficher la page

# Route pour afficher les résultats du modèle
@app.route('/compare-results', methods=['GET'])
def compare_results():
    best_model = request.args.get('best_model')
    best_mse = request.args.get('best_mse')
    prediction_file = request.args.get('prediction_file')

    # Afficher les résultats de comparaison
    return render_template('compare_results.html', 
                           best_model=best_model, 
                           best_mse=best_mse, 
                           prediction_file=prediction_file)


@app.route('/choose-model', methods=['GET', 'POST'])
def choose_model():
    if request.method == 'POST':
        # Gérer l'importation du fichier et le modèle choisi
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            if file:
                # Sauvegarder le fichier téléchargé
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                
                # Charger le fichier CSV dans un DataFrame
                df = pd.read_csv(file_path)
                
                # Récupérer le modèle choisi
                model_choice = request.form.get('model')

                # Enregistrer le temps d'entraînement
                start_time = time.time()

                # Entraîner le modèle choisi
                if model_choice == "linear_regression":
                    model_results = train_linear_model(df)  # Passer le DataFrame directement
                elif model_choice == "random_forest":
                    model_results = train_random_forest_model(df)  # Passer le DataFrame directement
                
                # Calculer le temps d'entraînement
                end_time = time.time()
                training_time = end_time - start_time
                
                # Ajouter le temps d'entraînement aux résultats
                model_results['training_time'] = training_time

                # Rediriger vers la page des résultats avec les données en paramètres d'URL
                return redirect(url_for('show_results', mse=model_results['mse'], training_time=model_results['training_time'], output_file=model_results['output_file']))

    return render_template('train_with_single_model.html')  # Si GET, afficher la page

# Route pour afficher les résultats
@app.route('/results')
def show_results():
    mse = request.args.get('mse')  # Récupérer les résultats passés dans l'URL
    training_time = request.args.get('training_time')
    output_file = request.args.get('output_file')

    # Si aucun résultat n'est disponible, rediriger vers la page d'accueil
    if mse is None or training_time is None:
        return redirect(url_for('index'))

    return render_template('results.html', mse=mse, training_time=training_time, output_file=output_file)


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug=True, host="127.0.0.1", port=8080)  # Changer ici le port
