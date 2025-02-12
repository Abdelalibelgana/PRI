from flask import Flask, render_template, request, redirect, url_for, session
from flask import send_from_directory
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
import time
import uuid 
from datetime import timedelta

from models.RandomForest import train_random_forest_Gene
from models.LinearRegression import train_linear_regression_Gene
from models.RL_model import train_q_learning
from models.LR_QL import regression_linear_q_learning
from models.RF_QL import random_forest_q_learning   
from models.MLP import mlp_model
from models.RF_QL_test import random_forest_q_learning_test
from models.LR_QL_test import linear_regression_q_learning_test
from datetime import datetime
import matplotlib.pyplot as plt

app = Flask(__name__)

# Définir un dossier pour enregistrer les fichiers téléchargés
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/view-data', methods=['GET', 'POST'])  
def view_results():
    if request.method == 'POST':
        # Vérifier si un fichier a été téléchargé
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            # Générer un nom de fichier unique pour éviter les conflits
            unique_filename = str(uuid.uuid4()) + ".csv"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)  # Sauvegarder le fichier sur le serveur
            print("Path of the uploaded file shows data: ", file_path)

            # Passer le chemin du fichier via l'URL avec un paramètre
            return redirect(url_for('process_columns', file_path=file_path))  # Passer file_path dans l'URL

        else:
            return "Invalid file format. Please upload a CSV file."

    # Si c'est une requête GET, afficher simplement le formulaire pour télécharger un fichier
    return render_template('upload_csv.html')

@app.route('/view-data-Comapraisson', methods=['GET', 'POST'])  
def view_results_comparaison():
    if request.method == 'POST':
        # Vérifier si un fichier a été téléchargé
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            # Générer un nom de fichier unique pour éviter les conflits
            unique_filename = str(uuid.uuid4()) + ".csv"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)  # Sauvegarder le fichier sur le serveur
            print("Path of the uploaded file shows data: ", file_path)

            # Passer le chemin du fichier via l'URL avec un paramètre
            return redirect(url_for('process_columns_comparaisson', file_path=file_path))  # Passer file_path dans l'URL

        else:
            return "Invalid file format. Please upload a CSV file."

    # Si c'est une requête GET, afficher simplement le formulaire pour télécharger un fichier
    return render_template('upload_csv_comparaison.html')

 
@app.route('/process-columns', methods=['GET', 'POST'])
def process_columns():
    # Si la méthode est GET, récupérez le chemin du fichier depuis les paramètres d'URL
    if request.method == 'GET':
        uploaded_file_path = request.args.get('file_path')
        #print("Path of the uploaded file:", uploaded_file_path)
        
        if not uploaded_file_path:
            return "No file found", 400  # Si aucun fichier n'est trouvé, renvoyer une erreur

        try:
            df = pd.read_csv(uploaded_file_path)
        except Exception as e:
            return f"Error reading the CSV file: {e}", 400

        columns = df.columns.tolist()
        return render_template('data.html', columns=columns, rows=df.head().values.tolist())
    
    # Si la méthode est POST, traiter les colonnes et effectuer l'entraînement
    if request.method == 'POST':
        #print("Je suis dans le process colonne POST")

        prix_column = request.form['prix']
        date_column = request.form['date']
        product_name_column = request.form['product_name']  # Prendre le produit sélectionné
        #print(f"Product name column selected: {product_name_column}")
        quantity_column = request.form.get('quantity')  # optionnel, 
        category_column = request.form.get('category')  # optionnel
        customer_review_column = request.form.get('customer_review')  # optionnel
        competing_price_column = request.form.get('competing_price')  # optionnel
        model_choice = request.form['model']

        uploaded_file_path = request.form.get('uploaded_file_path')
        #print("Path of the uploaded file:", uploaded_file_path)
        if not uploaded_file_path:
            return "No file found", 400  # Si aucun fichier n'est trouvé, renvoyer une erreur

        try:
            df = pd.read_csv(uploaded_file_path)
        except Exception as e:
            return f"Error reading the CSV file: {e}", 400
        
        # Vérifier que les colonnes obligatoires sont présentes dans le DataFrame
        required_columns = [prix_column, date_column, product_name_column]
        for col in required_columns:
            if col not in df.columns:
                return f"Missing required column: {col}", 400  # Retourner une erreur si une colonne est manquante

        # Supprimer les lignes contenant des valeurs manquantes dans les colonnes critiques
        df_cleaned = df.dropna(subset=[prix_column, date_column, product_name_column])  # Supprimer les lignes avec des NaN dans les colonnes critiques

        # Créer X avec les colonnes spécifiées
        X = pd.DataFrame()

        # Affecter les colonnes obligatoires à X
        X['Date'] = pd.to_datetime(df_cleaned[date_column], errors='coerce')  # Convertir la date en ordinal
        X['Price'] = df_cleaned[prix_column]
        X['Product'] = df_cleaned[product_name_column].astype(str)  # Assurez-vous que product_name est une chaîne

        # Ajouter les colonnes optionnelles si elles existent, sinon ajouter NaN
        X['Quantity'] = df_cleaned[quantity_column] if quantity_column in df_cleaned.columns else None
        X['Category'] = df_cleaned[category_column] if category_column in df_cleaned.columns else None
        if customer_review_column in df_cleaned.columns:
            # Convertir la colonne en float et gérer les erreurs en les forçant à NaN
           # X['Customer_Review'] = pd.to_numeric(df_cleaned[customer_review_column], errors='coerce')
           X['Customer_Review'] = df_cleaned[customer_review_column]
        else:
            # Si la colonne n'existe pas, on remplace par NaN
            X['Customer_Review'] = None
        X['Competing_Price'] = df_cleaned[competing_price_column] if competing_price_column in df_cleaned.columns else None
        #print ("pppppc")
        #print(X.head(3))
        # Définir Y avant le nettoyage
        Y = df_cleaned[prix_column]
        visualisation = 0
        # Entraîner le modèle en fonction du choix
        if model_choice == "linear_regression":
            visualisation = 1
            model, predictions, mse, mae, r2, y_test,training_time = train_linear_regression_Gene(X, Y)
            prediction_file = 'predicted_prices_Linear.csv'
            model_CH = "Linear Regression"
            visualize_predictions = "prix_comparaison_LR.png"
            scatter_predictions_with_ideal = "scatter_comparaison_ideal_LR.png"
            plot_error_distribution = "histogram_erreurs_LR.png"
            visualize_limited_predictions =  "limite_comparaison_LR.png"
        elif model_choice == "random_forest":
            visualisation = 1
            model, predictions,mse, mae, r2, y_test,training_time = train_random_forest_Gene(X, Y)
            prediction_file = 'predicted_prices_Forest.csv'
            model_CH = "Random Forest"
            visualize_predictions = "prix_comparaison_RF.png"
            scatter_predictions_with_ideal = "scatter_comparaison_ideal_RF.png"
            plot_error_distribution = "histogram_erreurs_RF.png"
            visualize_limited_predictions =  "limite_comparaison_RF.png"
        elif model_choice == "LR_QL": 
            visualisation = 1
            output_df, table, mse, mae, r2, training_time = regression_linear_q_learning(X,Y)
            prediction_file = 'LR_QL.csv'
            model_CH = "Linear Regression + Q_Learning"
            visualize_predictions = "prix_comparaison_LR_QL.png"
            scatter_predictions_with_ideal = "scatter_comparaison_ideal_LR_QL.png"
            plot_error_distribution = "histogram_erreurs_LR_QL.png"
            visualize_limited_predictions =  "limite_comparaison_LR_QL.png"
        elif model_choice == "RF_QL": 
            visualisation = 1
            output_df, table, mse, mae, r2, training_time = random_forest_q_learning(X,Y)
            prediction_file = 'RF_QL.csv'
            model_CH = "Random Forest + Q_Learning"
            visualize_predictions = "prix_comparaison_RF_QL.png"
            scatter_predictions_with_ideal = "scatter_comparaison_ideal_RF_QL.png"
            plot_error_distribution = "histogram_erreurs_RF_QL.png"
            visualize_limited_predictions =  "limite_comparaison_RF_QL.png"
        elif model_choice == "LR_Q_test": 
            visualisation = 1
            print ('je suis la') 
            output_df, table, mse, mae, r2, training_time = linear_regression_q_learning_test(X,Y)
            prediction_file = 'LR_QL_test.csv'
            model_CH = "Linear Regression + Q_Learning"
            visualize_predictions = "prix_comparaison_LR_QL_test.png"
            scatter_predictions_with_ideal = "scatter_comparaison_ideal_LR_QL_test.png"
            plot_error_distribution = "histogram_erreurs_LR_QL_test.png"
            visualize_limited_predictions =  "limite_comparaison_LR_QL_test.png"
        elif model_choice == "MLP": 
            visualisation = 1
            output_df, mse, mae, r2, training_time = mlp_model(X,Y)
            prediction_file = 'MLP.csv'
            model_CH = "MLP"
            visualize_predictions = "training_loss_mlp.png"
            scatter_predictions_with_ideal = "scatter_train_mlp.png"
            plot_error_distribution = "scatter_validation_mlp.png"
            visualize_limited_predictions =  "scatter_test_mlp.png"
        elif model_choice == "RF_QL_test": 
            visualisation = 1
            output_df, table, mse, mae, r2, training_time = random_forest_q_learning_test(X,Y)
            prediction_file = 'RF_QL_test.csv'
            model_CH = "Random Forest + Q_Learning"
            visualize_predictions = "prix_comparaison_RF_QL_test.png"
            scatter_predictions_with_ideal = "scatter_comparaison_ideal_RF_QL_test.png"
            plot_error_distribution = "histogram_erreurs_RF_QL_test.png"
            visualize_limited_predictions =  "limite_comparaison_RF_QL_test.png"

            

        elif model_choice == "RL": 
            output_df, table, training_time = train_q_learning(X,Y)
            mse = 0
            prediction_file = 'RL.csv'
            model_CH = "RL"
           



        # Créer une liste de tuples (actual, predicted)
        #actual_predicted = list(zip(y_test, predictions))
        # Formater le temps d'entraînement en hh:mm:ss
        print (training_time)
        formatted_training_time = str(timedelta(seconds=int(training_time)))
        prediction_file_path = os.path.join('static', prediction_file)
        predicted_df = pd.read_csv(prediction_file_path)
        top_10_rows = predicted_df.head(10).to_html(classes="table table-striped", index=False)
        # Après l'entraînement, rediriger ou afficher une page de résultats
        if (visualisation == 1) : 
            return render_template('view_data_result.html', model=model_CH, 
                                prix=prix_column, date=date_column, 
                                product_name=product_name_column,
                                quantity=quantity_column, category=category_column,
                                customer_review=customer_review_column, 
                                competing_price=competing_price_column, mse=mse, mae = mae, r2 = r2,
                                visualize_predictions = visualize_predictions ,
                                scatter_predictions_with_ideal = scatter_predictions_with_ideal,
                                plot_error_distribution= plot_error_distribution ,
                                visualize_limited_predictions = visualize_limited_predictions ,
                                training_time= formatted_training_time,
                                prediction_file =prediction_file, 
                                top_10_rows = top_10_rows)
        else : 
            return render_template('view_data.html', model=model_CH, 
                                prix=prix_column, date=date_column, 
                                product_name=product_name_column,
                                quantity=quantity_column, category=category_column,
                                customer_review=customer_review_column, 
                                competing_price=competing_price_column, mse=mse, 
                                training_time= formatted_training_time,
                                prediction_file =prediction_file, 
                                top_10_rows = top_10_rows)

    uploaded_file_path = request.args.get('file_path')
    #print("Path of the uploaded file:", uploaded_file_path)

    if not uploaded_file_path:
        return "No file found", 400  # Si aucun fichier n'est trouvé, renvoyer une erreur

    try:
        df = pd.read_csv(uploaded_file_path)
    except Exception as e:
        return f"Error reading the CSV file: {e}", 400

    columns = df.columns.tolist()

    return render_template('data.html', columns=columns, rows=df.head().values.tolist())

@app.route('/download_file/<filename>')
def download_file(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)
@app.route('/process-columns-comparaisson', methods=['GET', 'POST'])
def process_columns_comparaisson():
    # Si la méthode est GET, récupérez le chemin du fichier depuis les paramètres d'URL
    if request.method == 'GET':
        uploaded_file_path = request.args.get('file_path')
        #print("Path of the uploaded file:", uploaded_file_path)
        
        if not uploaded_file_path:
            return "No file found", 400  # Si aucun fichier n'est trouvé, renvoyer une erreur

        try:
            df = pd.read_csv(uploaded_file_path)
        except Exception as e:
            return f"Error reading the CSV file: {e}", 400

        columns = df.columns.tolist()
        return render_template('data_comparaison.html', columns=columns, rows=df.head().values.tolist())
    
    # Si la méthode est POST, traiter les colonnes et effectuer l'entraînement
    if request.method == 'POST':
        #print("Je suis dans le process colonne POST")

        prix_column = request.form['prix']
        date_column = request.form['date']
        product_name_column = request.form['product_name']  # Prendre le produit sélectionné
        #print(f"Product name column selected: {product_name_column}")
        quantity_column = request.form.get('quantity')  # optionnel, 
        category_column = request.form.get('category')  # optionnel
        customer_review_column = request.form.get('customer_review')  # optionnel
        competing_price_column = request.form.get('competing_price')  # optionnel
        with_reinforcement_learning = request.form.get('with_reinforcement_learning')
        

        uploaded_file_path = request.form.get('uploaded_file_path')
        #print("Path of the uploaded file:", uploaded_file_path)
        if not uploaded_file_path:
            return "No file found", 400  # Si aucun fichier n'est trouvé, renvoyer une erreur

        try:
            df = pd.read_csv(uploaded_file_path)
        except Exception as e:
            return f"Error reading the CSV file: {e}", 400
        
        # Vérifier que les colonnes obligatoires sont présentes dans le DataFrame
        required_columns = [prix_column, date_column, product_name_column]
        for col in required_columns:
            if col not in df.columns:
                return f"Missing required column: {col}", 400  # Retourner une erreur si une colonne est manquante

        # Supprimer les lignes contenant des valeurs manquantes dans les colonnes critiques
        df_cleaned = df.dropna(subset=[prix_column, date_column, product_name_column])  # Supprimer les lignes avec des NaN dans les colonnes critiques

        # Créer X avec les colonnes spécifiées
        X = pd.DataFrame()

        # Affecter les colonnes obligatoires à X
        X['Date'] = pd.to_datetime(df_cleaned[date_column], errors='coerce')  # Convertir la date en ordinal
        X['Price'] = df_cleaned[prix_column]
        X['Product'] = df_cleaned[product_name_column].astype(str)  # Assurez-vous que product_name est une chaîne

        # Ajouter les colonnes optionnelles si elles existent, sinon ajouter NaN
        X['Quantity'] = df_cleaned[quantity_column] if quantity_column in df_cleaned.columns else None
        X['Category'] = df_cleaned[category_column] if category_column in df_cleaned.columns else None
        if customer_review_column in df_cleaned.columns:
            # Convertir la colonne en float et gérer les erreurs en les forçant à NaN
           # X['Customer_Review'] = pd.to_numeric(df_cleaned[customer_review_column], errors='coerce')
           X['Customer_Review'] = df_cleaned[customer_review_column]
        else:
            # Si la colonne n'existe pas, on remplace par NaN
            X['Customer_Review'] = None
        X['Competing_Price'] = df_cleaned[competing_price_column] if competing_price_column in df_cleaned.columns else None
        #print ("pppppc")
        #print(X.head(3))
        # Définir Y avant le nettoyage
        Y = df_cleaned[prix_column]

        if with_reinforcement_learning == 'yes':
            print("Reinforcement Learning is enabled.")
            output_df3, mse3, mae3, r23, training_time3 = mlp_model(X,Y)
            output_df, table, mse1,mae1, r21, training_time1 = regression_linear_q_learning(X,Y)
            output_df2, table2, mse2,mae2, r22, training_time2 = random_forest_q_learning(X,Y)
            
            visualize_predictions = "prix_comparaison_LR_QL.png"
            scatter_predictions_with_ideal = "scatter_comparaison_ideal_LR_QL.png"
            plot_error_distribution = "histogram_erreurs_LR_QL.png"
            visualize_limited_predictions =  "limite_comparaison_LR_QL.png"
            visualize_predictions2 = "prix_comparaison_RF_QL.png"
            scatter_predictions_with_ideal2 = "scatter_comparaison_ideal_RF_QL.png"
            plot_error_distribution2 = "histogram_erreurs_RF_QL.png"
            visualize_limited_predictions2 =  "limite_comparaison_RF_QL.png"
               
        else:
            print("Reinforcement Learning is disabled.")
            output_df, mse3, mae3, r23, training_time3 = mlp_model(X,Y)
            model1, predictions1, mse1, mae1, r21, y_test1,training_time1 = train_linear_regression_Gene(X, Y)
            model2, predictions,mse2, mae2, r22, y_test2,training_time2 = train_random_forest_Gene(X, Y)
            
            visualize_predictions = "prix_comparaison_LR.png"
            scatter_predictions_with_ideal = "scatter_comparaison_ideal_LR.png"
            plot_error_distribution = "histogram_erreurs_LR.png"
            visualize_limited_predictions =  "limite_comparaison_LR.png"
            visualize_predictions2 = "prix_comparaison_RF.png"
            scatter_predictions_with_ideal2 = "scatter_comparaison_ideal_RF.png"
            plot_error_distribution2 = "histogram_erreurs_RF.png"
            visualize_limited_predictions2 =  "limite_comparaison_RF.png"
        # Stocker les métriques
        metrics = {
            "Linear Regression": {"MSE": mse1, "MAE": mae1, "R²": r21, "Time": training_time1},
            "Random Forest": {"MSE": mse2, "MAE": mae2, "R²": r22, "Time": training_time2},
            "MLP": {"MSE": mse3, "MAE": mae3, "R²": r23, "Time": training_time3}
        }
        # Identifier le meilleur modèle
        best_model = None
        best_metrics = {"MSE": float("inf"), "MAE": float("inf"), "R²": float("-inf")}
        for model_name, metric_values in metrics.items():
            if metric_values["MSE"] < best_metrics["MSE"]:
                best_model = model_name
                best_metrics = metric_values
            elif metric_values["MSE"] == best_metrics["MSE"] and metric_values["R²"] > best_metrics["R²"]:
                best_model = model_name
                best_metrics = metric_values

        # Générer des visualisations des métriques
        df_metrics = pd.DataFrame(metrics).T
        df_metrics.index.name = "Model"
        df_metrics.reset_index(inplace=True)
        # Passer les données des métriques sous forme de tableau
        df_metrics_html = df_metrics.to_html(classes="table table-striped", index=False)
                        

        # Préparer les temps d'entraînement
        formatted_training_time1 = str(timedelta(seconds=int(training_time1)))
        formatted_training_time2 = str(timedelta(seconds=int(training_time2)))
        formatted_training_time3 = str(timedelta(seconds=int(training_time3)))
        total_training_time = training_time1 + training_time2 + training_time3

        formatted_total_training_time = str(timedelta(seconds=int(total_training_time)))
                
       
        #prediction_file_path = os.path.join('static', prediction_file)
        #predicted_df = pd.read_csv(prediction_file_path)
        #top_10_rows = predicted_df.head(10).to_html(classes="table table-striped", index=False)
        #print( "visualize_predictions2 = ", visualize_predictions2)
        return render_template('compare_results.html', 
                                   best_model=best_model,  best_mse=best_metrics["MSE"],
                                   best_mae=best_metrics["MAE"],
                                   best_r2=best_metrics["R²"],
                                   formatted_training_time1 = formatted_training_time1,
                                   formatted_training_time2 = formatted_training_time2,
                                   formatted_training_time3=formatted_training_time3,
                                   formatted_training_time = formatted_total_training_time,
                                   metrics_table=df_metrics_html,
                                   visualize_predictions = visualize_predictions ,
                                   scatter_predictions_with_ideal = scatter_predictions_with_ideal,
                                   plot_error_distribution= plot_error_distribution ,
                                   visualize_limited_predictions = visualize_limited_predictions ,
                                   visualize_predictions2 = visualize_predictions2 ,
                                   scatter_predictions_with_ideal2 = scatter_predictions_with_ideal2,
                                   plot_error_distribution2 = plot_error_distribution2 ,
                                   visualize_limited_predictions2 = visualize_limited_predictions2 )
                                  # prediction_file=prediction_file, 
                                  # top_10_rows=top_10_rows)    

    
    uploaded_file_path = request.args.get('file_path')
    #print("Path of the uploaded file:", uploaded_file_path)

    if not uploaded_file_path:
        return "No file found", 400  # Si aucun fichier n'est trouvé, renvoyer une erreur

    try:
        df = pd.read_csv(uploaded_file_path)
    except Exception as e:
        return f"Error reading the CSV file: {e}", 400

    columns = df.columns.tolist()

    return render_template('data.html', columns=columns, rows=df.head().values.tolist())



# Route pour afficher la page d'accueil
@app.route('/')
def index():
    return render_template('index.html')





if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug=True, host="127.0.0.1", port=8080)  # Changer ici le port
