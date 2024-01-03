from flask import Flask, render_template, request, session,send_file, redirect,url_for
import pickle
import numpy as np
import pandas as pd
import joblib
from flask_mysqldb import MySQL
import MySQLdb.cursors
from werkzeug.utils import secure_filename
import io
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import seaborn as sns
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
from flask_compress import Compress

app = Flask(__name__)
Compress(app)

# *** Flask configuration
# Define folder to save uploaded files to process further
UPLOAD_FOLDER = os.path.join('static', 'uploads')

# Define allowed files (for this example I want only csv file)
ALLOWED_EXTENSIONS = {'csv'}

# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flask_app'

# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'

# Intialize MySQL
mysql = MySQL(app)

# Charger le modèle pré-entraîné
model = pickle.load(open('model.pkl', 'rb'))
# Load Dataset
data = pd.read_csv("creditcards.csv")

# Créer une instance Dash dans votre application Flask
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')
fig = px.histogram(data, x='V1', color='class',
                   title='Répartition des transactions en fonction de V1',
                   labels={'class': 'Type de transaction', 'V1': 'Valeur de la caractéristique'})

# Définir le layout Dash
dash_app.layout = html.Div([
    html.H1("Dashboard"),
    # Sélecteur de fonctionnalités (caractéristiques)
    dcc.Dropdown(
        id='feature-selector',
        options=[{'label': col, 'value': col} for col in data.columns if col not in ['class']],
        value='V1',  # Sélectionnez une caractéristique par défaut
        multi=False
    ),

    # Graphique pour visualiser la relation
    dcc.Graph(id='fraud-vs-nonfraud-graph')
])


@dash_app.callback(
    Output('fraud-vs-nonfraud-graph', 'figure'),
    [Input('feature-selector', 'value')]
)
def update_graph(selected_feature):
    # Créez un graphique de répartition des données en fonction de la caractéristique sélectionnée
    fig = px.histogram(data, x=selected_feature, color='class',
                       title=f'Répartition des transactions en fonction de {selected_feature}',
                       labels={'class': 'Type de transaction', selected_feature: 'Valeur de la caractéristique'})

    return fig


@app.route('/')
def index():
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        user = request.form['username']
        password = request.form['password']
        if user == 'admin123' and password == '123':

            return redirect(url_for('home_admin'))

        # Check if form exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (user, password,))
        # Fetch one record and return result
        account = cursor.fetchone()

        # If account exists in form table in our database
        if account:
            #Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            cursor.close()

            #if account['username']=='admin123' and account['password']=='123':
                #return render_template('admin/form.html')
            #else:
            return render_template('home.html')

        else:
            #Account doesn't exiat or username/ password incorrect
            msg ='Incorrect username/password!'
            cursor.close()

    return render_template('login.html', msg=msg)

@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    # Redirect to login page
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        user = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s', (user,))
        account = cursor.fetchone()

        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', user):
            msg = 'Username must contain only characters and numbers!'
        elif not user or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account
            cursor.execute('INSERT INTO users VALUES (NULL, %s, %s, %s)', (user, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
            cursor.close()
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'

    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)

### DownLoad File ###
@app.route("/download", methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        # Vérifier si un fichier a été soumis
        if 'uploaded-file' not in request.files:
            return "Aucun fichier n'a été soumis."

        # upload file
        uploaded_df = request.files['uploaded-file']
        if uploaded_df.filename == '':
            return "Aucun fichier sélectionné."

        #Extracting upload data file name
        data_filename = secure_filename(uploaded_df.filename)

        # flask upload file to database (defined uploaded folder
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))

        # Storing uploaded file path in flask session
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)

        return redirect(url_for('showData'))

    return render_template('home.html')
@app.route('/show_data')
def showData():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)

    if data_file_path is None:
        # Handle the case where the file path is not available
        return "File path not available"

    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    df = pd.read_csv(data_file_path)

    # Vérifier que le DataFrame contient les colonnes nécessaires pour la détection de fraude
    required_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12',
                        'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25',
                        'V26', 'V27', 'V28', 'Amount']
    if not all(col in df.columns for col in required_columns):
        return "Le fichier .csv doit contenir toutes les colonnes requises : time, amount, v1, v2, ..., v28."

    # Effectuer la détection de fraude pour chaque transaction
    df['fraud_prediction'] = model.predict(df[required_columns])

    # Convertir les prédictions (0 ou 1) en "fraudulent" ou "not fraudulent" pour la facilité d'affichage
    df['fraud_prediction'] = np.where(df['fraud_prediction'] == 1, 'fraudulent', 'not fraudulent')

    # Convert DataFrame to CSV format
    csv_data = df.to_csv(index=False)

    # Encode the CSV data to bytes
    csv_bytes = csv_data.encode('utf-8')

    # Create a file-like object to send as attachment
    csv_io = io.BytesIO(csv_bytes)

    # Set response headers for the CSV file
    response = send_file(
            csv_io,
            mimetype='text/csv',
            as_attachment=True,
            download_name='data.csv'  # Use 'download_name' instead of 'attachment_filename'
        )
    return response

### Upload File ###
@app.route('/upload')
def upload():
    return render_template('downloadFile.html')

@app.route('/import')
def importer():
    return render_template('importFile.html')

@app.route("/importer", methods=['GET', 'POST'])
def parcourir():
    if request.method == 'POST':
        # Vérifier si un fichier a été soumis
        if 'file' not in request.files:
            return "Aucun fichier n'a été soumis."

        file = request.files['file']
        if file.filename == '':
            return "Aucun fichier sélectionné."

        # Lire le fichier .csv dans un DataFrame
        df = pd.read_csv(file)

        # Vérifier que le DataFrame contient les colonnes nécessaires pour la détection de fraude
        required_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12',
                            'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25',
                            'V26', 'V27', 'V28', 'Amount']
        if not all(col in df.columns for col in required_columns):
            return "Le fichier .csv doit contenir toutes les colonnes requises : time, amount, v1, v2, ..., v28."

        # Effectuer la détection de fraude pour chaque transaction
        df['fraud_prediction'] = model.predict(df[required_columns])

        # Convertir les prédictions (0 ou 1) en "fraudulent" ou "not fraudulent" pour la facilité d'affichage
        df['fraud_prediction'] = np.where(df['fraud_prediction'] == 1, 'fraudulent', 'not fraudulent')

        # Renvoyer les résultats sous forme de DataFrame
        return df.to_html()

    return render_template('importFile.html')

### Form Transaction ###
@app.route('/transaction')
def transaction():
    return render_template('form.html')

### User show her infos ###
@app.route('/user/<int:id>')
def view_user(id):
    # Ici, vous devrez récupérer les informations de l'utilisateur depuis la base de données en utilisant son ID
    # Supposons que vous ayez une fonction "get_user_by_id" pour cela
    # Check if account exists using MySQL
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM users WHERE id = %s', (id,))
    user = cursor.fetchone()

    if user:
        return render_template('infoUser.html', user=user)
    else:
        # Gérer le cas où l'utilisateur n'existe pas
        return "Utilisateur non trouvé", 404

    return render_template('home.html')

### User update her infos ###
@app.route("/user/update/<int:id>", methods=['GET', 'POST'])
def modifier(id):
    msg = ''
    alert = ''
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM users WHERE id = %s', (id,))
    account = cursor.fetchone()

    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        user = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # If account exists show error and validation checks
        if not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
            alert = 'alert alert-danger'
        elif not re.match(r'[A-Za-z0-9]+', user):
            msg = 'Username must contain only characters and numbers!'
            alert = 'alert alert-danger'
        elif not user or not password or not email:
            msg = 'Please fill out the form!'
            alert = 'alert alert-danger'
        else:
            # Account doesnt exists and the form data is valid, now insert new account
            cursor.execute('UPDATE users SET username = %s, email = %s, password = %s WHERE id = %s', (user, email, password, id))
            mysql.connection.commit()
            msg = 'You have successfully updated!'
            alert = 'alert alert-success'
            account['username'] = user
            account['email'] = email
            account['password'] = password
            cursor.close()

    # Show registration form with message (if any)
    return render_template('updateUser.html', msg=msg, alert=alert, user=account)

### Admin links ###
@app.route('/admin')
def home_admin():
    return render_template('admin/home_admin.html')

@app.route('/users')
def users():
    msg = ''
    alert = ''
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    mysql.connection.commit()
    cursor.close()
    if 'alert' and 'msg' in request.args:
        msg = request.args.get('msg')
        alert = request.args.get('alert')

    return render_template('admin/users.html', users=users, msg=msg, alert=alert)

@app.route('/analyse')
def analyse():
    msg = data.describe()
    class_counts = data['class'].value_counts()
    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.index, class_counts.values)
    plt.xlabel('Classe')
    plt.ylabel('Nombre de transactions')
    plt.title('Distribution des classes de transaction')
    plt.xticks([0, 1], ['Non-fraude', 'Fraude'])
    # Convert plot to HTML image tag
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    ##########
    class_amount_mean = data.groupby('class')['Amount'].mean()
    plt.bar(class_amount_mean.index, class_amount_mean.values)
    plt.xlabel('Classe')
    plt.ylabel('Montant moyen')
    plt.title('Distribution de montant moyen par classe de transaction')
    plt.xticks([0, 1], ['Non-fraude', 'Fraude'])
    # Convert plot to HTML image tag
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_2 = base64.b64encode(img.getvalue()).decode()
    plt.close()
    ##########
    data["Heures_du_jour"] = (data["Time"] // 3600) % 24
    # Regrouper les données par heures du jour et compter le nombre de transactions dans chaque groupe
    transactions_par_heure = data.groupby("Heures_du_jour")["class"].count()
    # Créer un diagramme pour visualiser le nombre de transactions par heures du jour
    plt.figure(figsize=(10, 6))
    plt.bar(transactions_par_heure.index, transactions_par_heure.values)
    plt.xlabel("Heures du jour")
    plt.ylabel("Nombre de transactions")
    plt.title("Distribution des transactions non frauduleuses et frauduleuses en fonction des heures du jour")
    plt.xticks(range(0, 24))
    plt.grid(axis="y")
    # Convert plot to HTML image tag
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_3 = base64.b64encode(img.getvalue()).decode()
    plt.close()
    ##########
    # Conversion de la colonne "Time" en heures du jour (supposant que "Time" est en secondes)
    ds = data[data['class'] == 1]
    ds["Heures_du_jour"] = (ds["Time"] // 3600) % 24
    # Regrouper les données par heures du jour et compter le nombre de transactions dans chaque groupe
    transactions_par_heure = ds.groupby("Heures_du_jour")["class"].count()
    # Créer un diagramme pour visualiser le nombre de transactions par heures du jour
    plt.figure(figsize=(10, 6))
    plt.bar(transactions_par_heure.index, transactions_par_heure.values)
    plt.xlabel("Heures du jour")
    plt.ylabel("Nombre de transactions frauduleuses")
    plt.title("Distribution des transactions frauduleuses en fonction des heures du jour")
    plt.xticks(range(0, 24))
    plt.grid(axis="y")
    # Convert plot to HTML image tag
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_4 = base64.b64encode(img.getvalue()).decode()
    plt.close()
    ##########
    # Convertir la colonne 'Time' en heures
    data['Time_hours'] = data['Time'] / 3600  # Convertir les secondes en heures
    # Visualisation pour l'ensemble des données
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x='Time_hours', bins=48, kde=True)
    plt.title('Distribution des transactions non frauduleuses au fil des heures')
    plt.xlabel('Heures')
    plt.ylabel('Nombre de transactions')
    # Convert plot to HTML image tag
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_5 = base64.b64encode(img.getvalue()).decode()
    plt.close()
    ##########
    fraud_data = data[data['class'] == 0]
    # Visualisation des transactions frauduleuses au fil des heures
    plt.figure(figsize=(12, 6))
    sns.histplot(data=fraud_data, x='Time_hours', bins=48, kde=True, color='red')
    plt.title('Distribution des transactions non frauduleuses au fil des heures')
    plt.xlabel('Heures')
    plt.ylabel('Nombre de transactions frauduleuses')
    # Convert plot to HTML image tag
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_6 = base64.b64encode(img.getvalue()).decode()
    plt.close()
    ##########
    fraud_data = data[data['class'] == 1]
    # Visualisation des transactions frauduleuses au fil des heures
    plt.figure(figsize=(12, 6))
    sns.histplot(data=fraud_data, x='Time_hours', bins=48, kde=True, color='red')
    plt.title('Distribution des transactions frauduleuses au fil des heures')
    plt.xlabel('Heures')
    plt.ylabel('Nombre de transactions frauduleuses')
    # Convert plot to HTML image tag
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_7 = base64.b64encode(img.getvalue()).decode()
    plt.close()
    ##########
    # Afficher la distribution de la variable 'Amount'
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x='Amount', bins=50, kde=True, color='blue')
    plt.title('Distribution du Montant des Transactions')
    plt.xlabel('Montant')
    plt.ylabel('Fréquence')
    # Convert plot to HTML image tag
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_8 = base64.b64encode(img.getvalue()).decode()
    plt.close()
    ##########
    # Appliquer une transformation logarithmique au 'Amount'
    data['Amount_log'] = np.log1p(data['Amount'])  # Appliquer la transformation log(1 + x)
    # Afficher la distribution du 'Amount' après la transformation
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x='Amount_log', bins=50, kde=True, color='green')
    plt.title('Distribution du Montant des Transactions (Transformation Logarithmique)')
    plt.xlabel('Montant Transformé par Logarithme')
    plt.ylabel('Fréquence')
    # Convert plot to HTML image tag
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_9 = base64.b64encode(img.getvalue()).decode()
    plt.close()
    ##########
    # Appliquer une transformation logarithmique sur les montants
    data['Amount_log'] = np.log1p(data['Amount'])
    # Afficher la distribution du montant transformé
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x='Amount_log', bins=50, kde=True, color='purple')
    plt.title('Distribution du Montant (Transformation Logarithmique)')
    plt.xlabel('Montant (Log)')
    plt.ylabel('Fréquence')
    # Convert plot to HTML image tag
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_10 = base64.b64encode(img.getvalue()).decode()
    plt.close()
    ##########
    # Séparer les données en transactions frauduleuses et non frauduleuses
    fraud_transactions = data[data['class'] == 1]
    non_fraud_transactions = data[data['class'] == 0]
    # Appliquer une transformation logarithmique au montant
    fraud_amount_log = np.log1p(fraud_transactions['Amount'])
    non_fraud_amount_log = np.log1p(non_fraud_transactions['Amount'])
    # Tracer les distributions transformées
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.hist(fraud_amount_log, bins=50, color='red', alpha=0.7)
    plt.title('Distribution du Montant (Transaction Frauduleuse)')
    plt.xlabel('Montant (Logarithme)')
    plt.ylabel('Fréquence')
    plt.subplot(1, 2, 2)
    plt.hist(non_fraud_amount_log, bins=50, color='green', alpha=0.7)
    plt.title('Distribution du Montant (Transaction Non Frauduleuse)')
    plt.xlabel('Montant (Logarithme)')
    plt.ylabel('Fréquence')
    # Convert plot to HTML image tag
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_11 = base64.b64encode(img.getvalue()).decode()
    plt.close()
    ##########
    # Séparer les données en transactions frauduleuses et non frauduleuses
    fraud_transactions = data[data['class'] == 1]
    non_fraud_transactions = data[data['class'] == 0]
    # Appliquer une transformation logarithmique au montant
    fraud_amount_log = np.log1p(fraud_transactions['Amount'])
    non_fraud_amount_log = np.log1p(non_fraud_transactions['Amount'])
    # Tracer l'histogramme avec les deux distributions
    plt.figure(figsize=(10, 6))
    plt.hist(fraud_amount_log, bins=50, color='red', alpha=0.7, label='Fraude')
    plt.hist(non_fraud_amount_log, bins=50, color='green', alpha=0.5, label='Non Fraude')
    plt.title('Distribution du Montant par Classe')
    plt.xlabel('Montant (Logarithme)')
    plt.ylabel('Fréquence')
    # Convert plot to HTML image tag
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_12 = base64.b64encode(img.getvalue()).decode()
    plt.close()
    ##########

    return render_template('admin/analyse.html', msg=msg,plot_url=plot_url,plot_url_2=plot_url_2, plot_url_3=plot_url_3,plot_url_4=plot_url_4,plot_url_5=plot_url_5, plot_url_6=plot_url_6, plot_url_7=plot_url_7, plot_url_8=plot_url_8, plot_url_9=plot_url_9, plot_url_10=plot_url_10, plot_url_11=plot_url_11, plot_url_12=plot_url_12)

@app.route('/transaction_admin')
def admin_transaction():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("SELECT * FROM transactions")
    transactions = cursor.fetchall()
    mysql.connection.commit()
    cursor.close()
    return render_template('admin/manage_transaction.html', transactions=transactions)

@app.route('/delete/<int:transaction_id>')
def delete_transaction(transaction_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('DELETE FROM transactions WHERE id = %s', (transaction_id,))
    mysql.connection.commit()
    cursor.close()
    return redirect(url_for('admin_transaction'))

@app.route("/addUser", methods=['GET', 'POST'])
def addUser():
    msg = ''
    alert = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        user = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s', (user,))
        account = cursor.fetchone()

        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
            alert = 'alert alert-danger'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
            alert = 'alert alert-danger'
        elif not re.match(r'[A-Za-z0-9]+', user):
            msg = 'Username must contain only characters and numbers!'
            alert = 'alert alert-danger'
        elif not user or not password or not email:
            msg = 'Please fill out the form!'
            alert = 'alert alert-danger'
        else:
            # Account doesnt exists and the form data is valid, now insert new account
            cursor.execute('INSERT INTO users VALUES (NULL, %s, %s, %s)', (user, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
            alert = 'alert alert-success'
            cursor.close()
            return redirect(url_for('users', msg=msg, alert=alert))

    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
        alert = 'alert alert-danger'

    # Show registration form with message (if any)
    return render_template('admin/addUser.html', msg=msg, alert=alert)

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def editUser(id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM users WHERE id = %s', (id,))
    account = cursor.fetchone()
    cursor.close()

    msg = ''
    alert = ''

    if account is None:
        return redirect(url_for('users'))

    if request.method == 'POST':
        userEdit = request.form['username']
        passwordEdit = request.form['password']
        emailEdit = request.form['email']

        if not re.match(r'[^@]+@[^@]+\.[^@]+', emailEdit):
            msg = 'Invalid email address!'
            alert = 'alert alert-danger'
        elif not re.match(r'[A-Za-z0-9]+', userEdit):
            msg = 'Username must contain only characters and numbers!'
            alert = 'alert alert-danger'
        elif not userEdit or not passwordEdit or not emailEdit:
            msg = 'Please fill out the form!'
            alert = 'alert alert-danger'
        else:
            cursor = mysql.connection.cursor()
            cursor.execute('UPDATE users SET username = %s, email= %s, password = %s WHERE id = %s',
                           (userEdit, emailEdit, passwordEdit, id))
            mysql.connection.commit()
            cursor.close()

            msg = 'You have successfully updated!'
            alert = 'alert alert-success'
            #account['username'] = userEdit
            #account['email'] = emailEdit
            #account['password'] = passwordEdit
            return redirect(url_for('users', msg=msg, alert=alert))
        

    return render_template('admin/editUser.html', msg=msg, alert=alert, user=account)

@app.route('/delete/<int:id>')
def delete_user(id):

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('DELETE FROM users WHERE id = %s', (id,))
    mysql.connection.commit()
    cursor.close()

    return redirect("/users")

@app.route("/profile")
def profile():
    return render_template('admin/profile.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtenir les informations du formulaire
    time = float(request.form['time'])
    amount = float(request.form['amount'])
    v1 = float(request.form['v1'])
    v2 = float(request.form['v2'])
    v3 = float(request.form['v3'])
    v4 = float(request.form['v4'])
    v5 = float(request.form['v5'])
    v6 = float(request.form['v6'])
    v7 = float(request.form['v7'])
    v8 = float(request.form['v8'])
    v9 = float(request.form['v9'])
    v10 = float(request.form['v10'])
    v11 = float(request.form['v11'])
    v12 = float(request.form['v12'])
    v13 = float(request.form['v13'])
    v14 = float(request.form['v14'])
    v15 = float(request.form['v15'])
    v16 = float(request.form['v16'])
    v17 = float(request.form['v17'])
    v18 = float(request.form['v18'])
    v19 = float(request.form['v19'])
    v20 = float(request.form['v20'])
    v21 = float(request.form['v21'])
    v22 = float(request.form['v22'])
    v23 = float(request.form['v23'])
    v24 = float(request.form['v24'])
    v25 = float(request.form['v25'])
    v26 = float(request.form['v26'])
    v27 = float(request.form['v27'])
    v28 = float(request.form['v28'])
   

    # Préparer les données pour la prédiction
    data = np.array([[time, amount, v1, v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28]])

    # Faire la prédiction
    prediction = model.predict(data)

    # Afficher le résultat
    if prediction == 1:
        result = "fraude"
    else:
        result = "non fraude"

    # Exécuter la requête SQL d'INSERT pour ajouter la transaction
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    insert_query = "INSERT INTO transactions (time, amount, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, class) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    cursor.execute(insert_query, (time, amount, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
            v21, v22,
            v23, v24, v25, v26, v27, v28, int(prediction[0]) ))
    mysql.connection.commit()
    cursor.close()

    return render_template('resultat.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
