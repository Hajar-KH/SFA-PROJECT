import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Charger l'ensemble de données
data = pd.read_csv('creditcards.csv')

# Séparer les caractéristiques et les étiquettes
X = data.drop('class', axis=1)
y = data['class']

# Diviser les données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle de régression logistique
logreg_model = LogisticRegression(max_iter=1000)

# Entraîner le modèle de régression logistique
logreg_model.fit(X_train, y_train)

pickle.dump(logreg_model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))