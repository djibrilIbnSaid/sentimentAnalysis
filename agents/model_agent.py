import pandas as pd
from langchain_core.messages import HumanMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Classe de l'agent de modèle de classification des tweets
class ModelAgent:
    def __init__(self):
        self.name = 'ModelAgent'
    
    def invoke(self, state):
        df = pd.read_csv(state['data'])
        X = df['tweet']
        y = pd.Categorical(df['sentiment'])
        
        # Créer un vecteur de caractéristiques TF-IDF
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(X)
        
        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Y_train: {y_train}")
        
        # Créer un classificateur Naive Bayes multinomial
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        
        # Prédire les étiquettes des données de test
        y_pred = clf.predict(X_test)
        
        # Calculer la précision du modèle
        accuracy = accuracy_score(y_test, y_pred)
        
        # Afficher le rapport de classification
        report = classification_report(y_test, y_pred)
        
        print(f"Accuracy: {accuracy}")
        print(report)
        
        return {
            "messages": state["messages"] + [HumanMessage(content=f"Action effectuée par l'agent {self.name}")],
            "data": 'data/tweets_dataset_clean.csv',
            "context": state.get("context", {})
        }

