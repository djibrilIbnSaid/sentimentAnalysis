import re
import joblib
from pathlib import Path
import pandas as pd
from langchain_core.messages import HumanMessage
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

class ModelAgent:
    def __init__(self):
        self.name = 'ModelAgent'
        self.models_dir = Path('saved_models')
        self.models_dir.mkdir(exist_ok=True)
        
    def invoke(self, state):
        dataset = pd.read_csv(state['data'])

        # Séparation des données en X et y
        X = dataset.text
        y = dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42)

        vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
        # Vectorisation des données
        vectorizer.fit(X_train)
        X_train = vectorizer.transform(X_train)
        X_test  = vectorizer.transform(X_test)

        # Entrainement des modèles

        # Modele 1 : BernoulliNB
        BNBmodel = BernoulliNB()
        BNBmodel.fit(X_train, y_train)

        # Modele 2 : LinearSVC
        SVCmodel = LinearSVC()
        SVCmodel.fit(X_train, y_train)

        # Modele 3 : LogisticRegression
        LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
        LRmodel.fit(X_train, y_train)

        

        # Evaluation des modèles (choisir le meilleur)
        def evaluate_models(X_train, X_test, y_train, y_test, models_dict):
            """
            Évalue plusieurs modèles et retourne le meilleur selon le score F1
            
            Parameters:
            -----------
            X_train, X_test : matrices de caractéristiques d'entraînement et de test
            y_train, y_test : labels d'entraînement et de test
            models_dict : dictionnaire des modèles à évaluer
            
            Returns:
            --------
            tuple : (meilleur_modèle, nom_meilleur_modèle, scores_détaillés)
            """
            best_f1 = 0
            best_model = None
            best_model_name = None
            detailed_scores = {}
            
            for model_name, model in models_dict.items():
                # Prédictions sur l'ensemble de test
                y_pred = model.predict(X_test)
                
                # Calcul des métriques
                report = classification_report(y_test, y_pred, output_dict=True)
                f1_weighted = report['weighted avg']['f1-score']
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # Stockage des scores détaillés
                detailed_scores[model_name] = {
                    'classification_report': report,
                    'confusion_matrix': conf_matrix,
                    'f1_weighted': f1_weighted
                }
                
                # Mise à jour du meilleur modèle
                if f1_weighted > best_f1:
                    best_f1 = f1_weighted
                    best_model = model
                    best_model_name = model_name
            
            return best_model, best_model_name, detailed_scores

        models = {
            'BernoulliNB': BNBmodel,
            'LinearSVC': SVCmodel,
            'LogisticRegression': LRmodel
        }

        best_model, best_model_name, scores = evaluate_models(
            X_train, X_test, y_train, y_test, models
        )

        # Sauvegarde du meilleur modèle et du vectorizer
        model_path = self.models_dir / f'best_model_{best_model_name}.joblib'
        vectorizer_path = self.models_dir / 'vectorizer.joblib'
        
        joblib.dump(best_model, model_path)
        joblib.dump(vectorizer, vectorizer_path)

        # Mise à jour du contexte avec les chemins des fichiers sauvegardés
        state["context"].update({
            "best_model": best_model,
            "model_name": best_model_name,
            "evaluation_scores": scores,
            "vectorizer": vectorizer,
        })

        return {
            "messages": state["messages"] + [
                HumanMessage(content=f"Le modèle {best_model_name} a été sélectionné et sauvegardé dans {model_path}")
            ],
            "data": state["data"],
            "context": state["context"]
        }