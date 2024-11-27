import pandas as pd
from langchain_core.messages import HumanMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

class ModelAgent:
    def __init__(self, lang='french'):
        self.name = 'ModelAgent'
        self.lang = lang
    
    def _create_model(self, df):
        X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['label'], test_size=0.25, random_state=42)
        
        # Vectorisation des textes avec TF-IDF
        tfidf = TfidfVectorizer(max_features=500, stop_words=self.lang)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        # Étape 3 : Entraîner un modèle Naive Bayes
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nRapport de classification :\n", classification_report(y_test, y_pred))
        return model
    
    def invoke(self, state):
        df = state['data']
        return {
            "messages": state["messages"] + [HumanMessage(content=f"Action effectuée par l'agent {self.name}")],
            "data": self._create_model(df),
            "context": state.get("context", {})
        }





# # Étape 5 : Utiliser le modèle pour prédire de nouveaux textes
# new_texts = ['Ce produit est génial, je l’adore !', 'Très mauvais service, je suis déçu.']
# new_texts_tfidf = tfidf.transform(new_texts)
# predictions = model.predict(new_texts_tfidf)

# # Afficher les prédictions
# for text, label in zip(new_texts, predictions):
#     print(f"Texte: {text} → Prédiction: {label}")