import pandas as pd
import re

from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate

class GeneratorTweetAgent:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2", verbose=False)
        self.name = 'GeneratorTweetAgent'
       
       
    
    def _calculer_tweets_a_generer(self, categories):
        """
        Calcule le nombre de tweets à générer pour équilibrer les données par catégorie.
        
        Args:
            categories (dict): Dictionnaire contenant les catégories et leurs nombres de tweets.
                            Exemple : {"positif": 100, "negatif": 50, "neutre": 50}
        
        Returns:
            dict: Dictionnaire indiquant combien de tweets à générer pour chaque catégorie.
        """
        # Trouver le maximum parmi les nombres de tweets des catégories
        max_tweets = max(categories.values())
        
        # Calculer combien de tweets il faut générer pour chaque catégorie
        tweets_a_generer = {categorie: max_tweets - nombre for categorie, nombre in categories.items()}
        
        return tweets_a_generer

    def _generate_tweet(self, context, dataset):
        """
        Génère des tweets en fonction du contexte et du dataset fournis.

        Args:
            context (str): Contexte pour lequel générer les tweets.
            dataset: Chemin du fichier CSV contenant les données.

        Returns:
            str: Chemin du fichier CSV contenant les tweets générés.
        """
        df = pd.read_csv(dataset)
        categories = {
            "POSITIVE": df[df['sentiment'] == 'POSITIVE'].shape[0],
            "NEGATIVE": df[df['sentiment'] == 'NEGATIVE'].shape[0],
            "NEUTRAL": df[df['sentiment'] == 'NEUTRAL'].shape[0]
        }
        resultats = self._calculer_tweets_a_generer(categories)
        lang = "french"
        prompt_template = PromptTemplate(
            input_variables=["context", "category", "lang"],
            template=(
                "Generate a tweet of up to 280 characters in a {category} tone. "
                "The tweet must be clear, concise, relevant, and aligned with the following context:\n\n"
                "Context: {context}\n\n"
                "Ensure that the tweet accurately reflects the requested tone (positive, negative, or neutral) "
                "and remains engaging without exceeding the character limit."
                "Language: {lang}"
                "Example: 'This product is amazing, I love it!'"
                "Example: 'Very bad service, I am disappointed.'"
                "In the context of {context}, generate a tweet in a {category} tone in {lang}."
            )
        )
        
        tweets = []
        for categorie, nombre in resultats.items():
            for _ in range(nombre):
                prompt = prompt_template.format(context=context, category=categorie, lang=lang)
                tweet = self.llm(prompt)
                print(f"Tweet généré ({categorie}): {tweet}")
                tweets.append({"tweet": self._clean_text(tweet), "sentiment": categorie})
        
        # enregistrer les tweets générés dans un fichier CSV
        df_generated = pd.DataFrame(tweets)
        df_generated.to_csv('data/tweets_generated.csv', index=False)
        df = pd.concat([df, df_generated])
        df.to_csv('data/tweets_dataset_aug.csv', index=False)
        return 'data/tweets_dataset_aug.csv'

    def _clean_text(self, text):
        """
        Nettoie le texte en supprimant les URLs, les hashtags, les mentions et la ponctuation.

        Args:
            text (str): Texte à nettoyer.

        Returns:
            str: Texte nettoyé.
        """
        text = re.sub(r"http[s]?://\S+|www\.\S+", '', text)
        text = re.sub(r"@\w+|\#\w+", '', text)
        text = re.sub(r'[^\w\s,]', '', text)
        text = text.lower()
        return text

    def invoke(self, state):
        """
        Méthode principale pour l'agent

        Args:
            state: l'état actuel de l'agent

        Returns:
            dict: l'état mis à jour de l'agent
        """
        self.query = state["context"]
        print(f"Contexte: {self.query}")
        print(f"Dataset: {state['data']}")
        path_tweet = self._generate_tweet(self.query, state['data'])
        return {
            "messages": state["messages"] + [HumanMessage(content=f"Action effectuée par l'agent {self.name}")],
            "data": path_tweet,
            "context": state.get("context", {})
        }