import pandas as pd

from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate

class GeneratorTweetAgent:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2", verbose=False)
        self.name = 'GeneratorTweetAgent'
        self.query = None
    
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
                tweets.append({"tweet": tweet, "sentiment": categorie})
        
        # enregistrer les tweets générés dans un fichier CSV
        df_generated = pd.DataFrame(tweets)
        df_generated.to_csv('tweets_generated.csv', index=False)
        return tweets
    
    def invoke(self, state):
        self.query = state["context"]
        print(f"Contexte: {self.query}")
        print(f"Dataset: {state['data']}")
        tweets = self._generate_tweet(self.query, state['data'])
        return {
            "messages": state["messages"] + [HumanMessage(content=f"Action effectuée par l'agent {self.name}")],
            "data": tweets,
            "context": state.get("context", {})
        }