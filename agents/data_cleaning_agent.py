import re
import pandas as pd
from langchain_core.messages import HumanMessage

class DataCleaningAgent:
    def __init__(self):
        self.name = 'DataCleaningAgent'
        
    def invoke(self, state):
        """
        Méthode principale pour l'agent

        Args:
            state: l'état actuel de l'agent

        Returns:
            dict: l'état mis à jour de l'agent
        """
        
        df = pd.read_json(state['data'])
        if df.shape[0] == 0:
            return {
                "messages": state["messages"] + [HumanMessage(content=f"Le dataset est vide")],
                "data": state['data'],
                "context": state.get("context", {})
            }
        df = df[['tweet_content']]
        print(df.head())
        def clean_text(text):
            text = re.sub(r"http[s]?://\S+|www\.\S+", '', text)
            text = re.sub(r"@\w+|\#\w+", '', text)
            text = re.sub(r'[^\w\s,]', '', text)
            text = text.lower()
            return text
        
        df['tweet'] = df['tweet_content'].apply(clean_text)
        df = df[['tweet']]
        # supprimer les doublons
        df.drop_duplicates(inplace=True)
        df.to_csv('data/tweets_dataset_clean.csv', index=False)
        
        return {
            "messages": state["messages"] + [HumanMessage(content=f"Action effectuée par l'agent {self.name}")],
            "data": 'data/tweets_dataset_clean.csv',
            "context": state.get("context", {})
        }