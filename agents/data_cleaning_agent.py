import re
import pandas as pd
from langchain_core.messages import HumanMessage

class DataCleaningAgent:
    def __init__(self):
        self.name = 'DataCleaningAgent'
        
    def invoke(self, state):
        df = pd.read_csv(state['data'])
        df = df[['tweet']]
        print(df.head())
        def clean_text(text):
            text = re.sub(r"http[s]?://\S+|www\.\S+", '', text)
            text = re.sub(r"@\w+|\#\w+", '', text)
            text = re.sub(r'[^\w\s,]', '', text)
            text = text.lower()
            return text
        
        df['tweet'] = df['tweet'].apply(clean_text)
        # supprimer les doublons
        df.drop_duplicates(inplace=True)
        df.to_csv('tweets_dataset_clean.csv', index=False)
        
        return {
            "messages": state["messages"] + [HumanMessage(content=f"Action effectuée par l'agent {self.name}")],
            "data": 'tweets_dataset_clean.csv',
            "context": state.get("context", {})
        }