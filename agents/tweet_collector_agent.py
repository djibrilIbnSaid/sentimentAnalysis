import pandas as pd

from langchain_core.messages import HumanMessage

class TweetCollectorAgent:
    def __init__(self, mode='term', number=100):
        super().__init__()
        self.name = 'TweetCollectorAgent'
        self.mode = mode
        self.number = number
        self.query = None
    
    def invoke(self, state):
        self.query = state["data"]
        
        # simulation de la collecte de tweets
        data = pd.read_json("tweets.json")
        data['tweet'] = data[['tweet_content']]
        data.drop(columns=['tweet_content'], inplace=True)
        data.to_csv('tweets.csv', index=False)
        # fin de la simulation
        
        return {
            "messages": state["messages"] + [HumanMessage(content=f"Action effectuée par l'agent {self.name}")],
            "data": 'tweets.csv',
            "context": state.get("context", {})
        }