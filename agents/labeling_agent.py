import pandas as pd

from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate

class LabelingAgent:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2", verbose=False)
        self.name = 'LabelingAgent'
        self.query = None

    def get_sentiment_langchain(self, text):
        prompt_template = PromptTemplate(
            input_variables=["text", "query"],
            template="By reading the following tweet, determine if it conveys a POSITIVE, NEGATIVE, or NEUTRAL attitude in the context of {query}. Respond with only one of these three options nothing else. Here is the tweet: {text}"
        )
        
        print(f"Query: {self.query}")
        
        prompt = prompt_template.format(text=text, query=self.query)
        
        result = self.llm(prompt)
        return result.strip()

    
    def invoke(self, state):
        self.query = state["context"]
        data = pd.read_csv(state['data'])
        data['sentiment'] = data['tweet'].apply(self.get_sentiment_langchain)
        data.to_csv('data/tweets_labeled.csv', index=False)
        return {
            "messages": state["messages"] + [HumanMessage(content=f"Action effectuée par l'agent {self.name}")],
            "data": 'data/tweets_labeled.csv',
            "context": state.get("context", {})
        }