from langchain_core.messages import HumanMessage

class TestModelAgent:
    def __init__(self):
        self.name = 'TestModelAgent'
    
    def invoke(self, state):
        model = state["context"]["model"]
        tokenizer = state["context"]["tokenizer"]
        history = state["data"]
        
        
        
        print(f"Vous utilisez le modèle pour prédire le sentiment du tweet, avec un score de {history.history['accuracy'][-1]:.2f} d'accuracy.")
        while True:
            tweet = input("Entrez le tweet ou FINISH pour quitter: ")
            if tweet == "FINISH":
                break
            tweet_vectorized = tokenizer.texts_to_sequences([tweet])
            prediction = model.predict(tweet_vectorized)
            
            print(f"Le sentiment du tweet est: {prediction}")
        
        return {
            "messages": [
                HumanMessage(content=f"Context provided by human:")
            ],
            "data": "",
            "context": ""
        }