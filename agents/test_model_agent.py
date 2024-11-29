from langchain_core.messages import HumanMessage

class TestModelAgent:
    def __init__(self):
        self.name = 'TestModelAgent'
    
    def invoke(self, state):
        model = state["context"]["best_model"]
        vectorizer = state["context"]["vectorizer"]
        model_name = state["context"]["model_name"]
        
        print(f"Vous utilisez le modèle {model_name} pour prédire le sentiment du tweet, avec un score de {model.score}")
        while True:
            tweet = input("Entrez le tweet ou FINISH pour quitter: ")
            if tweet == "FINISH":
                break
            tweet_vectorized = vectorizer.transform([tweet])
            prediction = model.predict(tweet_vectorized)
            
            print(f"Le sentiment du tweet est: {prediction}")
        
        return {
            "messages": [
                HumanMessage(content=f"Context provided by human:")
            ],
            "data": "",
            "context": ""
        }