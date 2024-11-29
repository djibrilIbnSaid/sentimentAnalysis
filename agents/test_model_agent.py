from langchain_core.messages import HumanMessage

class TestModelAgent:
    def __init__(self):
        self.name = 'TestModelAgent'
    
    def invoke(self, state):
        model_path = state["context"]["model_path"]
        tokenizer_path = state["context"]["tokenizer_path"]
        emotions = {-1:'NEGATIVE', 0: 'NEUTRE', 1: 'POSITIVE'}
        
        def predict(text, model_path, token_path):
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            import matplotlib.pyplot as plt
            import pickle
            from tensorflow.keras.models import load_model

            model = load_model(model_path)


            with open(token_path, 'rb') as f:
                tokenizer = pickle.load(f)

            sequences = tokenizer.texts_to_sequences([text])
            x_new = pad_sequences(sequences, maxlen=500)
            predictions = model.predict([x_new])


            label = list(emotions.values())
            probs = list(predictions[0])
            labels = label
            plt.subplot(1, 1, 1)
            bars = plt.barh(labels, probs)
            plt.xlabel('Probability', fontsize=15)
            ax = plt.gca()
            ax.bar_label(bars, fmt = '%.2f')
            plt.show()
            return probs
        
        
        while True:
            tweet = input("Entrez le tweet ou FINISH pour quitter: ")
            if tweet == "FINISH":
                break
            prediction = predict(tweet, model_path, tokenizer_path)
            
            print(f"Le sentiment du tweet est: {emotions[prediction.index(max(prediction))]}")
        
        return {
            "messages": [
                HumanMessage(content=f"Context provided by human:")
            ],
            "data": "",
            "context": ""
        }
    



