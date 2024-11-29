import re
import pandas as pd
from langchain_core.messages import HumanMessage
import nltk
from nltk.corpus import stopwords # nltk.download('stopwords') si non téléchargé
from nltk.tokenize import RegexpTokenizer
import string
nltk.download('wordnet')

class DatasetPreparatorAgent:
    def __init__(self):
        self.name = 'DatasetPreparatorAgent'
        
    def invoke(self, state):
        # Charger le dataset
        df = pd.read_csv(state['data'])
        # Remplacer les valeurs directement
        df['sentiment'] = df['sentiment'].replace({
            'POSITIVE': '1',
            'NEUTRAL': '0',
            'NEGATIVE': '-1'
        })
        pos_data = df[df['sentiment'] == '1'] # positif
        neu_data = df[df['sentiment'] == '0'] # neutre
        neg_data = df[df['sentiment'] == '-1'] # négatif
        dataset = pd.concat([pos_data, neu_data, neg_data])

        # On ne traite que le cas des tweets en français
        stops = set(stopwords.words('french'))
        # Fonction pour nettoyer les stopwords
        def cleaning_stopwords(text):
            return " ".join([word for word in str(text).split() if word not in stops])
        dataset['tweet'] = dataset['tweet'].apply(lambda text: cleaning_stopwords(text))

        punctuations = string.punctuation
        # Fonction pour nettoyer la ponctuation
        def cleaning_punctuations(text):
            translator = str.maketrans('', '', punctuations)
            return text.translate(translator)
        dataset['tweet']= dataset['tweet'].apply(lambda x: cleaning_punctuations(x))

        # Fonction pour nettoyer les caractères répétitifs
        def cleaning_repeating_char(text):
            return re.sub(r'(.)1+', r'1', text)
        dataset['tweet'] = dataset['tweet'].apply(lambda x: cleaning_repeating_char(x))
        
        # Fonction pour nettoyer les URLs (peut être déjà fait dans l'agent avant la labélisation)
        def cleaning_URLs(data):
            return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
        dataset['tweet'] = dataset['tweet'].apply(lambda x: cleaning_URLs(x))

        # Fonction pour nettoyer les chiffres
        def cleaning_numbers(data):
            return re.sub('[0-9]+', '', data)
        dataset['tweet'] = dataset['tweet'].apply(lambda x: cleaning_numbers(x))

        # Fonction pour tokeniser le texte
        tokenizer = RegexpTokenizer(r'\w+')
        dataset['tweet'] = dataset['tweet'].apply(tokenizer.tokenize)

        st = nltk.PorterStemmer()
        # Fonction pour stemmer le texte
        def stemming_on_text(data):
            text = [st.stem(word) for word in data]
            return text
        dataset['tweet']= dataset['tweet'].apply(lambda x: stemming_on_text(x))

        lm = nltk.WordNetLemmatizer()
        # Fonction pour lemmatizer le texte
        def lemmatizer_on_text(data):
            text = [lm.lemmatize(word) for word in data]
            return text
        dataset['tweet'] = dataset['tweet'].apply(lambda x: lemmatizer_on_text(x))

        df.to_csv('data/tweets_dataset.csv', index=False)
        return {
            "messages": state["messages"] + [HumanMessage(content=f"Action effectuée par l'agent {self.name}")],
            "data": 'data/tweets_dataset.csv',
            "context": state.get("context", {})
        }