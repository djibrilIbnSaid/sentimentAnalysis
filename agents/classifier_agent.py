from langchain_core.messages import HumanMessage

import pandas as pd
import numpy as np
import re
import string
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords # nltk.download('stopwords') si non téléchargé
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout, BatchNormalization, ReLU ,Concatenate,Input
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

class ClassifierAgent:

    def __init__(self):
        self.name = 'ClassifierAgent'

    def clean_and_prepare_data(self,file_path):
        """
        Nettoyer et préparer les données pour l'entraînement du modèle.

        Args:
            file_path (str): Chemin du fichier CSV contenant les données.

        Returns:
            Tuple: X (features), y (labels).
        """
        df = pd.read_csv(file_path)
        df['sentiment'] = df['sentiment'].replace({
            'POSITIVE': 1,
            'NEUTRAL': 0,
            'NEGATIVE': -1
        })
        pos_data = df[df['sentiment'] == 1] # positif
        neu_data = df[df['sentiment'] == 0] # neutre
        neg_data = df[df['sentiment'] == -1] # négatif
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
        X = df['tweet'].values
        y = np.array([label + 1 for label in df['sentiment'].values])  # Convertir -1, 0, 1 en 0, 1, 2
        return X, y

    def detect_imbalance(self, y, threshold=0.2):
        """
        Détecter les déséquilibres dans les données.

        Args:
            y (int): les labels.
            threshold (float, optional): _description_. Defaults to 0.2.

        Returns:
            str: Message d'avertissement sur les déséquilibres.
        """
        label_counts = pd.Series(y).value_counts(normalize=True)
        imbalance_message = ""
        for label, proportion in label_counts.items():
            if proportion < threshold:
                imbalance_message += (
                    f"La classe '{label}' est sous-représentée ({proportion:.2%}).\n"
                )
        return imbalance_message if imbalance_message else None
    
    def main_pipeline(self,X,y):
        """
        Entraîner un modèle de classification de texte.

        Args:
            X : les features
            y : les labels

        Returns:
            Tuple: Chemin du modèle et du tokenizer.
        """

        # Diviser les données
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        

        # Label Encoding and One-Hot Encoding
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_val = encoder.transform(y_val)
        y_test = encoder.transform(y_test)

        # One-hot encoding
        y_train = to_categorical(y_train, num_classes=3)
        y_val = to_categorical(y_val, num_classes=3)
        y_test = to_categorical(y_test, num_classes=3)

        # Tokenisation et padding
        max_words = 10000
        max_len = 500
        tokenizer = Tokenizer(num_words=max_words)

        X_train = np.array(pd.Series(X_train).fillna("").tolist())
        X_val = np.array(pd.Series(X_val).fillna("").tolist())
        X_test = np.array(pd.Series(X_test).fillna("").tolist())

        tokenizer.fit_on_texts(X_train)

        train_x_padded = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding='post')
        val_x_padded = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=max_len)
        test_x_padded = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

        # Modèle
        input_layer = Input(shape=(max_len,))
        branch1 = Embedding(input_dim=max_words, output_dim=100, input_length=max_len)(input_layer)
        branch1 = Conv1D(64, 3, padding='same', activation='relu')(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = ReLU()(branch1)
        branch1 = Dropout(0.5)(branch1)
        branch1 = GlobalMaxPooling1D()(branch1)

        branch2 = Embedding(input_dim=max_words, output_dim=100, input_length=max_len)(input_layer)
        branch2 = Conv1D(64, 3, padding='same', activation='relu')(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = ReLU()(branch2)
        branch2 = Dropout(0.5)(branch2)
        branch2 = GlobalMaxPooling1D()(branch2)

        concatenated = Concatenate()([branch1, branch2])
        hid_layer = Dense(128, activation='relu')(concatenated)
        dropout = Dropout(0.3)(hid_layer)
        output_layer = Dense(3, activation='softmax')(dropout)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

        model.summary()

        # Entraînement
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model.fit(
            train_x_padded, y_train,
            epochs=10,
            batch_size=8,
            validation_data=(val_x_padded, y_val),
            callbacks=[early_stopping]
        )

        # Évaluation
        loss, accuracy, precision, recall = model.evaluate(test_x_padded, y_test)
        print(f"Loss: {loss:.2f}, Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
        import pickle

        tokenizer_path = 'data/tokenizer.pkl'
        model_path = 'data/tweet_classifier.h5'
        with open(tokenizer_path, 'wb') as tokenizer_file:
            pickle.dump(tokenizer, tokenizer_file)

        # Sauvegarde du modèle
        model.save(model_path)
        print("Modèle sauvegardé dans 'tweet_classifier.h5'.")

        # Tracer les métriques
        return model_path , tokenizer_path
        

    def invoke(self, state):
        """
        Méthode principale pour l'agent

        Args:
            state: l'état actuel de l'agent

        Returns:
            dict: l'état mis à jour de l'agent
        """
        
        X,y = self.clean_and_prepare_data(state["data"])
        # Détection de déséquilibres
        imbalance_message = self.detect_imbalance(y, threshold=0.2)
        if imbalance_message:
            return {
                "messages": state["messages"] + [
                    HumanMessage(
                        content=f"Action effectuée par l'agent {self.name}. Cependant, des déséquilibres ont été détectés :\n"
                                + imbalance_message +
                                "Veuillez envisager un équilibrage avant de continuer."
                    )
                ],
                "data": state["data"],
                "context": state.get("context", {})
            }

        model_path,tokenizer_path = self.main_pipeline(X,y)

        return {
            "messages": state["messages"] + [
                HumanMessage(content=f"Action effectuée par l'agent {self.name}. Le modèle ANN est prêt et sauvegardé.")
            ],
            "data" : "",
            "context": {'model_path':model_path,
                        'tokenizer_path':tokenizer_path}
        }
