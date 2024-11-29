from langchain_core.messages import HumanMessage

import pandas as pd
import numpy as np
import re
import string
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

class ClassifierAgent:

    def __init__(self):
        self.name = 'ClassifierAgent'

    def detect_imbalance(self, y, threshold=0.2):
        """Détecter les déséquilibres dans les données."""
        label_counts = pd.Series(y).value_counts(normalize=True)
        imbalance_message = ""
        for label, proportion in label_counts.items():
            if proportion < threshold:
                imbalance_message += (
                    f"La classe '{label}' est sous-représentée ({proportion:.2%}).\n"
                )
        return imbalance_message if imbalance_message else None
    
    def main_pipeline(self, X,y):

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

        # Sauvegarde du modèle
        model.save('tweet_classifier.h5')
        print("Modèle sauvegardé dans 'tweet_classifier.h5'.")

        # Tracer les métriques
        return history
        

    def invoke(self, state):
        # Charger les données pré-traitées
        df = pd.read_csv(state['data'])

        # Vérifier les colonnes
        if df.shape[1] < 2:
            raise ValueError("Les données doivent avoir au moins deux colonnes : texte et label.")
        
        X = df.iloc[:, 0].values  # Texte pré-traité
        y = df.iloc[:, 1].values  # Labels

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

        history = self.main_pipeline(X,y)

        return {
            "messages": state["messages"] + [
                HumanMessage(content=f"Action effectuée par l'agent {self.name}. Le modèle ANN est prêt et sauvegardé.")
            ],
            "data": history,
            "context": state.get("context", {})
        }
