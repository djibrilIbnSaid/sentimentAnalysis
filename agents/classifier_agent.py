import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.layers import Embedding, BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from langchain_core.messages import HumanMessage
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

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

        # Encodage des labels
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        # Diviser les données en Train, Validation, Test
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Tokenisation et padding
        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)
        X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
        X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=100)
        X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

        # Détection et équilibrage des classes si nécessaire
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        max_words = 10000
        embedding_dim = 32
        max_len = 100  # Longueur du texte après padding

        # Branch 1
        branch1 = Sequential()
        branch1.add(Embedding(max_words, embedding_dim, input_length=max_len))
        branch1.add(Conv1D(64, 3, padding='same', activation='relu'))
        branch1.add(BatchNormalization())
        branch1.add(ReLU())
        branch1.add(Dropout(0.5))
        branch1.add(GlobalMaxPooling1D())

        # Branch 2
        branch2 = Sequential()
        branch2.add(Embedding(max_words, embedding_dim, input_length=max_len))
        branch2.add(Conv1D(64, 3, padding='same', activation='relu'))
        branch2.add(BatchNormalization())
        branch2.add(ReLU())
        branch2.add(Dropout(0.5))
        branch2.add(GlobalMaxPooling1D())

        concatenated = Concatenate()([branch1.output, branch2.output])

        hid_layer = Dense(128, activation='relu')(concatenated)
        dropout = Dropout(0.3)(hid_layer)
        output_layer = Dense(3, activation='softmax')(dropout)  # 3 classes

        model = Model(inputs=[branch1.input, branch2.input], outputs=output_layer)

        # Compiler le modèle
        model.compile(optimizer=Adamax(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy', Precision(), Recall()])

        # Entraînement du modèle
        batch_size = 256
        epochs = 25
        history = model.fit([X_train_seq, X_train_seq], y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=([X_val_seq, X_val_seq], y_val), class_weight=class_weight_dict)

        # Évaluation du modèle
        (loss, accuracy, precision, recall) = model.evaluate([X_train_seq, X_train_seq], y_train)
        print(f'Loss: {round(loss, 2)}, Accuracy: {round(accuracy, 2)}, Precision: {round(precision, 2)}, Recall: {round(recall, 2)}')

        (loss, accuracy, precision, recall) = model.evaluate([X_test_seq, X_test_seq], y_test)
        print(f'Loss: {round(loss, 2)}, Accuracy: {round(accuracy, 2)}, Precision: {round(precision, 2)}, Recall: {round(recall, 2)}')

        # Sauvegarder le modèle
        model.save('model.h5')

        # Visualisation des résultats d'entraînement
        plt.figure(figsize=(20, 12))
        plt.style.use('fivethirtyeight')

        # Loss graph
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy graph
        plt.subplot(2, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.suptitle('Model Training Metrics', fontsize=16)
        plt.show()

        return {
            "messages": state["messages"] + [
                HumanMessage(content=f"Action effectuée par l'agent {self.name}. Le modèle ANN est prêt et sauvegardé.")
            ],
            "data": state["data"],
            "context": state.get("context", {})
        }
