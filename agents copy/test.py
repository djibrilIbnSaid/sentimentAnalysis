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
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout, BatchNormalization, ReLU ,Concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd

def add_positive_labels(file_path, num_to_add=50):
    # Charger le dataset existant
    df = pd.read_csv(file_path)
    
    # Créer de nouvelles lignes avec un sentiment positif (label = 1)
    positive_tweets = ['This is a great tweet!', 'I love this!', 'So happy about this!', 'Amazing day!', 'Feeling good today!'] * (num_to_add // 5)
    
    # Limiter le nombre de nouveaux tweets à ajouter
    positive_tweets = positive_tweets[:num_to_add]
    
    # Créer un DataFrame avec les tweets positifs et leur label (1 pour positif)
    positive_df = pd.DataFrame({
        'tweet': positive_tweets,
        'sentiment': [1] * len(positive_tweets)  # label 1 pour les positifs
    })
    
    # Ajouter ces lignes au DataFrame existant
    df = pd.concat([df, positive_df], ignore_index=True)
    
    # Sauvegarder le DataFrame mis à jour dans un nouveau fichier (ou remplacer l'existant)
    df.to_csv('updated_dataset.csv', index=False)
    
    # Afficher quelques exemples pour vérifier
    print(df.tail())  # Afficher les dernières lignes pour vérifier
    return df


# Exemple d'appel
def clean_and_prepare_data(file_path):
    df = add_positive_labels('tweets_dataset.csv', num_to_add=50)
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
        return data
    dataset['tweet'] = dataset['tweet'].apply(lambda x: lemmatizer_on_text(x))
    X = df['tweet'].values
    y = np.array([label + 1 for label in df['sentiment'].values])  # Convertir -1, 0, 1 en 0, 1, 2
    return X, y

def main_pipeline(file_path):
    # Nettoyage et préparation
    X, y = clean_and_prepare_data(file_path)
    # Diviser les données
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    encoder = LabelEncoder()
    tr_label = encoder.fit_transform(y_train)
    val_label = encoder.transform(y_val)
    ts_label = encoder.transform(y_test)
    # Tokenisation et padding
    max_words = 10000
    max_len = 500  # Longueur correcte
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    train_x_padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    
    sequences = tokenizer.texts_to_sequences(X_val)
    val_x_padded = pad_sequences(sequences, maxlen=max_len)

    sequences = tokenizer.texts_to_sequences(X_test)
    test_x_padded = pad_sequences(sequences, maxlen=max_len)
    # Vérification des dimensions
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}, y_test shape: {y_test.shape}")
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)
    # Modèle
    model = Sequential()     
    model.add(Embedding(10000, 8, input_length=max_len))  # Correct max_len
    model.add(Conv1D(64, 9, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(.5))
    model.add(GlobalMaxPooling1D())
    model.add(Conv1D(64, 9, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(.5))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(3, activation='softmax'))  # 3 classes
    model.compile(optimizer='adamax',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])

    model.summary()

    # Entraînement
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    batch_size = 8
    epochs = 10
    history = model.fit(
        train_x_padded, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_x_padded, y_val),
        callbacks=[early_stopping]
    )

    # Évaluation
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
    print(f"Loss: {loss:.2f}, Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

    # Sauvegarde du modèle
    model.save('text_classifier.h5')
    print("Modèle sauvegardé dans 'text_classifier.h5'.")

    # Tracer les métriques
    plot_metrics(history)

def plot_metrics(history):
    tr_acc = history.history['accuracy']
    tr_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    epochs = range(1, len(tr_acc) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, tr_loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'g', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, tr_acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

file_path = 'tweets_dataset.csv'

main_pipeline(file_path)


