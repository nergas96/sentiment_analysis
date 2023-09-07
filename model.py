import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
import pickle

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('spanish'))


def preprocess_text(text):
    stop_words2 = stop_words -{"no"}
    words = word_tokenize(text.lower(), language='spanish')
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words2]
    return ' '.join(words)


def predecir_sentimiento(texto, modelo, vectorizador, tf_transformer=None):
    # Preprocesar el texto de entrada
    texto_preprocesado = preprocess_text(texto)  # Asegúrate de tener la función preprocess_text definida
    print(texto_preprocesado)
    # Vectorizar el texto preprocesado
    texto_vectorizado = vectorizador.transform([texto_preprocesado])
    if tf_transformer!=None:
        texto_vectorizado = tf_transformer.transform(texto_vectorizado)
    # Realizar la predicción
    prediccion = modelo.predict(texto_vectorizado)
    prob = modelo.predict_proba(texto_vectorizado)

    # Devolver la clasificación predicha
    return prediccion[0], prob