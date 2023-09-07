import streamlit as st
import pickle
from model import predecir_sentimiento


with open('models/model.pkl', 'rb') as f:
    modelo = pickle.load(f)

with open('models/tf_transformer.pkl', 'rb') as f:
    tf_transformer = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


st.title('Escribe algo y te diremos si es un comentario positivo o negativo!')
with st.form("my_form"):
    texto = st.text_input('Escribe un comentario', '')
    clasificar = st.form_submit_button("Submit")

clasificacion = predecir_sentimiento(texto, modelo, vectorizer, tf_transformer=tf_transformer)
st.write("### Clasificación:")
st.write(clasificacion[0])
st.write("### Probabilidad de que el comentario sea positivo:")
d = {"Bueno":clasificacion[1][0][0], "Malo": clasificacion[1][0][1]}
st.write(d)