import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.header("Iris Flower Prediction App")

st.sidebar.header('Ui Parameters')

def ui_patameters():
    sepal_length = st.sidebar.slider('Sepal lenght',4.3,7.9, 5.0)
    sepal_width = st.sidebar.slider('Sepal width',2.0,4.4,3.2)
    petal_lenght = st.sidebar.slider('Petal length',1.0,6.9,2.5)
    petal_width = st.sidebar.slider('Petal width',0.1,2.5,0.5)

    data = {
        'sepal_length ': sepal_length,
        'sepal_width ': sepal_width ,
        'petal_length ': petal_lenght,
        'petal_width ': petal_width
    }

    parametros = pd.DataFrame(data,index=[0])
    return parametros

df = ui_patameters()
st.subheader("UI Parâmetros")
st.write(df)
st.divider() 

iris = datasets.load_iris()

x = iris.data
y = iris.target

rfc = RandomForestClassifier()
rfc.fit(x, y)

previsao = rfc.predict(df)
previsao_probabilidade = rfc.predict_proba(df)

st.subheader('Rótulos de classificação')
st.write(iris.target_names)
st.divider() 

st.subheader('Previsão')
st.write(previsao)
st.divider() 

st.subheader('Probabilidade %')
st.write(previsao_probabilidade)


