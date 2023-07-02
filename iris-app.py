import streamlit as st
import pandas as pd
import pickle
from sklearn import datasets

st.write("""
# สวัสดี App สำหรับทำนายดอก Iris
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4) # input
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4) # input
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3) # input
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2) # input
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters') 
st.write(df)

iris = datasets.load_iris()

clf = pickle.load(open('finalized_model_knn.sav', 'rb')) # load model จากที่ทำเอาไว้

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df) # ความน่าจะเป็น ที่จะได้เป็น class อะไร

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)