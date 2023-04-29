# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime

startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename, 'rb'))
# otwieramy wcześniej wytrenowany model

# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():
    st.set_page_config(page_title="Health app")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://www.apple.com/newsroom/images/values/health/Apple-Health-study-July-2022-hero_big.jpg.large_2x.jpg")

    with overview:
        st.title("Health app")

    with left:
        symptoms_slider = st.slider("Objawy", value=1, min_value=0, max_value=5)
        age_slider = st.slider("Wiek", value=1, min_value=0, max_value=77)
        disease_slider = st.slider("Choroby", min_value=0, max_value=5)

    with right:
        height_slider = st.slider("Wzrost", min_value=0, max_value=200)
        medication_slider = st.slider("Leki", min_value=0, max_value=4, step=1)

    data = [[symptoms_slider, age_slider, disease_slider, height_slider, medication_slider]]
    health = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy taka osoba jest zdrowa?")
        st.subheader(("Tak" if health[0] == 1 else "Nie"))
        st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][health][0] * 100))


if __name__ == "__main__":
    main()
