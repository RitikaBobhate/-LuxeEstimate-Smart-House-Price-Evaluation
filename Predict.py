import streamlit as st
import pandas as pd
from Predict import predict_price

st.title("\U0001F4C8 Predict House Price")

city = st.text_input("City")
area = st.number_input("Area (sq ft)", step=10.0)
bhk = st.slider("Bedrooms (BHK)", 1, 10, 3)
weather_temp = st.number_input("Current Temperature (in Celsius)")
weather_humidity = st.slider("Humidity %", 0, 100, 60)

if st.button("Predict Price"):
    result = predict_price(city, area, bhk, weather_temp, weather_humidity)
    st.success(f"Estimated Price: \U0001F4B8 ₹{result:,.2f}")