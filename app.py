
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(page_title="Taxi Fare Predictor")


@st.cache_resource
def load_model():
    file = Path(__file__).with_name("taxi_fare_model.joblib")
    return joblib.load(file)

artifact = load_model()
model = artifact["model"]

st.title("Taxi Fare Predictor")
st.write("Predict a taxi fare from the distance.")

distance = st.slider("Distance (km)", 1.0, 30.0, 10.0, 0.5)


if st.button("Predict fare"):
    fare = float(model.predict([[distance]])[0])
    st.success(f"Estimated fare: {fare:.2f} JD")
    st.caption("Demo estimate; not an actual fare.")
