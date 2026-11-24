from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

base_dir = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    saved_data = joblib.load(base_dir / "house_price_model.pkl")
    return saved_data["model"], saved_data["columns"]

@st.cache_data
def load_dataset():
    return pd.read_csv(base_dir / "House Price Prediction Dataset.csv")

model, feature_columns = load_model()
df = load_dataset()

st.title("🏠 House Price Prediction")
st.write("Adjust the property details below to estimate the house price.")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area (sq ft)",
        min_value=int(df["Area"].min()),
        max_value=int(df["Area"].max()),
        value=int(df["Area"].median()),
        step=50,
    )
    bedrooms = st.slider(
        "Bedrooms",
        min_value=int(df["Bedrooms"].min()),
        max_value=int(df["Bedrooms"].max()),
        value=int(df["Bedrooms"].median()),
    )
    bathrooms = st.slider(
        "Bathrooms",
        min_value=int(df["Bathrooms"].min()),
        max_value=int(df["Bathrooms"].max()),
        value=int(df["Bathrooms"].median()),
    )
    floors = st.slider(
        "Floors",
        min_value=int(df["Floors"].min()),
        max_value=int(df["Floors"].max()),
        value=int(df["Floors"].median()),
    )

with col2:
    year_built = st.slider(
        "Year Built",
        min_value=int(df["YearBuilt"].min()),
        max_value=int(df["YearBuilt"].max()),
        value=int(df["YearBuilt"].median()),
    )
    location = st.selectbox("Location", sorted(df["Location"].dropna().unique().tolist()))
    condition = st.selectbox("Condition", sorted(df["Condition"].dropna().unique().tolist()))
    garage = st.selectbox("Garage", sorted(df["Garage"].dropna().unique().tolist()))

if st.button("Predict Price", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "YearBuilt": year_built,
        "Location": location,
        "Condition": condition,
        "Garage": garage,
        "HouseAge": 2026 - year_built,
        "TotalRooms": bedrooms + bathrooms,
    }])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    predicted_price = model.predict(input_encoded)[0]
    st.success(f"### Estimated House Price: {predicted_price:,.2f}")