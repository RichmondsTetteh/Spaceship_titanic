import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow_decision_forests as tfdf
import os

# Cache resources
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

@st.cache_resource
def load_training_columns():
    return joblib.load('training_columns.pkl')

# Load model and columns
try:
    model = load_model()
    training_columns = load_training_columns()
except FileNotFoundError as e:
    st.error(f"File not found: {str(e)}. Ensure 'model.pkl' and 'training_columns.pkl' are in the app directory.")
    st.stop()
except ModuleNotFoundError as e:
    st.error(f"Missing module: {str(e)}. Ensure TensorFlow and tensorflow_decision_forests are installed.")
    st.stop()
except Exception as e:
    st.error(f"Error loading files: {str(e)}")
    st.stop()

st.title('Space Titanic Survival Prediction')
st.write('Enter passenger details to predict if they will be transported.')

# Collect user input
passenger_id = st.text_input('Passenger ID')
home_planet = st.selectbox('Home Planet', ['Earth', 'Europa', 'Mars', 'Unknown'])
cryo_sleep = st.selectbox('Cryo Sleep', [True, False, 'Unknown'])
cabin = st.text_input('Cabin (e.g., B/0/P)')
destination = st.selectbox('Destination', ['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e', 'Unknown'])
age = st.number_input('Age', min_value=0, max_value=100, step=1)
vip = st.selectbox('VIP', [True, False, 'Unknown'])
room_service = st.number_input('Room Service Bills', min_value=0.0)
food_court = st.number_input('Food Court Bills', min_value=0.0)
shopping_mall = st.number_input('Shopping Mall Bills', min_value=0.0)
spa = st.number_input('Spa Bills', min_value=0.0)
vr_deck = st.number_input('VR Deck Bills', min_value=0.0)
name = st.text_input('Name')

# Create DataFrame
input_data = {
    'PassengerId': [passenger_id],
    'HomePlanet': [home_planet],
    'CryoSleep': [cryo_sleep],
    'Cabin': [cabin],
    'Destination': [destination],
    'Age': [float(age)],
    'VIP': [vip],
    'RoomService': [float(room_service)],
    'FoodCourt': [float(food_court)],
    'ShoppingMall': [float(shopping_mall)],
    'Spa': [float(spa)],
    'VRDeck': [float(vr_deck)],
    'Name': [name]
}
input_df = pd.DataFrame(input_data)

# Preprocess input
def preprocess_input(df):
    # Impute missing values
    numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    # Handle categorical and boolean columns
    categorical_cols = ['HomePlanet', 'Cabin', 'Destination', 'Name']
    boolean_cols = ['CryoSleep', 'VIP']
    # Convert boolean-like inputs ('Unknown' to False for simplicity)
    for col in boolean_cols:
        df[col] = df[col].apply(lambda x: False if x == 'Unknown' else x)
    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)
    # Convert boolean columns to integers
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)
    # Align with training columns
    df_aligned = df_encoded.reindex(columns=training_columns, fill_value=0)
    return df_aligned

# Make prediction
if st.button('Predict'):
    if not passenger_id:
        st.warning("Please enter Passenger ID.")
    elif not (cabin == '' or '/' in cabin):
        st.warning("Cabin must follow the format 'Letter/Number/Letter' (e.g., B/0/P) or be empty.")
    elif any(x < 0 for x in [room_service, food_court, shopping_mall, spa, vr_deck]):
        st.warning("Bills cannot be negative.")
    else:
        try:
            processed_input = preprocess_input(input_df.copy())
            # Convert to TensorFlow dataset
            input_ds = tfdf.keras.pd_dataframe_to_tf_dataset(processed_input)
            # Predict
            predictions = model.predict(input_ds, verbose=0)
            prob = predictions[0][0]  # Probability of positive class (Transported=True)
            st.write(f"Probability of being Transported: {prob:.2%}")
            if prob >= 0.5:
                st.success(f'Prediction: Transported (Passenger ID: {passenger_id})')
            else:
                st.info(f'Prediction: Not Transported (Passenger ID: {passenger_id})')
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")