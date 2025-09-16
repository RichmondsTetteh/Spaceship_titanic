import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'model.pkl' is in the same directory.")
    st.stop()

st.title('Space Titanic Survival Prediction')

st.write('Enter the passenger details to predict if they will be transported.')

# Collect user input
passenger_id = st.text_input('Passenger ID')
home_planet = st.selectbox('Home Planet', ['Earth', 'Europa', 'Mars'])
cryo_sleep = st.selectbox('Cryo Sleep', [True, False])
cabin = st.text_input('Cabin (e.g., B/0/P)')
destination = st.selectbox('Destination', ['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'])
age = st.number_input('Age', min_value=0, max_value=100)
vip = st.selectbox('VIP', [True, False])
room_service = st.number_input('Room Service Bills', min_value=0.0)
food_court = st.number_input('Food Court Bills', min_value=0.0)
shopping_mall = st.number_input('Shopping Mall Bills', min_value=0.0)
spa = st.number_input('Spa Bills', min_value=0.0)
vr_deck = st.number_input('VR Deck Bills', min_value=0.0)
name = st.text_input('Name') # Name might not be used in prediction based on preprocessing


# Create a DataFrame from user input
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


# Preprocess the input data (similar to how training data was preprocessed)
def preprocess_input(df):
    # Handle missing values (using the same imputation strategy as training)
    # For demonstration, using hardcoded median/mode. In a real app, you'd load these from training data.
    # Assuming we have train_df available or saved median/mode values

    # Simple imputation for demonstration
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Replace with median from training data if available, otherwise a default
            df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 0.0)
        elif df[col].dtype == 'object':
             # Replace with mode from training data if available, otherwise a default
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].isnull().all() else 'Unknown')
        elif df[col].dtype == 'bool':
             df[col] = df[col].fillna(df[col].mode()[0] if not df[col].isnull().all() else False)


    # Encode categorical features (using the same one-hot encoding as training)
    # This is tricky as the test data needs to have the exact same columns as the training data after encoding.
    # A robust approach would save the list of columns from the training data after encoding.
    # For this example, we'll assume the categorical columns are consistent and
    # reindex the input data with the training columns after encoding.

    # Assuming train_df_encoded is available or its columns are saved
    # This requires access to the columns of the *trained* features dataframe (X_train or train_df_encoded)

    # A simplified approach for demo assumes we know the categorical columns and can re-apply encoding
    categorical_cols = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Name']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)

    # To ensure same columns as training data, we need the training columns list.
    # Let's assume we have saved the column list from X_train or train_df_encoded.
    # For demonstration, we'll use a placeholder. In a real application, load the saved column list.
    # PLACEHOLDER: Replace with actual columns from trained features
    # e.g., loaded_training_columns = joblib.load('training_columns.pkl')
    # For now, let's try to use the columns from X_train if available in the kernel
    try:
        training_columns = X_train.columns
    except NameError:
        st.error("Training columns not available in the kernel. Cannot align input data.")
        st.stop()


    df_aligned = df_encoded.reindex(columns=training_columns, fill_value=0)

    # Ensure boolean columns are int after reindexing if they were not included in get_dummies
    for col in df_aligned.columns:
         if df_aligned[col].dtype == 'bool':
              df_aligned[col] = df_aligned[col].astype(int)


    return df_aligned


# Make prediction when button is clicked
if st.button('Predict'):
    if not passenger_id:
        st.warning("Please enter Passenger ID.")
    else:
        processed_input = preprocess_input(input_df.copy())
        prediction = model.predict(processed_input)

        # Assuming binary classification where prediction is 0 or 1
        if prediction[0] == 1:
            st.success('Prediction: Transported')
        else:
            st.info('Prediction: Not Transported')