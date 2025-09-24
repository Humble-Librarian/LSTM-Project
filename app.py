import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Binge Session Predictor",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model and Preprocessing Objects ---
@st.cache_resource
def load_resources():
    """
    Loads the trained model and preprocessing objects.
    """
    try:
        model = tf.keras.models.load_model('lstm_model.h5')
        label_encoders = load('label_encoders.joblib')
        scaler = load('minmax_scaler.joblib')
        return model, label_encoders, scaler
    except FileNotFoundError:
        st.error("Model or preprocessing files not found. Please ensure 'train_lstm_model.py' has been run successfully.")
        return None, None, None

model, label_encoders, scaler = load_resources()

# --- Main App UI ---
st.title("📺 Binge Session Predictor")
st.markdown(
    """
    Welcome to the Binge Session Predictor! 
    This app uses a machine learning model to estimate the remaining minutes of a binge-watching session based on various factors.
    Enter the details of your current session below to get a prediction.
    """
)

st.markdown("---")

if model is not None and label_encoders is not None and scaler is not None:
    # --- User Input Widgets in Columns ---
    with st.expander("🎬 **Enter Your Session Details**", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Content Information")
            genre = st.selectbox(
                "🎬 **Genre**", 
                options=list(label_encoders['genre'].classes_), 
                help="The genre of the show you are watching."
            )
            episode_length = st.number_input(
                "⏱️ **Episode Length (minutes)**", 
                min_value=1.0, 
                max_value=200.0, 
                value=25.0, 
                step=1.0,
                help="Total duration of the episode."
            )
            watch_fraction = st.number_input(
                "📊 **Watch Fraction (0.0 to 1.0)**", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01,
                help="What percentage of the episode has been watched so far."
            )

        with col2:
            st.markdown("### User & Device Details")
            device = st.selectbox(
                "📱 **Device**", 
                options=list(label_encoders['device'].classes_),
                help="The device used for watching."
            )
            subscription = st.selectbox(
                "💳 **Subscription**", 
                options=list(label_encoders['subscription'].classes_),
                help="Your subscription tier."
            )
            time_of_day = st.number_input(
                "⏰ **Time of Day (24hr format)**", 
                min_value=0.0, 
                max_value=24.0, 
                value=12.0, 
                step=0.1,
                help="The current time of day."
            )
            watched_minutes = st.number_input(
                "▶️ **Watched Minutes**", 
                min_value=0.0, 
                max_value=200.0, 
                value=15.0, 
                step=1.0,
                help="Number of minutes watched so far in the current session."
            )
    
    st.markdown("---")
    
    # --- Prediction Logic and UI ---
    if st.button("🚀 **Predict Remaining Minutes**", use_container_width=True):
        with st.spinner('Predicting...'):
            # Create a single data point from user input
            data_point = pd.DataFrame([{
                'genre': genre,
                'device': device,
                'subscription': subscription,
                'episode_length': episode_length,
                'time_of_day': time_of_day,
                'watch_fraction': watch_fraction,
                'watched_minutes': watched_minutes,
                'pos': 10 # Arbitrary pos for single-session prediction
            }])
            
            # Preprocess the user input using the loaded preprocessors
            for col in ['genre', 'device', 'subscription']:
                data_point[f'{col}_encoded'] = label_encoders[col].transform(data_point[col])

            numerical_features = data_point[['episode_length', 'time_of_day', 'watch_fraction', 'watched_minutes']]
            data_point[numerical_features.columns] = scaler.transform(numerical_features)
            
            # Prepare the input for the LSTM model. 
            feature_names = ['pos', 'episode_length', 'time_of_day', 'watch_fraction', 'watched_minutes',
                             'genre_encoded', 'device_encoded', 'subscription_encoded']
            
            input_sequence = np.zeros((1, 10, len(feature_names)))
            input_sequence[0, 9, :] = data_point[feature_names].values

            prediction = model.predict(input_sequence)
            
        st.success(f"🎉 **Prediction Complete!**")
        st.balloons()
        
        st.info(f"The model predicts you will watch for **{prediction[0][0]:.2f} more minutes**.")

else:
    st.warning("Please run the `train_lstm_model.py` script first to generate the necessary model and preprocessing files.")
    st.info("Make sure the files `lstm_model.h5`, `label_encoders.joblib`, and `minmax_scaler.joblib` are in the same directory.")

st.markdown("---")
st.markdown("`Powered by Streamlit and TensorFlow`")
