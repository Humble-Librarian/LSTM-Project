import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import save_model
from joblib import dump, load

# Load the dataset from the uploaded CSV file
try:
    df = pd.read_csv('binge_sessions_150k.csv')
except FileNotFoundError:
    print("Error: The file 'binge_sessions_150k.csv' was not found.")
    print("Please make sure the file is in the same directory as the script.")
    exit()

# --- Data Preprocessing ---

# Sort the DataFrame by user_id and pos to create sequences
df = df.sort_values(by=['user_id', 'pos']).reset_index(drop=True)

# Identify categorical and numerical columns for preprocessing
categorical_cols = ['genre', 'device', 'subscription']
numerical_cols = ['episode_length', 'time_of_day', 'watch_fraction', 'watched_minutes']

# Create and save LabelEncoder objects for categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
# Save the fitted encoders
dump(label_encoders, 'label_encoders.joblib')

# Min-Max scale the numerical features and save the scaler
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
# Save the fitted scaler
dump(scaler, 'minmax_scaler.joblib')

# --- Create Sequences for LSTM ---

def create_sequences(data, sequence_length):
    """
    This function groups the data by user and creates input sequences for the LSTM model.
    Each sequence consists of a fixed number of timesteps (e.g., a user's last 10 sessions).
    The target is the 'remaining_minutes' of the session immediately following the sequence.

    Args:
        data (pd.DataFrame): The preprocessed DataFrame.
        sequence_length (int): The number of sessions in each input sequence.

    Returns:
        tuple: A tuple containing the sequences (X) and the corresponding targets (y).
    """
    sequences = []
    targets = []
    
    # Iterate through each user to create their individual sequences
    for user_id, group in data.groupby('user_id'):
        # Features to be used as input for the LSTM
        user_features = group[['pos', 'episode_length', 'time_of_day', 'watch_fraction', 'watched_minutes',
                               'genre_encoded', 'device_encoded', 'subscription_encoded']].values
        
        # The target variable to be predicted
        user_targets = group['remaining_minutes'].values
        
        # Ensure the user has enough data to create a sequence
        if len(user_features) > sequence_length:
            for i in range(len(user_features) - sequence_length):
                # The input sequence is a slice of the user's features
                sequences.append(user_features[i:i + sequence_length])
                # The target is the remaining_minutes of the next session
                targets.append(user_targets[i + sequence_length])
                
    return np.array(sequences), np.array(targets)

# Define the length of each sequence
SEQUENCE_LENGTH = 10
X, y = create_sequences(df, SEQUENCE_LENGTH)

# --- Model Building and Training ---

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model architecture
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1)) 

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print a summary of the model's architecture
print("Model Summary:")
model.summary()

# --- Training (Uncomment and run in your environment) ---
print("\nStarting model training...")
history = model.fit(
    X_train, 
    y_train, 
    epochs=20,
    batch_size=64, 
    validation_data=(X_test, y_test)
)
print("Training finished.")

# Save the trained model and preprocessing objects
print("\nSaving the trained model and preprocessing objects...")
save_model(model, 'lstm_model.h5')
print("Model, scalers, and encoders saved successfully.")
