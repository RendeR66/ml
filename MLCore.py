import os
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import sys
from sklearn.model_selection import train_test_split
from keras import backend as K

# Set standard output to UTF-8
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

MODEL_PATH = "saved_model/my_model.keras"


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def load_and_preprocess_data(file_path):
    try:
        data = pd.read_excel(file_path, engine="openpyxl", sheet_name=None)
        if not data:
            raise FileNotFoundError("No sheets found in the Excel file")

        df = pd.concat(data.values(), ignore_index=True)

        df.rename(columns={
            'date': 'date_from',
            'from': 'route_from',
            'to': 'route_to',
            'price': 'cost',
            'time_taken': 'duration'
        }, inplace=True)

        required_columns = ['date_from', 'route_from', 'route_to', 'cost', 'duration']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the dataset")

        df['date_from'] = pd.to_datetime(df['date_from'], format="%d %m %Y", errors='coerce')
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        df['duration'] = df['duration'].apply(convert_duration_to_minutes)

        df.dropna(subset=['date_from', 'cost', 'duration'], inplace=True)
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error: {e}")
        raise


def convert_duration_to_minutes(duration_str):
    pattern = re.compile(r'(\d+)h (\d+)m')
    match = pattern.match(duration_str)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        total_minutes = hours * 60 + minutes
        return total_minutes
    else:
        raise ValueError(f"Duration string '{duration_str}' does not match expected format 'Xh Ym'")


# Assuming the Excel file is in the same directory as this script
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "DateSet.xlsx")

df = load_and_preprocess_data(file_path)


def create_and_train_model(df):
    X = df[['date_from', 'route_from', 'route_to']].copy()
    y = df[['cost', 'duration']].copy()

    X['date_from'] = X['date_from'].astype('int64') / 10 ** 9  # Convert to seconds
    X = pd.get_dummies(X, columns=['route_from', 'route_to'])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(y.shape[1])
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse',
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    model.fit(X, y, epochs=10, batch_size=32)

    model.save(MODEL_PATH)

    return model


if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = create_and_train_model(df)


def evaluate_model(df, model):
    X = df[['date_from', 'route_from', 'route_to']].copy()
    y_true = df[['cost', 'duration']].copy()

    X['date_from'] = X['date_from'].astype('int64') / 10 ** 9  # Convert to seconds
    X = pd.get_dummies(X, columns=['route_from', 'route_to'])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y_pred = model.predict(X)

    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values

    return y_true, y_pred



y_true = [100, 200, 300, 400]
y_pred = [110, 190, 310, 420]
mape = calculate_mape(np.array(y_true), np.array(y_pred))

print(f"MAPE: {mape}")
