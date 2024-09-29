import requests
import streamlit as st
import pandas as pd
from netCDF4 import Dataset
import os
import numpy as np
import datetime
import math
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import base64
from streamlit_folium import st_folium
import folium
import gdown

# Set TensorFlow to use CPU for computations
os.environ['TF_ENABLE_MLIR_OPTIMIZATIONS'] = '1'

# Google Drive file ID for the dataset
file_id = '1RVnGXF9RMYb4qgfHFawlASp1dtZaYwwM'
local_filename = '5days_combined.nc'

# Check if the file already exists
if not os.path.exists(local_filename):
    print(f"File {local_filename} not found locally. Downloading...")
    gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", local_filename, quiet=True)
else:
    print(f"File {local_filename} already exists. Skipping download.")

# Load the NetCDF dataset
L3_data1 = Dataset(local_filename)
print(L3_data1.variables.keys())

# Extract necessary data from the dataset
lat = L3_data1.variables['latitude'][:]
lon = L3_data1.variables['longitude'][:]
time_data = L3_data1.variables['time'][:]
no2 = L3_data1.variables['tropospheric_NO2_column_number_density'][:,:,:]

# Model file path
model_file = 'no2_prediction_model.h5'

# Function to build the LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Function to predict NO2 concentration based on latitude, longitude, and date
def predict(latitude_input, longitude_input, date):
    predict_days = abs((datetime.datetime.strptime('2024-09-05',"%Y-%m-%d") - datetime.datetime.strptime(date,"%Y-%m-%d")).days)
    st.write("--------predict days-----:", predict_days)
    
    sq_diff_lat = (lat - latitude_input)**2
    sq_diff_lon = (lon - longitude_input)**2
    
    min_index_lat = sq_diff_lat.argmin()
    min_index_lon = sq_diff_lon.argmin()
    
    start_date = L3_data1.variables['time'].units[14:24]
    end_date = L3_data1.variables['time'].units[14:18] + '-09-05'
    
    date_range = pd.date_range(start=start_date, end=end_date)
    st.success(date_range)   
    df = pd.DataFrame(0, columns=['NO2'], index=date_range)
    
    dt = np.arange(0, 4)
    for i in dt:
        df.iloc[i] = no2[i, min_index_lat, min_index_lon]
        
    df.to_csv(r"5days_combined.csv")
    
    # Load the CSV data for prediction
    data_frame = pd.read_csv(r"5days_combined.csv")
    df1 = data_frame.reset_index()['NO2']
    
    # Preprocess the data
    df1 = df1.fillna(0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    
    # Splitting dataset into train and test
    training_size = int(len(df1) * 0.70)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[test_size:len(df1), :1]
    
    # Convert dataset into matrix form for LSTM
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 1
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    # Reshape the input to [samples, time steps, features] for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Check if a saved model exists, load it; otherwise, train a new one
    if os.path.exists(model_file):
        model = load_model(model_file)
        print("Loaded saved model.")
    else:
        model = build_model()
        print("Training a new model.")
        model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=50, batch_size=8, verbose=1)
        model.save(model_file)
    
    # Prediction using the model
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform the predictions
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    
    # Predict the next days' NO2 concentration
    x_input = test_data[len(test_data)-1:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    
    lst_output = []
    n_steps = 1
    i = 0
    while i < predict_days:
        if len(temp_input) > 1:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i += 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1
    
    # Inverse transform the predicted NO2 values
    no2_output = pd.DataFrame(scaler.inverse_transform(lst_output), columns=['NO2 Concentration'])
    st.write(no2_output)
    output = no2_output.at[predict_days-1, 'NO2 Concentration']
    return output

# Main function for Streamlit app
def main():
    st.markdown("<h1 style ='color:green; text_align:center;font-family:times new roman;font-weight: bold;font-size:20pt;'>NO2 PREDICTION </h1>", unsafe_allow_html=True)  
    st.markdown("<h1 style='text-align: left; font-weight:bold;color:black;background-color:white;font-size:11pt;'> Enter the Location Details</h1>", unsafe_allow_html=True)
    
    m = folium.Map()
    m.add_child(folium.LatLngPopup())
    map = st_folium(m, height=500, width=700)
    try:
        latitude_input = float(map['last_clicked']['lat'])
        longitude_input = float(map['last_clicked']['lng'])
        st.write(latitude_input)
        st.write(longitude_input)
    except:
        st.warning("No location chosen")
    
    # Enter the date for prediction
    date = st.date_input('Date', value=pd.to_datetime('2024-09-06'), min_value=pd.to_datetime('2024-09-06'), max_value=pd.to_datetime('2025-09-06'))
    
    if st.button("Predict"):
        latitude_input = float(latitude_input)
        longitude_input = float(longitude_input)
        st.success(latitude_input)
        st.success(longitude_input)
        date = str(date)
        st.success(date)
        with st.spinner("Predicting the results...."):
            result = predict(latitude_input, longitude_input, date)
        st.success(f'Predicted NO2 Concentration is {round(result, 4)} molecules/cm²')
        st.balloons()

if __name__ == "__main__":
    main()
