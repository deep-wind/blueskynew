# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 19:57:21 2022

@author: PRAMILA
"""
import requests
import streamlit as st
import pandas as pd
from netCDF4 import Dataset
import os
import xarray as xr
import numpy as np
import datetime
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from streamlit_folium import folium_static, st_folium
import folium
import gdown
import math

# specify that TensorFlow performs computations using the CPU
os.environ['TF_ENABLE_MLIR_OPTIMIZATIONS'] = '1'

# BlueSky Above: Pollution estimation using hyper-spectral satellite imagery and maps

# Google Drive file ID
file_id = '1RVnGXF9RMYb4qgfHFawlASp1dtZaYwwM'
local_filename = '5days_combined.nc'

# Check if the file already exists
if not os.path.exists(local_filename):
    print(f"File {local_filename} not found locally. Downloading...")
    gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", local_filename, quiet=True)
else:
    print(f"File {local_filename} already exists. Skipping download.")

# Load the dataset
L3_data1 = Dataset(local_filename)
print("Dataset variables loaded:", L3_data1.variables.keys())
lat = L3_data1.variables['latitude'][:]
lon = L3_data1.variables['longitude'][:]
time_data = L3_data1.variables['time'][:]
no2 = L3_data1.variables['tropospheric_NO2_column_number_density'][:,:,:]

def predict(latitude_input, longitude_input, date):
    predict_days = abs((datetime.datetime.strptime('2024-09-05', "%Y-%m-%d") - datetime.datetime.strptime(date, "%Y-%m-%d")).days)
    st.write("Days to predict:", predict_days)

    # Finding the closest lat and lon indices
    sq_diff_lat = (lat - latitude_input) ** 2
    sq_diff_lon = (lon - longitude_input) ** 2
    min_index_lat = sq_diff_lat.argmin()
    min_index_lon = sq_diff_lon.argmin()

    start_date = L3_data1.variables['time'].units[14:24]
    end_date = L3_data1.variables['time'].units[14:18] + '-09-05'
    
    date_range = pd.date_range(start=start_date, end=end_date)
    st.success("Date range of the data: " + str(date_range))
    
    df = pd.DataFrame(0, columns=['NO2'], index=date_range)
    dt = np.arange(0, 4)

    for i in dt:
        df.iloc[i] = no2[i, min_index_lat, min_index_lon]
        
    df.to_csv(r"5days_combined.csv")
    
    ##############################################
    #            PREDICTION MODULE               #
    ##############################################
    # Load the data for prediction
    data_frame = pd.read_csv(r"5days_combined.csv")
    df1 = data_frame.reset_index()['NO2']
    st.write("Original NO2 data:", df1)
    
    df1 = df1.fillna(0)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    
    # Splitting the dataset into training and testing sets
    training_size = int(len(df1) * 0.70)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[test_size:len(df1), :1]
    
    print("Train data size:", train_data.shape)
    print("Test data size:", test_data.shape)
    
    # Prepare the dataset for LSTM
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):         
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    
    time_step = 1
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    print("Training model summary:")
    model.summary()

    model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=50, batch_size=8, verbose=1)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform the predictions
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    
    st.info("Train predictions complete")
    st.info("Test predictions complete")

    # Calculate RMSE
    st.write("Train RMSE:", math.sqrt(mean_squared_error(y_train, train_predict)))
    st.write("Test RMSE:", math.sqrt(mean_squared_error(ytest, test_predict)))

    # Future predictions
    x_input = test_data[len(test_data) - 1:].reshape(1, -1)
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

    no2_output = pd.DataFrame(scaler.inverse_transform(lst_output), columns=['NO2 Concentration'])
    st.write(no2_output)
    output = no2_output.at[predict_days - 1, 'NO2 Concentration']
    return output

def main():
    st.markdown("<h1 style ='color:green; text_align:center;font-family:times new roman;font-weight: bold;font-size:20pt;'>NO2 PREDICTION </h1>", unsafe_allow_html=True)
    m = folium.Map()
    m.add_child(folium.LatLngPopup())
    map = st_folium(m, height=500, width=700)
    try:
        latitude_input = float(map['last_clicked']['lat'])
        longitude_input = float(map['last_clicked']['lng'])
        st.write("Selected Latitude:", latitude_input)
        st.write("Selected Longitude:", longitude_input)
    except:
        st.warning("No location chosen")

    date = st.date_input('Date', value=pd.to_datetime('2024-09-06'), min_value=pd.to_datetime('2024-09-06'), max_value=pd.to_datetime('2025-09-06'))
    
    if st.button("Predict"):
        st.success("Latitude: " + str(latitude_input))
        st.success("Longitude: " + str(longitude_input))
        date = str(date)
        st.success("Selected date: " + date)
        with st.spinner("Predicting the results..."):
            result = predict(latitude_input, longitude_input, date)
        st.success(f'Predicted NO2 Concentration is {round(result, 4)} molecules/cm2')
        st.balloons()

if __name__ == "__main__":
    main()
