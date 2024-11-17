"""
Created on Mon Oct 31 19:57:21 2022

@author: PRAMILA
"""
import requests
import os

# Disable GPU usage
import streamlit as st
import pandas as pd
#import harp
from netCDF4 import Dataset
from os.path import join

import glob
import xarray as xr
import numpy as np
from numpy import array
import itertools
import datetime
import time
import math
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt 
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import base64
from streamlit_folium import folium_static
from streamlit_folium import st_folium
import folium
import gdown
import plotly.express as px

# specify that TensorFlow performs computations using the CPU
os.environ['TF_ENABLE_MLIR_OPTIMIZATIONS'] = '1'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# BlueSky Above: Pollution estimation using hyper-spectral satellite imagery and maps

# Google Drive file ID
file_id = '1RVnGXF9RMYb4qgfHFawlASp1dtZaYwwM'
local_filename = '5days_combined.nc'

# Check if the file already exists
if not os.path.exists(local_filename):
    print(f"File {local_filename} not found locally. Downloading from Google Drive...")
    # Download the file from Google Drive
    gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", local_filename, quiet=True)
else:
    print(f"File {local_filename} already exists locally. Skipping download.")

# Install a pip package in the current Jupyter kernel
import sys

# Latitude and Longitude boundaries
LatMin=51.25
LatMax=51.75
LngMin=-0.6
LngMax=0.28
from sentinelsat import SentinelAPI, geojson_to_wkt
import json

geojsonstring='{{"type":"FeatureCollection","features":[{{"type":"Feature","properties":{{}},"geometry":{{"type":"Polygon","coordinates":[[[{LongiMin},{LatiMin}],[{LongiMax},{LatiMin}],[{LongiMax},{LatiMax}],[{LongiMin},{LatiMax}],[{LongiMin},{LatiMin}]]]}}}}]}}'.format(LongiMin=LngMin,LatiMin=LatMin,LongiMax=LngMax,LatiMax=LatMax)

# Sentinel API setup
api = SentinelAPI('s5pguest', 's5pguest' , 'https://s5phub.copernicus.eu/dhus')
footprint = geojson_to_wkt(json.loads(geojsonstring))

startdate='20240901'
enddate='20240905'

L3_data1 = Dataset(local_filename)
print("Dataset loaded successfully.")
print("Available variables in the dataset:", L3_data1.variables.keys())

lat = L3_data1.variables['latitude'][:]
lon = L3_data1.variables['longitude'][:]
time_data = L3_data1.variables['time'][:]
no2 = L3_data1.variables['tropospheric_NO2_column_number_density'][:,:,:]


def predict(latitude_input, longitude_input, date):
    predict_days = abs((datetime.strptime('2024-09-05',"%Y-%m-%d") - datetime.strptime(date,"%Y-%m-%d")).days)
    st.write("Predict days:", predict_days)
    
    if predict_days > 4:
        predict_days = 4

    sq_diff_lat = (lat - latitude_input) ** 2
    sq_diff_lon = (lon - longitude_input) ** 2
    
    min_index_lat = sq_diff_lat.argmin()
    min_index_lon = sq_diff_lon.argmin()
    
    start_date = L3_data1.variables['time'].units[14:24]
    end_date = L3_data1.variables['time'].units[14:18] + '-09-05'
    date_range = pd.date_range(start=start_date, end=end_date)
    
    date_range = pd.date_range(start=start_date, end=end_date)
    st.success(f"Date range: {date_range}")
    
    df = pd.DataFrame(0, columns=['NO2'], index=date_range)
    
    dt = np.arange(0, 4)
    for i in dt:
        df.iloc[i] = no2[i, min_index_lat, min_index_lon]
        
    df.to_csv(r"5days_combined.csv")

    
    ##############################################
    #            PREDICTION MODULE               #
    ##############################################
   
    ### Data Collection
    data_frame=pd.read_csv(r"5days_combined.csv")
    df1=data_frame.reset_index()['NO2']
    st.write(df1)
    ### LSTM are sensitive to the scale of the data. so we apply MinMax scaler 

    df1 = df1.fillna(0)
    ### LSTM are sensitive to the scale of the data. so we apply MinMax scaler 

    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    
    ##splitting dataset into train and test split


    training_size=int(len(df1)*0.70)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[test_size:len(df1),:1]
    print(train_data)
    print(test_data)
    
    # st.write(train_data)
    # st.write(test_data)
       
    
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
     	dataX, dataY = [], []
     	for i in range(len(dataset)-time_step-1):         
        		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        		dataX.append(a)
        		dataY.append(dataset[i + time_step, 0])
     	return np.array(dataX), np.array(dataY)
    
    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 1
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

  
    # reshape input to be [samples, time steps, features] which is required for LSTM
    
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    # st.info(X_train.shape)
    # st.info(y_train.shape)
    
    # print(X_test.shape), print(ytest.shape)
    
    ### Create the Stacked LSTM model
    # model=Sequential()
    # model.add(LSTM(50,return_sequences=True,input_shape=(1,1)))
    # # model.add(LSTM(50,return_sequences=True))
    # model.add(LSTM(50))
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_error',optimizer='adam')
    
    
    # model.summary()
    # model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=50,batch_size=8,verbose=1)
    # model.save('my_model.h5')  
    model = load_model('my_model.h5')
    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    
    ##Transformback to original form
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    
    ### Calculate RMSE performance metrics
    
    math.sqrt(mean_squared_error(y_train,train_predict))
    
    ### Test Data RMSE
    math.sqrt(mean_squared_error(ytest,test_predict))
    
    ### Plotting 
    # shift train predictions for plotting
    look_back=1
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
      
    x_input=test_data[len(test_data)-1:].reshape(1,-1)
    
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    
    # demonstrate prediction for next days
    
    lst_output=[]
    n_steps=1
    i=0
    while(i<predict_days):
        
        if(len(temp_input)>1):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
        
    
    print(lst_output)
    
    
    # Create a DataFrame from the scaled NO2 output with the date range
    no2_output = pd.DataFrame(scaler.inverse_transform(lst_output), 
                              columns=['NO2 Concentration (mol/m²)'])
    
    # Add the date range as a column
    no2_output['Date'] = pd.date_range(start=date, periods=predict_days, freq='D').date
    
    # Reorder the columns to have 'Date' first
    no2_output = no2_output[['Date', 'NO2 Concentration (mol/m²)']]
    
    # Style the DataFrame to highlight NO2 values using a color gradient
    styled_no2_output = no2_output.style.background_gradient(cmap='YlOrRd', subset=['NO2 Concentration (mol/m²)'])
    
    # Display the styled DataFrame with date and NO2 concentration
    st.write("Predicted NO2 Concentration for Future Days with Date:")
    st.dataframe(styled_no2_output)
    
    # Extract the NO2 concentration for the specific number of predicted days
    output = no2_output.at[predict_days - 1, 'NO2 Concentration (mol/m²)']
    
    # Return the formatted output, limiting the concentration to 4 decimal places
    return output

    # st.write(df3)
    # no2_output=pd.DataFrame(scaler.inverse_transform(lst_output),columns=['NO2 Concentration ðŸ­'])
    # st.write(no2_output)
    # output= (no2_output.at[predict_days-1,'NO2 Concentration ðŸ­'])
    # return output
    


def main():
    st.markdown("<h1 style='color:green; text-align:center; font-family:times new roman; font-weight:bold; font-size:20pt;'>NO2 Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-weight:bold;color:black;background-color:white;font-size:11pt;'> Choose any Location on the Map📌</h1>",unsafe_allow_html=True)
    st.markdown(
        """
    <style>
        iframe {
            height: 400px !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Define the bounds fot the map
    southwest = [-9.015302333420586, 422.05078125000006]
    northeast = [46.37725420510028, 485.77148437500006]
    center_lat = (southwest[0] + northeast[0]) / 2
    center_lng = (southwest[1] + northeast[1]) / 2
    
    # Create a Folium map centered on the calculated coordinates with the specified zoom level
    m = folium.Map(location=[center_lat, center_lng], zoom_start=4)
    folium.Rectangle(bounds=[southwest, northeast], color='blue', fill=True, fill_opacity=0.1).add_to(m)
    m.add_child(folium.LatLngPopup())
    
    # Call to render Folium map in Streamlit
    map = st_folium(m, height=400, width=700)
    
    try:
        latitude_input = float(map['last_clicked']['lat'])
        longitude_input = float(map['last_clicked']['lng'])
        st.write("Selected Latitude:", latitude_input)
        st.write("Selected Longitude:", longitude_input)
    except:
        st.warning("No location selected.")
    
    date = st.date_input('Date', value=pd.to_datetime('2024-09-06'), min_value=pd.to_datetime('2024-09-06'), max_value=pd.to_datetime('2025-09-06'))
    
    if st.button("Predict"):
        latitude_input = float(latitude_input)
        longitude_input = float(longitude_input)
        date = str(date)
        st.success(f"Predicting NO2 level for the location (Lat: {latitude_input}, Lon: {longitude_input}) on {date}")
        output = predict(latitude_input, longitude_input, date)
        st.info(f"NO2 level predicted: {output} mol/m²")
        
        # NO₂ concentration data
        no2_genereal_data = {
            "NO₂ Concentrations (mol/m²) / ppb": [
                "0 - 0.02 mol/m² (0 - 40 ppb)", 
                "0.02 - 0.04 mol/m² (41 - 80 ppb)", 
                "0.04 - 0.09 mol/m² (81 - 180 ppb)", 
                "0.09 mol/m² and above (181 ppb)"
            ],
            "Health Impact": [
                "Good (Safe) Levels", 
                "Moderate Levels", 
                "Unhealthy Levels", 
                "Hazardous Levels"
            ]
        }
        
        # Create DataFrame
        df_no2 = pd.DataFrame(no2_genereal_data)
        
        # Function to color rows based on levels
        def color_rows(row):
            if "Good" in row["Health Impact"]:
                return ['background-color: lightgreen'] * len(row)
            elif "Moderate" in row["Health Impact"]:
                return ['background-color: yellow'] * len(row)
            elif "Unhealthy" in row["Health Impact"]:
                return ['background-color: orange'] * len(row)
            elif "Hazardous" in row["Health Impact"]:
                return ['background-color: red'] * len(row)
            return [''] * len(row)
        
        # Style the DataFrame with colors
        styled_df = df_no2.style.apply(color_rows, axis=1)
        
        # Streamlit app
        st.title("NO₂ Concentration Overview")
        
        # Display the styled table
        st.dataframe(styled_df)


if __name__ == '__main__':
    main()
