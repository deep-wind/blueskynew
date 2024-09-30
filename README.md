> #### https://www.hackerearth.com/challenges/hackathon/ieee-machine-learning-hackathon/
### Project Outline: Solar Image-Based NO2 Prediction System

**Objective**: Develop an interactive system that predicts NO2 levels based on satellite imagery, time-series LSTM models, and user-selected geographic locations. The tool leverages hyperspectral satellite data and geospatial visualization techniques to display predictions of NO2 concentrations over specified regions and dates.

---

### Workflow Breakdown:

#### 1. **Data Acquisition**:
   - **Sources**: Satellite NO2 data is obtained using the Sentinel-5P satellite’s public dataset via the `sentinelsat` API. Data files are downloaded and preprocessed from Copernicus Open Access Hub (e.g., using NetCDF4 format).
   - **Boundaries**: Define geospatial boundaries for acquiring NO2 concentration levels using latitude and longitude points.
   
#### 2. **Preprocessing**:
   - **Input Data**: The NO2 concentration data (`L3_data`) is stored in a `.nc` file. This data includes:
     - Latitude, longitude
     - Time dimension for days
     - NO2 concentration levels
   - **Spatial & Temporal Alignment**: Input data is preprocessed using `xarray`, where the geospatial and temporal elements are aligned to allow modeling over selected geographic regions.
   - **Missing Values**: Handling of missing or noisy NO2 concentration values using `pandas`.

#### 3. **Modeling**:
   - **Model Type**: A Sequential LSTM neural network model is used for predicting future NO2 levels based on past data.
   - **Data Scaling**: The NO2 concentration data is normalized using MinMaxScaler before feeding it into the LSTM model.
   - **Training**: Historical data (e.g., last 5 days) is split into training and testing sets, and the LSTM is trained on this data to predict future concentrations.

#### 4. **User Interface (UI)**:
   - **Front-End**: Built using **Streamlit**, providing an interactive and easy-to-use web application interface.
   - **Map-Based Location Selection**: The `streamlit-folium` package is integrated to allow users to choose geographic locations directly on a map.
   - **Prediction Input**: Users provide:
     - A **location** via latitude/longitude using the map.
     - A **date** for the desired NO2 level prediction (future dates).
   - **Results Display**: NO2 concentration prediction is displayed along with map-based results, with the possibility to view the predicted concentrations over time and space.

#### 5. **Visualization**:
   - **Map**: The interactive map (via **Folium**) displays the user’s chosen location for prediction.
   - **Charts**: Plot results (NO2 concentrations over time) using **Matplotlib** to visually show past vs. predicted values.
   
---

### Diagram of Workflow:

1. **User Input**: Location + Date → Interactive Folium Map → Latitude/Longitude Coordinates Extracted
2. **Data Preprocessing**: Retrieve satellite data → Apply preprocessing and scaling → Prepare for model input.
3. **LSTM Model**: Feed past NO2 data → LSTM Model → Predict future NO2 values.
4. **Result Display**: Show results on the map → Display NO2 levels in a readable, user-friendly format.

Here is a high-level flow diagram to illustrate the workflow:

```
+---------------------+       +----------------------------+       +----------------------+
|                     |       |                            |       |                      |
|   User selects      |       |  Preprocess satellite data |       |   Train LSTM Model   |
|   location and date | ----> |  (Download, clean, scale)  | ----> |   Predict NO2 levels |
|                     |       |                            |       |                      |
+---------------------+       +----------------------------+       +----------------------+
        |                                                                 |
        |                                                                 |
        v                                                                 v
+-------------------+                                     +-----------------------+
|                   |                                     |                       |
|   Visualize on    |                                     |    Display predicted  |
|   Folium Map      | ----------------------------------> |    NO2 concentration  |
|                   |                                     |                       |
+-------------------+                                     +-----------------------+
```

---


## Hyperspectral Satelite Data downloaded from Copernicus:
> #### https://browser.dataspace.copernicus.eu/ 

- We download the hyperspectral sentinel 5P satellite data from the copernicus datahub.
- This may download upto 9.4 GB of netCDF data. More info navigate to this page - https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-5p.
- For reliable functioning of the NO2 estimation model, satellite data upto 5 days prior to the time of estimation needs to be available.
-  **Note:** When prompted to enter the 'start date', enter a date at least 3 days prior to the date when the first estimation is required.

https://github.com/user-attachments/assets/abeaa807-b535-47ae-82f8-4ff14bdca667


## Steps to run the project in Local:

1. Create a folder

![image](https://github.com/user-attachments/assets/a9821684-61c5-499e-85b3-ad2202289c45)
![image](https://github.com/user-attachments/assets/ff84ba92-93f3-45f9-a8e0-e6a21dd47c82)

2. download harp zip , extract and place inside the above folder - https://github.com/stcorp/harp/releases/tag/1.23
3. conda install -c conda-forge visan
4. conda install -c conda-forge dask
5. cd harp > conda activate visan
6. cd..
7. pip install requirements.txt
8. python newbluesky.py
9. streamlit run newbluesky.py

   drive - https://drive.google.com/drive/folders/1w9Pqe55qZDsKQeonXtv3qUZo-4xJ45Ei?usp=drive_link
