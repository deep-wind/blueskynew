### NO2 Prediction System using a combination of hyper-spectral Satellite imagery data and maps

> #### The Bluesky above challenge - https://www.hackerearth.com/challenges/hackathon/ieee-machine-learning-hackathon/

> #### Video Reference - https://vimeo.com/672448685
---
 

**Objective**: To develop an interactive system that predicts NO2 levels based on satellite imagery, time-series LSTM (Long Short-Term Memory) network models, and user-selected geographic locations. The tool leverages hyperspectral satellite data and geospatial visualization techniques to display predictions of NO2 concentrations over specified regions and dates.


### Diagram of Workflow:

1. **User Input**: Location + Date → Interactive Folium Map → Latitude/Longitude Coordinates Extracted
2. **Data Preprocessing**: Retrieve satellite data → Apply preprocessing and scaling → Prepare for model input.
3. **LSTM Model**: Feed past NO2 data → LSTM Model → Predict future NO2 values.
4. **Result Display**: Show results on the map → Display NO2 levels.

Flow diagram to illustrate the workflow:

```
+---------------------+       +----------------------------+       +----------------------+
|                     |       |                            |       |                      |
|   User selects      |       |  Preprocess satellite data |       |   Train LSTM Model   |
|   location and date | ----> |  (Download, clean, scale)  | ----> |   Predict NO2 levels |
|                     |       |                            |       |                      |
+---------------------+       +----------------------------+       +----------------------+
       👆🏻
        |                                                                 |
        |                                                                 |
                                                                          v
+-------------------+                                     +-----------------------+
|                   |                                     |                       |
|   Streamlit       |                                     |    Display predicted  |
|   Folium Map      |                                     |    NO2 concentration  |
|                   |                                     |                       |
+-------------------+                                     +-----------------------+
```

---

### OverAll Flow:

      1. Start
      
      2. Download Satellite Data
         └─> Check if file exists
             ├─> Yes: Load dataset
             └─> No: Download from Google Drive
             
      3. Input Latitude and Longitude
         ├─> Example: Latitude = 51.5, Longitude = -0.1
         
      4. Input Date Range
         ├─> Example: Date = '2024-09-06'
         
      5. Predict NO2 Concentration
         ├─> Calculate Distance to Nearest Coordinates
         ├─> Extract NO2 Data from Dataset
         ├─> Preprocess Data
         ├─> Scale Data using MinMaxScaler
         ├─> Split Data into Training and Testing Sets
         ├─> Create LSTM Model
         ├─> Train LSTM Model
         ├─> Make Predictions
         └─> Transform Predictions Back to Original Scale
         
      6. Generate Prediction Output
         ├─> Create DataFrame with Dates and Predictions
         └─> Example Output:
         
             ┌─────────────┬───────────────────────────────┐
             │     Date    │ NO2 Concentration (mol/m²)    │
             ├─────────────┼───────────────────────────────┤
             │ 2024-09-07  │              0.0325           │
             │ 2024-09-08  │              0.0347           │
             │ 2024-09-09  │              0.0301           │
             │ 2024-09-10  │              0.0298           │
             └─────────────┴───────────────────────────────┘
      
      7. End


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

---



## Hyperspectral Satelite Data downloaded from Copernicus:
> #### https://browser.dataspace.copernicus.eu/ 

- We download the hyperspectral sentinel 5P satellite data from the copernicus datahub.
- This may download upto 9.4 GB of netCDF data. More info navigate to this page - https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-5p.
- For reliable functioning of the NO2 estimation model, satellite data upto 5 days prior to the time of estimation needs to be available.

https://github.com/user-attachments/assets/abeaa807-b535-47ae-82f8-4ff14bdca667


## Steps to run the project in Local:

### *Recommended to run this project in Local

1. Create a folder

      <img width="560" src="https://github.com/user-attachments/assets/a9821684-61c5-499e-85b3-ad2202289c45">
      <img width="560" src="https://github.com/user-attachments/assets/ff84ba92-93f3-45f9-a8e0-e6a21dd47c82">

2. download harp zip , extract and place inside the above folder - https://github.com/stcorp/harp/releases/tag/1.23

3. conda install -c conda-forge visan

4. conda install -c conda-forge dask

5. cd harp > conda activate visan

6. cd..

7. pip install requirements.txt

8. python newbluesky.py

9. streamlit run newbluesky.py

   drive - https://drive.google.com/drive/folders/1w9Pqe55qZDsKQeonXtv3qUZo-4xJ45Ei?usp=drive_link



---

## Basic Info

<img width="525" alt="{D2603589-764B-4442-9626-13E6BF1AF23E}" src="https://github.com/user-attachments/assets/b2d0d973-2b1d-4da0-aeb9-54babc899b39">

      +---------------------------------+
      |         Health Effects          |
      +---------------------------------+
      | Short-Term: Coughing, Wheezing  |
      | Long-Term: Respiratory issues   |
      |                                 |
      +---------------------------------+
      
      +---------------------------------+
      |         Sources of NO₂          |
      +---------------------------------+
      | - Motor Vehicles                |
      | - Power Plants                  |
      +---------------------------------+
      
      +---------------------------------+
      |       Control Measures          |
      +---------------------------------+
      | 1. Limit Vehicle Use            |
      | 2. Use Cleaner Energy Sources   |
      | 3. Improve Home Ventilation     |
      | 4. Stay Informed on Air Quality |
      +---------------------------------+
