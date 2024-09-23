
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
