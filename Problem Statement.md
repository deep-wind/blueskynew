Sub-theme 1: Blue Sky Above
Goal: Pollution estimation with improved accuracy using a combination of hyper-spectral satellite imagery data and maps. 

Background:

Air pollution is one of the major public health concerns which leads to a  number of respiratory/cardiovascular diseases and an estimated 7 million mortalities  worldwide [WHO]. Tracking the quality of the air we breathe using a mobile device  will help us be more informed about the air quality condition around and to make an  informed decision. However, accessible data from local air quality measurement  ground stations are not available everywhere. There are some recent advancements in  terms of the use of map-based land-use regression (LUR) models (Steininger et al.).  However, such models may suffer from spatial and temporal inaccuracy due to  artefacts, non-man-made sources of pollution, wind speed, pressure, precipitation and  temperature. Satellite imagery data could give a better estimate of the pollutants in a  given area. The challenges with satellite images include availability of only temporal  snapshots of data instead of continuous data, huge data-size of the imagery data  makes them not suitable to be downloaded through mobile networks to consumer  devices, and processing of such data with limited computational capacity in the  mobile devices. This hackathon aims to find solutions to these challenges through a  hybrid approach.

Problem Statement: 

The schematic diagram below illustrates the challenge. A code template with APIs is  provided to assist you with the implementation. The solution requires two levels of  data or image processing. One involves processing of the hyper-spectral satellite  images/data to extract hyper-parameters or other suitable compressed  representation/feature set. This compressed information should be sent to the second  regression machine learning (ML) model as data input. The regression machine  learning model should process the data from the satellite imagery along with the map data from the open street map to find the regression estimate of pollution (Nitrogen Dioxide) in the region for specific time instances. The solution should be submitted as a python code (.py) or (.ipynb) file



Dataset: https://s3-ap-southeast-1.amazonaws.com/he-public-data/BlueSkyAbove0287eea4.ipynb

Click here to read more on the dataset - https://s3-ap-southeast-1.amazonaws.com/he-public-data/BlueSkyChallenge39c7d73.pdf

Judging/scoring Criteria: 

1) A dataset similar to the example dataset will be used for judging and testing the same trained network. The solution should be submitted as a python code/ipynb file.
 
2) Location (Eg. 51.5219,-0.1280) and the local time (Eg. 'April 4 2021 1:33PM') at which the NO2 concentration must be estimated will be provided as input. Your model can take up to one week of prior data and extract key information/data from this. The data extracted by your model from the satellite data should be stored as a file. This compressed data/features file size (os.path.getsize python function) from the satellite-imagery will be taken into consideration to calculate the score. There will be a penalty of -1 point for every kilobyte of the compressed features file extracted from the satellites-imagery data and fed to the regression model. The goal is to achieve maximum accuracy with minimum compressed data/features transferred from the satellite imagery to the regression model.
 
3) For a time-instance, 5 sample spatial locations will be chosen. The locations will be within a 50 km circumference from the center of London (51° 30' 35.5140'' N and 0° 7' 5.1312'' W). These locations will be at least 7 kms apart. 100 time-instances will be evaluated. Absolute percentage correctness (i.e., 100%-(percentage of error)) of the estimate from the ground truth (8 hour running average) for each of the time-space samples (5x100) will be summed to get the first score component.
 
4) The time of execution of the code will be scored as -1 point for every millisecond of execution (rounded off to nearest milliseconds from microseconds time of run estimation (time.time python function)). The execution will be benchmarked in the amazon elastic cloud compute unit as specified in the problem statement.
 
5) Proper comments in the code, explanation of the algorithm and presentation/report will involve additional scores up to 10000 points.
Example Scores: 

Team A: 

Score from regression: 43000 

Score from the hyper-parameter datasize (2048 kb): -2048 points Score from time of execution (2.242 seconds): -2242 points 

Comments/Presentation: 4200 points 

Total: 42910 

Team B: 

Score from regression: 46000 

Score from the hyper-parameter datasize (10084 kb): -10084 points Score from time of execution (4.648 seconds): -4648 points 

Comments/Presentation: 2200 points 

Total: 33468 

Teams Information: Please check the Rules section

Benchmarking compute cloud information: 

Amazon EC2 

t2.micro, 1 GiB of Memory, 1 vCPUs, EBS only, 64-bit platform 

Code template: 

https://github.com/williamnavaraj/BlueSkyChallenge.git

Potential FAQs: 

1) Should the code run on a mobile device? 

No. While the problem statement aims towards a mobile application/webapp, the current  algorithms/code development is aimed to be tested in a consumer grade computer and will  be benchmarked in an equivalent computing elastic cloud computing unit in the cloud.

2) What dataset will be used for testing? 

A similar dataset as the dataset provided in the challenge (From same region and similar  satellite data will be used for testing). 

3) Will the ML training time/resources be taken into consideration for the grading? 

No. Training can be carried out in any computing system. However, the resulting trained  models should not exceed more than 4 GB for the compression/hyperparameter extraction  and should not be more than 1 GB for the regression model. 

4) What if we get negative total scores? 

Given the penalty points, total negative scores are possible and are acceptable. The overall  goal is to maximize the accuracy within the limited time and computing resources. The  solutions will be ranked based on whoever gets the maximum in the positive direction. 

5) Can we publish this work? 

Yes. You are free to publish the work. 

6) Who owns the IP? 

You own the IP. On submission of the solution, you are contributing to IEEE a license to  use your algorithm for potential app development to tackle and create awareness about air  pollution. 

References: 

https://www.who.int/health-topics/air-pollution#tab=tab_1

Schmitz, O., Beelen, R., Strak, M. et al. High resolution annual average air pollution  concentration maps for the Netherlands. Sci Data 6, 190035 (2019).  https://doi.org/10.1038/sdata.2019.35 

Michael Steininger, Konstantin Kobs, Albin Zehe, Florian Lautenschlager, Martin  Becker, and Andreas Hotho. 2020. MapLUR: Exploring a New Paradigm for  Estimating Air Pollution Using Deep Learning on Map Images. ACM Trans. Spatial  Algorithms Syst. 6, 3, Article 19 (May 2020), 24 pages.  DOI: https://doi.org/10.1145/3380973
