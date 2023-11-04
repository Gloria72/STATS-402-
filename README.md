# STAT_402_Project

*This is a repository for STAT-402 course.Contributer: Peng Wang, Yantao Mei, Yu Cheng*

This project is aiming to use data of Bilibili videos to discuss the existing video recommendation systems and to produce our recommendation system baised on the purpose to break the echo chamber effect. The inspiration of this project comes from a project of DUKE's CS216 course project.

## Major parts of the project

- Data Scraping (API sources: [bilibili-API-collect](https://github.com/SocialSisterYi/bilibili-API-collect))
- Data Analysis (Baised on GNN)
- Application


## How to run the code

### Data scraping (in folder Data_collecting and Data_collecting_v_2):
*Before running the code, make sure you get the python envrioment basied on python 3.9 including libiaries of: os, time, requests, sys, pandas.*

1.  Data collect has two version, v_1 is based on our assumptions to filter the top videos with greater weight. The v_2 is simply get all the data without any preprocessing, which we believe is better than v_1.

2.  To use v_2, just run the Data_collecting_v_2/main_v_2.py, then you will get the csv files contain different layers of videos. Since git hub can only upload files smaller than 50M, we seperate our data into small files. To wrap up them, you can run the wrap_up.py in the layer_2 folder to get an whole csv file.

### Model & application
*This code requires the following libraries to be installed:
numpy
pandas
matplotlib
pyvis
gravis
scikit-learn
networkx
torch
torch_geometric*

- The code provides a recommender system for recommending videos based on tags and the relationships between videos in the graph G1. It utilizes Graph Neural Networks to learn embeddings for the videos and calculates distances between videos to make recommendations. The recommendations are made based on the similarity between the initial video and other videos with the target tag. The user can interact with the system by inputting the initial video and target tag and selecting the next video from the recommended options.

- To run this part, you should find Analysis folder and run recommendation system.py
