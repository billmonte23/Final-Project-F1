#Formula 1 Data Cleaning & Feature Engineering
#This project focuses on data preprocessing and feature engineering using Formula 1 datasets.
#The goal is to merge multiple datasets, clean the data, and create useful features for analysis or machine learning.
#Datasets Used
#The following datasets are used in this project:
#drivers.csv
#constructors.csv
#races.csv
#results.csv
#qualifying.csv

#Import Libraries
import pandas as pd
#Load Datasets
drivers = pd.read_csv('/content/drivers.csv')
constructors = pd.read_csv('/content/constructors.csv')
races = pd.read_csv('/content/races.csv')
results = pd.read_csv('/content/results.csv')
qualifying = pd.read_csv('/content/qualifying.csv')
#Merge Datasets
#We combine datasets step-by-step using common keys.
df = results.merge(races[['raceId', 'year']], on='raceId', how='left')

df = df.merge(
    qualifying[['raceId', 'driverId', 'q1', 'q2', 'q3']],
    on=['raceId', 'driverId'],
    how='left'
)
#Select Relevant Features
df = df[['raceId', 'year', 'driverId', 'constructorId',
         'grid', 'positionOrder', 'q1', 'q2', 'q3']]

#Data Cleaning
#Remove missing or invalid values
#Handle \N values
#Drop null rows
df = df.replace('\\N', None)
df = df.dropna()

#Data Type Conversion
#Convert columns to numeric format for analysis.
df['final_position'] = pd.to_numeric(df['positionOrder'])
df['grid'] = pd.to_numeric(df['grid'])

#Feature Engineering
#Create a new feature:
top3 → 1 if driver finished in top 3, else 0
df['top3'] = df['final_position'].apply(lambda x: 1 if x <= 3 else 0)

#Drop Unnecessary Columns
df = df.drop(columns=['positionOrder'])
#Save Clean Dataset
df.to_csv("f1_clean_dataset_final.csv", index=False)
