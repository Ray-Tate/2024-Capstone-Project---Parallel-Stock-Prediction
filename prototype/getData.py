#Python file for collecting stock data and exporting it to csv

import yfinance as yf
import csv
import warnings
warnings.filterwarnings('ignore')

from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

azn_df = yf.download("AZN.L", start="2014-07-10", end="2019-07-11")

# Create DataFrame for Adjusted Close price

azn_adj = azn_df[['Adj Close']]

# Convert DataFrame to numpy array
azn_adj_arr = azn_adj.values

# Find number of rows to train model on (80% of data set) 
training_data_len = int(0.8*len(azn_adj))

# Create train data set

train = azn_adj_arr[0:training_data_len, :]

# Normalise the data
scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train)


# Creating a data structure with 60 time-steps and 1 output

# Split data into X_train and y_train data sets
X_train = []
y_train = []

# Creating a data structure with 60 time-steps and 1 output
for i in range(60, len(train_scaled)):
    X_train.append(train_scaled[i-60:i, 0])
    y_train.append(train_scaled[i:i+1, 0])  
    if i <= 61:     # 60 days for first pass, 61 for second
      print(X_train)
      print(y_train)
      
#Create csv file of data
filename = "data.csv"
with open(filename, 'w',newline='') as csvfile:
   csvwriter = csv.writer(csvfile)
   fields = []
   for i in range(len(X_train[0])):
        fields.append("X"+ str(i))
   fields.append("Y")
   csvwriter.writerow(fields)
   for i in range(len(X_train)):
        row = []
        for j in range(len(X_train[0])) :
            row.append(X_train[i][j]) 
        row.append(y_train[i][0])
        csvwriter.writerow(row)
   
