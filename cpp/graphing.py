import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

import os

import sys

# Check if an argument is provided
if len(sys.argv) < 2:
    arg = 'config.json'
else:
    # Access the first argument
    arg = sys.argv[1]
    print(f"Argument received: {arg}")

# Get the directory of the currently running script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's directory
os.chdir(script_dir)

# Load the configuration from config.json
with open(arg, 'r') as f:
    config = json.load(f)

start_date = config['START_DATE']
end_date = config['END_DATE']
stocks = config['STOCKS']

# Plot each stock's data
plt.figure(figsize=(14, 8))

predictDaysAhead = config['PREDICT_DAYS_AHEAD']
maintxt = f"InputData/{config['STOCK_FOR_VALIDATION']}.txt"
trainedtxt = "Trainedpredicitons.txt"
veriftxt = "VerificationPredictions.txt"

with open(maintxt, 'r') as f:
    stock_data = [float(line.strip()) for line in f.readlines()]

for i in range(predictDaysAhead):
    stock_data.pop()

with open(trainedtxt, 'r') as f:
    train_data = [float(line.strip()) for line in f.readlines()]

with open(veriftxt, 'r') as f:
    verif_data = [float(line.strip()) for line in f.readlines()]


# Plot the stock data
count_array = np.arange(1, len(stock_data) + 1)
plt.plot(count_array,stock_data, label=maintxt)
count_array = np.arange(1, len(train_data) + 1)
plt.plot(train_data,  label="Training Set")
count_array = np.arange(len(stock_data) - len(verif_data), len(stock_data))
plt.plot(count_array, verif_data, label="Validation Set")




# Add labels and legend
plt.title(f'Rolling Stock Prediction Over Time ({config['PREDICT_DAYS_AHEAD']} days ahead)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig(f"{config['STOCK_FOR_VALIDATION']}Output.png", dpi=300)
