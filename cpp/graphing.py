import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import datetime
import os
import sys

# Check if an argument is provided
if len(sys.argv) < 2:
    arg = 'config.json'
else:
    arg = sys.argv[1]

# Load the configuration from config.json
with open(arg, 'r') as f:
    config = json.load(f)

start_date = config['START_DATE']
end_date = config['END_DATE']
stocks = config['STOCKS']

# Ensure the output directory exists
output_folder = "Output_Graphs"
os.makedirs(output_folder, exist_ok=True)

# Load the stock data
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

# Remove first 10 elements from train_data and verif_data
removeFirstElements = 10
train_data = train_data[removeFirstElements:]
verif_data = verif_data[removeFirstElements:]

# Generate business day date range
date_range = pd.bdate_range(start=start_date, end=end_date)
if len(stock_data) != len(date_range):
    date_range = date_range[:len(stock_data)]

# Plot the stock data
plt.figure(figsize=(14, 8))
plt.plot(date_range, stock_data, label="Actual Stock Data")

# Align the training and verification data with the correct dates
train_dates = date_range[removeFirstElements: removeFirstElements + len(train_data)]
verify_dates = date_range[-len(verif_data):]

plt.plot(train_dates, train_data, label="Training Set")
plt.plot(verify_dates, verif_data, label="Validation Set")

# Add labels and legend
plt.title(f"{config['STOCK_FOR_VALIDATION']} Rolling Stock Prediction Over Time ({predictDaysAhead} days ahead)")
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Save the stock prediction graph
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"{output_folder}/{config['STOCK_FOR_VALIDATION']}_{config['EPOCHS']} epochs_{config['LEARNING_RATE']}_{config['LSTM_UNITS']}_lstmUNITS_{config['BATCH_SIZE']}_batchSize_Output_{timestamp}.png", dpi=300)

# Plot loss history
losstxt = "LossHistory.txt"
if os.path.exists(losstxt):
    with open(losstxt, 'r') as f:
        loss_data = [float(line.strip()) for line in f.readlines()]

    plt.figure(figsize=(14, 8))
    plt.plot(np.arange(1, len(loss_data) + 1), loss_data, label="Loss History")
    plt.title(f"{config['STOCK_FOR_VALIDATION']} Model Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_folder}/{config['STOCK_FOR_VALIDATION']}_{config['EPOCHS']} epochs_{config['LEARNING_RATE']}_{config['LSTM_UNITS']}_lstmUNITS_{config['BATCH_SIZE']}_batchSize_Loss_{timestamp}.png", dpi=300)

print("Graphs saved successfully.")
