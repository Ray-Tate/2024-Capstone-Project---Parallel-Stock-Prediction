import yfinance as yf
import json
import os

# Change the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load the config file
with open("config.json", "r") as file:
    config = json.load(file)

START_DATE = config["START_DATE"]
END_DATE = config["END_DATE"]
STOCKS = config["STOCKS"]

# Function to load stock data from Yahoo Finance
def load_data(stock_symbol):
    data = yf.download(stock_symbol, start=START_DATE, end=END_DATE)
    #print(data)
    return data['Close'].values  # Use closing price

# Function to write an array to a file for cpp processing
def writeArr2File(data, filename):
    # Write multi-dimensional array to file
    with open("InputData/"+filename, "w") as file:
        for row in data:
            file.write(" ".join(map(str, row)) + "\n")  # Write each row as space-separated values

#MAIN
for stock in STOCKS:
    arr = load_data(stock)
    writeArr2File(arr, stock+".txt")