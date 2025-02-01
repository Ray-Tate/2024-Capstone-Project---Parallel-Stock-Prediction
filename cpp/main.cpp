#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include <algorithm>
#include "json.hpp"

nlohmann::json getConfig() {
    const std::string configPath = "config.json";
    std::ifstream configFile(configPath);
    if (!configFile.is_open()) { // If error opening the file
        std::cerr << "Unable to open file: " + configPath << std::endl;
        exit(1);
    }
    nlohmann::json jsonConfig; // Declare a nlohmann::json object
    configFile >> jsonConfig; // Parse JSON data from the file
    configFile.close();
    std::cout << "Data read from config file '" << configPath << ":'" << jsonConfig.dump(4) << std::endl;
    return jsonConfig;
}

std::vector<double> file2arr(const std::string& filePath) {
    std::string readLine;
    std::ifstream dataFile(filePath);
    if (!dataFile.is_open()) {
        std::cerr << "Unable to open file: " << filePath << std::endl;
        exit(1);
    }

    // Create vector to hold the data
    std::vector<double> array;

    while (std::getline(dataFile, readLine)) {
        array.push_back(std::stod(readLine));
    }

    return array;
}

std::tuple<std::vector<double>, double, double> normalize_data(const std::vector<double>& doubleArray) {
    std::vector<double> doubleArrayNormalized;
    double max = *std::max_element(doubleArray.begin(), doubleArray.end());
    double min = *std::min_element(doubleArray.begin(), doubleArray.end());
    for (double value : doubleArray) {
        doubleArrayNormalized.push_back((value - min) / (max - min));
    }
    return {doubleArrayNormalized, min, max};
}

std::vector<double> denormalize_data(const std::vector<double>& doubleArrayNormalized, double min, double max) {
    std::vector<double> doubleArrayDenormalized;
    for (double value : doubleArrayNormalized) {
        doubleArrayDenormalized.push_back(value * (max - min) + min);
    }
    return doubleArrayDenormalized;
}

std::vector<double> moving_average(const std::vector<double>& doubleArray, int windowSize) {
    std::vector<double> doubleArrayMA;
    double moving_total = 0;
    for (size_t i = 0; i < doubleArray.size(); ++i) {
        moving_total += doubleArray[i];
        if (i >= windowSize) {
            moving_total -= doubleArray[i - windowSize];
            doubleArrayMA.push_back(moving_total / windowSize);
        } else {
            doubleArrayMA.push_back(moving_total / (i + 1));
        }
    }
    return doubleArrayMA;
}

// Function to preprocess data with moving average and variable prediction days
std::tuple<std::vector<std::vector<double>>, std::vector<double>, double, double>
preprocess_data(const std::vector<double>& stock_data, int PREDICTION_DAYS, int PREDICTION_DAYS_AHEAD) {
    if (PREDICTION_DAYS > stock_data.size()) {
        std::cerr << "Error: PREDICTION_DAYS set to " << PREDICTION_DAYS
                  << " but stock_data is only " << stock_data.size() << " elements long" << std::endl;
        throw std::invalid_argument("PREDICTION_DAYS exceeds stock_data size");
    }

    auto [normalized_data, min_val, max_val] = normalize_data(stock_data);

    std::vector<std::vector<double>> x_train;
    std::vector<double> y_train;

    // Removes first portion of data. Cannot predict when we don't have at least PREDICTION_DAYS + PREDICTION_DAYS_AHEAD of history.
    for (size_t i = PREDICTION_DAYS + PREDICTION_DAYS_AHEAD - 1; i < normalized_data.size(); ++i) {
        std::vector<double> x_row;
        for (int j = i + 1 - (PREDICTION_DAYS + PREDICTION_DAYS_AHEAD); j <= i - PREDICTION_DAYS_AHEAD; ++j) {
            x_row.push_back(normalized_data[j]);
        }
        x_train.push_back(x_row);
        y_train.push_back(normalized_data[i]);
    }

    return {x_train, y_train, min_val, max_val};
}

class StockData {
private:
    std::string name;                // Name of the data holder
    std::vector<double> doubleArray; // Array of doubles
    static int arrayLength;
    std::vector<double> doubleArrayNormalized; // Array of normalized doubles
    double min_val, max_val; // Min and max values for normalization

public:
    // Constructor
    StockData(const std::string& name, const std::vector<double>& doubleArray) {
        StockData::name = name;
        StockData::doubleArray = doubleArray;
        auto [normalized_data, min, max] = normalize_data(doubleArray);
        StockData::doubleArrayNormalized = normalized_data;
        StockData::min_val = min;
        StockData::max_val = max;

        if (arrayLength == 0) {
            arrayLength = doubleArray.size();
        }
        if (doubleArray.size() != arrayLength) {
            std::cerr << "Invalid input data size! Stock '" << name << "' has '" << doubleArray.size() << "' elements, expected '" << arrayLength << "' because of previous input data size." << std::endl;
            exit(1);
        }
    }

    // Getters
    std::string getName() const {
        return name;
    }

    std::vector<double> getDoubleArray() const {
        return doubleArray;
    }

    std::vector<double> getNormalizedArray() const {
        return doubleArrayNormalized;
    }

    double getMinVal() const {
        return min_val;
    }

    double getMaxVal() const {
        return max_val;
    }

    // Method to print data
    void printStockData() const {
        std::cout << "Name: " << name << "\nArray (Size = " << doubleArray.size() << "): ";
        for (double value : doubleArray) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
        std::cout << "Name: " << name << "\nNormalized Array (Size = " << doubleArrayNormalized.size() << "): ";
        for (double value : doubleArrayNormalized) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
};

int StockData::arrayLength = 0;

int main() {
    nlohmann::json jsonConfig = getConfig();
    int PREDICTION_DAYS = jsonConfig["PREDICTION_DAYS"];
    int PREDICTION_DAYS_AHEAD = jsonConfig["PREDICTION_DAYS_AHEAD"];

    // Read the config for which stocks to use and read in the respective data text files
    std::vector<StockData> allStockData;
    for (const std::string& stock : jsonConfig["STOCKS"]) {
        StockData tmp(stock, file2arr("InputData/" + stock + ".txt"));
        allStockData.push_back(tmp);
    }

    // Print all the read-in stock data
    for (const StockData& stock : allStockData) {
        stock.printStockData();
    }

    // Preprocess data for each stock
    for (const StockData& stock : allStockData) {
        auto [x_train, y_train, min_val, max_val] = preprocess_data(stock.getDoubleArray(), PREDICTION_DAYS, PREDICTION_DAYS_AHEAD);

        // Print preprocessed data
        std::cout << "\nPreprocessed data for stock: " << stock.getName() << std::endl;
        std::cout << "x_train:" << std::endl;
        for (const auto& row : x_train) {
            for (double val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "y_train:" << std::endl;
        for (double val : y_train) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "min_val: " << min_val << ", max_val: " << max_val << std::endl;
    }

    std::cout << "DONE!!!!!!!!\n" << std::endl;

    return 0;
}