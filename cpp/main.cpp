#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "json.hpp"
#include "LSTM_cell.h"

nlohmann::json getConfig(){
    const std::string configPath = "config.json";
    std::ifstream configFile(configPath);
    if (!configFile.is_open()) { //If error opening the file
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

    while (std::getline (dataFile, readLine)) {
        array.push_back(std::stod(readLine));
    }

    return array;
}

std::vector<double> normalize_data(std::vector<double> doubleArray) {
    std::vector<double> doubleArrayNormalized;
    double max = *std::max_element(doubleArray.begin(), doubleArray.end());
    double min = *std::min_element(doubleArray.begin(), doubleArray.end());
    for (int i = 0; i < int(doubleArray.size()); i++) {
        doubleArrayNormalized.push_back((doubleArray[i] - min) / (max - min));
    }
    return doubleArrayNormalized;
    }

std::vector<double> denormalize_data(std::vector<double> doubleArrayNormalized, std::vector<double> doubleArray) {
    std::vector<double> doubleArrayDenormalized;
    double max = *std::max_element(doubleArray.begin(), doubleArray.end());
    double min = *std::min_element(doubleArray.begin(), doubleArray.end());
    for (int i = 0; i < int(doubleArrayNormalized.size()); i++) {
        doubleArrayDenormalized.push_back(doubleArrayNormalized[i] * (max - min) + min);
    }
    return doubleArrayDenormalized;
}

std::vector<double> flatten_2d_vector(const std::vector<std::vector<double>> vec2d) {
    std::vector<double> flattened;
    
    for (const auto& row : vec2d) {
        if (row.size() != 1) {
            throw std::invalid_argument("Each inner vector must contain exactly one element.");
        }
        flattened.push_back(row[0]);
    }

    return flattened;
}

std::vector<double> moving_average(std::vector<double> doubleArray, int windowSize) {
    std::vector<double> doubleArrayMA;
    double moving_total = 0;
    double denominator = 0;
    for (int i = 0; i < int(doubleArray.size()); i++) {
        moving_total += doubleArrayMA[i];
        denominator += 1;
        if (i > windowSize) {
            moving_total -= doubleArrayMA[i - windowSize];
            denominator -= 1;
        }
        doubleArrayMA.push_back(moving_total / denominator);
}
    return doubleArrayMA;
}

std::vector<double> getFirst(const std::vector<double>& vec, double N) {
    int count = std::ceil(N * vec.size());  // Round up to ensure all values are included
    return std::vector<double>(vec.begin(), vec.begin() + std::min(count, (int)vec.size()));
}

std::vector<double> getLast(const std::vector<double>& vec, double N) {
    int count = std::ceil(N * vec.size());  // Round up to ensure all values are included
    return std::vector<double>(vec.end() - std::min(count, (int)vec.size()), vec.end());
}

void write_vector_to_file(const std::vector<double>& vec, const std::string& filename) {
    std::ofstream outFile(filename);
    
    if (!outFile) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    for (double val : vec) {
        outFile << val << "\n";  // Write each value on a new line
    }

    outFile.close();
}

class StockData {
private:
    std::string name;                // Name of the data holder
    std::vector<double> doubleArray;   // Array of doubles
    static int arrayLength;
    std::vector<double> doubleArrayNormalized;   // Array of doubles

public:
    // Constructor
    StockData(const std::string name, const std::vector<double>& doubleArray){
        StockData::name = name;
        StockData::doubleArray = doubleArray;
        StockData::doubleArrayNormalized = normalize_data(doubleArray);
        if (arrayLength == 0) {
            arrayLength = doubleArray.size();
        }
        if(doubleArray.size() != arrayLength){
            std::cerr << "Invalid input data size! Stock '" << name << "' has '" << doubleArray.size() <<"' elements, expected '" << arrayLength <<"' because of previous input data size." << std::endl;
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

    std::vector<double> getDoubleArrayNormalized() const {
        return doubleArrayNormalized;
    }

    void removeDataFromBeginnnig(int n){
        doubleArray.erase(doubleArray.begin(), doubleArray.begin() + n);
        doubleArrayNormalized = normalize_data(doubleArray);
    }

    void removeDataFromEnd(int n){
        doubleArray.resize(doubleArray.size() - n);
        doubleArrayNormalized = normalize_data(doubleArray);
    }

    // Setters # We should never need to use these, data should be final after constructor
    //void setName(const std::string& newName) {
    //    name = newName;
    //}

    //void setdoubleArray(const std::vector<double>& newArray) {
    //    doubleArray = newArray;
    //}

    // Method to print data
    void printStockData() const {
        std::cout << "Name: " << name << "\nArray (Size = " << doubleArray.size() << "): ";
        for (double value : doubleArray) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
        //std::cout << "Name: " << name << "\nNormalized Array (Size = " << doubleArrayNormalized.size() << "): ";
        //for (double value : doubleArrayNormalized) {
        //    std::cout << value << " ";
        // }
        //std::cout << std::endl;
    }
};

int StockData::arrayLength = 0;


int main() {
    nlohmann::json jsonConfig = getConfig();
    
    //Read the config for which stocks to use and read in the respective data text files
    std::vector<StockData> allStockData;
    for(std::string stock : jsonConfig["STOCKS"]){
        StockData tmp(stock, file2arr("InputData/"+stock+".txt"));
        allStockData.push_back(tmp);
    }

    //Get a pointer to the main stock
    StockData* mainStockPtr;
    for(StockData& stock : allStockData){
        if(stock.getName() == jsonConfig["STOCK_FOR_VALIDATION"]){
            mainStockPtr = &stock;
        }
    }
    StockData target_Y(mainStockPtr->getName(), mainStockPtr->getDoubleArray()); //Create a copy of the main stock for Y
    
    //Resize X and Y (remove beggining from Y cuz no history to predict. Remove end of X to match X/Y lengths)
    target_Y.removeDataFromBeginnnig(jsonConfig["PREDICT_DAYS_AHEAD"]);
    for(StockData& stock : allStockData){
        stock.removeDataFromEnd(jsonConfig["PREDICT_DAYS_AHEAD"]);
    }

    //Print all the read in stock data
    for(StockData& stock : allStockData){
        stock.printStockData();
    }
    target_Y.printStockData();


    //Get training portions of data.
    std::vector<std::vector<double>> xTrain;
    std::vector tmp = getFirst(mainStockPtr->getDoubleArrayNormalized(),jsonConfig["TRAIN_SPLIT"]);
    xTrain.resize(tmp.size());
    for(int i = 0; i<tmp.size() ; i++){
        xTrain[i].push_back(tmp[i]);
    }
    std::cout << xTrain.size() << " THATS HOW BIG Xtrain is" << std::endl;

    std::vector<std::vector<double>> yTrain;
    tmp = getFirst(target_Y.getDoubleArrayNormalized(),jsonConfig["TRAIN_SPLIT"]);
    yTrain.resize(tmp.size());
    for(int i = 0; i<tmp.size() ; i++){
        yTrain[i].push_back(tmp[i]);
    }

    std::cout << yTrain.size() << " THATS HOW BIG Ytrain is" << std::endl;

    std::vector<std::vector<double>> xVerify;
    tmp = getLast(mainStockPtr->getDoubleArrayNormalized(), 1.0 - (double)jsonConfig["TRAIN_SPLIT"]);
    xVerify.resize(tmp.size());
    for(int i = 0; i<tmp.size() ; i++){
        xVerify[i].push_back(tmp[i]);
    }
    std::cout << xVerify.size() << " THATS HOW BIG Xverify is" << std::endl;

    int hidden_size = jsonConfig["LSTM_UNITS"];
    LSTM lstmLayer(xTrain[0].size()+hidden_size, hidden_size,xTrain[0].size(),jsonConfig["EPOCHS"],jsonConfig["LEARNING_RATE"]);
    
    lstmLayer.train(xTrain, yTrain);
    std::vector<std::vector<double>> trainedPredictionsNorm = lstmLayer.forward(xTrain);
    std::vector<std::vector<double>> verifiyPredictionsNorm = lstmLayer.forward(xVerify);

    std::vector<double> trainedPredictions = denormalize_data(flatten_2d_vector(trainedPredictionsNorm), mainStockPtr->getDoubleArray());
    std::vector<double> verifiyPredictions = denormalize_data(flatten_2d_vector(verifiyPredictionsNorm), mainStockPtr->getDoubleArray());

    write_vector_to_file(trainedPredictions, "Trainedpredicitons.txt");
    write_vector_to_file(verifiyPredictions, "VerificationPredictions.txt");

    
    std::cout << "DONE!!!!!!!!\n" << std::endl;

    return 0;
}