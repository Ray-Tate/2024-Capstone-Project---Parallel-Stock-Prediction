#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "json.hpp"

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

class StockData {
private:
    std::string name;                // Name of the data holder
    std::vector<double> doubleArray;   // Array of doubles

public:
    // Constructor
    StockData(const std::string name, const std::vector<double>& doubleArray)
        : name(name), doubleArray(doubleArray) {}

    // Default constructor
    StockData() : name("Unnamed"), doubleArray({}) {}

    // Getters
    std::string getName() const {
        return name;
    }

    std::vector<double> getDoubleArray() const {
        return doubleArray;
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
        std::cout << "Name: " << name << "\nArray: ";
        for (double value : doubleArray) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
};


int main() {
    nlohmann::json jsonConfig = getConfig();
    
    //Read the config for which stocks to use and read in the respective data text files
    std::vector<StockData> allStockData;
    for(std::string stock : jsonConfig["STOCKS"]){
        StockData tmp(stock, file2arr("InputData/"+stock+".txt"));
        allStockData.push_back(tmp);
    }

    //Print all the read in stock data
    for(StockData stock : allStockData){
        stock.printStockData();
    }

    std::cout << "DONE!!!!!!!!\n" << std::endl;

    return 0;
}