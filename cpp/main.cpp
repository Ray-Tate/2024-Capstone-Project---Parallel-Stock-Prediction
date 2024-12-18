#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "json.hpp"


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


int main() {
    
    //Reading in config.json file
    const std::string configPath = "config.json";
    std::ifstream configFile(configPath);
    if (!configFile.is_open()) { //If error opening the file
        std::cerr << "Unable to open file: " + configPath << std::endl;
        exit(1);
    }
    nlohmann::json jsonConfig; // Declare a nlohmann::json object
    configFile >> jsonConfig; // Parse JSON data from the file
    configFile.close();
    std::cout << "Parsed JSON data:\n" << jsonConfig.dump(4) << "\n";

    for(std::string stock : jsonConfig["STOCKS"]){
        std::vector<double> arr = file2arr("InputData/"+stock+".txt");
        std::cout << stock << std::endl;
    }

    return 0;
}