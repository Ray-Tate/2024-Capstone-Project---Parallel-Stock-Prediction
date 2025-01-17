#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>



//Using Xavier Initialization
std::vector<std::vector<double>> initWeights(int input_size,int output_size){
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<> dis(-1.0,1.0);
    std::vector<std::vector<double>> weights;
    const double val = sqrt(6.0/(input_size+output_size));
    for(int i = 0; i< output_size;i++){
        for(int j = 0; j<input_size;j++){
            
        }
    }

};

std::vector<double> sigmoid(std::vector<double> input, bool derivative){
    std::vector<double> output = input;
    if(derivative){
        for(int i = 0;i<input.size();i++){
            output[i] = input[i]*(1-input[i]);
        }
        return output;
    }
    for(int i = 0;i<input.size();i++){
        output[i] = 1/(1 + exp(-input[i]));
    }
    return output;
};

std::vector<double> vector_tanh(std::vector<double> input, bool derivative){
    std::vector<double> output = input;
    if(derivative){
        for(long i = 0;i<input.size();i++){
            output[i] = 1-input[i]*input[i];
        }
        return output;
    }
    for(long i = 0;i<input.size();i++){
        output[i] = tanh(input[i]);
    }
    return output;
};

std::vector<double> softmax(std::vector<double> input){
    std::vector<double> output = input;
    double denominator = 0;
    for(long i = 0; i<input.size();i++){
        denominator = denominator + exp(output[i]);
        output[i] = exp(output[i]);
    }
    for(long i = 0; i<input.size();i++){
        output[i] = output[i]/denominator;
    }
};


class LSTMCell {

};
