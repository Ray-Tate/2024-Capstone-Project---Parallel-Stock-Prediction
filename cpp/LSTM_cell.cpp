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


// Helper function to create a zero matrix
std::vector<std::vector<double>> zeroMatrix(int rows, int cols) {
    return std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0.0));
}

// Helper function to create a random matrix
std::vector<std::vector<double>> randomMatrix(int rows, int cols) {
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

class LSTM {
private:
    // Hyperparameters
    double learning_rate;
    int hidden_size;
    int num_epochs;

    // Weights and biases
    std::vector<std::vector<double>> wf, bf; // Forget gate
    std::vector<std::vector<double>> wi, bi; // Input gate
    std::vector<std::vector<double>> wc, bc; // Candidate gate
    std::vector<std::vector<double>> wo, bo; // Output gate
    std::vector<std::vector<double>> wy, by; // Final gate

public:
    // Constructor
    LSTM(int input_size, int hidden_size, int output_size, int num_epochs, double learning_rate) 
        : learning_rate(learning_rate), hidden_size(hidden_size), num_epochs(num_epochs) {
        
        // Initialize weights and biases for all gates
        wf = randomMatrix(hidden_size, input_size);
        bf = zeroMatrix(hidden_size, 1);

        wi = randomMatrix(hidden_size, input_size);
        bi = zeroMatrix(hidden_size, 1);

        wc = randomMatrix(hidden_size, input_size);
        bc = zeroMatrix(hidden_size, 1);

        wo = randomMatrix(hidden_size, input_size);
        bo = zeroMatrix(hidden_size, 1);

        wy = randomMatrix(output_size, hidden_size);
        by = zeroMatrix(output_size, 1);
    }

    // Debugging: print weight dimensions
    void printWeightInfo() {
        std::cout << "Weight Dimensions: " << std::endl;
        std::cout << "Forget Gate: wf (" << wf.size() << "x" << wf[0].size() << "), bf (" << bf.size() << "x" << bf[0].size() << ")" << std::endl;
        std::cout << "Input Gate: wi (" << wi.size() << "x" << wi[0].size() << "), bi (" << bi.size() << "x" << bi[0].size() << ")" << std::endl;
        std::cout << "Candidate Gate: wc (" << wc.size() << "x" << wc[0].size() << "), bc (" << bc.size() << "x" << bc[0].size() << ")" << std::endl;
        std::cout << "Output Gate: wo (" << wo.size() << "x" << wo[0].size() << "), bo (" << bo.size() << "x" << bo[0].size() << ")" << std::endl;
        std::cout << "Final Gate: wy (" << wy.size() << "x" << wy[0].size() << "), by (" << by.size() << "x" << by[0].size() << ")" << std::endl;
    }
};
