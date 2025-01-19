#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>
#include <random>

//+++++ Helper Functions +++++


//Using Xavier Initialization
std::vector<std::vector<double>> initWeights(int input_size,int output_size){
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<> dis(-1.0,1.0);
    std::vector<std::vector<double>> weights;
    std::vector<double> row;
    const double val = sqrt(6.0/(input_size+output_size));
    for(int i = 0; i< output_size;i++){
        row = {};
        for(int j = 0; j<input_size;j++){
            row.push_back(dis(gen));
        }
        weights.push_back(row);
    }
    return weights;
};

std::vector<double> sigmoid_vector(std::vector<double> input, bool derivative){
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

std::vector<std::vector<double>> sigmoid_matrix(std::vector<std::vector<double>> input, bool derivative){
    std::vector<std::vector<double>> output = {};
    for(int i =0; i<input.size();i++){
        output.push_back(sigmoid_vector(input[0],derivative));
    }
    return output;
};

std::vector<double> tanh_vector(std::vector<double> input, bool derivative){
    std::vector<double> output = input;
    if(derivative){
        for(int i = 0;i<input.size();i++){
            output[i] = 1-input[i]*input[i];
        }
        return output;
    }
    for(int i = 0;i<input.size();i++){
        output[i] = tanh(input[i]);
    }
    return output;
};

std::vector<std::vector<double>> tanh_matrix(std::vector<std::vector<double>> input, bool derivative){
    std::vector<std::vector<double>> output = {};
    for(int i =0; i<input.size();i++){
        output.push_back(tanh_vector(input[0],derivative));
    }
    return output;
}

std::vector<double> softmax_vector(std::vector<double> input){
    std::vector<double> output = input;
    double denominator = 0;
    for(int i = 0; i<input.size();i++){
        denominator = denominator + exp(output[i]);
        output[i] = exp(output[i]);
    }
    for(int i = 0; i<input.size();i++){
        output[i] = output[i]/denominator;
    }
    return output;
};

//+++++ LSTM Class +++++

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

std::vector<std::vector<double>> concatenateMatrix(const std::vector<std::vector<double>>& array1, const std::vector<std::vector<double>>& array2) {
    std::vector<std::vector<double>> result = array1;

    // Append rows of array2 to array1
    for (const auto& row : array2) {
        result.push_back(row);
    }

    return result;
}

std::vector<std::vector<double>> dotMatrix(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    // Check if multiplication is possible (colsA must equal rowsB)
    if (colsA != rowsB) {
        throw std::invalid_argument("Incompatible matrices for dot product");
    }

    // Initialize result matrix with appropriate dimensions (rowsA x colsB)
    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0.0));

    // Perform matrix multiplication (dot product)
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

class LSTM {
private:
    int input_size;
    int hidden_size;
    int output_size;
    int num_epochs;
    double learning_rate;

    // Weights and biases
    std::vector<std::vector<double>> wf, bf; // Forget gate
    std::vector<std::vector<double>> wi, bi; // Input gate
    std::vector<std::vector<double>> wc, bc; // Candidate gate
    std::vector<std::vector<double>> wo, bo; // Output gate
    std::vector<std::vector<double>> wy, by; // Final gate

    // State containers
    std::unordered_map<int, std::vector<double>> concat_inputs;
    std::unordered_map<int, std::vector<double>> hidden_states;
    std::unordered_map<int, std::vector<double>> cell_states;

    // Gate activations
    std::unordered_map<int, std::vector<double>> activation_outputs;
    std::unordered_map<int, std::vector<double>> candidate_gates;
    std::unordered_map<int, std::vector<double>> output_gates;
    std::unordered_map<int, std::vector<double>> forget_gates;
    std::unordered_map<int, std::vector<double>> input_gates;
    std::unordered_map<int, std::vector<double>> outputs;

public:
    // Constructor
    LSTM(int input_size, int hidden_size, int output_size, int num_epochs, double learning_rate)
        : input_size(input_size),
          hidden_size(hidden_size),
          output_size(output_size),
          num_epochs(num_epochs),
          learning_rate(learning_rate) {

        // Initialize weights and biases
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

        // Reset all states and gates
        reset();
    }

    // Reset network memory
    void reset() {
        concat_inputs.clear();

        // Initialize hidden and cell states with zeros
        hidden_states.clear();
        hidden_states[-1] = std::vector<double>(hidden_size, 0.0);

        cell_states.clear();
        cell_states[-1] = std::vector<double>(hidden_size, 0.0);

        // Clear gate activations
        activation_outputs.clear();
        candidate_gates.clear();
        output_gates.clear();
        forget_gates.clear();
        input_gates.clear();
        outputs.clear();
    }

    // Print the current states (for debugging)
    void printStates() const {
        std::cout << "Hidden State (t=-1): ";
        for (double val : hidden_states.at(-1)) {
            std::cout << val << " ";
        }
        std::cout << "\nCell State (t=-1): ";
        for (double val : cell_states.at(-1)) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
};

//+++++ Main +++++

int main(){
    //testing helper functions
    std::vector<std::vector<double>> weights = initWeights(4,6);
    for(int i;i<weights.size();i++){
        for(int j;j<weights[0].size();j++){
            std::cout << std::to_string(weights[i][j]) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}