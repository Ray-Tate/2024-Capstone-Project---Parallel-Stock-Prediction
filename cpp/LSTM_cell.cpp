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

std::vector<double> sigmoid_vector(std::vector<double> input, bool derivative = false){
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

std::vector<std::vector<double>> sigmoid_matrix(std::vector<std::vector<double>> input, bool derivative = false){
    std::vector<std::vector<double>> output = {};
    for(int i =0; i<input.size();i++){
        output.push_back(sigmoid_vector(input[0],derivative));
    }
    return output;
};

std::vector<double> tanh_vector(std::vector<double> input, bool derivative = false){
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

std::vector<std::vector<double>> tanh_matrix(std::vector<std::vector<double>> input, bool derivative = false){
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

// Helper function to create a zero vector
std::vector<double> zeroVector(int rows) {
    return std::vector<double>(rows, 0.0);
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

std::vector<double> addVectors(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }

    std::vector<double> result(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] + vec2[i];
    }
    return result;
}

std::vector<std::vector<double>> concatenateMatrix(const std::vector<std::vector<double>>& array1, const std::vector<std::vector<double>>& array2) {
    std::vector<std::vector<double>> result = array1;

    // Append rows of array2 to array1
    for (const auto& row : array2) {
        result.push_back(row);
    }

    return result;
}

std::vector<double> dotMatrix(const std::vector<std::vector<double>>& A, const std::vector<double>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = 1;

    // Check if multiplication is possible (colsA must equal rowsB)
    if (colsA != rowsB) {
        throw std::invalid_argument("Incompatible matrices for dot product");
    }

    // Initialize result matrix with appropriate dimensions (rowsA x colsB)
    std::vector<double> result = zeroVector(colsA);

    // Perform matrix multiplication (dot product)
    for (int i = 0; i < rowsA; ++i) {
        for (int k = 0; k < colsA; ++k) {
            result[i] += A[i][k] * B[k];
        }
    }

    return result;
}

std::vector<double> elementWiseMultiply(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    // Check if both vectors have the same size
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must be of the same size for element-wise multiplication.");
    }

    std::vector<double> result(vec1.size());

    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] * vec2[i];
    }

    return result;
}

std::vector<double> elementWiseAdd(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    // Check if both vectors have the same size
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must be of the same size for element-wise addition.");
    }

    std::vector<double> result(vec1.size());

    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] + vec2[i];
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

    // Weights
    std::vector<std::vector<double>> wf; // Forget gate
    std::vector<std::vector<double>> wi; // Input gate
    std::vector<std::vector<double>> wc; // Candidate gate
    std::vector<std::vector<double>> wo; // Output gate
    std::vector<std::vector<double>> wy; // Final gate

    // Biases
    std::vector<double> bf; // Forget gate
    std::vector<double> bi; // Input gate
    std::vector<double> bc; // Candidate gate
    std::vector<double> bo; // Output gate
    std::vector<double> by; // Final gate


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
        bf = zeroVector(hidden_size);

        wi = randomMatrix(hidden_size, input_size);
        bi = zeroVector(hidden_size);

        wc = randomMatrix(hidden_size, input_size);
        bc = zeroVector(hidden_size);

        wo = randomMatrix(hidden_size, input_size);
        bo = zeroVector(hidden_size);

        wy = randomMatrix(output_size, hidden_size);
        by = zeroVector(output_size);

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

    std::vector<double> forward( std::vector<std::vector<double>>& inputs) {
        // Reset the hidden and cell states
        reset();

        std::vector<double> forward_output;
        
        for (size_t q = 0; q < inputs.size(); ++q) {
            concat_inputs[q] = hidden_states[q - 1];
            concat_inputs[q].insert(concat_inputs[q].end(), inputs[q].begin(), inputs[q].end());
            forget_gates[q] = sigmoid_vector(addVectors(dotMatrix(wf, concat_inputs[q]),  bf));
            input_gates[q] = sigmoid_vector(addVectors(dotMatrix(wi, concat_inputs[q]), bi));
            candidate_gates[q] = tanh_vector(addVectors(dotMatrix(wc, concat_inputs[q]), bc));
            output_gates[q] = sigmoid_vector(addVectors(dotMatrix(wo, concat_inputs[q]), bo));

            cell_states[q] = elementWiseAdd(elementWiseMultiply(forget_gates[q], cell_states[q - 1]), elementWiseMultiply(input_gates[q], candidate_gates[q])); // check

            hidden_states[q] = elementWiseMultiply(output_gates[q], tanh_vector(cell_states[q]));//check

            forward_output = elementWiseAdd(elementWiseAdd(dotMatrix(wy, hidden_states[q]), by), forward_output);
        }
        return forward_output;
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
