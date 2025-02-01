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

//Computes softmax using the entire array.
std::vector<std::vector<double>> softmax_matrix(std::vector<std::vector<double>> input, bool derivative = false){
    std::vector<std::vector<double>> output = input;
    double denominator = 0;
    for(int i =0; i<input.size();i++){
        for(int j=0;j<input[0].size();j++){
            denominator = denominator + exp(output[i][j]);
            output[i][j] = exp(output[i][j]);
        }
    }
    for(int i =0; i<input.size();i++){
        for(int j=0;j<input[0].size();j++){
            output[i][j] = output[i][j]/denominator;
        }
    }
    return output;
}

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

std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix) {
    std::vector<std::vector<double>> result(matrix[0].size(), std::vector<double>(matrix.size()));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

std::vector<std::vector<double>> outerProduct(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    std::vector<std::vector<double>> result(vec1.size(), std::vector<double>(vec2.size()));
    for (size_t i = 0; i < vec1.size(); ++i) {
        for (size_t j = 0; j < vec2.size(); ++j) {
            result[i][j] = vec1[i] * vec2[j];
        }
    }
    return result;
}

std::vector<std::vector<double>> scaleMatrix(const std::vector<std::vector<double>>& matrix, double scale) {
    std::vector<std::vector<double>> result = matrix;
    for (auto& row : result) {
        for (double& val : row) {
            val *= scale;
        }
    }
    return result;
}

std::vector<double> scaleVector(const std::vector<double>& vec, double scale) {
    std::vector<double> result = vec;
    for (double& val : result) {
        val *= scale;
    }
    return result;
}

void clip(std::vector<std::vector<double>>& matrix) {
    for (auto& row : matrix) {
        for (double& val : row) {
            val = std::max(-1.0, std::min(1.0, val));
        }
    }
}

void clip(std::vector<double>& vec) {
    for (double& val : vec) {
        val = std::max(-1.0, std::min(1.0, val));
    }
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

std::vector<std::vector<double>> addMatrixVector(const std::vector<std::vector<double>>& mat2D, const std::vector<double>& vec1D) {
    // Ensure the number of columns in mat2D matches the size of vec1D
    if (mat2D.empty() || mat2D[0].size() != vec1D.size()) {
        std::cerr << "Dimensions mismatch: Cannot add 1D vector to 2D matrix." << std::endl;
        return {};  // Return empty matrix in case of error
    }

    // Create a new 2D matrix to store the result
    std::vector<std::vector<double>> result = mat2D;

    // Iterate through each row in the 2D matrix
    for (size_t i = 0; i < result.size(); ++i) {
        // Iterate through each element in the row and add corresponding element from vec1D
        for (size_t j = 0; j < result[i].size(); ++j) {
            result[i][j] += vec1D[j];  // Broadcasting happens here
        }
    }

    return result;
}

std::vector<std::vector<double>> elementWiseAdd(const std::vector<std::vector<double>>& mat1, const std::vector<std::vector<double>>& mat2) {
    // Ensure that both matrices have the same dimensions
    if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size()) {
        std::cerr << "Error: Matrices must have the same dimensions for element-wise addition." << std::endl;
        return {};  // Return empty matrix in case of error
    }

    // Create a new matrix to store the result
    std::vector<std::vector<double>> result(mat1.size(), std::vector<double>(mat1[0].size()));

    // Perform element-wise addition
    for (size_t i = 0; i < mat1.size(); ++i) {
        for (size_t j = 0; j < mat1[i].size(); ++j) {
            result[i][j] = mat1[i][j] + mat2[i][j];
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

    std::vector<std::vector<double>> forward( std::vector<std::vector<double>>& inputs) {
        // Reset the hidden and cell states
        reset();
        
        std::vector<std::vector<double>> forward_output;
        
        for (size_t q = 0; q < inputs.size(); ++q) {
            concat_inputs[q] = hidden_states[q - 1];
            concat_inputs[q].insert(concat_inputs[q].end(), inputs[q].begin(), inputs[q].end());
            forget_gates[q] = sigmoid_vector(addVectors(dotMatrix(wf, concat_inputs[q]),  bf));
            input_gates[q] = sigmoid_vector(addVectors(dotMatrix(wi, concat_inputs[q]), bi));
            candidate_gates[q] = tanh_vector(addVectors(dotMatrix(wc, concat_inputs[q]), bc));
            output_gates[q] = sigmoid_vector(addVectors(dotMatrix(wo, concat_inputs[q]), bo));

            cell_states[q] = elementWiseAdd(elementWiseMultiply(forget_gates[q], cell_states[q - 1]), elementWiseMultiply(input_gates[q], candidate_gates[q])); // check

            hidden_states[q] = elementWiseMultiply(output_gates[q], tanh_vector(cell_states[q]));//check
            
            //Chagned as outputs += tmp is a concatinate operation with tmp being a 1 x N matrix
            forward_output.push_back(elementWiseAdd(dotMatrix(wy, hidden_states[q]), by));
            
        }
        return forward_output;
    }
    void backward(const std::vector<std::vector<double>>& errors, const std::vector<std::vector<double>>& inputs) {
        std::vector<std::vector<double>> d_wf = zeroMatrix(wf.size(), wf[0].size());
        std::vector<std::vector<double>> d_wi = zeroMatrix(wi.size(), wi[0].size());
        std::vector<std::vector<double>> d_wc = zeroMatrix(wc.size(), wc[0].size());
        std::vector<std::vector<double>> d_wo = zeroMatrix(wo.size(), wo[0].size());
        std::vector<std::vector<double>> d_wy = zeroMatrix(wy.size(), wy[0].size());

        std::vector<double> d_bf = zeroVector(bf.size());
        std::vector<double> d_bi = zeroVector(bi.size());
        std::vector<double> d_bc = zeroVector(bc.size());
        std::vector<double> d_bo = zeroVector(bo.size());
        std::vector<double> d_by = zeroVector(by.size());
        
        std::vector<double> dh_next = zeroVector(hidden_states[0].size());
        std::vector<double> dc_next = zeroVector(cell_states[0].size());
        
        for (int q = inputs.size() - 1; q >= 0; --q) {
            std::vector<double> error = errors[q];

            // Final Gate Weights and Biases Errors
            d_wy = elementWiseAdd(d_wy, outerProduct(error, hidden_states[q]));
            d_by = addVectors(d_by, error);

            // Hidden State Error
            std::vector<double> d_hs = addVectors(dotMatrix(transpose(wy), error), dh_next);


            // Output Gate Weights and Biases Errors
            std::vector<double> tanh_cs = tanh_vector(cell_states[q]);
            std::vector<double> d_o = elementWiseMultiply(elementWiseMultiply(tanh_cs, d_hs), sigmoid_vector(output_gates[q], true));
            d_wo = elementWiseAdd(d_wo, outerProduct(d_o, inputs[q]));
            d_bo = addVectors(d_bo, d_o);

            // Cell State Error
            std::vector<double> d_cs = addVectors(elementWiseMultiply(elementWiseMultiply(tanh_vector(tanh_vector(cell_states[q]), true),output_gates[q]),d_hs),dc_next);

            // Forget Gate Weights and Biases Errors
            std::vector<double> d_f = elementWiseMultiply(elementWiseMultiply(d_cs, cell_states[q - 1]),sigmoid_vector(forget_gates[q], true));
            d_wf = elementWiseAdd(d_wf, outerProduct(d_f, inputs[q]));
            d_bf = addVectors(d_bf, d_f);

            // Input Gate Weights and Biases Errors
            std::vector<double> d_i = elementWiseMultiply(elementWiseMultiply(d_cs, candidate_gates[q]),sigmoid_vector(input_gates[q], true));
            d_wi = elementWiseAdd(d_wi, outerProduct(d_i, inputs[q]));
            d_bi = addVectors(d_bi, d_i);
            
            // Candidate Gate Weights and Biases Errors
            std::vector<double> d_c = elementWiseMultiply(elementWiseMultiply(d_cs, input_gates[q]),tanh_vector(candidate_gates[q], true));
            d_wc = elementWiseAdd(d_wc, outerProduct(d_c, inputs[q]));
            d_bc = addVectors(d_bc, d_c);

            // Concatenated Input Error (Sum of Error at Each Gate!)
            std::vector<double> d_z = addVectors(addVectors(dotMatrix(transpose(wf), d_f),dotMatrix(transpose(wi), d_i)),addVectors(dotMatrix(transpose(wc), d_c),dotMatrix(transpose(wo), d_o)));

            // Error of Hidden State and Cell State at Next Time Step
            dh_next = std::vector<double>(d_z.begin(), d_z.begin() + hidden_size);
            dc_next = elementWiseMultiply(forget_gates[q], d_cs);
        }

        clip(d_wf); clip(d_wi); clip(d_wc); clip(d_wo); clip(d_wy);
        clip(d_bf); clip(d_bi); clip(d_bc); clip(d_bo); clip(d_by);
        
        wf = elementWiseAdd(wf, scaleMatrix(d_wf, learning_rate));
        wi = elementWiseAdd(wi, scaleMatrix(d_wi, learning_rate));
        wc = elementWiseAdd(wc, scaleMatrix(d_wc, learning_rate));
        wo = elementWiseAdd(wo, scaleMatrix(d_wo, learning_rate));
        wy = elementWiseAdd(wy, scaleMatrix(d_wy, learning_rate));

        bf = addVectors(bf, scaleVector(d_bf, learning_rate));
        bi = addVectors(bi, scaleVector(d_bi, learning_rate));
        bc = addVectors(bc, scaleVector(d_bc, learning_rate));
        bo = addVectors(bo, scaleVector(d_bo, learning_rate));
        by = addVectors(by, scaleVector(d_by, learning_rate));
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

    void train(std::vector<std::vector<double>> xtrain, std::vector<std::vector<double>> ytrain){
        int i,j;
        std::vector<std::vector<double>> preditions;
        std::vector<std::vector<double>> errors;
        std::vector<std::vector<double>> concat_inputs_martix;
        for(i=0;i<num_epochs;i++){
            preditions = forward(xtrain);
            errors.clear();
            for(j=0;j<preditions.size();j++){
                errors.push_back(softmax_vector(scaleVector(preditions[j],-1)));
                //errors[errors.size() -1][char_to_idx[ytrain[q]]]++; //Likely not needed as it involves hot encoding
            }
            //Convert from map to matrix
            concat_inputs_martix.clear();
            for(auto input : concat_inputs){
                concat_inputs_martix.push_back(input.second);
            }
            backward(errors,concat_inputs_martix);

        }
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
