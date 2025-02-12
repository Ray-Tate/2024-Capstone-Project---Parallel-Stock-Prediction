#ifndef CUSTOMVECTORFUNCTIONS_H
#define CUSTOMVECTORFUNCTIONS_H

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
            output[i
] = input[i]*(1-input[i]);
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
    std::vector<double> result = zeroVector(rowsA);

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

double sum2DVector(const std::vector<std::vector<double>>& vec) {
    double sum = 0.0;
    
    for (const auto& row : vec) {
        for (double val : row) {
            sum += val;
        }
    }

    return sum;
}

#endif