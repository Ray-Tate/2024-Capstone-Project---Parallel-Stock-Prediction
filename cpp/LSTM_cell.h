#include "CustomVectorFunctions.h"

/*
    Data Matrix [x][y]
    x - feature sample
    y - feature
*/
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

    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& inputs) {
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

    std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& errors, const std::vector<std::vector<double>>& inputs) {
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
        
        std::vector<std::vector<double>> prev_layer_error;

        for (int q = inputs.size() - 1; q >= 0; --q) {
            std::vector<double> error = errors[q];
            //Errors are y - h(x)
            
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
            prev_layer_error.insert(prev_layer_error.begin(),d_z);

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

        return prev_layer_error;
    }

    //Droput mask version
    std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& errors, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& ignore_mask) {
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
        
        std::vector<std::vector<double>> prev_layer_error;
        std::vector<std::vector<double>> masked_input = elementWiseMultiply(inputs, ignore_mask);

        for (int q = inputs.size() - 1; q >= 0; --q) {
            std::vector<double> error = errors[q];
            //Errors are y - h(x)
            
            // Final Gate Weights and Biases Errors
            d_wy = elementWiseAdd(d_wy, outerProduct(error, hidden_states[q]));
            d_by = addVectors(d_by, error);

            // Hidden State Error
            std::vector<double> d_hs = addVectors(dotMatrix(transpose(wy), error), dh_next);


            // Output Gate Weights and Biases Errors 
            std::vector<double> tanh_cs = tanh_vector(cell_states[q]);
            std::vector<double> d_o = elementWiseMultiply(elementWiseMultiply(tanh_cs, d_hs), sigmoid_vector(output_gates[q], true));
            d_wo = elementWiseAdd(d_wo, outerProduct(d_o, masked_input[q]));
            d_bo = addVectors(d_bo, d_o);

            // Cell State Error
            std::vector<double> d_cs = addVectors(elementWiseMultiply(elementWiseMultiply(tanh_vector(tanh_vector(cell_states[q]), true),output_gates[q]),d_hs),dc_next);

            // Forget Gate Weights and Biases Errors
            std::vector<double> d_f = elementWiseMultiply(elementWiseMultiply(d_cs, cell_states[q - 1]),sigmoid_vector(forget_gates[q], true));
            d_wf = elementWiseAdd(d_wf, outerProduct(d_f, masked_input[q]));
            d_bf = addVectors(d_bf, d_f);

            // Input Gate Weights and Biases Errors
            std::vector<double> d_i = elementWiseMultiply(elementWiseMultiply(d_cs, candidate_gates[q]),sigmoid_vector(input_gates[q], true));
            d_wi = elementWiseAdd(d_wi, outerProduct(d_i, masked_input[q]));
            d_bi = addVectors(d_bi, d_i);
            
            // Candidate Gate Weights and Biases Errors
            std::vector<double> d_c = elementWiseMultiply(elementWiseMultiply(d_cs, input_gates[q]),tanh_vector(candidate_gates[q], true));
            d_wc = elementWiseAdd(d_wc, outerProduct(d_c, masked_input[q]));
            d_bc = addVectors(d_bc, d_c);

            // Concatenated Input Error (Sum of Error at Each Gate!)
            std::vector<double> d_z = addVectors(addVectors(dotMatrix(transpose(wf), d_f),dotMatrix(transpose(wi), d_i)),addVectors(dotMatrix(transpose(wc), d_c),dotMatrix(transpose(wo), d_o)));
            prev_layer_error.insert(prev_layer_error.begin(),d_z);

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

        return prev_layer_error;
    }

    std::vector<std::vector<double>> getConcatInputs(){
        std::vector<std::vector<double>> concat_inputs_martix;
        for(auto input : concat_inputs){
            concat_inputs_martix.push_back(input.second);
        }
        return concat_inputs_martix;
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

    void printOrigins(int epoc) const {
        std::cout << epoc << ", "
                  << wf[0][0] << ", " << bf[0] << ", "
                  << wi[0][0] << ", " << bi[0] << ", "
                  << wc[0][0] << ", " << bc[0] << ", "
                  << wo[0][0] << ", " << bo[0] << ", "
                  << wy[0][0] << ", " << by[0] << std::endl;
    }


    void train(const std::vector<std::vector<double>> xtrain, const std::vector<std::vector<double>> ytrain){
        int i,j,k;
        std::vector<std::vector<double>> preditions;
        std::vector<std::vector<double>> errors;
        std::vector<double> error_row;
        for(i=0;i<num_epochs;i++){
            preditions = forward(xtrain);
            errors.clear();
            for(j=0;j<preditions.size();j++){
                //Likely not needed as it involves hot encoding
                //errors.push_back(softmax_vector(scaleVector(preditions[j],-1)));
                //errors[errors.size() -1][char_to_idx[ytrain[q]]]++; 
                error_row.clear();
                for(k=0;k<preditions[0].size();k++){
                    error_row.push_back(ytrain[j][k] - preditions[j][k]);
                }
                errors.push_back(error_row);
            }
            std::cout << "Epoc: " << i+1 << " Error: " << absSum2DVector(errors) << std::endl;
            //Convert from map to matrix
            
            backward(errors,getConcatInputs());
            
            printOrigins(i);
        }
    }


    void test(const std::vector<std::vector<double>>& xtrain, const std::vector<std::vector<double>>& ytrain) {
        std::vector<std::vector<double>> probabilities = forward(xtrain);

        double mse = 0.0, mae = 0.0;
        size_t n = ytrain.size();
        size_t m = ytrain[0].size();  
        
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
                double error = probabilities[i][j] - ytrain[i][j];
                mse += error * error;
                mae += std::abs(error);
            }
        }

        mse /= (n * m);
        mae /= (n * m);

        // Display results
        std::cout << "Ground Truth:\n";
        for (const auto& row : ytrain) {
            for (double label : row) std::cout << label << " ";
            std::cout << "\n";
        }

        std::cout << "\nPredictions:\n";
        for (const auto& row : probabilities) {
            for (double pred : row) std::cout << pred << " ";
            std::cout << "\n";
        }

        std::cout << "\nMean Squared Error: " << mse << "\nMean Absolute Error: " << mae << "\n";
    }
};
