/*
    dense and dropout layers
*/
/*
    Data Matrix [x][y]
    x - feature sample
    y - feature
*/

#include "CustomVectorFunctions.h"
#include <random>

class Dropout{
private:
    float rate;
    double scale;
    

public:
    std::vector<std::vector<double>> ignore_mask; //contains the coordinates of each of the elements to ignore
    
    Dropout(float rate)
    :rate(rate){
        scale = 1/(1-rate);
    }
    
    
    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> inputs){
        int i,j;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dis(rate);
        std::vector<std::vector<double>> outputs = inputs;
        ignore_mask = zeroMatrix(inputs.size(),inputs[0].size());
        for(j=0;j<inputs[0].size();j++){
            dis.reset();
            for(i=0;i<inputs.size();i++){
                if(dis(gen)){
                    outputs[i][j] = 0;
                    ignore_mask[i][j] = 0;
                }else{
                    outputs[i][j] *= scale;
                    ignore_mask[i][j] = 1;
                }
                
            }
        }
        return outputs;
    }

};

class Dense{
    private:
        double bias;
        double learning_rate;
        std::vector<double> weights;
    public:
        Dense(double learning_rate,int size)
            :learning_rate(learning_rate){
            bias = initWeights(1,1)[0][0];
            weights = initWeights(size,1)[0];
            
        }

        std::vector<double> forward(std::vector<std::vector<double>> inputs){
            int i;
            std::vector<double> outputs;
            outputs = dotMatrix(inputs,weights);
            for(i=0;i<outputs.size();i++){
                outputs[i] += bias;
            }
            return outputs;
        }
        //Errors are (y - h(x))
        std::vector<std::vector<double>> backward(const std::vector<double>errors, const std::vector<std::vector<double>>& inputs){
            int i,j;
            std::vector<std::vector<double>> prev_layer_error = zeroMatrix(inputs.size(),inputs[0].size());
            double sum;
            for(i=0;i<inputs.size();i++){
                prev_layer_error[i] = scaleVector(weights,errors[i]);
            }
            for(j=0;j<=inputs[0].size();j++){
                sum =0;
                for(i=0;i<inputs.size();i++){
                    //std::cout << "testi" << i<<std::endl;
                    if(j==0){
                        sum += errors[i]; 
                    }else{
                        sum += errors[i]*inputs[i][j-1];
                    }
                    
                }
                if(j==0){
                    bias = bias - learning_rate*sum/inputs.size();
                }else{
                    weights[j-1] = weights[j-1] - learning_rate*sum/inputs.size();
                }
            }

            return prev_layer_error;
        }

        //With Ignore Mask
        std::vector<std::vector<double>> backward(const std::vector<double>errors, const std::vector<std::vector<double>>& inputs , std::vector<std::vector<double>>& ignore_mask){
            int i,j;
            std::vector<std::vector<double>> prev_layer_error = zeroMatrix(inputs.size(),inputs[0].size());
            double sum;
            
            for(i=0;i<inputs.size();i++){
                prev_layer_error[i] = scaleVector(weights,errors[i]);
            }

            for(j=0;j<=inputs[0].size();j++){
                sum =0;
                for(i=0;i<inputs.size();i++){
                    
                    if(j==0){
                        sum += errors[i]; 
                    }else{
                        if(ignore_mask[i][j-1] == 0){
                            continue;
                        }
                        sum += errors[i]*inputs[i][j-1];
                    }
                    
                }
                if(j==0){
                    bias = bias - learning_rate*sum/inputs.size();
                }else{
                    weights[j-1] = weights[j-1] - learning_rate*sum/inputs.size();
                }
            }

            

            return prev_layer_error;
        }

};