#include <iostream>
#include <vector>
#include <fstream>
#include <regex>
#include <sstream>

using namespace std;

double linearFunction(vector<double> params, vector<double> X){
    int i;
    double Y;
    Y = params[0];
    cout << "size params "+ to_string(params.size()) << endl;
    //cout << "size x "+ to_string(X.size()) << endl;
    for(i =0; i< X.size();i++){
        cout << "i: "+ to_string(i) << endl;
        Y = Y + X[i]*params[i];
    }
    return Y;
}

vector <double> linearRegression(vector<vector<double>> data,float learnRate){
    int i,j;
    int numFeatures = data[0].size() -1;
    int numRows = data.size();
    double sum;
    vector <double> X;
    vector <double> params(numFeatures+1,1.0);
    for(i=0; i<params.size();i++){
        sum = 0;
        for(j=0;j<numRows;j++){
            cout << "size out1 x "+ to_string(X.size()) << endl;
            X.clear();
            X = data[j];
            cout << "size out2 x "+ to_string(X.size()) << endl;
            X.insert(X.begin(),1.0);
            X.pop_back();
            
            if(j == 0){
                sum = sum + (linearFunction(params, X)-data[j][numFeatures]);
            }else{
                sum = sum + (linearFunction(params, X)-data[j][numFeatures])*data[j][i];
            }
        }
        sum = sum*learnRate*(1.0/numRows);
        params[i] = params[i] - sum;
    }

    return params;
}



int main(){
    int i,j,k;
    fstream fin;
    fin.open("data.csv",ios::in);
    string line;
    getline(fin,line);
    
    //Find row size based on the number of commas
    int rowSize = 1;
    for(auto c : line){
        if(c == ','){
            rowSize++;
        }
    }
    cout << "Row size: " + to_string(rowSize) << endl;

    //import data into vector
    vector<vector<double>> data = {}; // data[y][x]
    vector<double> row = {};
    string element;
    while(getline(fin,line)){
        stringstream ss(line);
        while(ss.good()){
            getline(ss,element,',');
            row.insert(row.end(),stod(element));
        }
        data.insert(data.end(),row);
    }
    cout << "data dimension: " + to_string(data.size()) + "," + to_string( data[0].size()) << endl;

    vector <double> params;
    params = linearRegression(data,0.1);
    cout << "Parameters output:" << endl;
    for(auto i : params){
        cout << to_string(i) << endl;
    }

    
}