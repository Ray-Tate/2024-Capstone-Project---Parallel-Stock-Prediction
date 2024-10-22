#include <iostream>
#include <vector>
#include <fstream>
#include <regex>
#include <sstream>

using namespace std;

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
    cout << "Row size: " +to_string(rowSize) << endl;

    //import data into vector
    vector<vector<double>> data = {}; // data[y][x]
    vector<double> row = {};
    string element;
    j=0;
    while(getline(fin,line)){
        stringstream ss(line);
        while(ss.good()){
            getline(ss,element,',');
            row.insert(row.end(),stod(element));
        }
        data.insert(data.end(),row);
    }
    cout << "data dimension: " + to_string(data.size()) + "," + to_string( data[0].size()) << endl;
}