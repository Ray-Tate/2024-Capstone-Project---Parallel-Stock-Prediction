#include <iostream>
#include <vector>
#include <fstream>
#include <regex>

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
    cout << rowSize;

}