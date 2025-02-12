#include <iostream>
#include <string>
#include <unordered_map>
#include <set>
#include <vector>
#include "LSTM_cell.h"

// Function to one-hot encode a string
std::vector<std::vector<double>> oneHotEncode(const std::string& text, const std::unordered_map<char, int>& char_to_idx, int char_size) {
    std::vector<std::vector<double>> output(text.size(), std::vector<double>(char_size, 0.0));

    for (size_t i = 0; i < text.size(); i++) {
        auto it = char_to_idx.find(text[i]);
        if (it != char_to_idx.end()) {
            output[i][it->second] = 1.0;
        }
    }
    
    return output;
}

int main() {
    std::string data = "To be, or not to be, that is the question: Whether "
                        "tis nobler in the mind to suffer The slings and arrows of ou"
                        "trageous fortune, Or to take arms against a sea of troubles A"
                        "nd by opposing end them. To die—to sleep, No more; and by a s"
                        "leep to say we end The heart-ache and the thousand natural sh"
                        "ocks That flesh is heir to: 'tis a consummation Devoutly to b"
                        "e wish'd. To die, to sleep; To sleep, perchance to dream—ay, "
                        "there's the rub: For in that sleep of death what dreams may c"
                        "ome, When we have shuffled off this mortal coil, Must give us"
                        " pause—there's the respect That makes calamity of so long lif"
                        "e. For who would bear the whips and scorns of time, Th'oppres"
                        "sor's wrong, the proud man's contumely, The pangs of dispriz'"
                        "d love, the law's delay, The insolence of office, and the spu"
                        "rns That patient merit of th'unworthy takes, When he himself "
                        "might his quietus make";
    
    // Convert to lowercase
    for (char &c : data) {
        c = std::tolower(c);
    }

    // Get unique characters
    std::set<char> chars(data.begin(), data.end());
    
    int data_size = data.size();
    int char_size = chars.size();
    
    std::cout << "Data size: " << data_size << ", Char size: " << char_size << std::endl;
    
    // Create character-to-index and index-to-character mappings
    std::unordered_map<char, int> char_to_idx;
    std::unordered_map<int, char> idx_to_char;
    int index = 0;
    for (char c : chars) {
        char_to_idx[c] = index;
        idx_to_char[index] = c;
        index++;
    }

    // Create training data
    std::string train_X_char = data.substr(0, data_size - 1);
    std::string train_y_char = data.substr(1, data_size - 1);

    // One-hot encode training data
    std::vector<std::vector<double>> train_X = oneHotEncode(train_X_char, char_to_idx, char_size);
    std::vector<std::vector<double>> train_Y = oneHotEncode(train_y_char, char_to_idx, char_size);

    std::cout << "Training data size: " << train_X.size() << ", " << train_Y.size() << std::endl;

    int hidden_size = 25;

    LSTM lstmLayer(char_size + hidden_size,hidden_size, char_size, 1000, 0.005);

    //##### Training #####
    lstmLayer.train(train_X, train_Y);



    return 0;
}
