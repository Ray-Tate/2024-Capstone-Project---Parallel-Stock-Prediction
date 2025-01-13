#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "tensorflow/c/c_api.h"

int main() {
  std::cout << "HELLO" << std::endl;
  std::cout << "The tensorflow version is: " << TF_Version() << std::endl;
  std::cout << "BYE" << std::endl;
  return 0;
}