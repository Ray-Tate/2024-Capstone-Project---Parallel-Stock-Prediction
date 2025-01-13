#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "tensorflow/c/c_api.h"

int main() {
  //printf("Hello from TensorFlow C library version %s\n", TF_Version());
  //printf("Hello from TensorFlow C library version.");
  std::cout << "HELLO" << std::endl;
  std::cout << "The tensorflow version is: " << TF_Version() << std::endl;
  std::cout << "HELLO2" << std::endl;
  return 0;
}