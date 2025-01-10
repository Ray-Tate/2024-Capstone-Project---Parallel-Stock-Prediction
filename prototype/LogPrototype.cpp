#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/platform/env.h>
#include <iostream>

using namespace tensorflow;
using namespace tensorflow::ops;

int main() {
    Scope root = Scope::NewRootScope();

    std::vector<tensorflow::DeviceAttributes> devices;
    Status status = tensorflow::DeviceFactory::ListAllPhysicalDevices(&devices);
    
    if (!status.ok()) {
        std::cout << "Error getting devices: " << status.ToString() << std::endl;
        return 1;
    }

    int gpu_count = 0;
    for (const auto& device : devices) {
        if (device.device_type() == "GPU") {
            gpu_count++;
        }
    }

    std::cout << "Number of GPUs Available: " << gpu_count << std::endl;

    return 0;
}