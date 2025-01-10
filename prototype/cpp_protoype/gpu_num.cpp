#include <stdio.h>
#include <tensorflow/c/c_api.h>

int gpu_num() {
    // Create a TF status object
    TF_Status* status = TF_NewStatus();
    
    // Get the list of available devices
    TF_SessionOptions* options = TF_NewSessionOptions();
    const char* device_type = "GPU";
    TF_DeviceList* devices = TF_SessionListDevices(nullptr, status);
    
    if (TF_GetCode(status) != TF_OK) {
        printf("Error getting device list: %s\n", TF_Message(status));
        TF_DeleteStatus(status);
        TF_DeleteSessionOptions(options);
        return 1;
    }
    
    // Count GPU devices
    int num_gpus = 0;
    int num_devices = TF_DeviceListCount(devices);
    for (int i = 0; i < num_devices; ++i) {
        const char* device_type = TF_DeviceListType(devices, i, status);
        if (strcmp(device_type, "GPU") == 0) {
            num_gpus++;
        }
    }
    
    printf("Num GPUs Available: %d\n", num_gpus);
    
    // Clean up
    TF_DeleteDeviceList(devices);
    TF_DeleteSessionOptions(options);
    TF_DeleteStatus(status);
    
    return 0;
}