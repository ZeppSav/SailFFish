//-----------------------------------------------------------------------------
//-------------------------DataType_VkFFT Functions-----------------------------
//-----------------------------------------------------------------------------

#include "DataType_VkFFT.h"
// #include "omp.h"

#ifdef VKFFT

namespace SailFFish
{

void DataType_VkFFT::Datatype_Setup()
{
    // Setup necessary handles, arrays, and check info

    // Generate VkFFT object
    VK = {};
    VkFFTResult res = OpenCLSetup(&VK);

    // Get device properties
    Print_Device_Info(VK.device);

}

VkFFTResult DataType_VkFFT::OpenCLSetup(VkGPU *vkGPU)
{
    // Carries out checks for devices etc.
    cl_int res = CL_SUCCESS;
    cl_uint numPlatforms;
    res = clGetPlatformIDs(0, 0, &numPlatforms);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
    if (!platforms) return VKFFT_ERROR_MALLOC_FAILED;
    res = clGetPlatformIDs(numPlatforms, platforms, 0);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
    uint64_t k = 0;
    for (uint64_t j = 0; j < numPlatforms; j++) {
        cl_uint numDevices;
        res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, 0, 0, &numDevices);
        std::cout << "N OpenCl capable devices = " << numDevices << std::endl;
        cl_device_id* deviceList = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
        if (!deviceList) return VKFFT_ERROR_MALLOC_FAILED;
        res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, numDevices, deviceList, 0);
        if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
        for (uint64_t i = 0; i < numDevices; i++) {
            if (k == vkGPU->device_id) {
                vkGPU->platform = platforms[j];
                vkGPU->device = deviceList[i];
                vkGPU->context = clCreateContext(NULL, 1, &vkGPU->device, NULL, NULL, &res);
                if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
                cl_command_queue commandQueue = clCreateCommandQueue(vkGPU->context, vkGPU->device, 0, &res);
                if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
                vkGPU->commandQueue = commandQueue;
                i=numDevices;
                j=numPlatforms;
            }
            else {
                k++;
            }
        }
        free(deviceList);
    }
    free(platforms);

    return VKFFT_SUCCESS;
}

void DataType_VkFFT::Print_Device_Info(cl_device_id device)
{
    // This calls a number of queries and prints out the OpenCL device information.
    cl_ulong ulongVal;
    size_t sizeVal;
    cl_uint uintVal;

    char deviceName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    std::cout << "Device Name: " << deviceName << std::endl;

    // Get device type
    cl_device_type deviceType;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr);
    std::cout << "Device Type: " << deviceType;
    std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;

    // Device memory (global)
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ulongVal), &ulongVal, nullptr);
    std::cout << "   Device memory: " << (ulongVal / (1024 * 1024)) << " MB" << std::endl;

    // Max work group size
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(sizeVal), &sizeVal, nullptr);
    std::cout << "   Maximum work group size: " << sizeVal << std::endl;

    // Local/shared memory per work group
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ulongVal), &ulongVal, nullptr);
    std::cout << "   Shared memory per work group: " << (ulongVal / 1024) << " KB" << std::endl;

    // Registers per work group/block: NOT EXPOSED in OpenCL (no direct equivalent)
    std::cout << "   Registers per block: N/A in OpenCL" << std::endl;

    // Max work item sizes (grid size)
    size_t workItemSizes[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workItemSizes), &workItemSizes, nullptr);
    std::cout << "   Max Grid sizes (work item sizes): "
              << workItemSizes[0] << " x "
              << workItemSizes[1] << " x "
              << workItemSizes[2] << std::endl;

    // Warp size: No warp concept in OpenCL; approximate via preferred vector width
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(uintVal), &uintVal, nullptr);
    std::cout << "   Approx. warp size (vector width for float): " << uintVal << std::endl;

    // Compute units (like multiprocessors)
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uintVal), &uintVal, nullptr);
    std::cout << "   Number of compute units: " << uintVal << std::endl;

    // Max threads per compute unit: Not directly exposed; can be approximated
    std::cout << "   Max threads per multiprocessor: N/A directly in OpenCL" << std::endl;

    // Memory clock rate and memory bus width: Vendor-specific, not available in standard OpenCL
    std::cout << "   Memory Clock Rate: N/A in OpenCL standard" << std::endl;
    std::cout << "   Memory Bus Width: N/A in OpenCL standard" << std::endl;

    // Shared memory per block (dynamic): Not distinguishable from static in OpenCL
    std::cout << "   Dynamic shared memory per block max: N/A in OpenCL" << std::endl;
}

//--- Array allocation


SFStatus DataType_VkFFT::Allocate_Arrays()
{
    // Depending on the type of solver and the chosen operator/outputs, the arrays which must be allocated vary.
    // This is simply controlled here by specifying the necessary flags during solver initialization

    // size_t free_mem = 0, total_mem = 0;
    // CUresult result = cuMemGetInfo(&free_mem, &total_mem);
    // std::cout << "Used GPU memory: 0 " << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB" << std::endl;

    // if (r_in1)      r_Input1 = (Real*)malloc(NT*sizeof(Real));
    // if (r_in2)      r_Input2 = (Real*)malloc(NT*sizeof(Real));
    // if (r_in3)      r_Input3 = (Real*)malloc(NT*sizeof(Real));

    // // // Allocate local arrays: (pinned)
    // // if (r_in1)      cudaMallocHost((void**)&cuda_r_Input1, sizeof(CUDAReal)*NT);
    // // if (r_in2)      cudaMallocHost((void**)&cuda_r_Input2, sizeof(CUDAReal)*NT);
    // // if (r_in3)      cudaMallocHost((void**)&cuda_r_Input3, sizeof(CUDAReal)*NT);

    // // Allocate local arrays: not pinned
    // if (r_in1)      cudaMalloc((void**)&cuda_r_Input1, sizeof(CUDAReal)*NT);
    // if (r_in2)      cudaMalloc((void**)&cuda_r_Input2, sizeof(CUDAReal)*NT);
    // if (r_in3)      cudaMalloc((void**)&cuda_r_Input3, sizeof(CUDAReal)*NT);

    // result = cuMemGetInfo(&free_mem, &total_mem);
    // std::cout << "Used GPU memory: cuda_r_Input1 " << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB" << std::endl;

    // if (InPlace){
    //     if (r_out_1)    cuda_r_Output1 = cuda_r_Input1;
    //     if (r_out_2)    cuda_r_Output2 = cuda_r_Input2;
    //     if (r_out_3)    cuda_r_Output3 = cuda_r_Input3;
    // }
    // else{
    //     if (r_out_1)    cudaMalloc((void**)&cuda_r_Output1, sizeof(CUDAReal)*NT);
    //     if (r_out_2)    cudaMalloc((void**)&cuda_r_Output2, sizeof(CUDAReal)*NT);
    //     if (r_out_3)    cudaMalloc((void**)&cuda_r_Output3, sizeof(CUDAReal)*NT);
    // }

    // if (r_out_1)      r_Output1 = (Real*)malloc(NT*sizeof(Real));
    // if (r_out_2)      r_Output2 = (Real*)malloc(NT*sizeof(Real));
    // if (r_out_3)      r_Output3 = (Real*)malloc(NT*sizeof(Real));
    // // if (r_out_1)      memset(r_Output1, 0, NT*sizeof(Real));
    // // if (r_out_2)      memset(r_Output2, 0, NT*sizeof(Real));
    // // if (r_out_3)      memset(r_Output3, 0, NT*sizeof(Real));

    // result = cuMemGetInfo(&free_mem, &total_mem);
    // std::cout << "Used GPU memory: cuda_r_Output1 " << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB" << std::endl;

    // // Arrays for real Green's function
    // // if (r_fg)       cudaMalloc((void**)&r_FG, sizeof(CUDAReal)*NT);
    // //    if (r_fg)       cudaMalloc((void**)&cu_r_FG, sizeof(CUDAReal)*NT);          // Not necessary!
    // if (r_fg)       r_FG = (Real*)malloc(NT*sizeof(Real));

    // // Complex-valued arrays
    // if (c_in1)      cudaMalloc((void**)&c_Input1, sizeof(CUDAComplex)*NT);
    // if (c_in2)      cudaMalloc((void**)&c_Input2, sizeof(CUDAComplex)*NT);
    // if (c_in3)      cudaMalloc((void**)&c_Input3, sizeof(CUDAComplex)*NT);

    // if (c_ft_in1)   cudaMalloc((void**)&c_FTInput1, sizeof(CUDAComplex)*NTM);
    // if (c_ft_in2)   cudaMalloc((void**)&c_FTInput2, sizeof(CUDAComplex)*NTM);
    // if (c_ft_in3)   cudaMalloc((void**)&c_FTInput3, sizeof(CUDAComplex)*NTM);

    // // Prepare transforms for case of either in-place or out-of-place operation
    // c_ft_out1 = c_ft_in1;
    // c_ft_out2 = c_ft_in2;
    // c_ft_out3 = c_ft_in3;

    // if (InPlace){
    //     if (c_ft_out1) c_FTOutput1 = c_FTInput1;
    //     if (c_ft_out2) c_FTOutput2 = c_FTInput2;
    //     if (c_ft_out3) c_FTOutput3 = c_FTInput3;
    // }
    // else{
    //     if (c_ft_out1) cudaMalloc((void**)&c_FTOutput1, sizeof(CUDAComplex)*NTM);
    //     if (c_ft_out2) cudaMalloc((void**)&c_FTOutput2, sizeof(CUDAComplex)*NTM);
    //     if (c_ft_out3) cudaMalloc((void**)&c_FTOutput3, sizeof(CUDAComplex)*NTM);
    // }

    // result = cuMemGetInfo(&free_mem, &total_mem);
    // std::cout << "Used GPU memory: c_FTOutput1 " << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB" << std::endl;

    // if (c_vel_1)    cudaMalloc((void**)&c_FTVel1, sizeof(CUDAComplex)*NTM);
    // if (c_vel_2)    cudaMalloc((void**)&c_FTVel2, sizeof(CUDAComplex)*NTM);
    // if (c_vel_3)    cudaMalloc((void**)&c_FTVel3, sizeof(CUDAComplex)*NTM);

    // if (c_out_1)    cudaMalloc((void**)&c_Output1, sizeof(CUDAComplex)*NT);
    // if (c_out_2)    cudaMalloc((void**)&c_Output2, sizeof(CUDAComplex)*NT);
    // if (c_out_3)    cudaMalloc((void**)&c_Output3, sizeof(CUDAComplex)*NT);

    // // Arrays for complex Green's function & spectral operators arrays
    // if (c_fg)       cudaMalloc((void**)&c_FG, sizeof(CUDAComplex)*NTM);
    // if (c_fg_i)     cudaMalloc((void**)&c_FGi, sizeof(CUDAComplex)*NTM);
    // if (c_fg_j)     cudaMalloc((void**)&c_FGj, sizeof(CUDAComplex)*NTM);
    // if (c_fg_k)     cudaMalloc((void**)&c_FGk, sizeof(CUDAComplex)*NTM);

    // result = cuMemGetInfo(&free_mem, &total_mem);
    // std::cout << "Used GPU memory: c_fg_x " << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB" << std::endl;

    // if (c_dbf_1)    cudaMalloc((void**)&c_DummyBuffer1, sizeof(CUDAComplex)*NTM);
    // if (c_dbf_2)    cudaMalloc((void**)&c_DummyBuffer2, sizeof(CUDAComplex)*NTM);
    // if (c_dbf_3)    cudaMalloc((void**)&c_DummyBuffer3, sizeof(CUDAComplex)*NTM);
    // if (c_dbf_4)    cudaMalloc((void**)&c_DummyBuffer4, sizeof(CUDAComplex)*NTM);
    // if (c_dbf_5)    cudaMalloc((void**)&c_DummyBuffer5, sizeof(CUDAComplex)*NTM);
    // if (c_dbf_6)    cudaMalloc((void**)&c_DummyBuffer6, sizeof(CUDAComplex)*NTM);

    // result = cuMemGetInfo(&free_mem, &total_mem);
    // std::cout << "Used GPU memory: c_dbf_x " << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB" << std::endl;

    return NoError;
}

SFStatus DataType_VkFFT::Deallocate_Arrays()
{
    // Depending on the type of solver and the chosen operator/outputs, the arrays which must be allocated vary.
    // This is simply controlled here by specifying the necessary flags during solver initialization

    // // Cpu arrays
    // if (r_Input1)       free(r_Input1);
    // if (r_Input2)       free(r_Input2);
    // if (r_Input3)       free(r_Input3);

    // // Real-valued arrays
    // if (r_Input1)      cudaFreeHost(cuda_r_Input1);
    // if (r_Input2)      cudaFreeHost(cuda_r_Input2);
    // if (r_Input3)      cudaFreeHost(cuda_r_Input3);

    // //    if (r_FTInput1)   cudaFree(r_FTInput1);
    // //    if (r_FTInput2)   cudaFree(r_FTInput2);
    // //    if (r_FTInput3)   cudaFree(r_FTInput3);

    // if (r_Output1)    free(r_Output1);
    // if (r_Output2)    free(r_Output2);
    // if (r_Output3)    free(r_Output3);

    // if (r_Output1)    cudaFree(cuda_r_Output1);
    // if (r_Output2)    cudaFree(cuda_r_Output2);
    // if (r_Output3)    cudaFree(cuda_r_Output3);

    // // Arrays for real Green's function
    // if (r_FG)       free(r_FG);

    // // Complex-valued arrays
    // if (c_Input1)      cudaFree(c_Input1);
    // if (c_Input2)      cudaFree(c_Input2);
    // if (c_Input3)      cudaFree(c_Input3);

    // if (c_FTInput1)   cudaFree(c_FTInput1);
    // if (c_FTInput2)   cudaFree(c_FTInput2);
    // if (c_FTInput3)   cudaFree(c_FTInput3);

    // if (!InPlace && c_FTOutput1)    {cudaFree(c_FTOutput1);}
    // if (!InPlace && c_FTOutput2)    {cudaFree(c_FTOutput2);}
    // if (!InPlace && c_FTOutput3)    {cudaFree(c_FTOutput3);}

    // if (c_vel_1)    cudaFree(c_FTVel1);
    // if (c_vel_2)    cudaFree(c_FTVel2);
    // if (c_vel_3)    cudaFree(c_FTVel3);

    // if (c_Output1)    cudaFree(c_Output1);
    // if (c_Output2)    cudaFree(c_Output2);
    // if (c_Output3)    cudaFree(c_Output3);

    // // Arrays for complex Green's function & spectral operators arrays
    // if (c_FG)       cudaFree(c_FG);
    // if (c_FGi)      cudaFree(c_FGi);
    // if (c_FGj)      cudaFree(c_FGj);
    // if (c_FGk)      cudaFree(c_FGk);

    // if (c_DummyBuffer1)    cudaFree(c_DummyBuffer1);
    // if (c_DummyBuffer2)    cudaFree(c_DummyBuffer2);
    // if (c_DummyBuffer3)    cudaFree(c_DummyBuffer3);
    // if (c_DummyBuffer4)    cudaFree(c_DummyBuffer4);
    // if (c_DummyBuffer5)    cudaFree(c_DummyBuffer5);
    // if (c_DummyBuffer6)    cudaFree(c_DummyBuffer6);

    return NoError;
}

}

#endif
