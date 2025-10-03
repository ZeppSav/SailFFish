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
    VkFFTResult res = VKFFT_SUCCESS;

    // Generate VkFFT object & initialize drivers
    vkGPU = new VkGPU{};
    res = OpenCLSetup(vkGPU);

    // Enable fused kernel convolution
    // FusedKernel = true;

    InPlace = true;      // Issues with out of
}

// Options:

VkFFTResult DataType_VkFFT::OpenCLSetup(VkGPU *VK)
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
            if (k == VK->device_id) {
                VK->platform = platforms[j];
                VK->device = deviceList[i];
                VK->context = clCreateContext(NULL, 1, &VK->device, NULL, NULL, &res);
                if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
                cl_command_queue commandQueue = clCreateCommandQueue(VK->context, VK->device, 0, &res);
                if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
                VK->commandQueue = commandQueue;
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

    char deviceName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    std::cout << "Device Name: " << deviceName << std::endl;

    // Get device type
    cl_device_type deviceType;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr);
    std::cout << "Device Type: " << deviceType;
    std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;

    // Device memory (global)
    cl_ulong ulongVal;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ulongVal), &ulongVal, nullptr);
    std::cout << "   Device memory: " << (ulongVal / (1024 * 1024)) << " MB" << std::endl;

    // Max work group size
    size_t sizeVal;
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
    cl_uint uintVal;
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

SFStatus DataType_VkFFT::Allocate_Buffer(cl_mem &buffer, uint64_t bufsize)
{
    // cl_mem buffer = 0;
    cl_int res = CL_SUCCESS;
    buffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufsize, 0, &res);
    return ConvertClError(res);
}

SFStatus DataType_VkFFT::Allocate_SubBuffer(cl_mem &bufferDest, cl_mem &bufferSrc, uint64_t bufstart, uint64_t bufsize)
{
    // cl_mem buffer = 0;
    cl_int res = CL_SUCCESS;
    cl_buffer_region region;
    region.origin = bufstart;   // byte offset
    region.size   = bufsize;    // length of the sub-buffer
    bufferDest = clCreateSubBuffer(bufferSrc, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &res);
    return ConvertClError(res);
}

SFStatus DataType_VkFFT::Zero_FloatBuffer(cl_mem &buffer, uint64_t bufsize)
{
    // This helper function zeros the buffer
    cl_int res = clEnqueueFillBuffer(vkGPU->commandQueue, buffer, &CLR0, sizeof(CLR0), 0, bufsize, 0, NULL, NULL);
    clFinish(vkGPU->commandQueue);  // wait for completion
    return ConvertClError(res);
    // err = clEnqueueFillBuffer(queue,          // command queue
    //                           buffer,         // cl_mem object
    //                           &pattern,       // pointer to pattern
    //                           pattern_size,   // size of pattern
    //                           0,              // offset in buffer
    //                           buffer_size,    // size to fill (in bytes)
    //                           0, NULL, NULL); // wait list, events
}

SFStatus DataType_VkFFT::Allocate_Arrays()
{
    // Depending on the type of solver and the chosen operator/outputs, the arrays which must be allocated vary.
    // This is simply controlled here by specifying the necessary flags during solver initialization

    // The buffer sizes must be set here.
    bufferSizeX =   (uint64_t)NX*sizeof(cl_real);
    bufferSizeY =   (uint64_t)NY*sizeof(cl_real);
    bufferSizeZ =   (uint64_t)NZ*sizeof(cl_real);
    bufferSizeNT =  (uint64_t)NT*sizeof(cl_real);           // Single value
    bufferSizeNTM = (uint64_t)NTM*sizeof(cl_real);

    c_bufferSizeX   = (uint64_t)NX*sizeof(cl_complex);
    c_bufferSizeY   = (uint64_t)NY*sizeof(cl_complex);
    c_bufferSizeZ   = (uint64_t)NZ*sizeof(cl_complex);
    c_bufferSizeNT  = (uint64_t)NT*sizeof(cl_complex);      // Single value
    c_bufferSizeNTM = (uint64_t)NTM*sizeof(cl_complex);

    if (r_in1)      r_Input1 = (Real*)malloc(NT*sizeof(Real));
    if (r_in2)      r_Input2 = (Real*)malloc(NT*sizeof(Real));
    if (r_in3)      r_Input3 = (Real*)malloc(NT*sizeof(Real));

    // Allocate local arrays
    if (r_in1)      Allocate_Buffer(cl_r_Input1, bufferSizeNT);
    if (r_in2)      Allocate_Buffer(cl_r_Input2, bufferSizeNT);
    if (r_in3)      Allocate_Buffer(cl_r_Input3, bufferSizeNT);

    if (InPlace){
        if (r_out_1)    cl_r_Output1 = cl_r_Input1;
        if (r_out_2)    cl_r_Output2 = cl_r_Input2;
        if (r_out_3)    cl_r_Output3 = cl_r_Input3;
    }
    else{
        if (r_out_1)    Allocate_Buffer(cl_r_Output1, bufferSizeNT);
        if (r_out_2)    Allocate_Buffer(cl_r_Output2, bufferSizeNT);
        if (r_out_3)    Allocate_Buffer(cl_r_Output3, bufferSizeNT);
    }

    if (r_out_1)      r_Output1 = (Real*)malloc(NT*sizeof(Real));
    if (r_out_2)      r_Output2 = (Real*)malloc(NT*sizeof(Real));
    if (r_out_3)      r_Output3 = (Real*)malloc(NT*sizeof(Real));

    // Arrays for real Green's function
    // if (r_fg)       cudaMalloc((void**)&r_FG, sizeof(CUDAReal)*NT);
    //    if (r_fg)       cudaMalloc((void**)&cu_r_FG, sizeof(CUDAReal)*NT);    // Not necessary!
    if (r_fg)       r_FG = (Real*)malloc(NT*sizeof(Real));                      // Allocate memory for real data on CPU

    // Complex-valued arrays
    if (c_in1)      Allocate_Buffer(c_Input1, c_bufferSizeNT);
    if (c_in2)      Allocate_Buffer(c_Input2, c_bufferSizeNT);
    if (c_in3)      Allocate_Buffer(c_Input3, c_bufferSizeNT);

    if (InPlace){
        if (c_out_1)    c_Output1 = c_Input1;
        if (c_out_2)    c_Output2 = c_Input2;
        if (c_out_3)    c_Output3 = c_Input3;
    }
    else{
        if (c_out_1)    Allocate_Buffer(c_Output1, c_bufferSizeNT);
        if (c_out_2)    Allocate_Buffer(c_Output2, c_bufferSizeNT);
        if (c_out_3)    Allocate_Buffer(c_Output3, c_bufferSizeNT);
    }

    // These are allocated to simplify data transfer in the case of a periodic (complex) solve
    if (c_in1)      r_Input1 = (Real*)malloc(NT*sizeof(Real));
    if (c_in2)      r_Input2 = (Real*)malloc(NT*sizeof(Real));
    if (c_in3)      r_Input3 = (Real*)malloc(NT*sizeof(Real));


    if (c_ft_in1)   Allocate_Buffer(c_FTInput1, c_bufferSizeNTM);
    if (c_ft_in2)   Allocate_Buffer(c_FTInput2, c_bufferSizeNTM);
    if (c_ft_in3)   Allocate_Buffer(c_FTInput3, c_bufferSizeNTM);

    if (InPlace){
        if (c_ft_in1) c_FTOutput1 = c_FTInput1;
        if (c_ft_in2) c_FTOutput2 = c_FTInput2;
        if (c_ft_in3) c_FTOutput3 = c_FTInput3;
    }
    else{
        if (c_ft_in1) Allocate_Buffer(c_FTOutput1, c_bufferSizeNTM);
        if (c_ft_in2) Allocate_Buffer(c_FTOutput2, c_bufferSizeNTM);
        if (c_ft_in3) Allocate_Buffer(c_FTOutput3, c_bufferSizeNTM);
    }

    // Arrays for Green's function & spectral operators arrays
    if      (Transform==DFT_C2C)    Allocate_Buffer(c_FG,   c_bufferSizeNTM);   // Periodic
    else                            Allocate_Buffer(cl_r_FG, bufferSizeNT);     // R2R + R2C (unbounded)
    if      (Transform==DFT_R2C)    Allocate_Buffer(c_FG,   c_bufferSizeNTM);   // R2C (unbounded)
    if (c_fg_i)     Allocate_Buffer(c_FGi, c_bufferSizeNTM);
    if (c_fg_j)     Allocate_Buffer(c_FGj, c_bufferSizeNTM);
    if (c_fg_k)     Allocate_Buffer(c_FGk, c_bufferSizeNTM);

    return NoError;
}

SFStatus DataType_VkFFT::Deallocate_Arrays()
{
    // Depending on the type of solver and the chosen operator/outputs, the arrays which must be allocated vary.
    // This is simply controlled here by specifying the necessary flags during solver initialization

    if (r_in1)      free(r_Input1);
    if (r_in2)      free(r_Input2);
    if (r_in3)      free(r_Input3);

    if (r_out_1)      free(r_Output1);
    if (r_out_2)      free(r_Output2);
    if (r_out_3)      free(r_Output3);

    // Allocate local arrays
    if (r_in1)      clReleaseMemObject(cl_r_Input1);
    if (r_in2)      clReleaseMemObject(cl_r_Input2);
    if (r_in3)      clReleaseMemObject(cl_r_Input3);

    if (!InPlace){
        if (r_out_1)    clReleaseMemObject(cl_r_Output1);
        if (r_out_2)    clReleaseMemObject(cl_r_Output2);
        if (r_out_3)    clReleaseMemObject(cl_r_Output3);
    }

    // Arrays for real Green's function
    // if (r_fg)       cudaMalloc((void**)&r_FG, sizeof(CUDAReal)*NT);
    //    if (r_fg)       cudaMalloc((void**)&cu_r_FG, sizeof(CUDAReal)*NT);    // Not necessary!
    if (r_fg)       free(r_FG);                      // Allocate memory for real data on CPU

    // Complex-valued arrays
    if (c_in1)      clReleaseMemObject(c_Input1);
    if (c_in2)      clReleaseMemObject(c_Input2);
    if (c_in3)      clReleaseMemObject(c_Input3);

    if (!InPlace){
        if (c_out_1)    clReleaseMemObject(c_Output1);
        if (c_out_2)    clReleaseMemObject(c_Output2);
        if (c_out_3)    clReleaseMemObject(c_Output3);
    }

    if (c_ft_in1)   clReleaseMemObject(c_FTInput1);
    if (c_ft_in2)   clReleaseMemObject(c_FTInput2);
    if (c_ft_in3)   clReleaseMemObject(c_FTInput3);

    if (!InPlace){
        if (c_ft_in1) clReleaseMemObject(c_FTOutput1);
        if (c_ft_in2) clReleaseMemObject(c_FTOutput2);
        if (c_ft_in3) clReleaseMemObject(c_FTOutput3);
    }

    // Arrays for Green's function & spectral operators arrays
    if      (Transform==DFT_C2C)    clReleaseMemObject(c_FG);       // Periodic
    else                            clReleaseMemObject(cl_r_FG);    // R2R + R2C (unbounded)
    if      (Transform==DFT_R2C)    clReleaseMemObject(c_FG);       // R2C (unbounded)
    if (c_fg_i)     clReleaseMemObject(c_FGi);
    if (c_fg_j)     clReleaseMemObject(c_FGj);
    if (c_fg_k)     clReleaseMemObject(c_FGk);

    return NoError;
}

//--- Specify Plan

SFStatus DataType_VkFFT::Prepare_Plan(VkFFTConfiguration &Config)
{
    // There are a number of generic setup steps required, which are not specific to the setup chosen.

    // Setup kernel options (for the case of a fused kernel)
    Config.device = &vkGPU->device;
    Config.context = &vkGPU->context;

    Config.FFTdim = FFTDim;
    Config.size[0] = NX;                    // Specify size of FFT
    Config.size[1] = NY;                    // Specify size of FFT
    Config.size[2] = NZ;                    // Specify size of FFT

    switch (Transform)
    {
        case DCT1:        {Config.performDCT = 1;                       break;}
        case DCT2:        {Config.performDCT = 2;                       break;}
        case DST1:        {Config.performDST = 1;                       break;}
        case DST2:        {Config.performDST = 2;                       break;}
        case DFT_C2C:     {                     break;} // No need to do anything-> Baseline case
        case DFT_R2C:     {Config.performR2C = 1;                       break;}
        default: {
            std::cout << "DataType_VkFFT::Prepare_Plan: Transform not recognised. " << std::endl;
            return SetupError;
        }
    }

    // Specify input buffer (Note: This must be corrected if using fused convolution step)
    Config.bufferSize = &InputbufferSize;   // Specify config buffer size
    Config.buffer = &InputBuffer;           // Specify input buffer

    if (Transform==DFT_R2C){
        // Trial first out-of-place transform.
        Config.inputBuffer = &InputBuffer;
        Config.inputBufferSize = &InputbufferSize;
        Config.buffer = &c_FTInput1;
        Config.bufferSize = &c_bufferSizeNTM;

        // To denote that it is NOT padded... specify this here with the following flag:
        Config.isInputFormatted = true;
        Config.inverseReturnToInputBuffer = true;           // Only valid if in-place

        // // Trial zero padding feature
        // Config.performZeropadding[0] = true; //Perform padding with zeros on GPU. Still need to properly align input data (no need to fill padding area with meaningful data) but this will increase performance due to the lower amount of the memory reads/writes and omitting sequences only consisting of zeros.
        // Config.performZeropadding[1] = true;
        // Config.performZeropadding[2] = true;
        // Config.fft_zeropad_left[0] = (uint64_t)ceil(Config.size[0] / 2.0);
        // Config.fft_zeropad_right[0] = Config.size[0];
        // Config.fft_zeropad_left[1] = (uint64_t)ceil(Config.size[1] / 2.0);
        // Config.fft_zeropad_right[1] = Config.size[1];
        // Config.fft_zeropad_left[2] = (uint64_t)ceil(Config.size[2] / 2.0);
        // Config.fft_zeropad_right[2] = Config.size[2];

        // Out of place Option 1          ---- WORKING ----
        // Copy c_FTInputi to c_FTOutputi
        // Change launch.buffer to c_FTOutputi
        // Change launch.inputbuffer to cl_r_Outputi

        // Out of place Option 2:  ---- WORKING ----
        if (!InPlace){
            Config.isOutputFormatted = true;
            Config.outputBuffer = &c_FTOutput1;
            Config.outputBufferSize = &c_bufferSizeNTM;

            // Directly priod to C2R step, define the following:
            // if (!InPlace) launchParams.outputBuffer = &cl_r_Output1;
        }
    }

    // Specify precision
    if (std::is_same<Real,float>::value)    Config.doublePrecision = 0;
    if (std::is_same<Real,double>::value)   Config.doublePrecision = 1;

    return NoError;
}

SFStatus DataType_VkFFT::Specify_1D_Plan()
{
    // This specifies the forward and backward plans for execution

    // Specify scaling terms for Green's functions and input buffers
    switch (Transform)
    {
        case DCT1:        {InputBuffer = cl_r_Input1;   InputbufferSize = bufferSizeNT;     BFac = 1.0/(2.0*(NX-1));    break;}
        case DCT2:        {InputBuffer = cl_r_Input1;   InputbufferSize = bufferSizeNT;     BFac = 1.0/(2.0*NX);        break;}
        case DST1:        {InputBuffer = cl_r_Input1;   InputbufferSize = bufferSizeNT;     BFac = 1.0/(2.0*(NX+1));    break;}
        case DST2:        {InputBuffer = cl_r_Input1;   InputbufferSize = bufferSizeNT;     BFac = 1.0/(2.0*NX);        break;}
        case DFT_C2C:     {InputBuffer = c_Input1;      InputbufferSize = c_bufferSizeNT;   BFac = 1.0/NT;              break;}
        case DFT_R2C:     {InputBuffer = cl_r_Input1;   InputbufferSize = bufferSizeNT;     BFac = 1.0/NT;              break;}
        default: {
            std::cout << "DataType_VkFFT::Specify_1D_Plan(): Transform not recognised. " << std::endl;
            return SetupError;
        }
    }

    // Setup VkFFT configuration & application structs
    FFTDim = 1;
    Prepare_Plan(configuration);

    if (FusedKernel){
        Prepare_Plan(kernel_configuration);
        kernel_configuration.kernelConvolution = true;
        kernel_configuration.coordinateFeatures = 1;
    }

    return NoError;
}

SFStatus DataType_VkFFT::Specify_2D_Plan()
{
    // This specifies the forward and backward plans for execution

    // Specify scaling terms for Green's functions and input buffers
    switch (Transform)
    {
        case DCT1:        {InputBuffer = cl_r_Input1;   InputbufferSize = bufferSizeNT;     BFac = 1.0/(2.0*(NX-1)*2.0*(NY-1));     break;}
        case DCT2:        {InputBuffer = cl_r_Input1;   InputbufferSize = bufferSizeNT;     BFac = 1.0/(2.0*NX*2.0*NY);             break;}
        case DST1:        {InputBuffer = cl_r_Input1;   InputbufferSize = bufferSizeNT;     BFac = 1.0/(2.0*(NX+1)*2.0*(NY+1));     break;}
        case DST2:        {InputBuffer = cl_r_Input1;   InputbufferSize = bufferSizeNT;     BFac = 1.0/(2.0*NX*2.0*NY);             break;}
        case DFT_C2C:     {InputBuffer = c_Input1;      InputbufferSize = c_bufferSizeNT;   BFac = 1.0/NT;                          break;}
        case DFT_R2C:     {InputBuffer = cl_r_Input1;   InputbufferSize = bufferSizeNT;     BFac = 1.0/NT;                          break;}
        default: {
            std::cout << "DataType_VkFFT::Specify_2D_Plan(): Transform not recognised. " << std::endl;
            return SetupError;
        }
    }

    // Setup VkFFT configuration & application structs
    FFTDim = 2;
    Prepare_Plan(configuration);

    if (FusedKernel){
        Prepare_Plan(kernel_configuration);
        kernel_configuration.kernelConvolution = true;
        kernel_configuration.coordinateFeatures = 1;
    }

    return NoError;
}

SFStatus DataType_VkFFT::Specify_3D_Plan()
{
    // This specifies the forward and backward plans for execution

    // Specify scaling terms for Green's functions and input buffers
    switch (Transform)
    {
        case DCT1:        {InputBuffer = cl_r_Input1;   InputbufferSize = bufferSizeNT;     BFac = 1.0/(2.0*(NX-1)*2.0*(NY-1)*2.0*(NZ-1));  break;}
        case DCT2:        {InputBuffer = cl_r_Input1;   InputbufferSize = bufferSizeNT;     BFac = 1.0/(2.0*NX*2.0*NY*2.0*NZ);              break;}
        case DST1:        {InputBuffer = cl_r_Input1;   InputbufferSize = bufferSizeNT;     BFac = 1.0/(2.0*(NX+1)*2.0*(NY+1)*2.0*(NZ+1));  break;}
        case DST2:        {InputBuffer = cl_r_Input1;   InputbufferSize = bufferSizeNT;     BFac = 1.0/(2.0*NX*2.0*NY*2.0*NZ);              break;}
        case DFT_C2C:     {InputBuffer = c_Input1;      InputbufferSize = c_bufferSizeNT;   BFac = 1.0/NT;                                  break;}
        case DFT_R2C:     {InputBuffer = cl_r_Input1;   InputbufferSize = bufferSizeNT;     BFac = 1.0/NT;                                  break;}
        default: {
            std::cout << "DataType_VkFFT::Specify_3D_Plan(): Transform not recognised. " << std::endl;
            return SetupError;
        }
    }

    // Setup VkFFT configuration & application structs
    FFTDim = 3;
    Prepare_Plan(configuration);

    if (FusedKernel){
        Prepare_Plan(kernel_configuration);
        kernel_configuration.kernelConvolution = true;
        kernel_configuration.coordinateFeatures = 1;
    }

    return NoError;
}

//--- Utility functions
//--- CPU-GPU Interface functions adapted directly from VkFFT/benchmark_scripts/vkFFT_scripts/src/utils_VkFFT.cpp

VkFFTResult DataType_VkFFT::transferDataToCPU(VkGPU* vkGPU, void* cpu_arr, void* output_buffer, uint64_t transferSize) {
    //a function that transfers data from the GPU to the CPU using staging buffer, because the GPU memory is not host-coherent
    VkFFTResult resFFT = VKFFT_SUCCESS;
    cl_int res = CL_SUCCESS;
    cl_mem* buffer = (cl_mem*)output_buffer;
    cl_command_queue commandQueue = clCreateCommandQueue(vkGPU->context, vkGPU->device, 0, &res);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
    res = clEnqueueReadBuffer(commandQueue, buffer[0], CL_TRUE, 0, transferSize, cpu_arr, 0, NULL, NULL);
    if (res != CL_SUCCESS) {
        return VKFFT_ERROR_FAILED_TO_COPY;
    }
    res = clReleaseCommandQueue(commandQueue);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE;
    return resFFT;
}

VkFFTResult DataType_VkFFT::transferDataFromCPU(VkGPU* vkGPU, void* cpu_arr, void* input_buffer, uint64_t transferSize) {
    VkFFTResult resFFT = VKFFT_SUCCESS;
    cl_int res = CL_SUCCESS;
    cl_mem* buffer = (cl_mem*)input_buffer;
    cl_command_queue commandQueue = clCreateCommandQueue(vkGPU->context, vkGPU->device, 0, &res);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
    res = clEnqueueWriteBuffer(commandQueue, buffer[0], CL_TRUE, 0, transferSize, cpu_arr, 0, NULL, NULL);
    if (res != CL_SUCCESS) {
        return VKFFT_ERROR_FAILED_TO_COPY;
    }
    res = clReleaseCommandQueue(commandQueue);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE;
    return resFFT;
}

VkFFTResult DataType_VkFFT::performVulkanFFT(VkGPU* vkGPU, VkFFTApplication* app, VkFFTLaunchParams* launchParams, int inverse)
{
    VkFFTResult resFFT = VKFFT_SUCCESS;
    cl_int res = CL_SUCCESS;
    launchParams->commandQueue = &vkGPU->commandQueue;
    // std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
    resFFT = VkFFTAppend(app, inverse, launchParams);
    if (resFFT != VKFFT_SUCCESS) return resFFT;
    res = clFinish(vkGPU->commandQueue);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
    // std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
    // double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
    return resFFT;
}

//--- Prepare input array

VkFFTResult DataType_VkFFT::ConvertArray_R2C(RVector &I, void* input_buffer, size_t N)
{
    // Helper function for complex array inputs
    std::vector<cl_complex> IC(N);
    std::transform(I.begin(), I.end(), IC.begin(), [](Real val) {return cl_complex{{val, 0.}};});
    return transferDataFromCPU(vkGPU, IC.data(), input_buffer, sizeof(cl_complex)*N);
}

SFStatus DataType_VkFFT::Set_Input(RVector &I)
{
    // Transfers input array to opencl buffer
    if (size(I)!=size_t(NT)){
        std::cout << "DataType_VkFFT::Set_Input(RVector &I): Input array has incorrect dimension." << std::endl;
        return DimError;
    }

    RVector I2(NT);
    switch (Dimension)
    {
        case 1: {Map_C2F_1D(I,I2);  break;}
        case 2: {Map_C2F_2D(I,I2);  break;}
        case 3: {Map_C2F_3D(I,I2);  break;}
        default: {break;}
    }

    VkFFTResult res = VKFFT_SUCCESS;
    if (r_in1)  res = transferDataFromCPU(vkGPU, I2.data(), &cl_r_Input1, bufferSizeNT);
    if (c_in1)  res = ConvertArray_R2C(I2,&c_Input1,NT);
    return ConvertVkFFTError(res);
}

SFStatus DataType_VkFFT::Set_Input(RVector &I1, RVector &I2, RVector &I3)
{
    // Transfers input array to opencl buffer
    if ((size(I1)!=(size_t)NT) ||
        (size(I2)!=(size_t)NT) ||
        (size(I3)!=(size_t)NT)){
        std::cout << "DataType_VkFFT::Set_Input: Input array has incorrect dimension." << std::endl;
        return DimError;
    }

    // Map to F-style ordering
    RVector mI1(NT), mI2(NT), mI3(NT);
    Map_C2F_3DV(I1,I2,I3,mI1,mI2,mI3);

    VkFFTResult res = VKFFT_SUCCESS;
    if (r_in1)  res = transferDataFromCPU(vkGPU, mI1.data(), &cl_r_Input1, bufferSizeNT);
    if (r_in2)  res = transferDataFromCPU(vkGPU, mI2.data(), &cl_r_Input2, bufferSizeNT);
    if (r_in3)  res = transferDataFromCPU(vkGPU, mI3.data(), &cl_r_Input3, bufferSizeNT);
    if (c_in1)  res = ConvertArray_R2C(mI1,&c_Input1,NT);
    if (c_in2)  res = ConvertArray_R2C(mI2,&c_Input2,NT);
    if (c_in3)  res = ConvertArray_R2C(mI3,&c_Input3,NT);
    return ConvertVkFFTError(res);
}

SFStatus DataType_VkFFT::Set_Input_Unbounded_1D(RVector &I)
{
    // This function takes the input vector and stores this in the appropriate cl input array
    int NXH = NX/2;
    if (size(I)!=size_t(NXH)){
        std::cout << "Input array has incorrect dimension." << std::endl;
        return DimError;
    }
    RVector R1(NT,0);
    if (r_in1) memcpy(R1.data(), I.data(), NXH*sizeof(Real));
    if (c_in1)  {}// Not yet implemented!

    VkFFTResult res = VKFFT_SUCCESS;
    if (r_in1)  res = transferDataFromCPU(vkGPU, R1.data(), &cl_r_Input1, bufferSizeNT);
    if (c_in1)  {}  // Not yet implemented!
    return ConvertVkFFTError(res);
}

SFStatus DataType_VkFFT::Set_Input_Unbounded_2D(RVector &I)
{
    // This function takes the input vector and stores this in the appropriate cuda input array
    int NXH = NX/2;
    int NYH = NY/2;
    if (int(I.size())!=NXH*NYH){
        std::cout << "DataType_VkFFT::Set_Input_Unbounded_2D: Input array has incorrect dimension." << std::endl;
        return DimError;
    }

    // Create dummy arrays to pass in one block to cuda buffers
    if (r_in1) memset(r_Input1, 0, NT*sizeof(Real));
    if (c_in1) {}   // Not yet implemented!

    // Fill nonzero elements of dummy arrays
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            int idg = j*NX + i;
            int idl = i*NYH + j;
            if (r_in1) r_Input1[idg] = I[idl];
            if (c_in1) {}   // Not yet implemented!
        }
    }

    // Now transfer block arrays to cuda buffers
    VkFFTResult res = VKFFT_SUCCESS;
    if (r_in1)  res = transferDataFromCPU(vkGPU, r_Input1, &cl_r_Input1, bufferSizeNT);
    if (c_in1)  {}  // Not yet implemented!
    return ConvertVkFFTError(res);
}

SFStatus DataType_VkFFT::Set_Input_Unbounded_3D(RVector &I)
{
    // This function takes the input vector and stores this in the appropriate cuda input array
    int NXH = NX/2;
    int NYH = NY/2;
    int NZH = NZ/2;
    if (int(I.size())!=NXH*NYH*NZH){
        std::cout << "DataType_VkFFT::Set_Input_Unbounded_3D: Input array has incorrect dimension." << std::endl;
        return DimError;
    }

    // Create dummy arrays to pass in one block to cuda buffers
    if (r_in1) memset(r_Input1, 0, NT*sizeof(Real));
    if (c_in1) {}   // Not yet implemented!

    // Fill nonzero elements of dummy arrays
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            for (int k=0; k<NZH; k++){
                // int idg = i*NY*NZ + j*NZ + k;
                int idg = k*NX*NY + j*NX + i;         // HERE JOE
                int idl = i*NYH*NZH + j*NZH + k;
                if (r_in1) r_Input1[idg] = I[idl];
                // if (c_in1) C1[idg].real(I[idl]);
            }
        }
    }
    // Now transfer block arrays to cuda buffers
    VkFFTResult res = VKFFT_SUCCESS;
    if (r_in1)  res = transferDataFromCPU(vkGPU, r_Input1, &cl_r_Input1, bufferSizeNT);
    if (c_in1)  {}  // Not yet implemented!
    return ConvertVkFFTError(res);
}

SFStatus DataType_VkFFT::Set_Input_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3)
{
    // This function takes the input vector and stores this in the appropriate cuda input array
    int NXH = NX/2;
    int NYH = NY/2;
    int NZH = NZ/2;
    if (int(I1.size())!=NXH*NYH*NZH || int(I2.size())!=NXH*NYH*NZH || int(I3.size())!=NXH*NYH*NZH){
        std::cout << "DataType_VkFFT::Set_Input_Unbounded_3D: Input array has incorrect dimension." << std::endl;
        return DimError;
    }

    // Create dummy arrays to pass in one block to cuda buffers
    if (r_in1) memset(r_Input1, 0., NT*sizeof(Real));
    if (r_in2) memset(r_Input2, 0., NT*sizeof(Real));
    if (r_in3) memset(r_Input3, 0., NT*sizeof(Real));
    // if (c_in1) {}   // Not yet implemented!
    // if (c_in2) {}   // Not yet implemented!
    // if (c_in3) {}   // Not yet implemented!


    // Fill nonzero elements of dummy arrays
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            for (int k=0; k<NZH; k++){
                int idg = k*NX*NY + j*NX + i;
                int idl = i*NYH*NZH + j*NZH + k;
                r_Input1[idg] = I1[idl];
                r_Input2[idg] = I2[idl];
                r_Input3[idg] = I3[idl];
                // if (c_in1) c_Input1[idg].real(I1[idl]);
                // if (c_in2) c_Input2[idg].real(I2[idl]);
                // if (c_in3) c_Input3[idg].real(I3[idl]);
            }
        }
    }

    // Now transfer block arrays to cuda buffers
    VkFFTResult res = VKFFT_SUCCESS;
    if (r_in1)  res = transferDataFromCPU(vkGPU, r_Input1, &cl_r_Input1, bufferSizeNT);
    if (r_in2)  res = transferDataFromCPU(vkGPU, r_Input2, &cl_r_Input2, bufferSizeNT);
    if (r_in3)  res = transferDataFromCPU(vkGPU, r_Input3, &cl_r_Input3, bufferSizeNT);
    // if (c_in1)  {}  // Not yet implemented!
    // if (c_in2)  {}  // Not yet implemented!
    // if (c_in3)  {}  // Not yet implemented!

    return ConvertVkFFTError(res);
}

//--- Prepare output array

VkFFTResult DataType_VkFFT::ConvertArray_C2R(RVector &I, void* input_buffer, size_t N)
{
    // Helper function for complex array inputs
    std::vector<cl_complex> IC(N,CLC0);
    VkFFTResult res = transferDataToCPU(vkGPU, IC.data(), input_buffer, sizeof(cl_complex)*N);
    std::transform(IC.begin(), IC.end(), I.begin(), [](cl_complex val) {return val.x;});
    return res;
}

void DataType_VkFFT::Get_Output(RVector &I)
{
    // This function converts the output array into an easily accesible format
    if (I.empty()) I.assign(NT,0);

    RVector mI(NT);
    VkFFTResult res = VKFFT_SUCCESS;
    if (r_out_1)    res = transferDataToCPU(vkGPU, mI.data(), &cl_r_Output1, bufferSizeNT);
    if (c_out_1)    res = ConvertArray_C2R(mI,&c_Output1,NT);

    switch (Dimension)
    {
        case 1: {Map_F2C_1D(mI,I);  break;}
        case 2: {Map_F2C_2D(mI,I);  break;}
        case 3: {Map_F2C_3D(mI,I);  break;}
        default: {break;}
    }
}

void DataType_VkFFT::Get_Output(RVector &I1, RVector &I2, RVector &I3)
{
    // Transfers input array to opencl buffer
    VkFFTResult res = VKFFT_SUCCESS;
    RVector mI1(NT), mI2(NT), mI3(NT);
    if (r_out_1)    res = transferDataToCPU(vkGPU, mI1.data(), &cl_r_Output1, bufferSizeNT);
    if (r_out_2)    res = transferDataToCPU(vkGPU, mI2.data(), &cl_r_Output2, bufferSizeNT);
    if (r_out_3)    res = transferDataToCPU(vkGPU, mI3.data(), &cl_r_Output3, bufferSizeNT);
    if (c_out_1)    res = ConvertArray_C2R(mI1,&c_Output1,NT);
    if (c_out_2)    res = ConvertArray_C2R(mI2,&c_Output2,NT);
    if (c_out_3)    res = ConvertArray_C2R(mI3,&c_Output3,NT);

    // Map to C-style ordering
    if (I1.empty()) I1.assign(NT,0);
    if (I2.empty()) I2.assign(NT,0);
    if (I3.empty()) I3.assign(NT,0);
    Map_F2C_3DV(mI1,mI2,mI3,I1,I2,I3);
}

void DataType_VkFFT::Get_Output_Unbounded_1D(RVector &I)
{
    // Extracts appropriate output
    int NXH = NX/2;
    if (size(I)!=size_t(NXH))   I.assign(NXH,0);

    // Create dummy arrays if required
    RVector R1;
    CVector C1;
    if (r_out_1) R1 = RVector(NT,0);
    if (c_out_1) C1 = CVector(NT,ComplexNull);

    // Copy memory from cl buffer
    VkFFTResult res = VKFFT_SUCCESS;
    if (r_out_1)    res = transferDataToCPU(vkGPU, R1.data(), &cl_r_Output1, bufferSizeNT);
    if (c_out_1)  {}// Not yet implemented!
    SFStatus ressf = ConvertVkFFTError(res);

    // Copy to output arrays
    std::memcpy(I.data(), R1.data(), NXH*sizeof(Real));   // Just copy over

    // // Hacked output for testing ONLY FFT (forward):
    // std::cout << "Outputting complex arrays" << std::endl;
    // std::vector<cl_complex> O(NTM,CLC0);
    // VkFFTResult res2 = transferDataToCPU(vkGPU, O.data(), &c_FTOutput1, c_bufferSizeNTM);
    // for (auto i : O) std::cout << i.x csp i.y << std::endl;
}

void DataType_VkFFT::Get_Output_Unbounded_2D(RVector &I)
{
    // Extracts appropriate output
    int NXH = NX/2;
    int NYH = NY/2;
    if (I.empty()) I.assign(NXH*NYH,0);

    // Retrieve data from cl buffer
    VkFFTResult res = VKFFT_SUCCESS;
    if (r_out_1)    res = transferDataToCPU(vkGPU, r_Output1, &cl_r_Output1, bufferSizeNT);
    // if (c_out_1)    {}// Not yet implemented!

    // Copy necessary memory into output arrays
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            int idg = j*NX + i;
            int idl = i*NYH + j;
            if (r_out_1) I[idl] = r_Output1[idg];
            // if (c_out_1)    {}// Not yet implemented!
        }
    }

    // Hack:: read output array... did FFT work properly?
    // std::vector<cl_complex> O(NTM,CLC0);
    // VkFFTResult res2 = transferDataToCPU(vkGPU, O.data(), &c_FTInput1, c_bufferSizeNTM);    // FFT of input
    // VkFFTResult res2 = transferDataToCPU(vkGPU, O.data(), &c_FG, c_bufferSizeNTM);          // FFT of Green's function
    // for (auto i : O) std::cout << i.x csp i.y << std::endl;
}

void DataType_VkFFT::Get_Output_Unbounded_3D(RVector &I)
{
    // Extracts appropriate output
    int NXH = NX/2;
    int NYH = NY/2;
    int NZH = NZ/2;
    if (I.empty()) I.assign(NXH*NYH*NZH,0);

    // Create dummy arrays if required
    VkFFTResult res = VKFFT_SUCCESS;
    if (r_out_1)    res = transferDataToCPU(vkGPU, r_Output1, &cl_r_Output1, bufferSizeNT);
    // if (c_out_1)    {}// Not yet implemented!

    // Copy necessary memory into output arrays
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            for (int k=0; k<NZH; k++){
                int idg = k*NX*NY + j*NX + i;
                int idl = i*NYH*NZH + j*NZH + k;
                if (r_out_1) I[idl] = r_Output1[idg];
                // if (c_out_1) I[idl] = C1[idg].real();
            }
        }
    }
}

void DataType_VkFFT::Get_Output_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3)
{
    // Extracts appropriate output
    int NXH = NX/2;
    int NYH = NY/2;
    int NZH = NZ/2;
    if (I1.empty()) I1.assign(NXH*NYH*NZH,0);
    if (I2.empty()) I2.assign(NXH*NYH*NZH,0);
    if (I3.empty()) I3.assign(NXH*NYH*NZH,0);

    // Create dummy arrays if required
    VkFFTResult res = VKFFT_SUCCESS;
    if (r_out_1)    res = transferDataToCPU(vkGPU, r_Output1, &cl_r_Output1, bufferSizeNT);
    if (r_out_2)    res = transferDataToCPU(vkGPU, r_Output2, &cl_r_Output2, bufferSizeNT);
    if (r_out_3)    res = transferDataToCPU(vkGPU, r_Output3, &cl_r_Output3, bufferSizeNT);
    // if (c_out_1)    {}// Not yet implemented!

    // Copy necessary memory into output arrays
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            for (int k=0; k<NZH; k++){
                int idg = k*NX*NY + j*NX + i;
                int idl = i*NYH*NZH + j*NZH + k;
                I1[idl] = r_Output1[idg];
                I2[idl] = r_Output2[idg];
                I3[idl] = r_Output3[idg];
                // if (c_out_1) I1[idl] = C1[idg].real();
                // if (c_out_2) I2[idl] = C2[idg].real();
                // if (c_out_3) I3[idl] = C3[idg].real();
            }
        }
    }
}

//--- Greens functions prep

void DataType_VkFFT::Prepare_Fused_Kernel(FTType TF)
{
    // Untested: We shall exploit the fused kernel feature of VkFFT
    VkFFTResult resFFT = initializeVkFFT(&kernel_app, kernel_configuration);

    // If R2C transform is being executed, we need to carry out the FFT on the input data
    if (TF==DFT_R2C){
        launchParams.buffer = kernel_configuration.buffer;
        resFFT = performVulkanFFT(vkGPU, &kernel_app, &launchParams, -1);
    }
    std::cout << "Kernel app prepared for fused FFT-Conv-iFFT. " << std::endl;

    // Exploit the in-built convolution capabilites of VkFFT
    configuration.performConvolution = true;
    // configuration.conjugateConvolution = 0;
    configuration.kernel = kernel_configuration.buffer; // Pass convolution buffer
    configuration.symmetricKernel = true;               // Specify if convolution kernel is symmetric.
    configuration.matrixConvolution = 1;                // We do matrix convolution, so kernel is 9 numbers (3x3), but vector dimension is 3
    configuration.coordinateFeatures = 1;               // Equal to matrixConvolution size
    // configuration.kernelSize = &c_bufferSizeNT;
    configuration.kernelSize = &InputbufferSize;

    // configuration.numberBatches = 4;
    // configuration.printMemoryLayout = true;
    // configuration.disableReorderFourStep = 1;
    // configuration.isInputFormatted = true;
    // configuration.isOutputFormatted = true;
    std::cout << "Application prepared for FUSED FFT-Conv-iFFT. " << std::endl;

    std::cout << "Convolution VkFFT Plan: Axis split 1: " << kernel_app.localFFTPlan->axisSplit[0][0] csp kernel_app.localFFTPlan->axisSplit[0][1] << std::endl;
    std::cout << "Convolution VkFFT Plan: Axis split 2: " << kernel_app.localFFTPlan->axisSplit[1][0] csp kernel_app.localFFTPlan->axisSplit[1][1] << std::endl;
}

void DataType_VkFFT::Compile_Convolution_Kernel(FTType TF)
{
    // A fused kernel setup of VkFFT is not being exploited
    // We need to compile the kernel directly.

    cl_int err;
    std::string Source;
    if (std::is_same<Real,float>::value)    Source.append(ocl_kernels_float);
    if (std::is_same<Real,double>::value)   Source.append(ocl_kernels_double);
    if (TF==DFT_R2R)    Source.append(ocl_multiply);
    if (TF==DFT_C2C)    Source.append(ocl_complexmultiply);
    if (TF==DFT_R2C)    Source.append(ocl_complexmultiply);
    const char* source_str = Source.c_str();
    // std::cout << Source << std::endl;
    conv_program = clCreateProgramWithSource(vkGPU->context, 1, &source_str, NULL, &err);
    err = clBuildProgram(conv_program, 1, &vkGPU->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(conv_program, vkGPU->device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "Build error:\n%s\n", buffer);
    }

    // Create kernel
    if (TF==DFT_R2R)    conv_kernel = clCreateKernel(conv_program, "multiply_in_place", &err);
    if (TF==DFT_C2C)    conv_kernel = clCreateKernel(conv_program, "complex_multiply_in_place", &err);
    if (TF==DFT_R2C)    conv_kernel = clCreateKernel(conv_program, "complex_multiply_in_place", &err);

    // Specify kernel arguments
    if (TF==DFT_R2R){
        clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &cl_r_Input1);
        clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &cl_r_FG);
    }
    if (TF==DFT_C2C){
        clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &c_Input1);
        clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &c_FG);
    }
    if (TF==DFT_R2C){
        clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &c_FTInput1);        // Input (post FFT)
        clSetKernelArg(conv_kernel, 1, sizeof(cl_mem), &c_FG);              // Greens function (post FFT)
    }
    std::cout << "Application prepared for SEQUENTIAL FFT-Conv-iFFT. " << std::endl;
}

void DataType_VkFFT::Prep_Greens_Function(FTType TF)
{
    // This process is substantially the same between Real or Complex inputs, so we shall bring them together into a single function.
    VkFFTResult resFFT = VKFFT_SUCCESS;

    // Step 1: Pass Greens function to cl buffer
    if (TF==DFT_R2R){       // Real Green's kernel
        resFFT = transferDataFromCPU(vkGPU, r_FG, &cl_r_FG, InputbufferSize);     // Transfer cpu data from r_FG to GPU buffer
        if (FusedKernel) kernel_configuration.buffer = &cl_r_FG;
    }
    if (TF==DFT_C2C){       // Complex Green's kernel
        RVector crFG(r_FG, r_FG+NT);
        resFFT = ConvertArray_R2C(crFG, &c_FG, NT);
        if (FusedKernel) kernel_configuration.buffer = &c_FG;
    }
    if (TF==DFT_R2C){       // Real Green's kernel
        // for (int i=0; i<NT; i++) r_FG[i] = BFac;                             // Trick for identity convolution (testing FFT-iFFT)
        resFFT = transferDataFromCPU(vkGPU, r_FG, &cl_r_FG, InputbufferSize);   // Transfer cpu data from r_FG to GPU buffer
        if (FusedKernel) kernel_configuration.buffer = &cl_r_FG;
    }

    // Step 2: Prepare convolution plan
    if (FusedKernel)    Prepare_Fused_Kernel(TF);
    else                Compile_Convolution_Kernel(TF);

    // Step 3: Initialize FFT solver
    // Note: If the fused kernel is not being selected, this can be done in an earlier step.

    // configuration.printMemoryLayout = true;
    // configuration.disableReorderFourStep = 1;
    resFFT = initializeVkFFT(&app, configuration);
    // configuration.printMemoryLayout = true;
    SFStatus resSF = ConvertVkFFTError(resFFT);
    std::cout << "FFT plans configured." << std::endl;

    // Additional outputs for debugging_
    size_t preferredWorkGroupSizeMultiple;
    clGetKernelWorkGroupInfo(conv_kernel, vkGPU->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferredWorkGroupSizeMultiple, NULL);
    std::cout << "Convolution kernel: Preferred work-group size multiple: " << preferredWorkGroupSizeMultiple << std::endl;
}

void DataType_VkFFT::Prep_Greens_Function_R2C()
{
    // An additional step is needed here. If the

    // Carry out standard setup
    Prep_Greens_Function(DFT_R2C);

    // If a fused kernel is not being used, then the buffer for the Green's function contains the REAl Green's function.
    // The forward FFT must be taken of this to ensure that it is correctly defined for the convolution.
    if (!FusedKernel){

        // // Hack:: Delta Kernel
        // std::vector<Real> I(NT,0);
        // I[0] = 1.0;
        // VkFFTResult res = transferDataFromCPU(vkGPU, I.data(), &cl_r_FG, bufferSizeNT);

        launchParams.inputBuffer = &cl_r_FG;
        launchParams.buffer = &c_FG;
        VkFFTResult resFFT = FFT_DFT(ForwardFFT);
        std::cout << "FFT plans configured for R2C convolution" << std::endl;

        // // Hack:: Convolution to return input
        // std::vector<Real> I(NTM,1.0/NT);
        // ConvertArray_R2C(I, &c_FG, NTM);
    }
}

//--- Fourier transforms

VkFFTResult DataType_VkFFT::FFT_DFT(int FFTType)
{
    // Keep in mind here: The convolution step has already been prepared in preprocessing within VkFFT.
    // We only need to execute the FFT with the specified plan.

    VkFFTResult resFFT = VKFFT_SUCCESS;
    cl_int res = CL_SUCCESS;
    launchParams.commandQueue = &vkGPU->commandQueue;
    // std::chrono::steady_clock::time_point timeSubmit = std::chrono::steady_clock::now();
    res = VkFFTAppend(&app, FFTType, &launchParams);
    if (resFFT != VKFFT_SUCCESS) return resFFT;
    res = clFinish(vkGPU->commandQueue);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
    // std::chrono::steady_clock::time_point timeEnd = std::chrono::steady_clock::now();
    // double totTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeSubmit).count() * 0.001;
    return resFFT;
}

void DataType_VkFFT::Forward_FFT_R2R()
{
    VkFFTResult resFFT = VKFFT_SUCCESS;
    if (r_in1 && (resFFT==VKFFT_SUCCESS))  {launchParams.buffer = &cl_r_Input1;   resFFT = FFT_DFT(ForwardFFT); }
    if (r_in2 && (resFFT==VKFFT_SUCCESS))  {launchParams.buffer = &cl_r_Input2;   resFFT = FFT_DFT(ForwardFFT); }
    if (r_in3 && (resFFT==VKFFT_SUCCESS))  {launchParams.buffer = &cl_r_Input3;   resFFT = FFT_DFT(ForwardFFT); }
    ConvertVkFFTError(resFFT);
}

void DataType_VkFFT::Backward_FFT_R2R()
{
    if (FusedKernel) return;
    VkFFTResult resFFT = VKFFT_SUCCESS;
    if (r_in1 && (resFFT==VKFFT_SUCCESS))  {launchParams.buffer = &cl_r_Input1;   resFFT = FFT_DFT(BackwardFFT); }
    if (r_in2 && (resFFT==VKFFT_SUCCESS))  {launchParams.buffer = &cl_r_Input2;   resFFT = FFT_DFT(BackwardFFT); }
    if (r_in3 && (resFFT==VKFFT_SUCCESS))  {launchParams.buffer = &cl_r_Input3;   resFFT = FFT_DFT(BackwardFFT); }
    ConvertVkFFTError(resFFT);
}

void DataType_VkFFT::Forward_FFT_DFT()
{
    VkFFTResult resFFT = VKFFT_SUCCESS;
    if (c_in1 && (resFFT==VKFFT_SUCCESS))  {launchParams.buffer = &c_Input1;   resFFT = FFT_DFT(ForwardFFT); }
    if (c_in2 && (resFFT==VKFFT_SUCCESS))  {launchParams.buffer = &c_Input2;   resFFT = FFT_DFT(ForwardFFT); }
    if (c_in3 && (resFFT==VKFFT_SUCCESS))  {launchParams.buffer = &c_Input3;   resFFT = FFT_DFT(ForwardFFT); }
    ConvertVkFFTError(resFFT);
}

void DataType_VkFFT::Backward_FFT_DFT()
{
    if (FusedKernel) return;
    VkFFTResult resFFT = VKFFT_SUCCESS;
    if (c_in1 && (resFFT==VKFFT_SUCCESS))  {launchParams.buffer = &c_Output1;   resFFT = FFT_DFT(BackwardFFT); }
    if (c_in2 && (resFFT==VKFFT_SUCCESS))  {launchParams.buffer = &c_Output2;   resFFT = FFT_DFT(BackwardFFT); }
    if (c_in3 && (resFFT==VKFFT_SUCCESS))  {launchParams.buffer = &c_Output3;   resFFT = FFT_DFT(BackwardFFT); }
    ConvertVkFFTError(resFFT);
}

void DataType_VkFFT::Forward_FFT_R2C()
{
    VkFFTResult resFFT = VKFFT_SUCCESS;
    // InPlace means: cl_r_Input1 = cl_r_Output1 &  c_FTInputi = c_FTOutputi
    // So for this case, we will always use specify inputBuffer & buffer separately
    if (r_in1 && (resFFT==VKFFT_SUCCESS))  {launchParams.inputBuffer = &cl_r_Input1; launchParams.buffer = &c_FTInput1;  resFFT = FFT_DFT(ForwardFFT); }
    if (r_in2 && (resFFT==VKFFT_SUCCESS))  {launchParams.inputBuffer = &cl_r_Input2; launchParams.buffer = &c_FTInput2;  resFFT = FFT_DFT(ForwardFFT); }
    if (r_in3 && (resFFT==VKFFT_SUCCESS))  {launchParams.inputBuffer = &cl_r_Input3; launchParams.buffer = &c_FTInput3;  resFFT = FFT_DFT(ForwardFFT); }
    SFStatus stat = ConvertVkFFTError(resFFT);
}

void DataType_VkFFT::Backward_FFT_C2R()
{
    if (FusedKernel) return;

    //---- HACK TO TEST OUT_OF_PLACE
    // if (!InPlace) launchParams.outputBuffer = &cl_r_Output1;
    //---- HACK TO TEST OUT_OF_PLACE

    VkFFTResult resFFT = VKFFT_SUCCESS;
    // Note: Specifying outputBuffer means something else...
    if (r_out_1 && (resFFT==VKFFT_SUCCESS))  {launchParams.inputBuffer = &cl_r_Output1; launchParams.buffer = &c_FTOutput1;  resFFT = FFT_DFT(BackwardFFT); }
    if (r_out_2 && (resFFT==VKFFT_SUCCESS))  {launchParams.inputBuffer = &cl_r_Output2; launchParams.buffer = &c_FTOutput2;  resFFT = FFT_DFT(BackwardFFT); }
    if (r_out_3 && (resFFT==VKFFT_SUCCESS))  {launchParams.inputBuffer = &cl_r_Output3; launchParams.buffer = &c_FTOutput3;  resFFT = FFT_DFT(BackwardFFT); }
    SFStatus stat = ConvertVkFFTError(resFFT);
}

//--- Convolution

cl_int DataType_VkFFT::Convolution()
{
    // If a sequential FFT-Conv-iFFT process is executed, carry out the convolution here.
    // the input buffers were specvified during initialization of the kernel
    if (FusedKernel) return CL_SUCCESS;    // If exploiting the convolution feature of VkFFT, jump out here.

    // Now carry out execution with opencl kernel
    size_t globalSize = NTM;

    // Option 1: Assign NULL to local work group size: Will automatically choose and ensure complete vector is multiplied
    // this enables us less options in specifying the size of the group-> Possibly less efficient, but dont require
    // catches in kernel to avoid out of bounds access.
    cl_int err = clEnqueueNDRangeKernel(vkGPU->commandQueue, conv_kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    // Option 2: Explicitly assign local work group size. Possibly more efficient, but requires catches in kernel
    // This is not working for now... avoid
    // size_t localSize = 1024;
    // cl_int err = clEnqueueNDRangeKernel(vkGPU->commandQueue, conv_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    // Ensure process completes
    // err = clFinish(vkGPU->commandQueue);
    return err;
}

void DataType_VkFFT::Convolution_Real3()
{
    // Carry out convolution for three arrays
    if (FusedKernel) return;    // If exploiting the convolution feature of VkFFT, jump out here.

    cl_int res = CL_SUCCESS;
    if (r_in1 & (res==CL_SUCCESS)) {clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &cl_r_Input1);  res = Convolution();}
    if (r_in2 & (res==CL_SUCCESS)) {clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &cl_r_Input2);  res = Convolution();}
    if (r_in3 & (res==CL_SUCCESS)) {clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &cl_r_Input3);  res = Convolution();}
    ConvertClError(res);
}

void DataType_VkFFT::Convolution_Complex3()
{
    // Carry out convolution for three arrays
    if (FusedKernel) return;    // If exploiting the convolution feature of VkFFT, jump out here.

    // std::cout << "CHECKING FLAGS " << c_in1 csp c_in2 csp c_in3 << std::endl;

    cl_int res = CL_SUCCESS;
    if (r_in1 && res==CL_SUCCESS) {clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &c_FTInput1);  res = Convolution();}
    if (r_in2 && res==CL_SUCCESS) {clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &c_FTInput2);  res = Convolution();}
    if (r_in3 && res==CL_SUCCESS) {clSetKernelArg(conv_kernel, 0, sizeof(cl_mem), &c_FTInput3);  res = Convolution();}
    res = clFinish(vkGPU->commandQueue);
    ConvertClError(res);
}

//--- Destructor

DataType_VkFFT::~DataType_VkFFT()
{
    //--- This clears the data associated with this FFTW object

    // Clear arrays
    Deallocate_Arrays();

    // Clear plans
    deleteVkFFT(&app);
    if (FusedKernel) deleteVkFFT(&kernel_app);
}


}

#endif
