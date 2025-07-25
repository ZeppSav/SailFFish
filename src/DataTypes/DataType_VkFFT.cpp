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

    // res = Test_Case();

    // Get device properties
    // Print_Device_Info(vkGPU->device);

    // Specify buffer size for memory allocation
    // bufferSizeX = NX;

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

SFStatus DataType_VkFFT::Allocate_Arrays()
{
    // Depending on the type of solver and the chosen operator/outputs, the arrays which must be allocated vary.
    // This is simply controlled here by specifying the necessary flags during solver initialization

    // The buffer sizes must be set here.
    bufferSizeX =   (uint64_t)NX*sizeof(cl_real);
    bufferSizeY =   (uint64_t)NY*sizeof(cl_real);
    bufferSizeZ =   (uint64_t)NZ*sizeof(cl_real);
    bufferSizeNT =  (uint64_t)NT*sizeof(cl_real);       // Single value
    bufferSizeNTM = (uint64_t)NTM*sizeof(cl_real);

    c_bufferSizeX   = (uint64_t)NX*sizeof(cl_complex);
    c_bufferSizeY   = (uint64_t)NY*sizeof(cl_complex);
    c_bufferSizeZ   = (uint64_t)NZ*sizeof(cl_complex);
    c_bufferSizeNT  = (uint64_t)NT*sizeof(cl_complex);       // Single value
    c_bufferSizeNTM = (uint64_t)NTM*sizeof(cl_complex);

    //Neumann example:
    // r_in1 = true;
    // r_ft_in1 = true;
    // r_out_1 = true;
    // r_fg = true;

    // // Specify transform  and kernel type
    // Grid = G;
    // Spect_Kernel = B;
    // if (Grid==REGULAR)    Transform = DCT1;
    // if (Grid==STAGGERED)  Transform = DCT2;

    // if (Transform==DFT_C2C)

    // size_t free_mem = 0, total_mem = 0;
    // CUresult result = cuMemGetInfo(&free_mem, &total_mem);
    // std::cout << "Used GPU memory: 0 " << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB" << std::endl;

    if (r_in1)      r_Input1 = (Real*)malloc(NT*sizeof(Real));
    if (r_in2)      r_Input2 = (Real*)malloc(NT*sizeof(Real));
    if (r_in3)      r_Input3 = (Real*)malloc(NT*sizeof(Real));

    // Allocate local arrays: not pinned
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
    //    if (r_fg)       cudaMalloc((void**)&cu_r_FG, sizeof(CUDAReal)*NT);          // Not necessary!
    if (r_fg)       r_FG = (Real*)malloc(NT*sizeof(Real));                      // Allocate memory for real data on CPU

    // Complex-valued arrays
    if (c_in1)      Allocate_Buffer(c_Input1, c_bufferSizeNT);
    if (c_in2)      Allocate_Buffer(c_Input2, c_bufferSizeNT);
    if (c_in3)      Allocate_Buffer(c_Input3, c_bufferSizeNT);

    if (c_in1 )std::cout << "c_Input1 has been allocated" <<std::endl;

    if (c_ft_in1)   Allocate_Buffer(c_FTInput1, c_bufferSizeNT);
    if (c_ft_in2)   Allocate_Buffer(c_FTInput2, c_bufferSizeNT);
    if (c_ft_in3)   Allocate_Buffer(c_FTInput3, c_bufferSizeNT);

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

    // Arrays for complex Green's function & spectral operators arrays
    if (c_fg)       Allocate_Buffer(c_FG,  c_bufferSizeNT);
    if (c_fg_i)     Allocate_Buffer(c_FGi, c_bufferSizeNT);
    if (c_fg_j)     Allocate_Buffer(c_FGj, c_bufferSizeNT);
    if (c_fg_k)     Allocate_Buffer(c_FGk, c_bufferSizeNT);

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

//--- Specify Plan

SFStatus DataType_VkFFT::Prepare_Plan(VkFFTConfiguration &Config)
{
    // There are a number of generic setup steps required, which are not specific to the setup chosen.

    // Setup kernel options (for the case of a fused kernel)
    Config.device = &vkGPU->device;
    Config.context = &vkGPU->context;

    Config.FFTdim = FFTDim;
    Config.size[0] = NX;                     // Specify size of FFT
    Config.size[1] = NY;                     // Specify size of FFT
    Config.size[2] = NZ;                     // Specify size of FFT

    if (Transform==DCT1)        Config.performDCT = 1;
    if (Transform==DCT2)        Config.performDCT = 2;
    if (Transform==DST1)        Config.performDST = 1;
    if (Transform==DST2)        Config.performDST = 2;
    if (Transform==DFT_C2C)     // No need to do anything-> Baseline case
    if (Transform==DFT_R2C)     Config.performR2C = 1;

    return NoError;
}

SFStatus DataType_VkFFT::Specify_1D_Plan()
{
    // This specifies the forward and backward plans for execution

    // Setup VkFFT configuration & application structs
    FFTDim = 1;
    if (FusedKernel)    Prepare_Plan(kernel_configuration);
    Prepare_Plan(configuration);

    // Specify scaling terms for Green's function:
    if (Transform==DCT1)        BFac = 1.0/(2.0*(NX-1));
    if (Transform==DCT2)        BFac = 1.0/(2.0*NX);
    if (Transform==DST1)        BFac = 1.0/(2.0*(NX+1));
    if (Transform==DST2)        BFac = 1.0/(2.0*NX);
    // if (Transform==DFT_C2C)     BFac = 1.0/NT/4.0/Hx/Hx; // This works for nx = 1024*8
    if (Transform==DFT_C2C)     BFac = 1.0/NT;
    if (Transform==DFT_R2C)     BFac = 1.0/NT;

    if (FusedKernel){
        kernel_configuration.kernelConvolution = true;
        kernel_configuration.coordinateFeatures = 1;
    }

    return NoError;
}

SFStatus DataType_VkFFT::Specify_2D_Plan()
{
    // This specifies the forward and backward plans for execution

    // Setup VkFFT configuration & application structs
    FFTDim = 2;
    if (FusedKernel)    Prepare_Plan(kernel_configuration);
    Prepare_Plan(configuration);

    // Specify scaling terms for Green's function:
    if (Transform==DCT1)         BFac = 1.0/(2.0*(NX-1)*2.0*(NY-1));
    if (Transform==DCT2)         BFac = 1.0/(2.0*NX*2.0*NY);
    if (Transform==DST1)         BFac = 1.0/(2.0*(NX+1)*2.0*(NY+1));
    if (Transform==DST2)         BFac = 1.0/(2.0*NX*2.0*NY);
    if (Transform==DFT_C2C)      BFac = 1.0/NT;
    if (Transform==DFT_R2C)      BFac = 1.0/NT;

    if (FusedKernel){
        kernel_configuration.kernelConvolution = true;
        kernel_configuration.coordinateFeatures = 1;
    }

    return NoError;
}

SFStatus DataType_VkFFT::Specify_3D_Plan()
{
    // This specifies the forward and backward plans for execution

    // Setup VkFFT configuration & application structs
    FFTDim = 3;
    if (FusedKernel)    Prepare_Plan(kernel_configuration);
    Prepare_Plan(configuration);

    // Specify scaling terms for Green's function:
    if (Transform==DCT1)        BFac = 1.0/(2.0*(NX-1)*2.0*(NY-1)*2.0*(NZ-1));
    if (Transform==DCT2)        BFac = 1.0/(2.0*NX*2.0*NY*2.0*NZ);
    if (Transform==DST1)        BFac = 1.0/(2.0*(NX+1)*2.0*(NY+1)*2.0*(NZ+1));
    if (Transform==DST2)        BFac = 1.0/(2.0*NX*2.0*NY*2.0*NZ);
    if (Transform==DFT_C2C)     BFac = 1.0/NT;
    if (Transform==DFT_R2C)     BFac = 1.0/NT;

    if (FusedKernel){
        kernel_configuration.kernelConvolution = true;
        kernel_configuration.coordinateFeatures = 1;
    }

    return NoError;
}

//--- Prepare input array

SFStatus DataType_VkFFT::Set_Input(RVector &I)
{
    // Transfers input array to opencl buffer
    if (int(I.size())!=NT){
        std::cout << "Input array has incorrect dimension." << std::endl;
        return DimError;
    }

    VkFFTResult res = VKFFT_SUCCESS;
    if (r_in1)  res = transferDataFromCPU(vkGPU, I.data(), &cl_r_Input1, bufferSizeNT);
    if (c_in1){
        // Need to generate intermediate complex array and then transfer data over
        std::vector<cl_complex> IC(NT,CLC0);
        for (int i=0; i<NT; i++) {IC[i].x = I[i];}
        res = transferDataFromCPU(vkGPU, IC.data(), &c_Input1, c_bufferSizeNT);
    }
    return ConvertVkFFTError(res);

}

//--- Prepare output array

void DataType_VkFFT::Get_Output(RVector &I)
{
    // This function converts the output array into an easily accesible format
    if (I.empty()) I.assign(NT,0);
    VkFFTResult res = VKFFT_SUCCESS;
    if (r_out_1)    res = transferDataToCPU(vkGPU, I.data(), &cl_r_Input1, bufferSizeNT);
    if (c_out_1){
        // Need to generate intermediate complex array and then transfer data over
        std::vector<cl_complex> IC(NT);
        res = transferDataToCPU(vkGPU, IC.data(), &c_Input1, c_bufferSizeNT);
        // Test 1: Greens function:
        // res = transferDataToCPU(vkGPU, IC.data(), &c_FG, c_bufferSizeNT);
        // Test 1: Greens function: within convolution FFT
        // res = transferDataToCPU(vkGPU, IC.data(), configuration.kernel, c_bufferSizeNT);

        for (int i=0; i<NT; i++)    I[i]= IC[i].x;      // Greens functions real in
    }
    // return ConvertVkFFTError(res);
}

//--- Greens functions prep

// Notes:
// With identity kernel, output is always scaled and returns the EXACT input... no additioanl scaling.
// With Forward/Backward FFT + normalize,  output is always scaled and returns the EXACT input
// No support for 1D R2C convolutions
// 1D transforms kernel compilations fail (see the notebook below)
// 3D transforms may be problematic
// https://github.com/DTolm/VkFFT/issues/200
// https://github.com/vincefn/pyvkfft/issues/33

void DataType_VkFFT::Prep_Greens_Function_C2C()
{
    // In the case of C2C convolutions, the Greens functions (defined in R2R) are already defined within the frequency space.
    // We therefore do not need to carry out the additional step of doing the convolution in VkFFt. We do however need to ensure that the
    // buffers are correctly defined for the convolution step.

    VkFFTResult resFFT = VKFFT_SUCCESS;

    // Step 1: Pass complex convolution array to Cl Buffers.
    std::vector<cl_complex> rFG2(NT,CLC1);          // Identity kernel
    for (int i=0; i<NT; i++) rFG2[i].x = r_FG[i];   // Set real components  (comment out for identify kernel)
    resFFT = transferDataFromCPU(vkGPU, rFG2.data(), &c_FG, c_bufferSizeNT);   // Transfer cpu data from r_FG to GPU buffer
    // resFFT = performVulkanFFT(vkGPU, &kernel_app, &kernel_launchParams, -1, 1);  // This step not necessary- the Greens function is defined in frequency space

    if (FusedKernel){
        kernel_configuration.buffer = &c_FG;
        kernel_configuration.bufferSize = &c_bufferSizeNT;
        resFFT = initializeVkFFT(&kernel_app, kernel_configuration);                    // Initialize kernel convolution
    }
    std::cout << "kernel app prepared" << std::endl;

    // Now specify convolution plan
    configuration.bufferSize = &c_bufferSizeNT;
    configuration.buffer = &c_Input1;                       // Periodic BCs

    // Prepare convolution
    if (FusedKernel)
    {
        // Exploit the in-built convolution capabilites of VkFFT
        configuration.performConvolution = true;
        configuration.conjugateConvolution = true;
        configuration.kernel = kernel_configuration.buffer; // Pass convolution buffer
        configuration.symmetricKernel = true;               // Specify if convolution kernel is symmetric.
        configuration.matrixConvolution = 1;                // We do matrix convolution, so kernel is 9 numbers (3x3), but vector dimension is 3
        configuration.coordinateFeatures = 1;               // Equal to matrixConvolution size
        configuration.kernelSize = &c_bufferSizeNT;

        // configuration.numberBatches = 4;
        // configuration.printMemoryLayout = true;
        // configuration.disableReorderFourStep = 1;
        // configuration.isInputFormatted = true;
        // configuration.isOutputFormatted = true;
        std::cout << "Forward, backward FFT and convolution plan merged using VkFFT performConvolution option." << std::endl;
    }
    else
    {
        // Compile a bespoke openCl kernel for the convolution
        configuration.normalize = 1;                    // Normalize output
    }

    // Initialize FFT solver
    resFFT = initializeVkFFT(&app, configuration);
    std::cout << "Axis split: " << NX csp app.localFFTPlan->axisSplit[0][0] csp app.localFFTPlan->axisSplit[0][1] << std::endl;
    SFStatus resSF = ConvertVkFFTError(resFFT);
    std::cout << "convolution app prepared" << std::endl;
}

//--- Fourier transforms

void DataType_VkFFT::Forward_FFT_DFT()
{
    // Keep in mind here: The convolution step has already been prepared in preprocessing within VkFFT.
    // We only need to execute the FFT with the specified plan.
    VkFFTResult resFFT = VKFFT_SUCCESS;
    resFFT = performVulkanFFT(vkGPU, &app, &launchParams, -1, 1);
    SFStatus ressf = ConvertVkFFTError(resFFT);
    std::cout << "Forward FFT executed" << std::endl;
}

void DataType_VkFFT::Backward_FFT_DFT()
{
    if (FusedKernel) return;    // If exploiting the convolution feature of VkFFT, this step must not be executed

    VkFFTResult resFFT = VKFFT_SUCCESS;
    resFFT = performVulkanFFT(vkGPU, &app, &launchParams, 1, 1);
    SFStatus ressf = ConvertVkFFTError(resFFT);
    std::cout << "Backward FFT executed" << std::endl;
}

//--- Convolution

void DataType_VkFFT::Convolution_Complex()
{
    if (FusedKernel) return;    // If exploiting the convolution feature of VkFFT, jump out here.

    // Execute convolution kernel.
}

//--- Test case

VkFFTResult DataType_VkFFT::Test_Case()
{
    // This is a verbatim paste of the test from VkFFT

    uint64_t file_output = false;
    FILE* output = nullptr;
    // uint64_t isCompilerInitialized = false;

    VkFFTResult resFFT = VKFFT_SUCCESS;
    cl_int res = CL_SUCCESS;
    if (file_output)
        fprintf(output, "50 - VkFFT convolution example with identitiy kernel\n");
    printf("50 - VkFFT convolution example with identitiy kernel\n");
    //7 - convolution
    //Configuration + FFT application.
    VkFFTConfiguration configuration = {};
    VkFFTConfiguration convolution_configuration = {};
    VkFFTApplication app_convolution = {};
    VkFFTApplication app_kernel = {};
    //Convolution sample code
    //Setting up FFT configuration. FFT is performed in-place with no performance loss.

    configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
    // configuration.size[0] = 1024 * 1024 * 8; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.
    configuration.size[0] = 1024*8; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.
    configuration.size[1] = 1;
    configuration.size[2] = 1;

    configuration.kernelConvolution = true; //specify if this plan is used to create kernel for convolution
    // configuration.coordinateFeatures = 9; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc).
    configuration.coordinateFeatures = 1; //Specify dimensionality of the input feature vector (default 1). Each component is stored not as a vector, but as a separate system and padded on it's own according to other options (i.e. for x*y system of 3-vector, first x*y elements correspond to the first dimension, then goes x*y for the second, etc).
    //coordinateFeatures number is an important constant for convolution. If we perform 1x1 convolution, it is equal to number of features, but matrixConvolution should be equal to 1. For matrix convolution, it must be equal to matrixConvolution parameter. If we perform 2x2 convolution, it is equal to 3 for symmetric kernel (stored as xx, xy, yy) and 4 for nonsymmetric (stored as xx, xy, yx, yy). Similarly, 6 (stored as xx, xy, xz, yy, yz, zz) and 9 (stored as xx, xy, xz, yx, yy, yz, zx, zy, zz) for 3x3 convolutions.
    // configuration.normalize = 1;//normalize iFFT

    //After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
    configuration.device = &vkGPU->device;
    configuration.context = &vkGPU->context;

    //In this example, we perform a convolution for a real vectorfield (3vector) with a symmetric kernel (6 values). We use configuration to initialize convolution kernel first from real data, then we create convolution_configuration for convolution. The buffer object from configuration is passed to convolution_configuration as kernel object.
    //1. Kernel forward FFT.
    uint64_t kernelSize = ((uint64_t)configuration.coordinateFeatures) * sizeof(float) * 2 * (configuration.size[0]) * configuration.size[1] * configuration.size[2];;

    cl_mem kernel = 0;
    kernel = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, kernelSize, 0, &res);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    configuration.buffer = &kernel;
    configuration.bufferSize = &kernelSize;

    // std::cout << "HERE T1" << std::endl;

    if (file_output)
        fprintf(output, "Total memory needed for kernel: %" PRIu64 " MB\n", kernelSize / 1024 / 1024);
    printf("Total memory needed for kernel: %" PRIu64 " MB\n", kernelSize / 1024 / 1024);
    //Fill kernel on CPU.
    float* kernel_input = (float*)malloc(kernelSize);
    if (!kernel_input) return VKFFT_ERROR_MALLOC_FAILED;
    for (uint64_t v = 0; v < configuration.coordinateFeatures; v++) {
        for (uint64_t k = 0; k < configuration.size[2]; k++) {
            for (uint64_t j = 0; j < configuration.size[1]; j++) {

                //for (uint64_t i = 0; i < configuration.size[0]; i++) {
                //	kernel_input[i + j * configuration.size[0] + k * (configuration.size[0] + 2) * configuration.size[1] + v * (configuration.size[0] + 2) * configuration.size[1] * configuration.size[2]] = 1;

                //Below is the test identity kernel for 3x3 nonsymmetric FFT
                for (uint64_t i = 0; i < configuration.size[0]; i++) {
                    if ((v == 0) || (v == 4) || (v == 8))

                    kernel_input[2 * (i + j * (configuration.size[0]) + k * (configuration.size[0]) * configuration.size[1] + v * (configuration.size[0]) * configuration.size[1] * configuration.size[2])] = 1;

                    else
                        kernel_input[2 * (i + j * (configuration.size[0]) + k * (configuration.size[0]) * configuration.size[1] + v * (configuration.size[0]) * configuration.size[1] * configuration.size[2])] = 0;
                    kernel_input[2 * (i + j * (configuration.size[0]) + k * (configuration.size[0]) * configuration.size[1] + v * (configuration.size[0]) * configuration.size[1] * configuration.size[2]) + 1] = 0;

                }
            }
        }
    }
    //Sample buffer transfer tool. Uses staging buffer (if needed) of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
    resFFT = transferDataFromCPU(vkGPU, kernel_input, &kernel, kernelSize);
    if (resFFT != VKFFT_SUCCESS) return resFFT;
    //Initialize application responsible for the kernel. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.
    resFFT = initializeVkFFT(&app_kernel, configuration);
    if (resFFT != VKFFT_SUCCESS) return resFFT;
    //Sample forward FFT command buffer allocation + execution performed on kernel. Second number determines how many times perform application in one submit. FFT can also be appended to user defined command buffers.

    // std::cout << "HERE T2" << std::endl;

    //Uncomment the line below if you want to perform kernel FFT. In this sample we use predefined identitiy kernel.
    //performVulkanFFT(vkGPU, &app_kernel, -1, 1);

    //The kernel has been trasnformed.


    //2. Buffer convolution with transformed kernel.
    //Copy configuration, as it mostly remains unchanged. Change specific parts.
    convolution_configuration = configuration;
    configuration.kernelConvolution = false;
    convolution_configuration.performConvolution = true;
    // convolution_configuration.symmetricKernel = false;//Specify if convolution kernel is symmetric. In this case we only pass upper triangle part of it in the form of: (xx, xy, yy) for 2d and (xx, xy, xz, yy, yz, zz) for 3d.
    // convolution_configuration.matrixConvolution = 3;//we do matrix convolution, so kernel is 9 numbers (3x3), but vector dimension is 3
    // convolution_configuration.coordinateFeatures = 3;//equal to matrixConvolution size
    convolution_configuration.symmetricKernel = true;//Specify if convolution kernel is symmetric. In this case we only pass upper triangle part of it in the form of: (xx, xy, yy) for 2d and (xx, xy, xz, yy, yz, zz) for 3d.
    convolution_configuration.matrixConvolution = 1;//we do matrix convolution, so kernel is 9 numbers (3x3), but vector dimension is 3
    convolution_configuration.coordinateFeatures = 1;//equal to matrixConvolution size

    convolution_configuration.kernel = &kernel;

    //Allocate separate buffer for the input data.
    uint64_t bufferSize = ((uint64_t)convolution_configuration.coordinateFeatures) * sizeof(float) * 2 * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2];;
    cl_mem buffer = 0;
    buffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSize, 0, &res);
    if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
    configuration.buffer = &buffer;
    convolution_configuration.bufferSize = &bufferSize;
    convolution_configuration.kernelSize = &kernelSize;

    // std::cout << "HERE T3" << std::endl;

    if (file_output)
        fprintf(output, "Total memory needed for buffer: %" PRIu64 " MB\n", bufferSize / 1024 / 1024);
    printf("Total memory needed for buffer: %" PRIu64 " MB\n", bufferSize / 1024 / 1024);
    //Fill data on CPU. It is best to perform all operations on GPU after initial upload.
    float* buffer_input = (float*)malloc(bufferSize);
    if (!buffer_input) return VKFFT_ERROR_MALLOC_FAILED;
    for (uint64_t v = 0; v < convolution_configuration.coordinateFeatures; v++) {
        for (uint64_t k = 0; k < convolution_configuration.size[2]; k++) {
            for (uint64_t j = 0; j < convolution_configuration.size[1]; j++) {
                for (uint64_t i = 0; i < convolution_configuration.size[0]; i++) {
                    buffer_input[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2])] = (float)(i % 8 - 3.5);
                    buffer_input[2 * (i + j * convolution_configuration.size[0] + k * (convolution_configuration.size[0]) * convolution_configuration.size[1] + v * (convolution_configuration.size[0]) * convolution_configuration.size[1] * convolution_configuration.size[2]) + 1] = (float)(i % 4 - 1.5);
                }
            }
        }
    }
    //Transfer data to GPU using staging buffer.
    resFFT = transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
    if (resFFT != VKFFT_SUCCESS) return resFFT;

    // std::cout << "HERE T4" << std::endl;

    //Initialize application responsible for the convolution.
    resFFT = initializeVkFFT(&app_convolution, convolution_configuration);
    if (resFFT != VKFFT_SUCCESS) return resFFT;
    //Sample forward FFT command buffer allocation + execution performed on kernel. FFT can also be appended to user defined command buffers.

    // std::cout << "HERE T5" << std::endl;

    VkFFTLaunchParams launchParams = {};
    resFFT = performVulkanFFT(vkGPU, &app_convolution, &launchParams, -1, 1);
    if (resFFT != VKFFT_SUCCESS) return resFFT;

    // std::cout << "HERE T6" << std::endl;
    //The kernel has been transformed.
    free(kernel_input);
    free(buffer_input);
    // free(buffer_output);

    clReleaseMemObject(buffer);
    clReleaseMemObject(kernel);
    deleteVkFFT(&app_kernel);
    deleteVkFFT(&app_convolution);

    std::cout << "Test case complete! " << std::endl;

    return VKFFT_SUCCESS;
}

}

#endif
