/****************************************************************************
    SailFFish Library
    Copyright (C) 2022-present Joseph Saverin j.saverin@tu-berlin.de

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Information on file:

    -> This is the base datatype class when using VkFFT to calculate FFTs.

*****************************************************************************/

#ifndef DATATYPE_VKFFT_H
#define DATATYPE_VKFFT_H

#ifdef VKFFT

#include "DataType_Base.h"
#include "utils_VkFFT.h"

namespace SailFFish
{

inline SFStatus ConvertClError(cl_int res){
    switch (res)
    {
        case 0: 	{return NoError; }
        case -1: 	{std::cout <<  "CL_DEVICE_NOT_FOUND" << std::endl; return NoError; }
        case -2: 	{std::cout <<  "CL_DEVICE_NOT_AVAILABLE" << std::endl; return NoError; }
        case -3: 	{std::cout <<  "CL_COMPILER_NOT_AVAILABLE" << std::endl; return NoError; }
        case -4: 	{std::cout <<  "CL_MEM_OBJECT_ALLOCATION_FAILURE" << std::endl; return NoError; }
        case -5: 	{std::cout <<  "CL_OUT_OF_RESOURCES" << std::endl; return NoError; }
        case -6: 	{std::cout <<  "CL_OUT_OF_HOST_MEMORY" << std::endl; return NoError; }
        case -7: 	{std::cout <<  "CL_PROFILING_INFO_NOT_AVAILABLE" << std::endl; return NoError; }
        case -8: 	{std::cout <<  "CL_MEM_COPY_OVERLAP" << std::endl; return NoError; }
        case -9: 	{std::cout <<  "CL_IMAGE_FORMAT_MISMATCH" << std::endl; return NoError; }
        case -10: 	{std::cout <<  "CL_IMAGE_FORMAT_NOT_SUPPORTED" << std::endl; return NoError; }
        case -12: 	{std::cout <<  "CL_MAP_FAILURE" << std::endl; return NoError; }
        case -13: 	{std::cout <<  "CL_MISALIGNED_SUB_BUFFER_OFFSET" << std::endl; return NoError; }
        case -14: 	{std::cout <<  "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST" << std::endl; return NoError; }
        case -15: 	{std::cout <<  "CL_COMPILE_PROGRAM_FAILURE" << std::endl; return NoError; }
        case -16: 	{std::cout <<  "CL_LINKER_NOT_AVAILABLE" << std::endl; return NoError; }
        case -17: 	{std::cout <<  "CL_LINK_PROGRAM_FAILURE" << std::endl; return NoError; }
        case -18: 	{std::cout <<  "CL_DEVICE_PARTITION_FAILED" << std::endl; return NoError; }
        case -19: 	{std::cout <<  "CL_KERNEL_ARG_INFO_NOT_AVAILABLE" << std::endl; return NoError; }
        case -30: 	{std::cout <<  "CL_INVALID_VALUE" << std::endl; return NoError; }
        case -31: 	{std::cout <<  "CL_INVALID_DEVICE_TYPE" << std::endl; return NoError; }
        case -32: 	{std::cout <<  "CL_INVALID_PLATFORM" << std::endl; return NoError; }
        case -33: 	{std::cout <<  "CL_INVALID_DEVICE" << std::endl; return NoError; }
        case -34: 	{std::cout <<  "CL_INVALID_CONTEXT" << std::endl; return NoError; }
        case -35: 	{std::cout <<  "CL_INVALID_QUEUE_PROPERTIES" << std::endl; return NoError; }
        case -36: 	{std::cout <<  "CL_INVALID_COMMAND_QUEUE" << std::endl; return NoError; }
        case -37: 	{std::cout <<  "CL_INVALID_HOST_PTR" << std::endl; return NoError; }
        case -38: 	{std::cout <<  "CL_INVALID_MEM_OBJECT" << std::endl; return NoError; }
        case -39: 	{std::cout <<  "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR" << std::endl; return NoError; }
        case -40: 	{std::cout <<  "CL_INVALID_IMAGE_SIZE" << std::endl; return NoError; }
        case -41: 	{std::cout <<  "CL_INVALID_SAMPLER" << std::endl; return NoError; }
        case -42: 	{std::cout <<  "CL_INVALID_BINARY" << std::endl; return NoError; }
        case -43: 	{std::cout <<  "CL_INVALID_BUILD_OPTIONS" << std::endl; return NoError; }
        case -44: 	{std::cout <<  "CL_INVALID_PROGRAM" << std::endl; return NoError; }
        case -45: 	{std::cout <<  "CL_INVALID_PROGRAM_EXECUTABLE" << std::endl; return NoError; }
        case -46: 	{std::cout <<  "CL_INVALID_KERNEL_NAME" << std::endl; return NoError; }
        case -47: 	{std::cout <<  "CL_INVALID_KERNEL_DEFINITION" << std::endl; return NoError; }
        case -48: 	{std::cout <<  "CL_INVALID_KERNEL" << std::endl; return NoError; }
        case -49: 	{std::cout <<  "CL_INVALID_ARG_INDEX" << std::endl; return NoError; }
        case -50: 	{std::cout <<  "CL_INVALID_ARG_VALUE" << std::endl; return NoError; }
        case -51: 	{std::cout <<  "CL_INVALID_ARG_SIZE" << std::endl; return NoError; }
        case -52: 	{std::cout <<  "CL_INVALID_KERNEL_ARGS" << std::endl; return NoError; }
        case -53: 	{std::cout <<  "CL_INVALID_WORK_DIMENSION" << std::endl; return NoError; }
        case -54: 	{std::cout <<  "CL_INVALID_WORK_GROUP_SIZE" << std::endl; return NoError; }
        case -55: 	{std::cout <<  "CL_INVALID_WORK_ITEM_SIZE" << std::endl; return NoError; }
        case -56: 	{std::cout <<  "CL_INVALID_GLOBAL_OFFSET" << std::endl; return NoError; }
        case -57: 	{std::cout <<  "CL_INVALID_EVENT_WAIT_LIST" << std::endl; return NoError; }
        case -58: 	{std::cout <<  "CL_INVALID_EVENT" << std::endl; return NoError; }
        case -59: 	{std::cout <<  "CL_INVALID_OPERATION" << std::endl; return NoError; }
        case -60: 	{std::cout <<  "CL_INVALID_GL_OBJECT" << std::endl; return NoError; }
        case -61: 	{std::cout <<  "CL_INVALID_BUFFER_SIZE" << std::endl; return NoError; }
        case -62: 	{std::cout <<  "CL_INVALID_MIP_LEVEL" << std::endl; return NoError; }
        case -63: 	{std::cout <<  "CL_INVALID_GLOBAL_WORK_SIZE" << std::endl; return NoError; }
        case -64: 	{std::cout <<  "CL_INVALID_PROPERTY" << std::endl; return NoError; }
        case -65: 	{std::cout <<  "CL_INVALID_IMAGE_DESCRIPTOR" << std::endl; return NoError; }
        case -66: 	{std::cout <<  "CL_INVALID_COMPILER_OPTIONS" << std::endl; return NoError; }
        case -67: 	{std::cout <<  "CL_INVALID_LINKER_OPTIONS" << std::endl; return NoError; }
        case -68: 	{std::cout <<  "CL_INVALID_DEVICE_PARTITION_COUNT" << std::endl; return NoError; }
        case -69: 	{std::cout <<  "CL_INVALID_PIPE_SIZE" << std::endl; return NoError; }
        case -70: 	{std::cout <<  "CL_INVALID_DEVICE_QUEUE" << std::endl; return NoError; }
        case -71: 	{std::cout <<  "CL_INVALID_SPEC_ID" << std::endl; return NoError; }
        case -72: 	{std::cout <<  "CL_MAX_SIZE_RESTRICTION_EXCEEDED" << std::endl; return NoError; }
        case -1002: {std::cout <<  "CL_INVALID_D3D10_DEVICE_KHR" << std::endl; return NoError; }
        case -1003: {std::cout <<  "CL_INVALID_D3D10_RESOURCE_KHR" << std::endl; return NoError; }
        case -1004: {std::cout <<  "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR" << std::endl; return NoError; }
        case -1005: {std::cout <<  "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR" << std::endl; return NoError; }
        case -1006: {std::cout <<  "CL_INVALID_D3D11_DEVICE_KHR" << std::endl; return NoError; }
        case -1007: {std::cout <<  "CL_INVALID_D3D11_RESOURCE_KHR" << std::endl; return NoError; }
        case -1008: {std::cout <<  "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR" << std::endl; return NoError; }
        case -1009: {std::cout <<  "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR" << std::endl; return NoError; }
        case -1010: {std::cout <<  "CL_INVALID_DX9_MEDIA_ADAPTER_KHR" << std::endl; return NoError; }
        case -1011: {std::cout <<  "CL_INVALID_DX9_MEDIA_SURFACE_KHR" << std::endl; return NoError; }
        case -1012: {std::cout <<  "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR" << std::endl; return NoError; }
        case -1013: {std::cout <<  "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR" << std::endl; return NoError; }
        case -1093: {std::cout <<  "CL_INVALID_EGL_OBJECT_KHR" << std::endl; return NoError; }
        case -1092: {std::cout <<  "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR" << std::endl; return NoError; }
        case -1001: {std::cout <<  "CL_PLATFORM_NOT_FOUND_KHR" << std::endl; return NoError; }
        case -1057: {std::cout <<  "CL_DEVICE_PARTITION_FAILED_EXT" << std::endl; return NoError; }
        case -1058: {std::cout <<  "CL_INVALID_PARTITION_COUNT_EXT" << std::endl; return NoError; }
        case -1059: {std::cout <<  "CL_INVALID_PARTITION_NAME_EXT" << std::endl; return NoError; }
        case -1094: {std::cout <<  "CL_INVALID_ACCELERATOR_INTEL" << std::endl; return NoError; }
        case -1095: {std::cout <<  "CL_INVALID_ACCELERATOR_TYPE_INTEL" << std::endl; return NoError; }
        case -1096: {std::cout <<  "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL" << std::endl; return NoError; }
        case -1097: {std::cout <<  "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL" << std::endl; return NoError; }
        case -1000: {std::cout <<  "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR" << std::endl; return NoError; }
        case -1098: {std::cout <<  "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL" << std::endl; return NoError; }
        case -1099: {std::cout <<  "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL" << std::endl; return NoError; }
        case -1100: {std::cout <<  "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL" << std::endl; return NoError; }
        case -1101: {std::cout <<  "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL" << std::endl; return NoError; }
        default: 	{std::cout <<  "CL_UNKNOWN_ERROR" << std::endl; return NoError; }
    }
}

inline SFStatus ConvertVkFFTError(VkFFTResult res){
    switch (res)
    {
        case VKFFT_SUCCESS:                 {return NoError;}
        case VKFFT_ERROR_FAILED_TO_COPY:    {std::cout <<  "VKFFT_ERROR_FAILED_TO_COPY" << std::endl; return MemError; }
        default: {return SetupError;}
    }
}

#ifdef SinglePrec
    typedef cl_float    cl_real;
    typedef cl_float2   cl_complex;
#endif
#ifdef DoublePrec
    typedef cl_double    cl_real;
    typedef cl_double2   cl_complex;
#endif

static cl_real      CLR0 = 0.;
static cl_complex   CLC0 = []{cl_complex V = {{0., 0.}}; return V;}();
static cl_complex   CLC1 = []{cl_complex V = {{1., 0.}}; return V;}();
// static cl_complex   CLC01 = []{cl_complex V = {{0., 1.}}; return V;}();

//--- Dimension & index structs & functions

typedef unsigned uint;
struct dim3  {uint x, y, z; dim3(int x_ = 1, int y_ = 1, int z_ = 1) : x(x_), y(y_), z(z_) {}};
struct dim3s {int x, y, z; dim3s(int x_ = 1, int y_ = 1, int z_ = 1) : x(x_), y(y_), z(z_) {}};
inline uint GID(const uint &i, const uint &j, const uint &NX, const uint &NY)                                   {return i*NY + j;}
inline uint GID(const uint &i, const uint &j, const uint &k, const uint &NX, const uint &NY, const uint &NZ)    {return i*NY*NZ + j*NZ + k;}
inline uint GID(const uint &i, const uint &j, const uint &k, const dim3 &D)                                     {return i*D.y*D.z + j*D.z + k;}
inline uint GID(const dim3 &P, const dim3 &D)                                                                   {return P.x*D.y*D.z + P.y*D.z + P.z;}
inline uint GID(const uint &i, const uint &j, const uint &NX, const uint &NY, const Dim &D1)
{
    if (D1==EX) return i*NY + j;    // Row-major
    else        return j*NX + i;    // Column-major
}

//--- OpenCL kernels for convolution

const std::string ocl_kernels_float = R"CLC(
typedef float Real;
typedef float2 Complex;
)CLC";

const std::string ocl_kernels_double = R"CLC(
typedef double Real;
typedef double2 Complex;
)CLC";

const std::string ocl_multiply = R"CLC(
__kernel void multiply_in_place(__global Real* A, __global const Real* B) {
    int i = get_global_id(0);
    // int ig = get_global_size(0);
    // if (i<ig){
    Real a_real = A[i];
    Real b_real = B[i];
    Real result = a_real * b_real;
    A[i] = result;
    // }
}
)CLC";

const std::string ocl_complexmultiply = R"CLC(
__kernel void complex_multiply_in_place(__global Complex* A, __global Complex* B) {
    int i = get_global_id(0);
    // int ig = get_global_size(0);
    // if (i<ig){
        Complex a = A[i];
        Complex b = B[i];
        Real res_re = a.x * b.x - a.y * b.y;
        Real res_im = a.x * b.y + a.y * b.x;
        A[i] = (Complex){res_re,res_im};
    // }
}
)CLC";

class DataType_VkFFT : public DataType
{

protected:

    //--- Grid dimension
    dim3 GridDim;

    //--- Memory objects (cpu)
    Real *r_Input1, *r_Output1;
    Real *r_Input2, *r_Output2;
    Real *r_Input3, *r_Output3;

    //--- Memory objects (gpu)
    // In reality these are actually stored on the GPU, we shall use these simply as interfacing arrays.
    cl_mem cl_r_Input1, cl_r_Output1;
    cl_mem cl_r_Input2, cl_r_Output2;
    cl_mem cl_r_Input3, cl_r_Output3;
    cl_mem cl_r_FG;

    cl_mem c_Input1, c_FTInput1, c_FTOutput1, c_Output1, c_FTVel1;
    cl_mem c_Input2, c_FTInput2, c_FTOutput2, c_Output2, c_FTVel2;
    cl_mem c_Input3, c_FTInput3, c_FTOutput3, c_Output3, c_FTVel3;
    cl_mem c_DummyBuffer1, c_DummyBuffer2, c_DummyBuffer3;
    cl_mem c_DummyBuffer4, c_DummyBuffer5, c_DummyBuffer6;
    cl_mem c_FG, c_FGi, c_FGj, c_FGk;
    cl_mem InputBuffer;

    //--- VkFFT Objects
    int FFTDim = 1;
    bool FusedKernel = false; // Is convolution compiled into the FFT kernel? This is a feature of VkFFT
    VkGPU *vkGPU;
    uint64_t bufferSizeX = 0, bufferSizeY = 0, bufferSizeZ = 0, bufferSizeNT = 0, bufferSizeNTM = 0;
    uint64_t c_bufferSizeX = 0, c_bufferSizeY = 0, c_bufferSizeZ = 0, c_bufferSizeNT = 0, c_bufferSizeNTM = 0;
    uint64_t InputbufferSize = 0;

    //--- VkFFT Kernel objects (standard FFT + kernel config)
    VkFFTConfiguration configuration = {}, kernel_configuration = {};
    VkFFTApplication app = {}, kernel_app = {};
    VkFFTLaunchParams launchParams = {}, kernel_launchParams = {};

    //--- Convolution kernels (if not using fused CL kernels)
    cl_program conv_program;
    cl_kernel conv_kernel;
    // void Build_complex_Kernel()

    //--- Memory management
    SFStatus Allocate_Buffer(cl_mem &buffer, uint64_t bufsize);

    //--- Debugging
    VkFFTResult Test_Case();


    //--- OpenCL objects
    // cl_device device;

public:

    //--- Constructor
    DataType_VkFFT()  {}

    void Datatype_Setup()   override;
    VkFFTResult OpenCLSetup(VkGPU* VK);
    void Print_Device_Info(cl_device_id device);

    //--- Specify Plan
    SFStatus Prepare_Plan(VkFFTConfiguration &Config);
    SFStatus Specify_1D_Plan()  override;
    SFStatus Specify_2D_Plan()  override;
    SFStatus Specify_3D_Plan()  override;

    //--- Array allocation
    SFStatus Allocate_Arrays()      override;
    SFStatus Deallocate_Arrays()    override;

    //--- Specify Input
    VkFFTResult ConvertArray_R2C(RVector &I, void* input_buffer, size_t N);
    VkFFTResult ConvertArray_C2R(RVector &I, void* input_buffer, size_t N);
    SFStatus Set_Input(RVector &I) override;
    SFStatus Set_Input(RVector &I1, RVector &I2, RVector &I3)   override;
    SFStatus Set_Input_Unbounded_1D(RVector &I)                 override;
    // virtual SFStatus Set_Input_Unbounded_2D(RVector &I)                             override;
    // virtual SFStatus Set_Input_Unbounded_3D(RVector &I)                             override;
    // virtual SFStatus Set_Input_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3)  override;
    // virtual SFStatus Transfer_Data_Device()                                         override;

    //--- Retrieve output array
    void Get_Output(RVector &I)         override;
    void Get_Output(RVector &I1, RVector &I2, RVector &I3)  override;
    void Get_Output_Unbounded_1D(RVector &I)                override;
    // virtual void Get_Output_Unbounded_2D(RVector &I)                                {}
    // virtual void Get_Output_Unbounded_2D(RVector &I1, RVector &I2)                  {}
    // virtual void Get_Output_Unbounded_3D(RVector &I)                                {}
    // virtual void Get_Output_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3)     {}

    //--- Greens functions prep
    void Prep_Greens_Function(FTType TF);
    void Prep_Greens_Function_R2R()     override    {Prep_Greens_Function(DFT_R2R);};
    void Prep_Greens_Function_C2C()     override    {Prep_Greens_Function(DFT_C2C);};
    void Prep_Greens_Function_R2C()     override    ;
    // void Prepare_Dif_Operators_1D(Real Hx)                      {}
    // void Prepare_Dif_Operators_2D(Real Hx, Real Hy)             {}
    // void Prepare_Dif_Operators_3D(Real Hx, Real Hy, Real Hz)    {}

    //--- Fourier transforms (Note: Backward FFT/iFFT should not be called if Fused Kernel most employed!)
    VkFFTResult FFT_DFT(bool Forward);
    void Forward_FFT_R2R()  override;
    void Backward_FFT_R2R() override;
    void Forward_FFT_DFT()  override;
    void Backward_FFT_DFT() override;
    void Forward_FFT_R2C()  override;
    void Backward_FFT_C2R() override;

    //--- Convolution
    // We can completely avoid the convolution step if we are using a VkFFT backend, as we will exploit the
    // convolution feature of the VkFFT to carry out the convolution in one single step.
    cl_int Convolution();
    void Convolution_Real()     override    {Convolution();}
    void Convolution_Real3()    override;
    void Convolution_Complex()  override    {Convolution();}
    void Convolution_Complex3() override;

    //--- Spectral gradients- these are dummies for now
    void Spectral_Gradients_2D_Grad()       {}
    void Spectral_Gradients_3DV_Curl()      {}
    void Spectral_Gradients_3DV_Nabla()     {}
    void Transfer_FTInOut_Comp()            {}

};

}

#endif

#endif // DATATYPE_VKFFT_H
