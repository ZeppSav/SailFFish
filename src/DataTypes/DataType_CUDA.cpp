//-----------------------------------------------------------------------------
//-------------------------DataType_CUDA Functions-----------------------------
//-----------------------------------------------------------------------------

#include "DataType_CUDA.h"
#include <chrono>

#ifdef CUFFT

namespace SailFFish
{

void DataType_CUDA::Datatype_Setup()
{
    // Setup necessary handles, arrays, and check info

    // How many cuda devices are available on this host?
    int count;
    cudaGetDeviceCount(&count);
    cout << "N Cuda capable devices = " << count << endl;

    // Get device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    if (deviceProp.major == 9999 && deviceProp.minor == 9999)
    {
        cout << "No CUDA GPU has been detected" << endl;
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device Number: %d\n", 0);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

    //--- Specify cuda device
//    cudaSetDevice(0);

    //--- Create cuBLAS handle
    cublasCreate(&cublashandle);

}

//--- Array allocation

SFStatus DataType_CUDA::Allocate_Arrays()
{
    // Depending on the type of solver and the chosen operator/outputs, the arrays which must be allocated vary.
    // This is simply controlled here by specifying the necessary flags during solver initialization

//    cout << "Allocating arrays: NT = " << NT << endl;
//    if (r_in1) cout << "r_in1" << endl;
//    if (r_in2) cout << "r_in2" << endl;
//    if (r_in3) cout << "r_in3" << endl;
//    if (c_in1) cout << "c_in1" << endl;
//    if (c_in2) cout << "c_in2" << endl;
//    if (c_in3) cout << "c_in3" << endl;
//    if (c_fg) cout << "c_fg" << endl;

    // Real-valued arrays

    if (r_in1)      cudaMalloc((void**)&r_Input1, sizeof(CUDAReal)*NT);
    if (r_in2)      cudaMalloc((void**)&r_Input2, sizeof(CUDAReal)*NT);
    if (r_in3)      cudaMalloc((void**)&r_Input3, sizeof(CUDAReal)*NT);

    if (r_ft_in1)   cudaMalloc((void**)&r_FTInput1, sizeof(CUDAReal)*NT);
    if (r_ft_in2)   cudaMalloc((void**)&r_FTInput2, sizeof(CUDAReal)*NT);
    if (r_ft_in3)   cudaMalloc((void**)&r_FTInput3, sizeof(CUDAReal)*NT);

    if (r_out_1)    cudaMalloc((void**)&r_Output1, sizeof(CUDAReal)*NT);
    if (r_out_2)    cudaMalloc((void**)&r_Output2, sizeof(CUDAReal)*NT);
    if (r_out_3)    cudaMalloc((void**)&r_Output3, sizeof(CUDAReal)*NT);

    // Arrays for real Green's function
//    if (r_fg)       cudaMalloc((void**)&r_FG, sizeof(CUDAReal)*NT);
//    if (r_fg)       cudaMalloc((void**)&cu_r_FG, sizeof(CUDAReal)*NT);          // Not necessary!
    if (r_fg)       r_FG = (Real*)malloc(NT*sizeof(Real));

    // Complex-valued arrays
    if (c_in1)      cudaMalloc((void**)&c_Input1, sizeof(CUDAComplex)*NT);
    if (c_in2)      cudaMalloc((void**)&c_Input2, sizeof(CUDAComplex)*NT);
    if (c_in3)      cudaMalloc((void**)&c_Input3, sizeof(CUDAComplex)*NT);

    if (c_ft_in1)   cudaMalloc((void**)&c_FTInput1, sizeof(CUDAComplex)*NTM);
    if (c_ft_in2)   cudaMalloc((void**)&c_FTInput2, sizeof(CUDAComplex)*NTM);
    if (c_ft_in3)   cudaMalloc((void**)&c_FTInput3, sizeof(CUDAComplex)*NTM);

    if (c_out_1)    cudaMalloc((void**)&c_Output1, sizeof(CUDAComplex)*NT);
    if (c_out_2)    cudaMalloc((void**)&c_Output2, sizeof(CUDAComplex)*NT);
    if (c_out_3)    cudaMalloc((void**)&c_Output3, sizeof(CUDAComplex)*NT);

    // Arrays for complex Green's function & spectral operators arrays
    if (c_fg)       cudaMalloc((void**)&c_FG, sizeof(CUDAComplex)*NTM);
    if (c_fg_i)     cudaMalloc((void**)&c_FGi, sizeof(CUDAComplex)*NTM);
    if (c_fg_j)     cudaMalloc((void**)&c_FGj, sizeof(CUDAComplex)*NTM);
    if (c_fg_k)     cudaMalloc((void**)&c_FGk, sizeof(CUDAComplex)*NTM);

    if (c_dbf_1)    cudaMalloc((void**)&c_DummyBuffer1, sizeof(CUDAComplex)*NTM);
    if (c_dbf_2)    cudaMalloc((void**)&c_DummyBuffer2, sizeof(CUDAComplex)*NTM);
    if (c_dbf_3)    cudaMalloc((void**)&c_DummyBuffer3, sizeof(CUDAComplex)*NTM);
    if (c_dbf_4)    cudaMalloc((void**)&c_DummyBuffer4, sizeof(CUDAComplex)*NTM);
    if (c_dbf_5)    cudaMalloc((void**)&c_DummyBuffer5, sizeof(CUDAComplex)*NTM);
    if (c_dbf_6)    cudaMalloc((void**)&c_DummyBuffer6, sizeof(CUDAComplex)*NTM);

    return NoError;
}

SFStatus DataType_CUDA::Deallocate_Arrays()
{
    // Depending on the type of solver and the chosen operator/outputs, the arrays which must be allocated vary.
    // This is simply controlled here by specifying the necessary flags during solver initialization

    // Real-valued arrays
    if (r_in1)      cudaFree(r_Input1);
    if (r_in2)      cudaFree(r_Input2);
    if (r_in3)      cudaFree(r_Input3);

    if (r_ft_in1)   cudaFree(r_FTInput1);
    if (r_ft_in2)   cudaFree(r_FTInput2);
    if (r_ft_in3)   cudaFree(r_FTInput3);

    if (r_out_1)    cudaFree(r_Output1);
    if (r_out_2)    cudaFree(r_Output2);
    if (r_out_3)    cudaFree(r_Output3);

    // Arrays for real Green's function
    if (r_fg)       free(r_FG);

    // Complex-valued arrays
    if (c_in1)      cudaFree(c_Input1);
    if (c_in2)      cudaFree(c_Input2);
    if (c_in3)      cudaFree(c_Input3);

    if (c_ft_in1)   cudaFree(c_FTInput1);
    if (c_ft_in2)   cudaFree(c_FTInput2);
    if (c_ft_in3)   cudaFree(c_FTInput3);

    if (c_out_1)    cudaFree(c_Output1);
    if (c_out_2)    cudaFree(c_Output2);
    if (c_out_3)    cudaFree(c_Output3);

    // Arrays for complex Green's function & spectral operators arrays
    if (c_fg)       cudaFree(c_FG);
    if (c_fg_i)     cudaFree(c_FGi);
    if (c_fg_j)     cudaFree(c_FGj);
    if (c_fg_k)     cudaFree(c_FGk);

    if (c_dbf_1)    cudaFree(c_DummyBuffer1);
    if (c_dbf_2)    cudaFree(c_DummyBuffer2);
    if (c_dbf_3)    cudaFree(c_DummyBuffer3);
    if (c_dbf_4)    cudaFree(c_DummyBuffer4);
    if (c_dbf_5)    cudaFree(c_DummyBuffer5);
    if (c_dbf_6)    cudaFree(c_DummyBuffer6);

    return NoError;
}

//--- Specify Plan

SFStatus DataType_CUDA::Specify_1D_Plan()
{
    // This specifies the forward and backward plans
    if (Transform==DCT1 || Transform==DCT2 || Transform==DST1 || Transform==DST2)
    {
        cout << "DataType_CUDA::Specify_1D_Plan(): Real-to-real transforms is not supported by cuFFT. \n SailFFish must be linked with the FFTW library." << endl;
        return SetupError;
    }
    if (Transform==DFT_C2C)
    {
        BFac = 1.0/NT;
        cufftPlan1d(&FFT_Plan,NX,cufft_C2C, 1);
    }
    if (Transform==DFT_R2C)
    {
        BFac = 1.0/NT;
        cufftPlan1d(&Forward_Plan,NX,cufft_R2C,1);
        cufftPlan1d(&Backward_Plan,NX,cufft_C2R,1);
//        NXM = NX/2+1;
    }
    return NoError;
}

SFStatus DataType_CUDA::Specify_2D_Plan()
{
    // This specifies the forward and backward plans
    if (Transform==DCT1 || Transform==DCT2 || Transform==DST1 || Transform==DST2)
    {
        cout << "DataType_CUDA::Specify_2D_Plan(): Real-to-real transforms is not supported by cuFFT. \n SailFFish must be linked with the FFTW library." << endl;
        return SetupError;
    }
    if (Transform==DFT_C2C)
    {
        BFac = 1.0/NT;
        cufftPlan2d(&FFT_Plan,NX,NY,cufft_C2C);
    }
    if (Transform==DFT_R2C)
    {
        BFac = 1.0/NT;
        cufftPlan2d(&Forward_Plan,NX,NY,cufft_R2C);
        cufftPlan2d(&Backward_Plan,NX,NY,cufft_C2R);
//        NYM = NY/2+1;
    }
    return NoError;
}

SFStatus DataType_CUDA::Specify_3D_Plan()
{
    // This specifies the forward and backward plans

    if (Transform==DCT1 || Transform==DCT2 || Transform==DST1 || Transform==DST2)
    {
        cout << "DataType_CUDA::Specify_3D_Plan(): Real-to-real transforms is not supported by cuFFT. \n SailFFish must be linked with the FFTW library." << endl;
        return SetupError;
    }
    if (Transform==DFT_C2C)
    {
        BFac = 1.0/NT;
        cufftPlan3d(&FFT_Plan,NX,NY,NZ,cufft_C2C);
    }
    if (Transform==DFT_R2C)
    {
        BFac = 1.0/NT;
        cufftPlan3d(&Forward_Plan,NX,NY,NZ,cufft_R2C);
        cufftPlan3d(&Backward_Plan,NX,NY,NZ,cufft_C2R);
//        NZM = NZ/2+1;
    }
    return NoError;
}

//--- Greens functions prep

void DataType_CUDA::Prep_Greens_Function_C2C()
{
    // This is a hack to avoid using c_FG array within the solver class.
    // The Green's function is specified within the r_FG array and then converted here.
    // We need to generate first an array of std::complex, and then copy this memory to the cuda device

    // In reality this should be able to accept complex green's functions at some point

    CVector G = CVector(NT,ComplexNull);
    for (int i=0; i<NT; i++) {G[i].real(r_FG[i]);}
    cudaMemcpy(c_FG, G.data(), NT*sizeof(CUDAComplex), cudaMemcpyHostToDevice);
}

void DataType_CUDA::Prep_Greens_Function_R2C()
{
    // This function assumes that the greens function is purely real and undergoing an R2C transform.
    // We therefore only need to transform it into a cuda buffer and then carry out a forward FFT.

    // Copy real array over
    cudaMemcpy(r_Input1, r_FG, NT*sizeof(Real), cudaMemcpyHostToDevice);   // Just copy over

    // Now carry out forward FFT to get c_FG in real space...
    cufftResult R = cufft_Execute_R2C(Forward_Plan, r_Input1, c_FG);
    if (R != CUFFT_SUCCESS) {fprintf(stderr,"CUFFT Error: %s\n", cudaGetErrorString(cudaError_t(R))); return;}
}

void DataType_CUDA::Prepare_Dif_Operators_1D(Real Hx)
{
    // This prepares the differential operators in frequency space
//    c_FGi = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NX);

//    for (int i=0; i<NX; i++) {
//        if (2*i<NX)     c_FGi[i][1] = M_2PI*i/Hx/NX;
//        else            c_FGi[i][1] = M_2PI*(i-NX)/Hx/NX;
//        c_FGi[i][0] = 0.0;
//    }
}

void DataType_CUDA::Prepare_Dif_Operators_2D(Real Hx, Real Hy)
{
    // This prepares the differential operators in frequency space
    // A more suitable approach to this would be to write a corresponding kernel for this.
    // This would significantly reduce the memory overhead.

    CVector CI = CVector(NT,ComplexNull);
    CVector CJ = CVector(NT,ComplexNull);

    OpenMPfor
    for (int i=0; i<NXM; i++) {
        Real xfac;
        if (2*i<NX)     xfac = M_2PI*i/Hx/NX;
        else            xfac = M_2PI*(i-NX)/Hx/NX;
        for (int j=0; j<NYM; j++) {
            Real yfac;
            if (2*j<NY)     yfac = M_2PI*j/Hy/NY;
            else            yfac = M_2PI*(j-NY)/Hy/NY;

            CI[i*NYM+j].imag(xfac);
            CJ[i*NYM+j].imag(yfac);
        }
    }

    // Transfer to cuda buffers c_FGi, c_FGj
    cudaMemcpy(c_FGi, CI.data(), (NTM)*sizeof(CUDAComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(c_FGj, CJ.data(), (NTM)*sizeof(CUDAComplex), cudaMemcpyHostToDevice);
}

void DataType_CUDA::Prepare_Dif_Operators_3D(Real Hx, Real Hy, Real Hz)
{
    // This prepares the differential operators in frequency space
    // A more suitable approach to this would be to write a corresponding kernel for this.
    // This would significantly reduce the memory overhead.

    CVector CI = CVector(NT,ComplexNull);
    CVector CJ = CVector(NT,ComplexNull);
    CVector CK = CVector(NT,ComplexNull);

    OpenMPfor
    for (int i=0; i<NXM; i++) {
        Real xfac;
        if (2*i<NX)     xfac = M_2PI*i/Hx/NX;
        else            xfac = M_2PI*(i-NX)/Hx/NX;
        for (int j=0; j<NYM; j++) {
            Real yfac;
            if (2*j<NY)     yfac = M_2PI*j/Hy/NY;
            else            yfac = M_2PI*(j-NY)/Hy/NY;
            for (int k=0; k<NZM; k++) {
                Real zfac;
                if (2*k<NZ)     zfac = M_2PI*k/Hz/NZ;
                else            zfac = M_2PI*(k-NZ)/Hz/NZ;

                int id = i*NYM*NZM + j*NZM + k;
                CI[id].imag(xfac);
                CJ[id].imag(yfac);
                CK[id].imag(zfac);
            }
        }
    }

    // Transfer to cuda buffers c_FGi, c_FGj
    cudaMemcpy(c_FGi, CI.data(), (NTM)*sizeof(CUDAComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(c_FGj, CJ.data(), (NTM)*sizeof(CUDAComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(c_FGk, CK.data(), (NTM)*sizeof(CUDAComplex), cudaMemcpyHostToDevice);
}

void DataType_CUDA::Spectral_Gradients_1D()
{
    // The convolution of the input signal with the greens function was carried out in a previous step (DataType_CUDA::Convolution_Complex()).
    // If the gradients are to be calculated, the spectral differentiation is carried out here.
    // This is done in-place

    if (Operator==NONE)     return;
    if (Operator==CURL )    return; // Invalid in 1D
    if (Operator==DIV)      return; // Just gradient in 1D

    cout << "DataType_CUDA::Spectral_Gradients_1D() Not yet implemented!!!!" << endl;

//    if (Operator==GRAD)
//    {
//        OpenMPfor
//        for (int i=0; i<NXM; i++)   Multiply(c_FGi[i],c_FTInput1[i],c_FTInput1[i]);
//    }

//    if (Operator==NABLA)
//    {
//        OpenMPfor
//        for (int i=0; i<NXM; i++){
//                Multiply(c_FGi[i],c_FTInput1[i],c_FTInput1[i]);
//                Multiply(c_FGi[i],c_FTInput1[i],c_FTInput1[i]);
//        }
//    }
}

void DataType_CUDA::Spectral_Gradients_2D()
{
    // The convolution of the input signal with the greens function was carried out in a previous step
    // If the gradients are to be calculated, the spectral differentiation is carried out here.
    // This is done in-place to reduce data footprint

    if (Operator==NONE) return;

    if (Operator==GRAD)
    {
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGj, 1, c_FTInput2, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGi, 1, c_FTInput1, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
    }

    if (Operator==DIV)
    {
        cout << "DataType_CUDA::Spectral_Gradients_2D()-> Div option untested!" << endl;

        // Find gradients
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGj, 1, c_FTInput2, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGi, 1, c_FTInput1, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
        // Now add together
        cuComplex Alpha = {1.0,0.0};
        cublas_axpy(cublashandle, NTM, &Alpha, c_FTInput2, 1, c_FTInput1, 1);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
    }

    if (Operator==CURL)
    {
        cout << "DataType_CUDA::Spectral_Gradients_2D()-> Curl option untested!" << endl;

        // Find gradients
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGj, 1, c_FTInput2, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGi, 1, c_FTInput1, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
        // Transfer Uy to dummy buffer
        cudaMemcpy(c_DummyBuffer1,c_FTInput1,NTM*sizeof(CUDAComplex),cudaMemcpyDeviceToDevice);
        cuComplex Alpha = {-1.0,0.0};
        cublas_axpy(cublashandle, NTM, &Alpha, c_FTInput2, 1, c_FTInput1, 1);                       // X velocity
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
        cudaMemcpy(c_FTInput2,c_DummyBuffer1,NTM*sizeof(CUDAComplex),cudaMemcpyDeviceToDevice);     // Y velocity
    }

    if (Operator==NABLA)
    {
        cout << "DataType_CUDA::Spectral_Gradients_2D()-> Nabla option untested!" << endl;

        // Find gradients
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGj, 1, c_FTInput2, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput2, NTM, c_FGj, 1, c_FTInput2, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGi, 1, c_FTInput1, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGi, 1, c_FTInput1, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
    }
}

void DataType_CUDA::Spectral_Gradients_3D()
{
    // The convolution of the input signal with the greens function was carried out in a previous step (DataType_CUDA::Convolution_Complex()).
    // If the gradients are to be calculated, the spectral differentiation is carried out here.
    // This is done in-place

    if (Operator==NONE) return;

    if (Operator==GRAD)
    {
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGk, 1, c_FTInput3, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGj, 1, c_FTInput2, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGi, 1, c_FTInput1, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
    }

    if (Operator==DIV)
    {
        cout << "DataType_CUDA::Spectral_Gradients_3D()-> Div option untested!" << endl;

        // Find gradients
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGk, 1, c_FTInput3, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGj, 1, c_FTInput2, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGi, 1, c_FTInput1, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
        // Now add together
        cuComplex Alpha = {1.0,0.0};
        cublas_axpy(cublashandle, NTM, &Alpha, c_FTInput3, 1, c_FTInput2, 1);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
        cublas_axpy(cublashandle, NTM, &Alpha, c_FTInput2, 1, c_FTInput1, 1);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_2D Failed \n");  return;}
    }

    cout << "DataType_CUDA::Spectral_Gradients_3D() Not yet implemented!!!!" << endl;
}

void DataType_CUDA::Spectral_Gradients_3DV()
{
    // The convolution of the input signal with the greens function was carried out in a previous step (DataType_CUDA::Convolution_Complex()).
    // If the gradients are to be calculated, the spectral differentiation is carried out here.
    // This is done in-place

    if (Operator==NONE) return;

    if (Operator==DIV)
    {
        cout << "DataType_CUDA::Spectral_Gradients_3DV()-> Div option untested!" << endl;

        // Find gradients
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGi, 1, c_FTInput1, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput2, NTM, c_FGj, 1, c_FTInput2, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput3, NTM, c_FGk, 1, c_FTInput3, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}

        // Now add together
        cuComplex Alpha = {1.0,0.0};
        cublas_axpy(cublashandle, NTM, &Alpha, c_FTInput3, 1, c_FTInput2, 1);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}
        cublas_axpy(cublashandle, NTM, &Alpha, c_FTInput2, 1, c_FTInput1, 1);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}
    }

    if (Operator==GRAD)
    {
        // Ignore this case for now. We would require 9 arrays to store the outputs
        // Will be useful at some point, but not right now!
        cout << "DataType_CUDA::Spectral_Gradients_3DV() Grad option not yet implemented!!!!" << endl;
    }

    if (Operator==CURL)
    {
        // Velocity field extraction for 3D case! The holy grail!
        // This is currently achieved with an extremely ugly (in terms of memory) approach.
        // The better approach would be to write a cuda Kernel for this task.

        cuComplex Alpha = {-1.0,0.0};

        // Intermediate vars
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput3, NTM, c_FGj, 1, c_DummyBuffer2, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput2, NTM, c_FGk, 1, c_DummyBuffer1, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}
        cublas_axpy(cublashandle, NTM, &Alpha, c_DummyBuffer2, 1, c_DummyBuffer1, 1); // Add together (X velocity c_DummyBuffer1)
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}

        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGk, 1, c_DummyBuffer3, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput3, NTM, c_FGi, 1, c_DummyBuffer2, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}
        cublas_axpy(cublashandle, NTM, &Alpha, c_DummyBuffer3, 1, c_DummyBuffer2, 1); // Add together (Y velocity c_DummyBuffer2)
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}


        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput2, NTM, c_FGi, 1, c_DummyBuffer4, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGj, 1, c_DummyBuffer3, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}
        cublas_axpy(cublashandle, NTM, &Alpha, c_DummyBuffer4, 1, c_DummyBuffer3, 1); // Add together (Z velocity c_DummyBuffer3)
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}

        // Transfer back to appropriate positions
        cudaMemcpy(c_FTInput1,c_DummyBuffer1, NTM*sizeof(CUDAComplex), cudaMemcpyDeviceToDevice);     // X velocity
        cudaMemcpy(c_FTInput2,c_DummyBuffer2, NTM*sizeof(CUDAComplex), cudaMemcpyDeviceToDevice);     // Y velocity
        cudaMemcpy(c_FTInput3,c_DummyBuffer3, NTM*sizeof(CUDAComplex), cudaMemcpyDeviceToDevice);     // Z velocity
    }

    if (Operator==NABLA)
    {
        cout << "DataType_CUDA::Spectral_Gradients_3DV()-> Nabla option untested!" << endl;

        // Find gradients
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGi, 1, c_FTInput1, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FGi, 1, c_FTInput1, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}

        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput2, NTM, c_FGj, 1, c_FTInput2, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput2, NTM, c_FGj, 1, c_FTInput2, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}

        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput3, NTM, c_FGk, 1, c_FTInput3, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}
        cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput3, NTM, c_FGk, 1, c_FTInput3, NTM);
        if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Spectral_Gradients_3DV Failed \n");  return;}
    }

}

//--- Convolution

void DataType_CUDA::Convolution_Complex()
{
    // The multiplication will be carried out by using a modified cuBLAS tier-2 routine
    cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FG, 1, c_FTInput1, NTM);
    if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Convolution_Complex 1 Failed \n");  return;}
}

void DataType_CUDA::Convolution_Complex3()
{
    // The multiplication will be carried out by using a modified cuBLAS tier-2 routine
    // This is done in-place as with FFTW

    cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput1, NTM, c_FG, 1, c_FTInput1, NTM);
    if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Convolution_Complex 1 Failed \n");  return;}
    cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput2, NTM, c_FG, 1, c_FTInput2, NTM);
    if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Convolution_Complex 1 Failed \n");  return;}
    cublas_dgmm(cublashandle, CUBLAS_SIDE_LEFT, NTM, 1, c_FTInput3, NTM, c_FG, 1, c_FTInput3, NTM);
    if (cudaGetLastError() != cudaSuccess){ fprintf(stderr, "DataType_CUDA::Convolution_Complex 1 Failed \n");  return;}
}

//--- Prepare input array

SFStatus DataType_CUDA::Set_Input(RVector &I)
{
    // Transfers input array to cuda device
    if (int(I.size())!=NT){
        cout << "Input array has incorrect dimension." << endl;
        return DimError;
    }

    if (r_in1)   cudaMemcpy(r_Input1, I.data(), (NT)*sizeof(Real), cudaMemcpyHostToDevice);   // Just copy over
    if (c_in1){
        // Need to generate dummy complex array and then transfer data over
        CVector IC = CVector(NT,ComplexNull);
        for (int i=0; i<NT; i++) {IC[i].real(I[i]);}
        cudaMemcpy(c_Input1, IC.data(), NT*sizeof(CUDAComplex), cudaMemcpyHostToDevice);
    }
    return NoError;
}

SFStatus DataType_CUDA::Set_Input(RVector &I1, RVector &I2, RVector &I3)
{
    // Transfers input array to cuda device
    if (int(I1.size())!=NT || int(I2.size())!=NT || int(I3.size())!=NT){
        cout << "Input arrays have incorrect dimension." << endl;
        return DimError;
    }

    if (r_in1)   cudaMemcpy(r_Input1, I1.data(), (NT)*sizeof(Real), cudaMemcpyHostToDevice);   // Just copy over
    if (r_in2)   cudaMemcpy(r_Input2, I2.data(), (NT)*sizeof(Real), cudaMemcpyHostToDevice);   // Just copy over
    if (r_in3)   cudaMemcpy(r_Input3, I3.data(), (NT)*sizeof(Real), cudaMemcpyHostToDevice);   // Just copy over
    if (c_in1){
        // Need to generate dummy complex array and then transfer data over
        CVector IC = CVector(NT,ComplexNull);
        for (int i=0; i<NT; i++) {IC[i].real(I1[i]);}
        cudaMemcpy(c_Input1, IC.data(), NT*sizeof(CUDAComplex), cudaMemcpyHostToDevice);
    }
    if (c_in2){
        // Need to generate dummy complex array and then transfer data over
        CVector IC = CVector(NT,ComplexNull);
        for (int i=0; i<NT; i++) {IC[i].real(I2[i]);}
        cudaMemcpy(c_Input2, IC.data(), NT*sizeof(CUDAComplex), cudaMemcpyHostToDevice);
    }
    if (c_in3){
        // Need to generate dummy complex array and then transfer data over
        CVector IC = CVector(NT,ComplexNull);
        for (int i=0; i<NT; i++) {IC[i].real(I3[i]);}
        cudaMemcpy(c_Input3, IC.data(), NT*sizeof(CUDAComplex), cudaMemcpyHostToDevice);
    }
    return NoError;
}

SFStatus DataType_CUDA::Set_Input_Unbounded_1D(RVector &I)
{
    // This function takes the input vector and stores this in the appropriate cuda input array
    int NXH = NX/2;
    if (int(I.size())!=NXH){
        cout << "Input array has incorrect dimension." << endl;
        return DimError;
    }

    // Create dummy arrays to pass in one block to cuda buffers
    RVector R1;
    CVector C1;
    if (r_in1) R1 = RVector(NT,0);
    if (c_in1) C1 = CVector(NT,ComplexNull);

    // Fill nonzero elements of dummy arrays
    if (r_in1)  {for (int i=0; i<NXH; i++) R1[i] = I[i];}
    if (c_in1)  {for (int i=0; i<NXH; i++) C1[i].real(I[i]);}

    // Now transfer block arrays to cuda buffers
    if (r_in1) cudaMemcpy(r_Input1, R1.data(), NT*sizeof(CUDAReal), cudaMemcpyHostToDevice);
    if (c_in1) cudaMemcpy(c_Input1, C1.data(), NT*sizeof(CUDAComplex), cudaMemcpyHostToDevice);

    return NoError;
}

SFStatus DataType_CUDA::Set_Input_Unbounded_2D(RVector &I)
{
    // This function takes the input vector and stores this in the appropriate cuda input array
    int NXH = NX/2;
    int NYH = NY/2;
    if (int(I.size())!=NXH*NYH){
        cout << "Input array has incorrect dimension." << endl;
        return DimError;
    }

    // Create dummy arrays to pass in one block to cuda buffers
    RVector R1;
    CVector C1;
    if (r_in1) R1 = RVector(NT,0);
    if (c_in1) C1 = CVector(NT,ComplexNull);

    // Fill nonzero elements of dummy arrays
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            int idg = i*NY + j;
            int idl = i*NYH + j;
            if (r_in1) R1[idg] = I[idl];
            if (c_in1) C1[idg].real(I[idl]);
        }
    }

    // Now transfer block arrays to cuda buffers
    if (r_in1) cudaMemcpy(r_Input1, R1.data(), NT*sizeof(CUDAReal), cudaMemcpyHostToDevice);
    if (c_in1) cudaMemcpy(c_Input1, C1.data(), NT*sizeof(CUDAComplex), cudaMemcpyHostToDevice);

    return NoError;
}

SFStatus DataType_CUDA::Set_Input_Unbounded_3D(RVector &I)
{
    // This function takes the input vector and stores this in the appropriate cuda input array
    int NXH = NX/2;
    int NYH = NY/2;
    int NZH = NZ/2;
    if (int(I.size())!=NXH*NYH*NZH){
        cout << "Input array has incorrect dimension." << endl;
        return DimError;
    }

    // Create dummy arrays to pass in one block to cuda buffers
    RVector R1;
    CVector C1;
    if (r_in1) R1 = RVector(NT,0);
    if (c_in1) C1 = CVector(NT,ComplexNull);

    // Fill nonzero elements of dummy arrays
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            for (int k=0; k<NZH; k++){
                int idg = i*NY*NZ + j*NZ + k;
                int idl = i*NYH*NZH + j*NZH + k;
                if (r_in1) R1[idg] = I[idl];
                if (c_in1) C1[idg].real(I[idl]);
            }
        }
    }

    // Now transfer block arrays to cuda buffers
    if (r_in1) cudaMemcpy(r_Input1, R1.data(), NT*sizeof(CUDAReal), cudaMemcpyHostToDevice);
    if (c_in1) cudaMemcpy(c_Input1, C1.data(), NT*sizeof(CUDAComplex), cudaMemcpyHostToDevice);

    return NoError;
}

SFStatus DataType_CUDA::Set_Input_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3)
{
    // This function takes the input vector and stores this in the appropriate cuda input array
    int NXH = NX/2;
    int NYH = NY/2;
    int NZH = NZ/2;
    if (int(I1.size())!=NXH*NYH*NZH || int(I2.size())!=NXH*NYH*NZH || int(I3.size())!=NXH*NYH*NZH){
        cout << "Input array has incorrect dimension." << endl;
        return DimError;
    }

    // Create dummy arrays to pass in one block to cuda buffers
    RVector R1, R2, R3;
    CVector C1, C2, C3;
    if (r_in1) R1 = RVector(NT,0);
    if (r_in2) R2 = RVector(NT,0);
    if (r_in3) R3 = RVector(NT,0);
    if (c_in1) C1 = CVector(NT,ComplexNull);
    if (c_in2) C2 = CVector(NT,ComplexNull);
    if (c_in3) C3 = CVector(NT,ComplexNull);

    // Fill nonzero elements of dummy arrays
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            for (int k=0; k<NZH; k++){
                int idg = i*NY*NZ + j*NZ + k;
                int idl = i*NYH*NZH + j*NZH + k;
                if (r_in1) R1[idg] = I1[idl];
                if (r_in2) R2[idg] = I2[idl];
                if (r_in3) R3[idg] = I3[idl];
                if (c_in1) C1[idg].real(I1[idl]);
                if (c_in2) C2[idg].real(I2[idl]);
                if (c_in3) C3[idg].real(I3[idl]);
            }
        }
    }

    // Now transfer block arrays to cuda buffers
    if (r_in1) cudaMemcpy(r_Input1, R1.data(), NT*sizeof(CUDAReal), cudaMemcpyHostToDevice);
    if (r_in2) cudaMemcpy(r_Input2, R2.data(), NT*sizeof(CUDAReal), cudaMemcpyHostToDevice);
    if (r_in3) cudaMemcpy(r_Input3, R3.data(), NT*sizeof(CUDAReal), cudaMemcpyHostToDevice);
    if (c_in1) cudaMemcpy(c_Input1, C1.data(), NT*sizeof(CUDAComplex), cudaMemcpyHostToDevice);
    if (c_in2) cudaMemcpy(c_Input2, C2.data(), NT*sizeof(CUDAComplex), cudaMemcpyHostToDevice);
    if (c_in3) cudaMemcpy(c_Input3, C3.data(), NT*sizeof(CUDAComplex), cudaMemcpyHostToDevice);

    return NoError;
}

//--- Prepare output array

void DataType_CUDA::Get_Output(RVector &I)
{
    // This function converts the output array into an easily accesible format
    if (I.empty()) I.assign(NT,0);
    if (r_out_1)    cudaMemcpy(r_Output1, I.data(), NT*sizeof(CUDAReal), cudaMemcpyDeviceToHost);
    if (c_out_1){
        CVector C = CVector(NT,ComplexNull);    // Need to create intermediate array
        cudaMemcpy(C.data(), c_Output1, NT*sizeof(CUDAComplex), cudaMemcpyDeviceToHost);
        for (int i=0; i<NT; i++)    I[i]= C[i].real();
    }
}

void DataType_CUDA::Get_Output(RVector &I1, RVector &I2, RVector &I3)
{
    // This function converts the output array into an easily accesible format
    if (I1.empty()) I1.assign(NT,0);
    if (I2.empty()) I2.assign(NT,0);
    if (I3.empty()) I3.assign(NT,0);
    if (r_out_1)    cudaMemcpy(I1.data(), r_Output1, NT*sizeof(CUDAReal), cudaMemcpyDeviceToHost);
    if (r_out_2)    cudaMemcpy(I2.data(), r_Output2, NT*sizeof(CUDAReal), cudaMemcpyDeviceToHost);
    if (r_out_3)    cudaMemcpy(I3.data(), r_Output3, NT*sizeof(CUDAReal), cudaMemcpyDeviceToHost);
    if (c_out_1){
        CVector C = CVector(NT,ComplexNull);    // Need to create intermediate array
        cudaMemcpy(C.data(), c_Output1, NT*sizeof(CUDAComplex), cudaMemcpyDeviceToHost);
        for (int i=0; i<NT; i++)    I1[i]= C[i].real();
    }
    if (c_out_2){
        CVector C = CVector(NT,ComplexNull);    // Need to create intermediate array
        cudaMemcpy(C.data(), c_Output2, NT*sizeof(CUDAComplex), cudaMemcpyDeviceToHost);
        for (int i=0; i<NT; i++)    I2[i]= C[i].real();
    }
    if (c_out_3){
        CVector C = CVector(NT,ComplexNull);    // Need to create intermediate array
        cudaMemcpy(C.data(), c_Output3, NT*sizeof(CUDAComplex), cudaMemcpyDeviceToHost);
        for (int i=0; i<NT; i++)    I3[i]= C[i].real();
    }
}

void DataType_CUDA::Get_Output_Unbounded_1D(RVector &I)
{
    // Extracts appropriate output
    int NXH = NX/2;
    if (I.empty()) I.assign(NXH,0);

    // Create dummy arrays if required
    RVector R1;
    CVector C1;
    if (r_out_1) R1 = RVector(NT,0);
    if (c_out_1) C1 = CVector(NT,ComplexNull);

    // Copy memory from cuda buffer
    if (r_out_1) cudaMemcpy(R1.data(), r_Output1, NT*sizeof(CUDAReal), cudaMemcpyDeviceToHost);
    if (c_out_1) cudaMemcpy(C1.data(), c_Output1, NT*sizeof(CUDAComplex), cudaMemcpyDeviceToHost);

    // Copy necessary memory into output arrays
    OpenMPfor
    for (int i=0; i<NXH; i++){
        if (r_out_1) I[i] = R1[i];
        if (c_out_1) I[i] = C1[i].real();
    }
}

void DataType_CUDA::Get_Output_Unbounded_2D(RVector &I)
{
    // Extracts appropriate output
    int NXH = NX/2;
    int NYH = NY/2;
    if (I.empty()) I.assign(NXH*NYH,0);

    // Create dummy arrays if required
    RVector R1;
    CVector C1;
    if (r_out_1) R1 = RVector(NT,0);
    if (c_out_1) C1 = CVector(NT,ComplexNull);

    // Copy memory from cuda buffer
    if (r_out_1) cudaMemcpy(R1.data(), r_Output1, NT*sizeof(CUDAReal), cudaMemcpyDeviceToHost);
    if (c_out_1) cudaMemcpy(C1.data(), c_Output1, NT*sizeof(CUDAComplex), cudaMemcpyDeviceToHost);

    // Copy necessary memory into output arrays
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            int idg = i*NY + j;
            int idl = i*NYH + j;
            if (r_out_1) I[idl] = R1[idg];
            if (c_out_1) I[idl] = C1[idg].real();
        }
    }
}

void DataType_CUDA::Get_Output_Unbounded_2D(RVector &I1, RVector &I2)
{
    // Extracts appropriate output
    int NXH = NX/2;
    int NYH = NY/2;
    if (I1.empty()) I1.assign(NXH*NYH,0);
    if (I2.empty()) I2.assign(NXH*NYH,0);

    // Create dummy arrays if required
    RVector R1, R2;
    CVector C1, C2;
    if (r_out_1) R1 = RVector(NT,0);
    if (r_out_2) R2 = RVector(NT,0);
    if (c_out_1) C1 = CVector(NT,ComplexNull);
    if (c_out_2) C2 = CVector(NT,ComplexNull);

    // Copy memory from cuda buffer
    if (r_out_1) cudaMemcpy(R1.data(), r_Output1, NT*sizeof(CUDAReal), cudaMemcpyDeviceToHost);
    if (r_out_2) cudaMemcpy(R2.data(), r_Output2, NT*sizeof(CUDAReal), cudaMemcpyDeviceToHost);
    if (c_out_1) cudaMemcpy(C1.data(), c_Output1, NT*sizeof(CUDAComplex), cudaMemcpyDeviceToHost);
    if (c_out_2) cudaMemcpy(C2.data(), c_Output2, NT*sizeof(CUDAComplex), cudaMemcpyDeviceToHost);

    // Copy necessary memory into output arrays
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            int idg = i*NY + j;
            int idl = i*NYH + j;
            if (r_out_1) I1[idl] = R1[idg];
            if (r_out_2) I2[idl] = R2[idg];
            if (c_out_1) I1[idl] = C1[idg].real();
            if (c_out_2) I2[idl] = C2[idg].real();
        }
    }
}

void DataType_CUDA::Get_Output_Unbounded_3D(RVector &I)
{
    // Extracts appropriate output
    int NXH = NX/2;
    int NYH = NY/2;
    int NZH = NZ/2;
    if (I.empty()) I.assign(NXH*NYH*NZH,0);

    // Create dummy arrays if required
    RVector R1;
    CVector C1;
    if (r_out_1) R1 = RVector(NT,0);
    if (c_out_1) C1 = CVector(NT,ComplexNull);

    // Copy memory from cuda buffer
    if (r_out_1) cudaMemcpy(R1.data(), r_Output1, NT*sizeof(CUDAReal), cudaMemcpyDeviceToHost);
    if (c_out_1) cudaMemcpy(C1.data(), c_Output1, NT*sizeof(CUDAComplex), cudaMemcpyDeviceToHost);

    // Copy necessary memory into output arrays
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            for (int k=0; k<NZH; k++){
                int idg = i*NY*NZ + j*NZ + k;
                int idl = i*NYH*NZH + j*NZH + k;
                if (r_out_1) I[idl] = R1[idg];
                if (c_out_1) I[idl] = C1[idg].real();
            }
        }
    }
}

void DataType_CUDA::Get_Output_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3)
{
    // Extracts appropriate output
    int NXH = NX/2;
    int NYH = NY/2;
    int NZH = NZ/2;
    if (I1.empty()) I1.assign(NXH*NYH*NZH,0);
    if (I2.empty()) I2.assign(NXH*NYH*NZH,0);
    if (I3.empty()) I3.assign(NXH*NYH*NZH,0);

    // Create dummy arrays if required
    RVector R1, R2, R3;
    CVector C1, C2, C3;
    if (r_out_1) R1 = RVector(NT,0);
    if (r_out_2) R2 = RVector(NT,0);
    if (r_out_3) R3 = RVector(NT,0);
    if (c_out_1) C1 = CVector(NT,ComplexNull);
    if (c_out_2) C2 = CVector(NT,ComplexNull);
    if (c_out_3) C3 = CVector(NT,ComplexNull);

    // Copy memory from cuda buffer
    if (r_out_1) cudaMemcpy(R1.data(), r_Output1, NT*sizeof(CUDAReal), cudaMemcpyDeviceToHost);
    if (r_out_2) cudaMemcpy(R2.data(), r_Output2, NT*sizeof(CUDAReal), cudaMemcpyDeviceToHost);
    if (r_out_3) cudaMemcpy(R3.data(), r_Output3, NT*sizeof(CUDAReal), cudaMemcpyDeviceToHost);
    if (c_out_1) cudaMemcpy(C1.data(), c_Output1, NT*sizeof(CUDAComplex), cudaMemcpyDeviceToHost);
    if (c_out_2) cudaMemcpy(C2.data(), c_Output2, NT*sizeof(CUDAComplex), cudaMemcpyDeviceToHost);
    if (c_out_3) cudaMemcpy(C3.data(), c_Output3, NT*sizeof(CUDAComplex), cudaMemcpyDeviceToHost);

    // Copy necessary memory into output arrays
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            for (int k=0; k<NZH; k++){
                int idg = i*NY*NZ + j*NZ + k;
                int idl = i*NYH*NZH + j*NZH + k;
                if (r_out_1) I1[idl] = R1[idg];
                if (r_out_2) I2[idl] = R2[idg];
                if (r_out_3) I3[idl] = R3[idg];
                if (c_out_1) I1[idl] = C1[idg].real();
                if (c_out_2) I2[idl] = C2[idg].real();
                if (c_out_3) I3[idl] = C3[idg].real();
            }
        }
    }
}

//--- Fourier transforms

void DataType_CUDA::Forward_FFT_DFT()
{
    // Carry out forward FFTs
    cufftResult R = CUFFT_SUCCESS;
    if (c_in1 && c_ft_in1)  R = cufft_Execute_C2C(FFT_Plan, c_Input1, c_FTInput1, CUFFT_FORWARD);
    if (R != CUFFT_SUCCESS) {fprintf(stderr,"CUFFT Error: %s\n", cudaGetErrorString(cudaError_t(R))); return;}
    if (cudaDeviceSynchronize() != cudaSuccess){fprintf(stderr, "CUFFT error: Failed to synchronize\n"); return;}

    if (c_in2 && c_ft_in2)  R = cufft_Execute_C2C(FFT_Plan, c_Input2, c_FTInput2, CUFFT_FORWARD);
    if (R != CUFFT_SUCCESS) {fprintf(stderr,"CUFFT Error: %s\n", cudaGetErrorString(cudaError_t(R))); return;}
    if (cudaDeviceSynchronize() != cudaSuccess){fprintf(stderr, "CUFFT error: Failed to synchronize\n"); return;}

    if (c_in3 && c_ft_in3)  R = cufft_Execute_C2C(FFT_Plan, c_Input3, c_FTInput3, CUFFT_FORWARD);
    if (R != CUFFT_SUCCESS) {fprintf(stderr,"CUFFT Error: %s\n", cudaGetErrorString(cudaError_t(R))); return;}
    if (cudaDeviceSynchronize() != cudaSuccess){fprintf(stderr, "CUFFT error: Failed to synchronize\n"); return;}
}

void DataType_CUDA::Backward_FFT_DFT()
{
    // Carry out backward FFTs
    cufftResult R = CUFFT_SUCCESS;
    if (c_out_1 && c_ft_in1)  R = cufft_Execute_C2C(FFT_Plan, c_FTInput1, c_Output1, CUFFT_INVERSE);
    if (R != CUFFT_SUCCESS) {fprintf(stderr,"CUFFT Error: %s\n", cudaGetErrorString(cudaError_t(R))); return;}
    if (cudaDeviceSynchronize() != cudaSuccess){fprintf(stderr, "CUFFT error: Failed to synchronize\n"); return;}

    if (c_out_2 && c_ft_in2)  R = cufft_Execute_C2C(FFT_Plan, c_FTInput2, c_Output2, CUFFT_INVERSE);
    if (R != CUFFT_SUCCESS) {fprintf(stderr,"CUFFT Error: %s\n", cudaGetErrorString(cudaError_t(R))); return;}
    if (cudaDeviceSynchronize() != cudaSuccess){fprintf(stderr, "CUFFT error: Failed to synchronize\n"); return;}

    if (c_out_3 && c_ft_in3)  R = cufft_Execute_C2C(FFT_Plan, c_FTInput3, c_Output3, CUFFT_INVERSE);
    if (R != CUFFT_SUCCESS) {fprintf(stderr,"CUFFT Error: %s\n", cudaGetErrorString(cudaError_t(R))); return;}
    if (cudaDeviceSynchronize() != cudaSuccess){fprintf(stderr, "CUFFT error: Failed to synchronize\n"); return;}
}

void DataType_CUDA::Forward_FFT_R2C()
{
    // Carry out forward FFTs
    cufftResult R = CUFFT_SUCCESS;
    if (r_in1 && c_ft_in1)  R = cufft_Execute_R2C(Forward_Plan, r_Input1, c_FTInput1);
    if (R != CUFFT_SUCCESS) {fprintf(stderr,"CUFFT Error: %s\n", cudaGetErrorString(cudaError_t(R))); return;}
    if (cudaDeviceSynchronize() != cudaSuccess){fprintf(stderr, "CUFFT error: Failed to synchronize\n"); return;}

    if (r_in2 && c_ft_in2)  R = cufft_Execute_R2C(Forward_Plan, r_Input2, c_FTInput2);
    if (R != CUFFT_SUCCESS) {fprintf(stderr,"CUFFT Error: %s\n", cudaGetErrorString(cudaError_t(R))); return;}
    if (cudaDeviceSynchronize() != cudaSuccess){fprintf(stderr, "CUFFT error: Failed to synchronize\n"); return;}

    if (r_in3 && c_ft_in3)  R = cufft_Execute_R2C(Forward_Plan, r_Input3, c_FTInput3);
    if (R != CUFFT_SUCCESS) {fprintf(stderr,"CUFFT Error: %s\n", cudaGetErrorString(cudaError_t(R))); return;}
    if (cudaDeviceSynchronize() != cudaSuccess){fprintf(stderr, "CUFFT error: Failed to synchronize\n"); return;}
}

void DataType_CUDA::Backward_FFT_C2R()
{
    // Carry out backward FFTs
    cufftResult R = CUFFT_SUCCESS;
    if (r_out_1 && c_ft_in1)  R = cufft_Execute_C2R(Backward_Plan, c_FTInput1, r_Output1);
    if (R != CUFFT_SUCCESS) {fprintf(stderr,"CUFFT Error: %s\n", cudaGetErrorString(cudaError_t(R))); return;}
    if (cudaDeviceSynchronize() != cudaSuccess){fprintf(stderr, "CUFFT error: Failed to synchronize\n"); return;}

    if (r_out_2 && c_ft_in2)  R = cufft_Execute_C2R(Backward_Plan, c_FTInput2, r_Output2);
    if (R != CUFFT_SUCCESS) {fprintf(stderr,"CUFFT Error: %s\n", cudaGetErrorString(cudaError_t(R))); return;}
    if (cudaDeviceSynchronize() != cudaSuccess){fprintf(stderr, "CUFFT error: Failed to synchronize\n"); return;}

    if (r_out_3 && c_ft_in3)  R = cufft_Execute_C2R(Backward_Plan, c_FTInput3, r_Output3);
    if (R != CUFFT_SUCCESS) {fprintf(stderr,"CUFFT Error: %s\n", cudaGetErrorString(cudaError_t(R))); return;}
    if (cudaDeviceSynchronize() != cudaSuccess){fprintf(stderr, "CUFFT error: Failed to synchronize\n"); return;}
}

//--- Destructor

DataType_CUDA::~DataType_CUDA()
{
    //--- This clears the data associated with this FFTW object

    // Clear arrays
    Deallocate_Arrays();

    // Clear plans
    cufftDestroy(FFT_Plan);
    cufftDestroy(Forward_Plan);
    cufftDestroy(Backward_Plan);
}

}

#endif

