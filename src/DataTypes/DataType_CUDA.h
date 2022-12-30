/****************************************************************************
    SailFFish Library
    Copyright (C) 2022 Joseph Saverin j.saverin@tu-berlin.de

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

    -> This is the base datatype class when using CUDA to calculate FFTs.
    -> Note. Although it is tempting to use cuFFTW, to reduce the implementation effort, the reality is that
    part of the attraction to use CUDA in the first place is to avoid communciation between CPU & GPU. Aka all of the
    data is stored on the GPU and we don't need to go back and forth between the two.
    cuFFTW is configured such that essentially all of the data is stored on CPU and is passed to/from cuFFTW and converted to local data types.

*****************************************************************************/


#ifndef DATATYPE_CUDA_H
#define DATATYPE_CUDA_H

#ifdef CUFFT

#include "DataType_Base.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

namespace SailFFish
{

#ifdef SinglePrec
    typedef float               CUDAReal;
    typedef cuComplex           CUDAComplex;
    static const cufftType_t    cufft_C2C = CUFFT_C2C;
    static const cufftType_t    cufft_R2C = CUFFT_R2C;
    static const cufftType_t    cufft_C2R = CUFFT_C2R;
    #define cufft_Execute_C2C   cufftExecC2C
    #define cufft_Execute_R2C   cufftExecR2C
    #define cufft_Execute_C2R   cufftExecC2R
    #define cublas_dgmm         cublasCdgmm
    #define cublas_axpy         cublasCaxpy_v2
#endif
#ifdef DoublePrec
    typedef double              CUDAReal;
    typedef cuDoubleComplex     CUDAComplex;
    static const cufftType_t    cufft_C2C = CUFFT_Z2Z;
    static const cufftType_t    cufft_R2C = CUFFT_D2Z;
    static const cufftType_t    cufft_C2R = CUFFT_Z2D;
    #define cufft_Execute_C2C   cufftExecZ2Z
    #define cufft_Execute_R2C   cufftExecD2Z
    #define cufft_Execute_C2R   cufftExecZ2D
    #define cublas_dgmm         cublasZdgmm
    #define cublas_axpy         cublasZaxpy_v2
#endif

class DataType_CUDA : public DataType
{

protected:

    //--- Plan
    cufftHandle FFT_Plan;
    cufftHandle Forward_Plan;
    cufftHandle Backward_Plan;

    //--- Handles for cuBLAS
    cublasHandle_t cublashandle;

    //--- Memory objects (Complex)
    CUDAComplex *c_Input1, *c_FTInput1;
    CUDAComplex *c_Input2, *c_FTInput2;
    CUDAComplex *c_Input3, *c_FTInput3;
    CUDAComplex *c_Output1;
    CUDAComplex *c_Output2;
    CUDAComplex *c_Output3;
    CUDAComplex *c_DummyBuffer1, *c_DummyBuffer2, *c_DummyBuffer3;
    CUDAComplex *c_DummyBuffer4, *c_DummyBuffer5, *c_DummyBuffer6;
    CUDAComplex *c_FG, *c_FGi, *c_FGj, *c_FGk;
//    FFTWReal     *c_FG;            // Just a placeholder for compiling
//    CVector c_FG;

    //--- Memory objects (Real)
    // In reality these are actually stored on the GPU, we shall use these simply as interfacing arrays.
    CUDAReal *r_Input1, *r_FTInput1;
    CUDAReal *r_Input2, *r_FTInput2;
    CUDAReal *r_Input3, *r_FTInput3;
    CUDAReal *r_Output1;
    CUDAReal *r_Output2;
    CUDAReal *r_Output3;
//    CUDAReal *cu_r_FG;

public:

    //--- Constructor
    DataType_CUDA()  {}

    void Datatype_Setup();

    //--- Specify Plan
    SFStatus Specify_1D_Plan();
    SFStatus Specify_2D_Plan();
    SFStatus Specify_3D_Plan();

    //--- Array allocation
    SFStatus Allocate_Arrays();
    SFStatus Deallocate_Arrays();

    //--- Prepare input array
    SFStatus Set_Input(RVector &I);
    SFStatus Set_Input(RVector &I1, RVector &I2, RVector &I3);
    SFStatus Set_Input_Unbounded_1D(RVector &I);
    SFStatus Set_Input_Unbounded_2D(RVector &I);
    SFStatus Set_Input_Unbounded_3D(RVector &I);
    SFStatus Set_Input_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3);

    //--- Prepare output array
    void Get_Output(RVector &I);
    void Get_Output(RVector &I1, RVector &I2, RVector &I3);
    void Get_Output_Unbounded_1D(RVector &I);
    void Get_Output_Unbounded_2D(RVector &I);
    void Get_Output_Unbounded_2D(RVector &I1, RVector &I2);
    void Get_Output_Unbounded_3D(RVector &I);
    void Get_Output_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3);

    //--- Fourier transforms
    void Forward_FFT_R2R()      {}      // Not available in CUDA
    void Backward_FFT_R2R()     {}      // Not available in CUDA
    void Forward_FFT_DFT();
    void Backward_FFT_DFT();
    void Forward_FFT_R2C();
    void Backward_FFT_C2R();

    //--- Greens functions prep
    void Prep_Greens_Function_C2C();
    void Prep_Greens_Function_R2C();
    void Prepare_Dif_Operators_1D(Real Hx);
    void Prepare_Dif_Operators_2D(Real Hx, Real Hy);
    void Prepare_Dif_Operators_3D(Real Hx, Real Hy, Real Hz);

    //--- Convolution
    void Convolution_Real()                     {}
    void Convolution_Real3()                    {}
    void Convolution_Complex();
    void Convolution_Complex3();
    void Spectral_Gradients_1D();
    void Spectral_Gradients_2D();
    void Spectral_Gradients_3D();
    void Spectral_Gradients_3DV();

    //--- Destructor
    ~DataType_CUDA();
};

}

#endif

#endif // DATATYPE_CUDA_H
