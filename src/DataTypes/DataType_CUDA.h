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
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>

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
#define cublas_axpy         cublasCaxpy
#define cublas_dot          cublasCdotu
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
#define cublas_axpy         cublasZaxpy
#define cublas_dot          cublasZdotu
#endif

//--- Dimension & index structs & functions

typedef unsigned uint;
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

class DataType_CUDA : public DataType
{

protected:

    //--- Grid dimension
    dim3 GridDim;

    //--- Plan
    cufftHandle FFT_Plan;
    cufftHandle Forward_Plan;
    cufftHandle Backward_Plan;

    //--- Handles for cuBLAS
    cublasHandle_t cublashandle;

    //--- Memory objects (cpu)
    Real *r_Input1, *r_Output1;
    Real *r_Input2, *r_Output2;
    Real *r_Input3, *r_Output3;

    //--- Memory objects (gpu)
    // In reality these are actually stored on the GPU, we shall use these simply as interfacing arrays.
    CUDAReal *cuda_r_Input1, *cuda_r_Output1;
    CUDAReal *cuda_r_Input2, *cuda_r_Output2;
    CUDAReal *cuda_r_Input3, *cuda_r_Output3;

    CUDAComplex *c_Input1, *c_FTInput1, *c_FTOutput1, *c_Output1, *c_FTVel1;
    CUDAComplex *c_Input2, *c_FTInput2, *c_FTOutput2, *c_Output2, *c_FTVel2;
    CUDAComplex *c_Input3, *c_FTInput3, *c_FTOutput3, *c_Output3, *c_FTVel3;
    CUDAComplex *c_DummyBuffer1, *c_DummyBuffer2, *c_DummyBuffer3;
    CUDAComplex *c_DummyBuffer4, *c_DummyBuffer5, *c_DummyBuffer6;
    CUDAComplex *c_FG, *c_FGi, *c_FGj, *c_FGk;

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
    SFStatus Transfer_Data_Device();

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
    void Prep_Greens_Function_C2C()         override;
    void Prep_Greens_Function_R2C()         override;
    void Prepare_Dif_Operators_1D(Real Hx)  override;
    void Prepare_Dif_Operators_2D(Real Hx, Real Hy) override;
    void Prepare_Dif_Operators_3D(Real Hx, Real Hy, Real Hz)    override;

    //--- Convolution
    void Convolute(CUDAComplex *A, CUDAComplex *B, CUDAComplex *Result, int N);
    void Increment(CUDAComplex *A, CUDAComplex *B, Real Fac, int N);
    void Convolution_Real()     override        {}
    void Convolution_Real3()    override        {}
    void Convolution_Complex()  override;
    void Convolution_Complex3() override;

    //--- Spectral Gradients
    void Transfer_FTInOut_Comp();
    void Spectral_Gradients_1D_Grad();
    void Spectral_Gradients_1D_Nabla();
    void Spectral_Gradients_2D_Div();
    void Spectral_Gradients_2D_Grad();
    void Spectral_Gradients_2D_Curl();
    void Spectral_Gradients_2D_Nabla();
    void Spectral_Gradients_3D_Div();
    void Spectral_Gradients_3D_Grad();
    void Spectral_Gradients_3DV_Div();
    void Spectral_Gradients_3DV_Grad(Component I);
    void Spectral_Gradients_3DV_Curl();
    void Spectral_Gradients_3DV_Nabla();
    void Spectral_Gradients_3DV_Reprojection();

    //--- Destructor
    ~DataType_CUDA();
};

}

#endif

#endif // DATATYPE_CUDA_H
