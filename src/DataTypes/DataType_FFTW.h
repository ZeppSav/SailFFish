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

    -> This is the base datatype class when using FFTW to calculate FFTs.

*****************************************************************************/

#ifndef DATATYPE_FFTW_H
#define DATATYPE_FFTW_H

#ifdef FFTW

#include "DataType_Base.h"
#include <fftw3.h>

namespace SailFFish
{

// Define data types of FFTW Depending on floating point precision

#define FFTW_PLANSETUP         FFTW_ESTIMATE
//#define FFTW_PLANSETUP         FFTW_MEASURE
//#define FFTW_PLANSETUP         FFTW_EXHAUSTIVE

#ifdef SinglePrec
    typedef fftwf_complex       FFTWReal;
    typedef fftwf_plan          FFTWPlan;
    #define FFTW_malloc         fftwf_malloc
    #define FFTW_Plan_1D        fftwf_plan_dft_1d
    #define FFTW_Plan_2D        fftwf_plan_dft_2d
    #define FFTW_Plan_3D        fftwf_plan_dft_3d
    #define FFTW_Plan_1D_R2C    fftwf_plan_dft_r2c_1d
    #define FFTW_Plan_2D_R2C    fftwf_plan_dft_r2c_2d
    #define FFTW_Plan_3D_R2C    fftwf_plan_dft_r2c_3d
    #define FFTW_Plan_1D_C2R    fftwf_plan_dft_c2r_1d
    #define FFTW_Plan_2D_C2R    fftwf_plan_dft_c2r_2d
    #define FFTW_Plan_3D_C2R    fftwf_plan_dft_c2r_3d
    #define FFTW_Plan_1D_DCT    fftwf_plan_r2r_1d
    #define FFTW_Plan_2D_DCT    fftwf_plan_r2r_2d
    #define FFTW_Plan_3D_DCT    fftwf_plan_r2r_3d
    #define FFTW_Execute        fftwf_execute
    #define FFTW_Execute_R2R    fftwf_execute_r2r
    #define FFTW_Execute_DFT    fftwf_execute_dft
    #define FFTW_Execute_R2C    fftwf_execute_dft_r2c
    #define FFTW_Execute_C2R    fftwf_execute_dft_c2r
    #define FFTW_Destroy_Plan   fftwf_destroy_plan
    #define FFTW_Free           fftwf_free
#endif
#ifdef DoublePrec
    typedef fftw_complex        FFTWReal;
    typedef fftw_plan           FFTWPlan;
    #define FFTW_malloc         fftw_malloc
    #define FFTW_Plan_1D        fftw_plan_dft_1d
    #define FFTW_Plan_2D        fftw_plan_dft_2d
    #define FFTW_Plan_3D        fftw_plan_dft_3d
    #define FFTW_Plan_1D_R2C    fftw_plan_dft_r2c_1d
    #define FFTW_Plan_2D_R2C    fftw_plan_dft_r2c_2d
    #define FFTW_Plan_3D_R2C    fftw_plan_dft_r2c_3d
    #define FFTW_Plan_1D_C2R    fftw_plan_dft_c2r_1d
    #define FFTW_Plan_2D_C2R    fftw_plan_dft_c2r_2d
    #define FFTW_Plan_3D_C2R    fftw_plan_dft_c2r_3d
    #define FFTW_Plan_1D_DCT    fftw_plan_r2r_1d
    #define FFTW_Plan_2D_DCT    fftw_plan_r2r_2d
    #define FFTW_Plan_3D_DCT    fftw_plan_r2r_3d
    #define FFTW_Execute        fftw_execute
    #define FFTW_Execute_R2R    fftw_execute_r2r
    #define FFTW_Execute_DFT    fftw_execute_dft
    #define FFTW_Execute_R2C    fftw_execute_dft_r2c
    #define FFTW_Execute_C2R    fftw_execute_dft_c2r
    #define FFTW_Destroy_Plan   fftw_destroy_plan
    #define FFTW_Free           fftw_free
#endif

class DataType_FFTW : public DataType
{
protected:

    //--- Plan
    FFTWPlan Forward_Plan;
    FFTWPlan Backward_Plan;

    //--- Trial R2C
    FFTWPlan R2DPlan;

    //--- Memory objects (Complex)
    FFTWReal *c_Input1, *c_FTInput1;
    FFTWReal *c_Input2, *c_FTInput2;
    FFTWReal *c_Input3, *c_FTInput3;
    FFTWReal *c_Output1;
    FFTWReal *c_Output2;
    FFTWReal *c_Output3;
    FFTWReal *c_FG;
    FFTWReal *c_FGi, *c_FGj, *c_FGk;

    //--- Memory objects (Real)
    Real *r_Input1, *r_FTInput1;
    Real *r_Input2, *r_FTInput2;
    Real *r_Input3, *r_FTInput3;
    Real *r_Output1;
    Real *r_Output2;
    Real *r_Output3;
//    Real *r_FG;

public:

    //--- Constructor
    DataType_FFTW()  {}

    void Datatype_Setup();

    //--- Specify Plan
    SFStatus Specify_1D_Plan();
    SFStatus Specify_2D_Plan();
    SFStatus Specify_3D_Plan();

    //--- Array allocation
    SFStatus Allocate_Arrays();
    SFStatus Deallocate_Arrays();

    //--- Prepare input array
    void Set_Null(FFTWReal *A, int N)               {for (int i=0; i<N; i++) {A[i][0] = 0.; A[i][1] = 0.;}}
    SFStatus Set_Input(RVector &I);
    SFStatus Set_Input(RVector &I1, RVector &I2, RVector &I3);
    SFStatus Set_Input_Unbounded_1D(RVector &I);
    SFStatus Set_Input_Unbounded_2D(RVector &I);
    SFStatus Set_Input_Unbounded_3D(RVector &I);
    SFStatus Set_Input_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3);

    //--- Retrieve output array
    void Get_Output(RVector &I);
    void Get_Output(RVector &I1, RVector &I2, RVector &I3);
    void Get_Output_Unbounded_1D(RVector &I);
    void Get_Output_Unbounded_2D(RVector &I);
    void Get_Output_Unbounded_2D(RVector &I1, RVector &I2);
    void Get_Output_Unbounded_3D(RVector &I);
    void Get_Output_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3);

    //--- Fourier transforms
    void Forward_FFT_R2R()
    {
        if (r_in1 && r_ft_in1)      FFTW_Execute_R2R(Forward_Plan,r_Input1,r_FTInput1);
        if (r_in2 && r_ft_in2)      FFTW_Execute_R2R(Forward_Plan,r_Input2,r_FTInput2);
        if (r_in3 && r_ft_in3)      FFTW_Execute_R2R(Forward_Plan,r_Input3,r_FTInput3);
    }
    void Backward_FFT_R2R()
    {
        if (r_out_1 && r_ft_in1)    FFTW_Execute_R2R(Backward_Plan,r_FTInput1,r_Output1);
        if (r_out_2 && r_ft_in2)    FFTW_Execute_R2R(Backward_Plan,r_FTInput2,r_Output2);
        if (r_out_3 && r_ft_in3)    FFTW_Execute_R2R(Backward_Plan,r_FTInput3,r_Output3);
    }    
    void Forward_FFT_DFT()
    {
        if (c_in1 && c_ft_in1)      FFTW_Execute_DFT(Forward_Plan,c_Input1,c_FTInput1);
        if (c_in2 && c_ft_in2)      FFTW_Execute_DFT(Forward_Plan,c_Input2,c_FTInput2);
        if (c_in3 && c_ft_in3)      FFTW_Execute_DFT(Forward_Plan,c_Input3,c_FTInput3);
    } 
    void Backward_FFT_DFT()
    {
        if (c_out_1 && c_ft_in1)    FFTW_Execute_DFT(Backward_Plan,c_FTInput1,c_Output1);
        if (c_out_2 && c_ft_in2)    FFTW_Execute_DFT(Backward_Plan,c_FTInput2,c_Output2);
        if (c_out_3 && c_ft_in3)    FFTW_Execute_DFT(Backward_Plan,c_FTInput3,c_Output3);
    }   
    void Forward_FFT_R2C()
    {
        if (r_in1 && c_ft_in1)      FFTW_Execute_R2C(Forward_Plan,r_Input1,c_FTInput1);
        if (r_in2 && c_ft_in2)      FFTW_Execute_R2C(Forward_Plan,r_Input2,c_FTInput2);
        if (r_in3 && c_ft_in3)      FFTW_Execute_R2C(Forward_Plan,r_Input3,c_FTInput3);
    }
    void Backward_FFT_C2R()
    {
        if (r_out_1 && c_ft_in1)    FFTW_Execute_C2R(Backward_Plan,c_FTInput1,r_Output1);
        if (r_out_2 && c_ft_in2)    FFTW_Execute_C2R(Backward_Plan,c_FTInput2,r_Output2);
        if (r_out_3 && c_ft_in3)    FFTW_Execute_C2R(Backward_Plan,c_FTInput3,r_Output3);
    }

    //--- Greens functions prep
    void Prep_Greens_Function_C2C();
    void Prep_Greens_Function_R2C();
    void Prepare_Dif_Operators_1D(Real Hx);
    void Prepare_Dif_Operators_2D(Real Hx, Real Hy);
    void Prepare_Dif_Operators_3D(Real Hx, Real Hy, Real Hz);

    //--- Convolution
    void Convolution_Real();
    void Convolution_Real3();
    void Convolution_Complex();
    void Convolution_Complex3();
    void Spectral_Gradients_1D();
    void Spectral_Gradients_2D();
    void Spectral_Gradients_3D();
    void Spectral_Gradients_3DV();

    //--- Destructor
    ~DataType_FFTW();
};

}

#endif

#endif // DATATYPE_FFTW_H
