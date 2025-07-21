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

    -> This header contains the parent class for the desired data type

*****************************************************************************/

#ifndef DATATYPE_BASE_H
#define DATATYPE_BASE_H

#include "../SailFFish_Math_Types.h"
#include <filesystem>
#include <fstream>

namespace SailFFish
{

//--- Status messages

enum SFStatus       {NoError, DimError, MemError, SetupError, ExecError, GridError};
enum FTType         {DCT1, DCT2, DST1, DST2, DFT_C2C, DFT_R2C};
enum OperatorType   {NONE, DIV, CURL, GRAD, NABLA};
enum Component      {XComp, YComp, ZComp};
enum Dim            {EX, EY, EZ};

class DataType
{
protected:

    //--- Fourier transform vars
    FTType Transform = DFT_C2C;

    //--- Operator vars
    bool Kernel_Convolution = true;
    OperatorType Operator = NONE;

    //--- Grid vars
    int NT = 0, NTM = 0;
    int NX = 0, NXM = 0;
    int NY = 0, NYM = 0;
    int NZ = 0, NZM = 0;

    //--- Scaling factors
    Real FFac = 1.0, BFac = 1.0;

    //--- Allocation flags. These specify which arrays should be allocated.
    bool InPlace = false;       // Are the transforms being carried out in-place?

    // Real-valued arrays
    bool r_in1 = false, r_ft_in1 = false, r_ft_out1 = false, r_out_1 = false;
    bool r_in2 = false, r_ft_in2 = false, r_ft_out2 = false, r_out_2 = false;
    bool r_in3 = false, r_ft_in3 = false, r_ft_out3 = false, r_out_3 = false;

    // Complex-valued arrays
    bool c_in1 = false, c_ft_in1 = false, c_ft_out1 = false, c_out_1 = false, c_vel_1 = false;
    bool c_in2 = false, c_ft_in2 = false, c_ft_out2 = false, c_out_2 = false, c_vel_2 = false;
    bool c_in3 = false, c_ft_in3 = false, c_ft_out3 = false, c_out_3 = false, c_vel_3 = false;

    // Arrays for Green's functions & spectral operators
    bool r_fg = false, c_fg = false;
    bool c_fg_i = false, c_fg_j = false, c_fg_k = false;

    // Dummy buffers
    bool c_dbf_1 = false, c_dbf_2 = false, c_dbf_3 = false;
    bool c_dbf_4 = false, c_dbf_5 = false, c_dbf_6 = false;

    //--- Memory objects (Real)
    Real *r_FG;

    //--- Solver status
    SFStatus Status = NoError;

public:

    //--- Constructor
    DataType()  {}

    virtual void Datatype_Setup()       {}

    //--- Plan specification
    virtual SFStatus Setup_1D(int NX);
    virtual SFStatus Setup_2D(int NX, int NY);
    virtual SFStatus Setup_3D(int NX, int NY, int NZ);

    //--- Specify Plan
    virtual SFStatus Specify_1D_Plan()      {return NoError;}
    virtual SFStatus Specify_2D_Plan()      {return NoError;}
    virtual SFStatus Specify_3D_Plan()      {return NoError;}

    //--- Array allocation
    virtual SFStatus Allocate_Arrays()      {return NoError;}
    virtual SFStatus Deallocate_Arrays()    {return NoError;}

    //--- Specify Input
    virtual SFStatus Set_Input(RVector &I)                                          {return NoError;}
    virtual SFStatus Set_Input(RVector &I1, RVector &I2, RVector &I3)               {return NoError;}
    virtual SFStatus Set_Input_Unbounded_1D(RVector &I)                             {return NoError;}
    virtual SFStatus Set_Input_Unbounded_2D(RVector &I)                             {return NoError;}
    virtual SFStatus Set_Input_Unbounded_3D(RVector &I)                             {return NoError;}
    virtual SFStatus Set_Input_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3)  {return NoError;}
    virtual SFStatus Transfer_Data_Device()                                         {return NoError;}

    //--- Retrieve output array
    virtual void Get_Output(RVector &I)                                             {}
    virtual void Get_Output(RVector &I1, RVector &I2, RVector &I3)                  {}
    virtual void Get_Output_Unbounded_1D(RVector &I)                                {}
    virtual void Get_Output_Unbounded_2D(RVector &I)                                {}
    virtual void Get_Output_Unbounded_2D(RVector &I1, RVector &I2)                  {}
    virtual void Get_Output_Unbounded_3D(RVector &I)                                {}
    virtual void Get_Output_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3)     {}

    //--- Greens functions prep
    virtual void Prep_Greens_Function_C2C()                             {}
    virtual void Prep_Greens_Function_R2C()                             {}
    virtual void Prepare_Dif_Operators_1D(Real Hx)                      {}
    virtual void Prepare_Dif_Operators_2D(Real Hx, Real Hy)             {}
    virtual void Prepare_Dif_Operators_3D(Real Hx, Real Hy, Real Hz)    {}

    //--- Fourier transforms
    virtual void Forward_FFT_R2R()      {}
    virtual void Backward_FFT_R2R()     {}
    virtual void Forward_FFT_DFT()      {}
    virtual void Backward_FFT_DFT()     {}
    virtual void Forward_FFT_R2C()      {}
    virtual void Backward_FFT_C2R()     {}

    //--- Convolution
    virtual void Convolution_Real()     {}
    virtual void Convolution_Real3()    {}
    virtual void Convolution_Complex()  {}
    virtual void Convolution_Complex3() {}


    //--- Spectral gradients
    virtual void Spectral_Gradients_3DV_Reprojection()                              {}

    //--- Destructor
    virtual ~DataType()    {}
};

}

#endif // DATATYPE_BASE_H
