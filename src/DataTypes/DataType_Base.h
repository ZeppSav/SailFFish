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

namespace SailFFish
{

enum SFStatus       {NoError, DimError, MemError, SetupError, ExecError};
enum FTType         {DCT1, DCT2, DST1, DST2, DFT_C2C, DFT_R2C};
enum OperatorType   {NONE, DIV, CURL, GRAD, NABLA};

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

    // Real-valued arrays
    bool r_in1 = false, r_ft_in1 = false, r_out_1 = false;
    bool r_in2 = false, r_ft_in2 = false, r_out_2 = false;
    bool r_in3 = false, r_ft_in3 = false, r_out_3 = false;

    // Complex-valued arrays
    bool c_in1 = false, c_ft_in1 = false, c_out_1 = false;
    bool c_in2 = false, c_ft_in2 = false, c_out_2 = false;
    bool c_in3 = false, c_ft_in3 = false, c_out_3 = false;

    // Arrays for Green's functions & spectral operators
    bool r_fg = false, c_fg = false;
    bool c_fg_i = false, c_fg_j = false, c_fg_k = false;

    // Dummy buffers
    bool c_dbf_1 = false, c_dbf_2 = false, c_dbf_3 = false;
    bool c_dbf_4 = false, c_dbf_5 = false, c_dbf_6 = false;

    //--- Memory objects (Real)
    Real *r_FG;

public:

    //--- Constructor
    DataType()  {}

    virtual void Datatype_Setup()       {}

    //--- Plan specification
    virtual SFStatus Setup_1D(int NX);
    virtual SFStatus Setup_2D(int NX, int NY);
    virtual SFStatus Setup_3D(int NX, int NY, int NZ);

    //--- Specify Plan
    virtual SFStatus Specify_1D_Plan()      {}
    virtual SFStatus Specify_2D_Plan()      {}
    virtual SFStatus Specify_3D_Plan()      {}

    //--- Array allocation
    virtual SFStatus Allocate_Arrays()      {}
    virtual SFStatus Deallocate_Arrays()    {}

    //--- Specify Input
    virtual SFStatus Set_Input(RVector &I)                                          {}
    virtual SFStatus Set_Input(RVector &I1, RVector &I2, RVector &I3)               {}
    virtual SFStatus Set_Input_Unbounded_1D(RVector &I)                             {}
    virtual SFStatus Set_Input_Unbounded_2D(RVector &I)                             {}
    virtual SFStatus Set_Input_Unbounded_3D(RVector &I)                             {}
    virtual SFStatus Set_Input_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3)  {}

    //--- Retrieve output array
    virtual void Get_Output(RVector &I)                                             {}
    virtual void Get_Output(RVector &I1, RVector &I2, RVector &I3)                  {}
    virtual void Get_Output_Unbounded_1D(RVector &I)                                {}
    virtual void Get_Output_Unbounded_2D(RVector &I)                                {}
    virtual void Get_Output_Unbounded_2D(RVector &I1, RVector &I2)                  {}
    virtual void Get_Output_Unbounded_3D(RVector &I)                                {}
    virtual void Get_Output_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3)     {}

    //--- Destructor
    virtual ~DataType()    {}
};

}

#endif // DATATYPE_BASE_H
