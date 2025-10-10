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

enum SFStatus       {NoError, DimError, MemError, SetupError, InputError, ExecError, GridError};
enum FTType         {DCT1, DCT2, DST1, DST2, DFT_R2R, DFT_C2C, DFT_R2C};
enum OperatorType   {NONE, DIV, CURL, GRAD, NABLA};
enum Component      {XComp, YComp, ZComp};
enum Dim            {EX, EY, EZ};

typedef unsigned uint;
inline uint GID(uint i, uint j, uint NX, uint NY)                   {return i*NY + j;}
inline uint GID(uint i, uint j, uint k, uint NX, uint NY, uint NZ)  {return i*NY*NZ + j*NZ + k;}

class DataType
{
protected:

    //--- Dimension
    uint Dimension = 0;

    //--- Fourier transform vars
    FTType Transform = DFT_C2C;

    //--- Operator vars
    bool Kernel_Convolution = true;
    OperatorType Operator = NONE;

    //--- Grid vars
    int NT = 0;             // Number of grid nodes in _ direction
    int NX = 0;             // Number of grid nodes in _ direction
    int NY = 0;             // Number of grid nodes in _ direction
    int NZ = 0;             // Number of grid nodes in _ direction

    //--- Grid vars (unbounded solvers)
    // For the unbounded solver configuration the grid size is halved in each direction
    // depending on the solver type, on of the dimesnions is halved for the R2C/C2R transform
    int NTM = 0, NTH = 0;
    int NXM = 0, NXH = 0;
    int NYM = 0, NYH = 0;
    int NZM = 0, NZH = 0;

    int gNT = 0;
    int NCX=0, NCY=0, NCZ=0;    // Number of cells
    int gNX=0, gNY=0, gNZ=0;    // Number of grid nodes

    Real Xl, Xu;                // X Limits
    Real Yl, Yu;                // Y Limits
    Real Zl, Zu;                // Z Limits
    Real Lx, Ly, Lz;            // Domain widths
    Real Hx, Hy, Hz;            // Cell widths
    RVector fX, fY, fZ;         // X-Y values on FFT grid
    RVector gX, gY, gZ;         // X-Y values on solution grid

    //--- Grid sizes for R2C transforms
    void Set_NTM1D()   {NXM = NX;                           NTM = NXM;          }
    void Set_NTM2D()   {NXM = NX;   NYM = NY;               NTM = NXM*NYM;      }
    void Set_NTM3D()   {NXM = NX;   NYM = NY;   NZM = NZ;   NTM = NXM*NYM*NZM;  }
    virtual void Set_NTM1D_R2C()   {NXM = NX/2 + 1;                             NTM = NXM;          }
    virtual void Set_NTM2D_R2C()   {NXM = NX;   NYM = NY/2 + 1;                 NTM = NXM*NYM;      }
    virtual void Set_NTM3D_R2C()   {NXM = NX;   NYM = NY;       NZM = NZ/2 + 1; NTM = NXM*NYM*NZM;  }

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

    //--- Mapping functions (bounded solvers)
    virtual void Map_C2F_1D(const RVector &Src, RVector &Dest)  {std::copy(Src.begin(), Src.end(), Dest.begin());}
    virtual void Map_F2C_1D(const RVector &Src, RVector &Dest)  {std::copy(Src.begin(), Src.end(), Dest.begin());}
    virtual void Map_C2F_2D(const RVector &Src, RVector &Dest);
    virtual void Map_F2C_2D(const RVector &Src, RVector &Dest);
    virtual void Map_C2F_3D(const RVector &Src, RVector &Dest);
    virtual void Map_F2C_3D(const RVector &Src, RVector &Dest);
    virtual void Map_C2F_3DV(const RVector &Src1, const RVector &Src2, const RVector &Src3, RVector &Dest1, RVector &Dest2, RVector &Dest3);
    virtual void Map_F2C_3DV(const RVector &Src1, const RVector &Src2, const RVector &Src3, RVector &Dest1, RVector &Dest2, RVector &Dest3);

    //--- Mapping functions (unbounded solvers)
    virtual void Map_C2F_UB_1D(const RVector &Src, Real *Dest)      {memcpy(Dest, Src.data(), (NXH)*sizeof(Real));}
    virtual void Map_F2C_UB_1D(Real *Src, RVector &Dest)            {memcpy(Dest.data(), Src, (NXH)*sizeof(Real));}
    virtual void Map_C2F_UB_2D(const RVector &Src, Real *Dest);
    virtual void Map_F2C_UB_2D(Real *Src, RVector &Dest);
    virtual void Map_C2F_UB_3D(const RVector &Src, Real *Dest);
    virtual void Map_F2C_UB_3D(Real *Src, RVector &Dest);
    virtual void Map_C2F_UB_3DV(const RVector &Src1, const RVector &Src2, const RVector &Src3, Real *Dest1, Real *Dest2, Real *Dest3);
    virtual void Map_F2C_UB_3DV(Real *Src1, Real *Src2, Real *Src3, RVector &Dest1, RVector &Dest2, RVector &Dest3);

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
    virtual SFStatus Set_Input_Unbounded(RVector &I)                                {return NoError;}
    virtual SFStatus Set_Input_Unbounded_1D(RVector &I)                             {return NoError;}
    virtual SFStatus Set_Input_Unbounded_2D(RVector &I)                             {return NoError;}
    virtual SFStatus Set_Input_Unbounded_3D(RVector &I)                             {return NoError;}
    virtual SFStatus Set_Input_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3)  {return NoError;}
    virtual SFStatus Set_Input_Unbounded(RVector &I1, RVector &I2, RVector &I3)     {return NoError;}
    virtual SFStatus Transfer_Data_Device()                                         {return NoError;}

    //--- Retrieve output array
    virtual void Get_Output(RVector &I)                                             {}
    virtual void Get_Output(RVector &I1, RVector &I2, RVector &I3)                  {}
    virtual void Get_Output_Unbounded(RVector &I)                                   {}
    virtual void Get_Output_Unbounded_1D(RVector &I)                                {}
    virtual void Get_Output_Unbounded_2D(RVector &I)                                {}
    virtual void Get_Output_Unbounded_2D(RVector &I1, RVector &I2)                  {}
    virtual void Get_Output_Unbounded_3D(RVector &I)                                {}
    virtual void Get_Output_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3)     {}
    virtual void Get_Output_Unbounded(RVector &I1, RVector &I2, RVector &I3)        {}

    //--- Greens functions prep
    virtual void Prep_Greens_Function_R2R()                             {}
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

    //--- Get status
    SFStatus GetStatus()    {return Status;}
    void    ResetStatus()  {Status = NoError;}


    //--- Spectral gradients
    virtual void Spectral_Gradients_3DV_Reprojection()                              {}

    //--- Destructor
    virtual ~DataType()    {}
};

}

#endif // DATATYPE_BASE_H
