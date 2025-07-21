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
#include "vkFFT.h"
#include "utils_VkFFT.h"

namespace SailFFish
{

#ifdef SinglePrec
    typedef cl_float   cl_real;
    typedef cl_float2  cl_real2;
    typedef cl_float3  cl_real3;
    typedef cl_float4  cl_real4;
    typedef cl_float8  cl_real8;
    typedef cl_float16 cl_real16;
#endif
#ifdef DoublePrec
    typedef cl_double   cl_real;
    typedef cl_double2  cl_real2;
    typedef cl_double3  cl_real3;
    typedef cl_double4  cl_real4;
    typedef cl_double8  cl_real8;
    typedef cl_double16 cl_real16;
#endif

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

class DataType_VkFFT : public DataType
{

protected:

    //--- Grid dimension
    dim3 GridDim;

    //--- Memory objects (cpu)
    Real *r_Input1, *r_Output1;
    Real *r_Input2, *r_Output2;
    Real *r_Input3, *r_Output3;

    //--- VkFFT Object
    VkGPU VK;

    //--- OpenCL objects
    // cl_device device;

public:

    //--- Constructor
    DataType_VkFFT()  {}

    void Datatype_Setup();
    VkFFTResult OpenCLSetup(VkGPU* vkGPU);
    void Print_Device_Info(cl_device_id device);

    //--- Specify Plan
    // SFStatus Specify_1D_Plan();
    // SFStatus Specify_2D_Plan();
    // SFStatus Specify_3D_Plan();

    //--- Array allocation
    SFStatus Allocate_Arrays();
    SFStatus Deallocate_Arrays();

};

}

#endif

#endif // DATATYPE_VKFFT_H
