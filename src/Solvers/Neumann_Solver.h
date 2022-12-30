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

    -> The Poisson solvers with Neumann BCs are defined here

*****************************************************************************/

#ifndef NEUMANN_SOLVER_H
#define NEUMANN_SOLVER_H

#include "Solver_Base.h"

namespace SailFFish
{

class Poisson_Neumann_1D : public Solver_1D_Scalar
{
protected:

public:

    //--- Constructor
    Poisson_Neumann_1D(Grid_Type G = STAGGERED, Bounded_Kernel B = PS);

    //--- Solver setup
    SFStatus FFT_Data_Setup()
    {
        if (Grid==REGULAR)          return Setup_1D(gNX);
        else if (Grid==STAGGERED)   return Setup_1D(gNX);
        else return SetupError;
    }

    //--- Specify boundary conditions
    void Set_BC(Real AX, Real BX);

    //--- Frequency space operations
    void Forward_Transform()    {Forward_FFT_R2R();}
    void Backward_Transform()   {Backward_FFT_R2R();}
    void Convolution()          {Convolution_Real();}

    //--- Greens function spec
    void Specify_Greens_Function();
};

class Poisson_Neumann_2D : public Solver_2D_Scalar
{
protected:

public:

    //--- Constructor
    Poisson_Neumann_2D(Grid_Type G = STAGGERED, Bounded_Kernel B = PS);

    //--- Solver setup
    SFStatus FFT_Data_Setup()
    {
        if (Grid==REGULAR)          return Setup_2D(gNX,gNY);
        else if (Grid==STAGGERED)   return Setup_2D(gNX,gNY);
        else return SetupError;
    }

    //--- Specify boundary conditions
    void Set_BC(RVector AX, RVector BX, RVector AY, RVector BY);

    //--- Frequency space operations
    void Forward_Transform()    {Forward_FFT_R2R();}
    void Backward_Transform()   {Backward_FFT_R2R();}
    void Convolution()          {Convolution_Real();}

    //--- Greens function spec
    void Specify_Greens_Function();
};

class Poisson_Neumann_3D : public Solver_3D_Scalar
{
protected:

public:

    //--- Constructor
    Poisson_Neumann_3D(Grid_Type G = STAGGERED, Bounded_Kernel B = PS);

    //--- Solver setup
    SFStatus FFT_Data_Setup()
    {
        if (Grid==REGULAR)          return Setup_3D(gNX,gNY,gNZ);
        else if (Grid==STAGGERED)   return Setup_3D(gNX,gNY,gNZ);
        else return SetupError;
    }

    //--- Specify boundary conditions
    void Set_BC(RVector AX, RVector BX, RVector AY, RVector BY, RVector AZ, RVector BZ);

    //--- Frequency space operations
    void Forward_Transform()    {Forward_FFT_R2R();}
    void Backward_Transform()   {Backward_FFT_R2R();}
    void Convolution()          {Convolution_Real();}

    //--- Greens function spec
    void Specify_Greens_Function();
};

class Poisson_Neumann_3DV : public Solver_3D_Vector
{
protected:

public:

    //--- Constructor
    Poisson_Neumann_3DV(Grid_Type G = STAGGERED, Bounded_Kernel B = PS);

    //--- Solver setup
    SFStatus FFT_Data_Setup()
    {
        if (Grid==REGULAR)          return Setup_3D(gNX,gNY,gNZ);
        else if (Grid==STAGGERED)   return Setup_3D(gNX,gNY,gNZ);
        else return SetupError;
    }

    //--- Specify boundary conditions
    void Set_BC(RVector AX1, RVector BX1, RVector AX2, RVector BX2, RVector AX3, RVector BX3,
                RVector AY1, RVector BY1, RVector AY2, RVector BY2, RVector AY3, RVector BY3,
                RVector AZ1, RVector BZ1, RVector AZ2, RVector BZ2, RVector AZ3, RVector BZ3);

    //--- Frequency space operations
    void Forward_Transform()    {Forward_FFT_R2R();}
    void Backward_Transform()   {Backward_FFT_R2R();}
    void Convolution()          {Convolution_Real3();}

    //--- Greens function spec
    void Specify_Greens_Function();
};

}

#endif // NEUMANN_SOLVER_H
