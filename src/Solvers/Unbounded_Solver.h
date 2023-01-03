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

    -> Unbounded Poisson solver definition. The method of Hockney-Eastwood is used.

*****************************************************************************/

#ifndef UNBOUNDED_SOLVER_H
#define UNBOUNDED_SOLVER_H

#include "Periodic_Solver.h"

namespace SailFFish
{

class Unbounded_Solver_1D : public Solver_1D_Scalar
{
protected:

public:

    //--- Constructor
    Unbounded_Solver_1D(Grid_Type G = STAGGERED, Unbounded_Kernel B = HEJ_S0);

    //--- Solver setup
    SFStatus FFT_Data_Setup()                   {return Setup_1D(2*gNX);}
    void Specify_Operator(OperatorType O)   {Operator = O;  }   // No additional allocations necessary

    //--- Frequency space operations
    void Forward_Transform()    {Forward_FFT_R2C();}
    void Backward_Transform()   {Backward_FFT_C2R();}
    void Convolution()          {Convolution_Complex();}

    //--- Greens function spec
    void Specify_Greens_Function();
};

class Unbounded_Solver_2D : public Solver_2D_Scalar
{
protected:

public:

    //--- Constructor
    Unbounded_Solver_2D(Grid_Type G = STAGGERED, Unbounded_Kernel B = HEJ_S0);

    //--- Solver setup
    SFStatus FFT_Data_Setup()                   {return Setup_2D(2*gNX,2*gNY);}
    void Specify_Operator(OperatorType O);

    //--- Frequency space operations
    void Forward_Transform()    {Forward_FFT_R2C();}
    void Backward_Transform()   {Backward_FFT_C2R();}
    void Convolution()          {Convolution_Complex();}

    //--- Greens function spec
    void Specify_Greens_Function();
};

class Unbounded_Solver_3D : public Solver_3D_Scalar
{
protected:

public:

    //--- Constructor
    Unbounded_Solver_3D(Grid_Type G = STAGGERED, Unbounded_Kernel B = HEJ_S0);

    //--- Solver setup
    SFStatus FFT_Data_Setup()                   {return Setup_3D(2*gNX,2*gNY,2*gNZ);}
    void Specify_Operator(OperatorType O);

    //--- Frequency space operations
    void Forward_Transform()    {Forward_FFT_R2C();}
    void Backward_Transform()   {Backward_FFT_C2R();}
    void Convolution()          {Convolution_Complex();}

    //--- Greens function spec
    void Specify_Greens_Function();
};

class Unbounded_Solver_3DV : public Solver_3D_Vector
{
protected:

public:

    //--- Constructor
    Unbounded_Solver_3DV(Grid_Type G = STAGGERED, Unbounded_Kernel B = HEJ_S0);

    //--- Solver setup
    SFStatus FFT_Data_Setup()                   {return Setup_3D(2*gNX,2*gNY,2*gNZ);}
    void Specify_Operator(OperatorType O);

    //--- Frequency space operations
    void Forward_Transform()    {Forward_FFT_R2C();}
    void Backward_Transform()   {Backward_FFT_C2R();}
    void Convolution()          {Convolution_Complex3();}

    //--- Greens function spec
    void Specify_Greens_Function();
};

}

#endif // UNBOUNDED_SOLVER_H
