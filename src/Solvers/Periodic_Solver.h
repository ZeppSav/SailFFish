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

    -> Poisson solvers with periodic boundary conditions is defined here

*****************************************************************************/

#ifndef PERIODIC_SOLVER_H
#define PERIODIC_SOLVER_H

#include "Solver_Base.h"

namespace SailFFish
{

class Poisson_Periodic_1D : public Solver_1D_Scalar
{
protected:

public:

    //--- Constructor
    Poisson_Periodic_1D(Grid_Type G = STAGGERED, Bounded_Kernel B = PS);

    //--- Solver setup
    SFStatus FFT_Data_Setup()
    {
        SFStatus Stat = SetupError;
        if (Grid==REGULAR)      Stat =  Setup_1D(gNX-1);
        if (Grid==STAGGERED)    Stat = Setup_1D(gNX);
        return Stat;
    }

    //--- Solver setup
//    virtual SFStatus FFT_Data_Setup()       {return Setup_1D(gNX-1);}

    //--- Frequency space operations
    void Forward_Transform()    {Forward_FFT_DFT();}
    void Backward_Transform()   {Backward_FFT_DFT();}
    void Convolution()          {Convolution_Complex();}

    //--- Greens function spec
    void Specify_Greens_Function();
};

class Poisson_Periodic_2D : public Solver_2D_Scalar
{
protected:

public:

    //--- Constructor
    Poisson_Periodic_2D(Grid_Type G = STAGGERED, Bounded_Kernel B = PS);

    //--- Solver setup
    SFStatus FFT_Data_Setup()
    {
        if (Grid==REGULAR)          return Setup_2D(gNX-1,gNY-1);
        else if (Grid==STAGGERED)   return Setup_2D(gNX,gNY);
        else return SetupError;
    }

    //--- Frequency space operations
    void Forward_Transform()    {Forward_FFT_DFT();}
    void Backward_Transform()   {Backward_FFT_DFT();}
    void Convolution()          {Convolution_Complex();}

    //--- Greens function spec
    virtual void Specify_Greens_Function();
};

class Poisson_Periodic_3D : public Solver_3D_Scalar
{
protected:

public:

    //--- Constructor
    Poisson_Periodic_3D(Grid_Type G = STAGGERED, Bounded_Kernel B = PS);

    //--- Solver setup
    SFStatus FFT_Data_Setup()
    {
        if (Grid==REGULAR)          return Setup_3D(gNX-1,gNY-1,gNZ-1);
        else if (Grid==STAGGERED)   return Setup_3D(gNX,gNY,gNZ);
        else return SetupError;
    }

    //--- Frequency space operations
    void Forward_Transform()    {Forward_FFT_DFT();}
    void Backward_Transform()   {Backward_FFT_DFT();}
    void Convolution()          {Convolution_Complex();}

    //--- Greens function spec
    virtual void Specify_Greens_Function();
};

class Poisson_Periodic_3DV : public Solver_3D_Vector
{
protected:

public:

    //--- Constructor
    Poisson_Periodic_3DV(Grid_Type G = STAGGERED, Bounded_Kernel B = PS);

    //--- Solver setup
    SFStatus FFT_Data_Setup()
    {
        if (Grid==REGULAR)          return Setup_3D(gNX-1,gNY-1,gNZ-1);
        else if (Grid==STAGGERED)   return Setup_3D(gNX,gNY,gNZ);
        else return SetupError;
    }

    //--- Frequency space operations
    void Forward_Transform()    {Forward_FFT_DFT();}
    void Backward_Transform()   {Backward_FFT_DFT();}
    void Convolution()          {Convolution_Complex3();}

    //--- Greens function spec
    void Specify_Greens_Function();
};

}

#endif // PERIODIC_SOLVER_H
