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

    -> This is the MKL type datatype which acts as a verification
    -> platform for the solver

*****************************************************************************/

#ifndef DATATYPE_MKL_H
#define DATATYPE_MKL_H

#include "DataType_Base.h"

namespace SailFFish
{

#ifdef MKL

#include "mkl_poisson.h"

#ifdef SinglePrec
        // Do single float crap
#endif
#ifdef DoublePrec
    // Do double float crap
#endif

class DataType_MKL : public DataType
{
protected:

    //--- Grid vars
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int nt = 0;

    int N_Points = 0;
    int N_BPoints = 0;
    int N_Cells = 0;
    int N_FaceX = 0;
    int N_FaceY = 0;

    Real Hx=0, Hy=0, Hz=0;

    //--- Vars MKL solver
    Real ax=0, bx=0;                    // Domain limits xmax, xmin etc
    Real ay=0, by=0;
    Real az=0, bz=0;
    Real q;                             // RHS factor (zero for Poisson equation)
    Real *spar;                         // Intermediate var for Helmholtz solver
    Real *f, *f0;                       // RHS of poisson equation. (Vorticity values)
    char *BCtype;                       // Identification of boundary conditions

//    //--- Plan
//    FFTWPlan Forward_Plan;
//    FFTWPlan Backward_Plan;
    MKL_INT ipar[128];                  // Internal information for MKL solver
    MKL_INT stat;                       // status of calculation
    DFTI_DESCRIPTOR_HANDLE xhandle;// = 0; // FFT Handle
    DFTI_DESCRIPTOR_HANDLE yhandle = 0; // FFT Handle

    //--- Memory objects (Boundary conditions)
    Real **BD_AX, **BD_BX;
    Real **BD_AY, **BD_BY;
    Real **BD_AZ, **BD_BZ;

    //--- Memory objects (Real)
    Real *r_Input1, *r_FTInput1;
    Real *r_Output1, *r_Output1Bar;
    Real *r_FG;

public:

    //--- Constructor
    DataType_MKL()  {}

//    //--- Plan specification
    void Setup_2D_DFT(int iNX, int iNY);

    //--- Data clearing
    void Reset_Array(Real *A, const int &N)     {memset(A, 0, N*sizeof(Real));}

    //--- Destructor
    ~DataType_MKL()     {}
};

#endif

}

#endif // DATATYPE_MKL_H
