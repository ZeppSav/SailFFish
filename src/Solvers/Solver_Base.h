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

    -> Parent class for Fast Poisson solver
    -> Note: The datatype object each use a specific type of number format.
    -> For simplicity I will do everything within the Poisson solver with std data types and then convert when necessary.

*****************************************************************************/

#ifndef SOLVER_BASE_H
#define SOLVER_BASE_H

#include "../DataTypes/DataType_FFTW.h"
#include "../DataTypes/DataType_CUDA.h"
#include "../DataTypes/DataType_VkFFT.h"

namespace SailFFish
{

enum Grid_Type          {REGULAR, STAGGERED};
enum Bounded_Kernel     {PS, FD2};
enum Unbounded_Kernel   {HEJ_S0, HEJ_G2, HEJ_G4, HEJ_G6, HEJ_G8, HEJ_G10};

#if defined(FFTW)
    class Solver : public DataType_FFTW
#elif defined(CUFFT)
    class Solver : public DataType_CUDA
#elif defined(VKFFT)
    class Solver : public DataType_VkFFT
#endif
// class Solver : public DataType_FFTW
{
protected:

    //--- Solver parameters
    Grid_Type Grid = REGULAR;
    Bounded_Kernel Spect_Kernel = FD2;
    Unbounded_Kernel Greens_Kernel = HEJ_S0;

    //--- Output parameters
    std::string OutputFolder;           // Where are results exported to?

public:

    //--- Constructor
    Solver()        {}

    //--- Solver parameter getters
    Grid_Type           Get_Grid_Type()                 {return Grid;}
    OperatorType        Get_Operator_Type()             {return Operator;}
    Bounded_Kernel      Get_Bounded_Kernel_Type()       {return Spect_Kernel;}
    Unbounded_Kernel    Get_Unbounded_Kernel_Type()     {return Greens_Kernel;}

    //--- Solver setup
    virtual void X_Grid_Setup(Real X[2], int iNX);
    virtual void Y_Grid_Setup(Real Y[2], int iNY);
    virtual void Z_Grid_Setup(Real Z[2], int iNZ);
    virtual SFStatus FFT_Data_Setup()               {return NoError;}
    virtual void Specify_Operator(OperatorType O)   {}

    //--- Frequency space operations
    virtual void Forward_Transform()    {}
    virtual void Backward_Transform()   {}
    virtual void Convolution()          {}

    //--- Greens function spec
    virtual void Specify_Greens_Function()      {}

    //--- Grid visualisation
    virtual void Create_vtk()                       {}

    //--- Import external fields
    virtual void Import_vtk(std::string &Inputfile) {}

    //--- Extract grid positions
    void Get_XGrid(RVector &X)      {for (int i=0; i<gNX; i++) X.push_back(gX[i]);}
    void Get_YGrid(RVector &Y)      {for (int j=0; j<gNY; j++) Y.push_back(gY[j]);}
    void Get_ZGrid(RVector &Z)      {for (int k=0; k<gNZ; k++) Z.push_back(gZ[k]);}
    void Get_Grid_Dims(Real &L1, Real &L2, Real &L3)    {L1 = Lx; L2 = Ly; L3 = Lz;}
    void Get_Grid_Res(Real &L1, Real &L2, Real &L3)     {L1 = Hx; L2 = Hy; L3 = Hz;}
    void Get_Grid_Lower_Bdry(Real &L1, Real &L2, Real &L3)  {L1 = gX[0]; L2 = gY[0]; L3 = gZ[0];}
    void Get_Grid_Upper_Bdry(Real &L1, Real &L2, Real &L3)  {L1 = gX[gNX-1]; L2 = gY[gNY-1]; L3 = gZ[gNZ-1];}

    //--- Destructor
    ~Solver();

};

//--- 1D Scalar solver
class Solver_1D_Scalar : public Solver
{
protected:

    // Default output name
    std::string vtk_Name = "Mesh_1D.vtk";

public:

    //--- Constructor
    Solver_1D_Scalar()        {Dimension = 1;}

    //--- Solver setup
    virtual SFStatus Setup(Real X[2], int iNX);
};

//--- 2D Scalar solver
class Solver_2D_Scalar : public Solver
{
protected:

    // Default output name
    std::string vtk_Name = "Mesh_2D.vtk";

public:

    //--- Constructor
    Solver_2D_Scalar()        {Dimension = 2;}

    //--- Solver setup
    virtual SFStatus Setup(Real X[2], Real Y[2], int iNX, int iNY);

    //--- Grid visualisation
    virtual void Create_vtk();
};

//--- 3D Scalar solver
class Solver_3D_Scalar : public Solver
{
protected:

    // Default output name
    std::string vtk_Name = "Mesh_3D.vtk";

public:

    //--- Constructor
    Solver_3D_Scalar()        {Dimension = 3;}

    //--- Solver setup
    virtual SFStatus Setup(Real X[2], Real Y[2], Real Z[2], int iNX, int iNY, int iNZ);

    //--- Grid visualisation
    virtual void Create_vtk();
};

//--- 3D Vector solver
class Solver_3D_Vector : public Solver
{
protected:

    // Default output name
    std::string vtk_Name = "Mesh_3DV.vtk";

public:

    //--- Constructor
    Solver_3D_Vector()        {Dimension = 3;}

    //--- Solver setup
    virtual SFStatus Setup(Real X[2], Real Y[2], Real Z[2], int iNX, int iNY, int iNZ);

    //--- Grid visualisation
    virtual void Create_vtk() override;

    //--- import grid
    void Import_vtk(std::string &Inputfile) override;
};

}

#endif // SOLVER_BASE_H
