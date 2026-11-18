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

    -> 3D Vector Test cases for each of the solver types.

*****************************************************************************/

#ifndef TEST_3DV_H
#define TEST_3DV_H

#include <gtest/gtest.h>
#include "SailFFish_Math_Types.h"
#include "Solvers/Periodic_Solver.h"
#include "Solvers/Dirichlet_Solver.h"
#include "Solvers/Neumann_Solver.h"
#include "Solvers/Unbounded_Solver.h"
#include "Test_Functions.h"

static const int NX_3DV = 64;               // Grid resolution for all 3DV tests (x)
static const int NY_3DV = 64;               // Grid resolution for all 3DV tests (y)
static const int NZ_3DV = 64;               // Grid resolution for all 3DV tests (z)
static const bool ExportVTK_3DV = false;    // Export grids as VTK
static const Real Tol_3DV = 1.0e-4;         // E_inf tolerance (calibrate)

// Bounded Dirichlet/Neumann solvers use real-to-real transforms (DST/DCT), which cuFFT does not provide
#if defined(CUFFT)
#define SKIP_IF_CUFFT() GTEST_SKIP() << "Real-to-real transforms (DCT/DST) are not supported by cuFFT."
#else
#define SKIP_IF_CUFFT()
#endif

//---------------------
//--- Bounded solvers
//---------------------

TEST(SailFFish_3DV_Tests, Test_Dirichlet_3DV)
{
    SKIP_IF_CUFFT();

    const int NX = NX_3DV;
    const int NY = NY_3DV;
    const int NZ = NZ_3DV;
    const bool ExportVTK = ExportVTK_3DV;
    // Test case for 3D (vector field) bounded Poisson solver with Dirichlet BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    auto Solver = std::make_unique<SailFFish::Poisson_Dirichlet_3DV>(SailFFish::STAGGERED, SailFFish::PS);
    // auto Solver = std::make_unique<SailFFish::Poisson_Dirichlet_3DV>(SailFFish::REGULAR, SailFFish::PS);
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    ASSERT_EQ(Status, SailFFish::NoError);
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_ZGrid(ZGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Generate Input & solution arrays

    int IS = 0;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) IS = 1;

    int NT = (NX-IS)*(NY-IS)*(NZ-IS);
    RVector Input1 = RVector(NT,0);
    RVector Input2 = RVector(NT,0);
    RVector Input3 = RVector(NT,0);
    RVector Output1 = RVector(NT,0);
    RVector Output2 = RVector(NT,0);
    RVector Output3 = RVector(NT,0);
    RVector Solution = RVector(NT,0);

    OpenMPfor
    for (int i=IS; i<NX; i++){
        for (int j=IS; j<NY; j++){
            for (int k=IS; k<NZ; k++){
                int id = (i-IS)*(NY-IS)*(NZ-IS) + (j-IS)*(NZ-IS) + (k-IS);
                Real O = STest_Omega(XGrid[i]-UnitX[0],Lx,YGrid[j]-UnitY[0],Ly,ZGrid[k]-UnitZ[0],Lz);
                Input1[id] = O;
                Input2[id] = O;
                Input3[id] = O;
                Solution[id] = STest_Phi(XGrid[i]-UnitX[0],Lx,YGrid[j]-UnitY[0],Ly,ZGrid[k]-UnitZ[0],Lz);   // X Velocity
            }
        }
    }
    Status = Solver->Set_Input(Input1,Input2,Input3);
    ASSERT_EQ(Status, SailFFish::NoError);
    unsigned int t3 = stopwatch();  // Timer

//    // Set BC
//    std::vector<Real> AX; AX.assign((NY+1)*(NZ+1),0.0);
//    std::vector<Real> BX; BX.assign((NY+1)*(NZ+1),10.0);
//    std::vector<Real> AY; AY.assign((NX+1)*(NZ+1),3.0);
//    std::vector<Real> BY; BY.assign((NX+1)*(NZ+1),7.0);
//    std::vector<Real> AZ; AZ.assign((NX+1)*(NY+1),2.0);
//    std::vector<Real> BZ; BZ.assign((NX+1)*(NY+1),9.0);
//    Solver->Set_BC(AX, BX, AX, BX, AX, BX,
//                   AX, BX, AX, BX, AX, BX,
//                   AX, BX, AX, BX, AX, BX);

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output(Output1,Output2,Output3);
    ASSERT_EQ(Output1.size(), Solution.size());
    ASSERT_EQ(Output2.size(), Solution.size());
    ASSERT_EQ(Output3.size(), Solution.size());
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    std::cout << "Trial Calculation: Solution of the 3D Poisson equation (vector field) with Dirichlet boundary conditions." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<","<< NY <<","<< NZ <<"] cells." << std::endl;
    std::cout << "E_inf Error (x component) =" csp std::scientific << E_Inf(Output1,Solution) << std::endl;
    std::cout << "E_inf Error (y component) =" csp std::scientific << E_Inf(Output2,Solution) << std::endl;
    std::cout << "E_inf Error (z component) =" csp std::scientific << E_Inf(Output3,Solution) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;
    std::cout << std::scientific;

    // If selected, export grid as VTK
    if (ExportVTK) Solver->Create_vtk();

    EXPECT_LT(E_Inf(Output1,Solution), Tol_3DV);
    EXPECT_LT(E_Inf(Output2,Solution), Tol_3DV);
    EXPECT_LT(E_Inf(Output3,Solution), Tol_3DV);
}

TEST(SailFFish_3DV_Tests, Test_Neumann_3DV)
{
    SKIP_IF_CUFFT();

    const int NX = NX_3DV;
    const int NY = NY_3DV;
    const int NZ = NZ_3DV;
    const bool ExportVTK = ExportVTK_3DV;
    // Test case for 3D (vector field) bounded Poisson solver with Neumann BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    auto Solver = std::make_unique<SailFFish::Poisson_Neumann_3DV>(SailFFish::STAGGERED, SailFFish::PS);
    // auto Solver = std::make_unique<SailFFish::Poisson_Neumann_3DV>(SailFFish::REGULAR, SailFFish::PS);
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    ASSERT_EQ(Status, SailFFish::NoError);
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_ZGrid(ZGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Generate Input & solution arrays
    int IS = 0;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) IS = 1;

    int NXM = NX + IS;
    int NYM = NY + IS;
    int NZM = NZ + IS;
    int NT = (NXM)*(NYM)*(NZM);
    RVector Input1 = RVector(NT,0);
    RVector Input2 = RVector(NT,0);
    RVector Input3 = RVector(NT,0);
    RVector Output1 = RVector(NT,0);
    RVector Output2 = RVector(NT,0);
    RVector Output3 = RVector(NT,0);
    RVector Solution = RVector(NT,0);
    OpenMPfor
    for (int i=0; i<NXM; i++){
        for (int j=0; j<NYM; j++){
            for (int k=0; k<NZM; k++){
                int id = i*(NYM)*(NZM) + j*(NZM) + k;
                Real O = CTest_Omega(XGrid[i]-UnitX[0],Lx,YGrid[j]-UnitY[0],Ly,ZGrid[k]-UnitZ[0],Lz);
                Input1[id] = O;
                Input2[id] = O;
                Input3[id] = O;
                Solution[id] = CTest_Phi(XGrid[i]-UnitX[0],Lx,YGrid[j]-UnitY[0],Ly,ZGrid[k]-UnitZ[0],Lz);   // X Velocity
            }
        }
    }
    Status = Solver->Set_Input(Input1,Input2,Input3);
    ASSERT_EQ(Status, SailFFish::NoError);
    unsigned int t3 = stopwatch();  // Timer

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output(Output1,Output2,Output3);
    ASSERT_EQ(Output1.size(), Solution.size());
    ASSERT_EQ(Output2.size(), Solution.size());
    ASSERT_EQ(Output3.size(), Solution.size());
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    std::cout << "Trial Calculation: Solution of the 3D Poisson equation (vector field) with Neumann boundary conditions." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<","<< NY <<","<< NZ <<"] cells." << std::endl;
    std::cout << "E_inf Error (x component) =" csp std::scientific << E_Inf(Output1,Solution) << std::endl;
    std::cout << "E_inf Error (y component) =" csp std::scientific << E_Inf(Output2,Solution) << std::endl;
    std::cout << "E_inf Error (z component) =" csp std::scientific << E_Inf(Output3,Solution) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;
    std::cout << std::scientific;

    // If selected, export grid as VTK
    if (ExportVTK) Solver->Create_vtk();

    EXPECT_LT(E_Inf(Output1,Solution), Tol_3DV);
    EXPECT_LT(E_Inf(Output2,Solution), Tol_3DV);
    EXPECT_LT(E_Inf(Output3,Solution), Tol_3DV);
}

TEST(SailFFish_3DV_Tests, Test_Periodic_3DV)
{
    const int NX = NX_3DV;
    const int NY = NY_3DV;
    const int NZ = NZ_3DV;
    const bool ExportVTK = ExportVTK_3DV;
    // Test case for 3D (vector field) bounded Poisson solver with periodic BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    auto Solver = std::make_unique<SailFFish::Poisson_Periodic_3DV>(SailFFish::STAGGERED, SailFFish::PS);
    // auto Solver = std::make_unique<SailFFish::Poisson_Periodic_3DV>(SailFFish::REGULAR, SailFFish::PS);
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    ASSERT_EQ(Status, SailFFish::NoError);
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_ZGrid(ZGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Generate Input & solution arrays
    RVector Input1 = RVector(NX*NY*NZ,0);
    RVector Input2 = RVector(NX*NY*NZ,0);
    RVector Input3 = RVector(NX*NY*NZ,0);
    RVector Output1 = RVector(NX*NY*NZ,0);
    RVector Output2 = RVector(NX*NY*NZ,0);
    RVector Output3 = RVector(NX*NY*NZ,0);
    RVector Solution = RVector(NX*NY*NZ,0);
    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++){
            for (int k=0; k<NZ; k++){
                int id = i*NY*NZ + j*NZ + k;
                Real O = CTest_Omega(XGrid[i],Lx,YGrid[j],Ly,ZGrid[k],Lz);
                Input1[id] = O;   // X Vorticity
                Input2[id] = O;   // Y Vorticity
                Input3[id] = O;   // Z Vorticity
                Solution[id] = CTest_Phi(XGrid[i],Lx,YGrid[j],Ly,ZGrid[k],Lz);   // X Velocity
            }
        }
    }
    Status = Solver->Set_Input(Input1,Input2,Input3);
    ASSERT_EQ(Status, SailFFish::NoError);
    unsigned int t3 = stopwatch();  // Timer

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output(Output1,Output2,Output3);
    ASSERT_EQ(Output1.size(), Solution.size());
    ASSERT_EQ(Output2.size(), Solution.size());
    ASSERT_EQ(Output3.size(), Solution.size());
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    std::cout << "Trial Calculation: Solution of the 3D Poisson equation (vector field) with periodic boundary conditions." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<","<< NY <<","<< NZ <<"] cells." << std::endl;
    std::cout << "E_inf Error (x component) =" csp std::scientific << E_Inf(Output1,Solution) << std::endl;
    std::cout << "E_inf Error (y component) =" csp std::scientific << E_Inf(Output2,Solution) << std::endl;
    std::cout << "E_inf Error (z component) =" csp std::scientific << E_Inf(Output3,Solution) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;
    std::cout << std::scientific;

    // If selected, export grid as VTK
    if (ExportVTK) Solver->Create_vtk();

    EXPECT_LT(E_Inf(Output1,Solution), Tol_3DV);
    EXPECT_LT(E_Inf(Output2,Solution), Tol_3DV);
    EXPECT_LT(E_Inf(Output3,Solution), Tol_3DV);
}

//---------------------
//--- Unbounded solvers
//---------------------

TEST(SailFFish_3DV_Tests, Test_Unbounded_3DV)
{
    const int NX = NX_3DV;
    const int NY = NY_3DV;
    const int NZ = NZ_3DV;
    const bool ExportVTI = ExportVTK_3DV;
    // Test case for 2D unbounded Poisson solver

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    auto Solver = std::make_unique<SailFFish::Unbounded_Solver_3DV>();
    // Solver->Specify_Operator(SailFFish::CURL);
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    ASSERT_EQ(Status, SailFFish::NoError);
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Real Hx, Hy, Hz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_ZGrid(ZGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);
    Solver->Get_Grid_Res(Hx, Hy, Hz);

    // If this is a grid-boundary solve, we have 1 extra grid point
    int NXM = NX, NYM = NY, NZM = NZ;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) {NXM++; NYM++; NZM++;}

    // Generate Input & solution arrays
    int NT = NXM*NYM*NZM;
    RVector Input1 = RVector(NT,0);
    RVector Input2 = RVector(NT,0);
    RVector Input3 = RVector(NT,0);
    RVector Output1 = RVector(NT,0);
    RVector Output2 = RVector(NT,0);
    RVector Output3 = RVector(NT,0);
    RVector Solution1 = RVector(NT,0);
    RVector Solution2 = RVector(NT,0);
    RVector Solution3 = RVector(NT,0);

    OpenMPfor
    for (int i=0; i<NXM; i++){
        for (int j=0; j<NYM; j++){
            for (int k=0; k<NZM; k++){
                int id = i*NYM*NZM + j*NZM + k;
                UTest_Omega_Hejlesen(   XGrid[i],YGrid[j],ZGrid[k],0.5, Input1[id], Input2[id], Input3[id]);
                UTest_Phi_Hejlesen(     XGrid[i],YGrid[j],ZGrid[k],0.5, Solution1[id], Solution2[id], Solution3[id]);
            }
        }
    }

    // Scale input/output for LInf- otherwise we encounter floating point errors...
    Real EFac = 1.0/exp(-Cbf);
    for (auto& i : Input1)  i *= EFac;
    for (auto& i : Input2)  i *= EFac;
    for (auto& i : Input3)  i *= EFac;
    for (auto& i : Solution1) i *= EFac;
    for (auto& i : Solution2) i *= EFac;
    for (auto& i : Solution3) i *= EFac;

    Status = Solver->Set_Input_Unbounded_3D(Input1,Input2,Input3);
    ASSERT_EQ(Status, SailFFish::NoError);
    unsigned int t3 = stopwatch();  // Timer

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output_Unbounded_3D(Output1,Output2,Output3);
    ASSERT_EQ(Output1.size(), Solution1.size());
    ASSERT_EQ(Output2.size(), Solution2.size());
    ASSERT_EQ(Output3.size(), Solution3.size());
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    std::cout << "Trial Calculation: Solution of the unbounded 3D Poisson equation (vector field)." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<" , "<< NY <<" , "<< NZ <<"] cells. " << std::endl;
    std::cout << "E_inf Error (x component) =" csp std::scientific << E_Inf(Output1,Solution1) << std::endl;
    std::cout << "E_inf Error (y component) =" csp std::scientific << E_Inf(Output2,Solution2) << std::endl;
    std::cout << "E_inf Error (z component) =" csp std::scientific << E_Inf(Output3,Solution3) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    // std::cout << "Execution Time:" <<  std::endl;
    // std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    // std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    // std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    // std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    // std::cout << "Forward FFT  "     << std::setw(10) << t32 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    // std::cout << "Convolution  "     << std::setw(10) << t33 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    // std::cout << "Backward FFT "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;
    std::cout << std::scientific;

    // If selected, export grid as VTI
    if (ExportVTI) Solver->Create_vtk();

    EXPECT_LT(E_Inf(Output1,Solution1), Tol_3DV);
    EXPECT_LT(E_Inf(Output2,Solution2), Tol_3DV);
    EXPECT_LT(E_Inf(Output3,Solution3), Tol_3DV);
}

TEST(SailFFish_3DV_Tests, Test_Unbounded_3DV_Curl)
{
    const int NX = NX_3DV;
    const int NY = NY_3DV;
    const int NZ = NZ_3DV;
    const bool ExportVTI = ExportVTK_3DV;
    // Test case for 2D unbounded Poisson solver

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    auto Solver = std::make_unique<SailFFish::Unbounded_Solver_3DV>();
    Solver->Specify_Operator(SailFFish::CURL);
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    ASSERT_EQ(Status, SailFFish::NoError);
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Real Hx, Hy, Hz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_ZGrid(ZGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);
    Solver->Get_Grid_Res(Hx, Hy, Hz);

    // If this is a grid-boundary solve, we have 1 extra grid point
    int NXM = NX, NYM = NY, NZM = NZ;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) {NXM++; NYM++; NZM++;}

    // Generate Input & solution arrays
    int NT = NXM*NYM*NZM;
    RVector Input1 = RVector(NT,0);
    RVector Input2 = RVector(NT,0);
    RVector Input3 = RVector(NT,0);
    RVector Output1 = RVector(NT,0);
    RVector Output2 = RVector(NT,0);
    RVector Output3 = RVector(NT,0);
    RVector Solution1 = RVector(NT,0);
    RVector Solution2 = RVector(NT,0);
    RVector Solution3 = RVector(NT,0);

    OpenMPfor
    for (int i=0; i<NXM; i++){
        for (int j=0; j<NYM; j++){
            for (int k=0; k<NZM; k++){
                int id = i*NYM*NZM + j*NZM + k;
                UTest_Omega_Hejlesen(     XGrid[i],YGrid[j],ZGrid[k],0.5, Input1[id], Input2[id], Input3[id]);
                UTest_Velocity_Hejlesen(  XGrid[i],YGrid[j],ZGrid[k],0.5, Solution1[id], Solution2[id], Solution3[id]);
            }
        }
    }

    // Scale input/output for LInf- otherwise we encounter floating point errors...
    Real EFac = 1.0/exp(-Cbf);
    for (auto& i : Input1)  i *= EFac;
    for (auto& i : Input2)  i *= EFac;
    for (auto& i : Input3)  i *= EFac;
    for (auto& i : Solution1) i *= EFac;
    for (auto& i : Solution2) i *= EFac;
    for (auto& i : Solution3) i *= EFac;

    Status = Solver->Set_Input_Unbounded_3D(Input1,Input2,Input3);
    ASSERT_EQ(Status, SailFFish::NoError);
    unsigned int t3 = stopwatch();

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();                      // This must be called for test to work
    Solver->Spectral_Gradients_3DV_Curl();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output_Unbounded_3D(Output1,Output2,Output3);
    ASSERT_EQ(Output1.size(), Solution1.size());
    ASSERT_EQ(Output2.size(), Solution2.size());
    ASSERT_EQ(Output3.size(), Solution3.size());
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    std::cout << "Trial Calculation: Solution of the unbounded 3D Poisson equation (vector field)." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<" , "<< NY <<" , "<< NZ <<"] cells. " << std::endl;
    std::cout << "E_inf Error (x component) =" csp std::scientific << E_Inf(Output1,Solution1) << std::endl;
    std::cout << "E_inf Error (y component) =" csp std::scientific << E_Inf(Output2,Solution2) << std::endl;
    std::cout << "E_inf Error (z component) =" csp std::scientific << E_Inf(Output3,Solution3) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;
    std::cout << std::scientific;

    // // If selected, export grid as VTI
    if (ExportVTI) Solver->Create_vtk();

    EXPECT_LT(E_Inf(Output1,Solution1), Tol_3DV);
    EXPECT_LT(E_Inf(Output2,Solution2), Tol_3DV);
    EXPECT_LT(E_Inf(Output3,Solution3), Tol_3DV);
}

TEST(SailFFish_3DV_Tests, DISABLED_Test_Unbounded_3DV_Nabla)
{
    // Analytical gradients should be checked!

    const int NX = NX_3DV;
    const int NY = NY_3DV;
    const int NZ = NZ_3DV;
    const bool ExportVTI = ExportVTK_3DV;
    // Test case for 2D unbounded Poisson solver

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    auto Solver = std::make_unique<SailFFish::Unbounded_Solver_3DV>();
    Solver->Specify_Operator(SailFFish::NABLA);
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    ASSERT_EQ(Status, SailFFish::NoError);
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    //    Real Hx, Hy, Hz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_ZGrid(ZGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);
    //    Solver->Get_Grid_Res(Hx, Hy, Hz);

    // If this is a grid-boundary solve, we have 1 extra grid point
    int NXM = NX, NYM = NY, NZM = NZ;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) {NXM++; NYM++; NZM++;}

    // Generate Input & solution arrays
    int NT = NXM*NYM*NZM;
    RVector Input1 = RVector(NT,0);
    RVector Input2 = RVector(NT,0);
    RVector Input3 = RVector(NT,0);
    RVector Output1 = RVector(NT,0);
    RVector Output2 = RVector(NT,0);
    RVector Output3 = RVector(NT,0);
    RVector Solution1 = RVector(NT,0);
    RVector Solution2 = RVector(NT,0);
    RVector Solution3 = RVector(NT,0);

    OpenMPfor
    for (int i=0; i<NXM; i++){
        for (int j=0; j<NYM; j++){
            for (int k=0; k<NZM; k++){
                int id = i*NYM*NZM + j*NZM + k;
                UTest_Phi_Hejlesen(       XGrid[i],YGrid[j],ZGrid[k],0.5, Input1[id], Input2[id], Input3[id]);
                UTest_Omega_Hejlesen(     XGrid[i],YGrid[j],ZGrid[k],0.5, Solution1[id], Solution2[id], Solution3[id]);
            }
        }
    }
    Status = Solver->Set_Input_Unbounded_3D(Input1,Input2,Input3);
    ASSERT_EQ(Status, SailFFish::NoError);
    Status = Solver->Transfer_Data_Device();
    ASSERT_EQ(Status, SailFFish::NoError);
    unsigned int t3 = stopwatch();  // Timer

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Transfer_FTInOut_Comp();        // Transfer input to output array
    Solver->Spectral_Gradients_3DV_Nabla();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output_Unbounded_3D(Output1,Output2,Output3);
    ASSERT_EQ(Output1.size(), Solution1.size());
    ASSERT_EQ(Output2.size(), Solution2.size());
    ASSERT_EQ(Output3.size(), Solution3.size());
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    std::cout << "Trial Calculation: Solution of the unbounded 3D Poisson equation - nabla type (vector field)." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<" , "<< NY <<" , "<< NZ <<"] cells. " << std::endl;
    std::cout << "E_inf Error (x component) =" csp std::scientific << E_Inf(Output1,Solution1) << std::endl;
    std::cout << "E_inf Error (y component) =" csp std::scientific << E_Inf(Output2,Solution2) << std::endl;
    std::cout << "E_inf Error (z component) =" csp std::scientific << E_Inf(Output3,Solution3) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;
    std::cout << std::scientific;

    // If selected, export grid as VTI
    if (ExportVTI) Solver->Create_vtk();

    EXPECT_LT(E_Inf(Output1,Solution1), Tol_3DV);
    EXPECT_LT(E_Inf(Output2,Solution2), Tol_3DV);
    EXPECT_LT(E_Inf(Output3,Solution3), Tol_3DV);
}

#endif // TEST_3DV_H
