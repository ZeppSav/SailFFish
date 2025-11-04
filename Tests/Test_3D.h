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

    -> 3D Test cases for each of the solver types.

*****************************************************************************/

#ifndef TEST_3D_H
#define TEST_3D_H

#include "../src/SailFFish.h"
#include "Test_Functions.h"

//---------------------
//--- Bounded solvers
//---------------------

void Test_Dirichlet_3D(int NX, int NY, int NZ, bool ExportVTK = false)
{
    // Test case for 3D bounded Poisson solver with Dirichlet BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Dirichlet_3D *Solver = new SailFFish::Poisson_Dirichlet_3D(SailFFish::STAGGERED, SailFFish::PS);
    // SailFFish::Poisson_Dirichlet_3D *Solver = new SailFFish::Poisson_Dirichlet_3D(SailFFish::REGULAR, SailFFish::PS);
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_ZGrid(ZGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Generate Input & Solution Arrays
    RVector Input, Output, Solution;
    int IS = 0;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) IS = 1;
    for (int i=IS; i<NX; i++){                       // Shifted: with the dirichlet solver the boundary values are ignored.
        for (int j=IS; j<NY; j++){                       // Shifted: with the dirichlet solver the boundary values are ignored.
            for (int k=IS; k<NZ; k++){                       // Shifted: with the dirichlet solver the boundary values are ignored.
                Real xs = XGrid[i]-UnitX[0];
                Real ys = YGrid[j]-UnitY[0];
                Real zs = ZGrid[k]-UnitZ[0];
                Input.push_back(    STest_Omega(xs,Lx,ys,Ly,zs,Lz));    // Input field
                Solution.push_back( STest_Phi(xs,Lx,ys,Ly,zs,Lz));      // Solution field
            }
        }
    }
    Status = Solver->Set_Input(Input);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output(Output);
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    std::cout << "Trial Calculation: Solution of the 3D Poisson equation with Dirichlet boundary conditions." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<","<< NY <<","<< NZ <<"] cells." << std::endl;
    std::cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;
    std::cout << std::scientific;

    // If selected, export grid as VTK
    if (ExportVTK) Solver->Create_vtk();

    delete Solver;
}

void Test_Dirichlet_3D_IHBC(int NX, int NY, int NZ, bool ExportVTK = false)
{
    // Test case for 3D bounded Poisson solver with Dirichlet BCs
    // IHBC: Inhomogeneous boundary conditions

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Dirichlet_3D *Solver = new SailFFish::Poisson_Dirichlet_3D(SailFFish::STAGGERED, SailFFish::FD2);
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_ZGrid(ZGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Boundary conditions (these are arbitrary and refer to the boundary a (left) and b (right) of the domain)
    Real a = 4.0, b = -1.0;     // X BCs
    Real c = 2.0, d = -7.0;     // Y BCs
    Real e = 1.0, f = -12.0;    // Z BCs

    // Generate Input & Solution Arrays
    RVector Input, Output, Solution;
    int IS = 0;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) IS = 1;
    for (int i=IS; i<NX; i++){                       // Shifted: with the dirichlet solver the boundary values are ignored.
        for (int j=IS; j<NY; j++){                       // Shifted: with the dirichlet solver the boundary values are ignored.
            for (int k=IS; k<NZ; k++){                       // Shifted: with the dirichlet solver the boundary values are ignored.
                Real xs = XGrid[i]-UnitX[0];
                Real ys = YGrid[j]-UnitY[0];
                Real zs = ZGrid[k]-UnitZ[0];
                Input.push_back(    IH_BC_Omega  (xs,Lx,a,b, ys,Ly,c,d, zs,Lz,e,f) + STest_Omega(xs,Lx,ys,Ly,zs,Lz)  );    // Input field
                Solution.push_back( IH_BC_Phi    (xs,Lx,a,b, ys,Ly,c,d, zs,Lz,e,f) + STest_Phi(xs,Lx,ys,Ly,zs,Lz)  );    // Solution field
            }
        }
    }
    Status = Solver->Set_Input(Input);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Set BC
    std::vector<Real> AX, BX, AY, BY, AZ, BZ;
    for (int j=0; j<NY-IS; j++){
        for (int k=0; k<NZ-IS; k++)    AX.push_back(IH_BC_Phi(0,Lx,a,b,    YGrid[j+IS]-UnitY[0],Ly,c,d,    ZGrid[k+IS]-UnitZ[0],Lz,e,f));
        for (int k=0; k<NZ-IS; k++)    BX.push_back(IH_BC_Phi(Lx,Lx,a,b,   YGrid[j+IS]-UnitY[0],Ly,c,d,    ZGrid[k+IS]-UnitZ[0],Lz,e,f));
    }
    for (int i=0; i<NX-IS; i++){
        for (int k=0; k<NZ-IS; k++)    AY.push_back(IH_BC_Phi(XGrid[i+IS]-UnitX[0],Lx,a,b,     0,Ly,c,d,   ZGrid[k+IS]-UnitZ[0],Lz,e,f));
        for (int k=0; k<NZ-IS; k++)    BY.push_back(IH_BC_Phi(XGrid[i+IS]-UnitX[0],Lx,a,b,     Ly,Ly,c,d,  ZGrid[k+IS]-UnitZ[0],Lz,e,f));
    }
    for (int i=0; i<NX-IS; i++){
        for (int j=0; j<NY-IS; j++)    AZ.push_back(IH_BC_Phi(XGrid[i+IS]-UnitX[0],Lx,a,b, YGrid[j+IS]-UnitY[0],Ly,c,d,    0,Lz,e,f));
        for (int j=0; j<NY-IS; j++)    BZ.push_back(IH_BC_Phi(XGrid[i+IS]-UnitX[0],Lx,a,b, YGrid[j+IS]-UnitY[0],Ly,c,d,    Lz,Lz,e,f));
    }
    Solver->Set_BC(AX, BX, AY, BY, AZ, BZ);

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output(Output);
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    std::cout << "Trial Calculation: Solution of the 3D Poisson equation with Dirichlet boundary conditions." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<","<< NY <<","<< NZ <<"] cells." << std::endl;
    std::cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;
    std::cout << std::scientific;

    // If selected, export grid as VTK
    if (ExportVTK) Solver->Create_vtk();

    delete Solver;
}

void Test_Neumann_3D(int NX, int NY, int NZ, bool ExportVTK = false)
{
    // Test case for 3D bounded Poisson solver with Neumann BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Neumann_3D *Solver = new SailFFish::Poisson_Neumann_3D(SailFFish::STAGGERED, SailFFish::PS);
    // SailFFish::Poisson_Neumann_3D *Solver = new SailFFish::Poisson_Neumann_3D(SailFFish::REGULAR, SailFFish::PS);
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_ZGrid(ZGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Generate Input & Solution Arrays
    RVector Input, Output, Solution;
    int IS = 0;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) IS = 1;      // With the Neumann solver the boundary values are included.
    for (int i=0; i<NX+IS; i++){
        for (int j=0; j<NY+IS; j++){
            for (int k=0; k<NZ+IS; k++){
                Input.push_back(    CTest_Omega(XGrid[i]-UnitX[0],Lx,YGrid[j]-UnitY[0],Ly,ZGrid[k]-UnitZ[0],Lz));    // Input field
                Solution.push_back( CTest_Phi(XGrid[i]-UnitX[0],Lx,YGrid[j]-UnitY[0],Ly,ZGrid[k]-UnitZ[0],Lz));      // Solution field
            }
        }
    }
    Status = Solver->Set_Input(Input);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output(Output);
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    std::cout << "Trial Calculation: Solution of the 3D Poisson equation with Neumann boundary conditions." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<","<< NY <<","<< NZ <<"] cells." << std::endl;
    std::cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;
    std::cout << std::scientific;

    // If selected, export grid as VTK
    if (ExportVTK) Solver->Create_vtk();

    delete Solver;
}

void Test_Neumann_3D_IHBC(int NX, int NY, int NZ, bool ExportVTK = false)
{
    // Test case for 3D bounded Poisson solver with Neumann BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Neumann_3D *Solver = new SailFFish::Poisson_Neumann_3D(SailFFish::REGULAR,SailFFish::FD2);
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_ZGrid(ZGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Boundary conditions (these are arbitrary and refer to the boundary a (left) and b (right) of the domain)
    Real a = 2.0, b = -1.0;     //  X BCs
    Real c = 1.0, d = -2.0;     //  Y BCs
    Real e = 2.0, f = -1.0;     //  Z BCs

    // Generate Input & Solution Arrays
    RVector Input, Output, Solution;
    int IS = 0;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) IS = 1;      // With the Neumann solver the boundary values are included.
    for (int i=0; i<NX+IS; i++){
        for (int j=0; j<NY+IS; j++){
            for (int k=0; k<NZ+IS; k++){
                Real xs = XGrid[i]-UnitX[0];
                Real ys = YGrid[j]-UnitY[0];
                Real zs = ZGrid[k]-UnitZ[0];
                Input.push_back(    IH_BC_Omega(xs,Lx,a,b, ys,Ly,c,d, zs,Lz,e,f) + CTest_Omega(xs,Lx,ys,Ly,zs,Lz)   );     // Input field
                Solution.push_back( IH_BC_Phi(  xs,Lx,a,b, ys,Ly,c,d, zs,Lz,e,f) + CTest_Phi(xs,Lx,ys,Ly,zs,Lz)     );     // Solution field
            }
        }
    }
    Status = Solver->Set_Input(Input);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Set BC
    std::vector<Real> AX, BX, AY, BY, AZ, BZ;
    for (int j=0; j<NY+IS; j++){
        for (int k=0; k<NZ+IS; k++)    AX.push_back(IH_BC_GradPhi(0,Lx,a,b));
        for (int k=0; k<NZ+IS; k++)    BX.push_back(-IH_BC_GradPhi(Lx,Lx,a,b));
    }
    for (int i=0; i<NX+IS; i++){
        for (int k=0; k<NZ+IS; k++)    AY.push_back(IH_BC_GradPhi(0,Ly,c,d));
        for (int k=0; k<NZ+IS; k++)    BY.push_back(-IH_BC_GradPhi(Ly,Ly,c,d));
    }
    for (int i=0; i<NX+IS; i++){
        for (int j=0; j<NY+IS; j++)    AZ.push_back(IH_BC_GradPhi(0,Lz,e,f));
        for (int j=0; j<NY+IS; j++)    BZ.push_back(-IH_BC_GradPhi(Lz,Lz,e,f));
    }
    Solver->Set_BC(AX, BX, AY, BY, AZ, BZ);

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output(Output);
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    // Remember: The solution for fully Neumann BCs is correct up to a constant, so eliminate this here.
    Real MaxSol = *std::max_element(Solution.begin(), Solution.end());
    Real MaxOut = *std::max_element(Output.begin(), Output.end());
    for (int i=0; i<Output.size(); i++) Output[i] += (MaxSol-MaxOut);

    std::cout << "Trial Calculation: Solution of the 3D Poisson equation with Neumann boundary conditions." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<","<< NY <<","<< NZ <<"] cells." << std::endl;
    std::cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;
    std::cout << std::scientific;

    // If selected, export grid as VTK
    if (ExportVTK) Solver->Create_vtk();

    delete Solver;
}

void Test_Periodic_3D(int NX, int NY, int NZ, bool ExportVTK = false)
{
    // Test case for 3D bounded Poisson solver with periodic BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Periodic_3D *Solver = new SailFFish::Poisson_Periodic_3D();
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_ZGrid(ZGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Generate Input & Solution Arrays
    RVector Input, Output, Solution;
    for (int i=0; i<NX; i++){                           // Shifted: with the Periodic (regular) solver we have 1 less unknown than grid nodes
        for (int j=0; j<NY; j++){                       // Shifted: with the Periodic (regular) solver we have 1 less unknown than grid nodes
            for (int k=0; k<NZ; k++){                   // Shifted: with the Periodic (regular) solver we have 1 less unknown than grid nodes
                Real xs = XGrid[i];                     // For periodic solve, this doesn't need to be shifted
                Real ys = YGrid[j];                     // For periodic solve, this doesn't need to be shifted
                Real zs = ZGrid[k];                     // For periodic solve, this doesn't need to be shifted
                Input.push_back(    CTest_Omega(xs,Lx,ys,Ly,zs,Lz));    // Input field
                Solution.push_back( CTest_Phi(xs,Lx,ys,Ly,zs,Lz));      // Solution field
            }
        }
    }
    Status = Solver->Set_Input(Input);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output(Output);
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    std::cout << "Trial Calculation: Solution of the 3D Poisson equation with periodic boundary conditions." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<","<< NY <<","<< NZ <<"] cells." << std::endl;
    std::cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;
    std::cout << std::scientific;

    // If selected, export grid as VTK
    if (ExportVTK) Solver->Create_vtk();

    delete Solver;
}

//---------------------
//--- Unbounded solvers
//---------------------

void Test_Unbounded_3D(int NX, int NY, int NZ, bool ExportVTI = false)
{
    // Test case for 2D unbounded Poisson solver

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Unbounded_Solver_3D *Solver = new SailFFish::Unbounded_Solver_3D();
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
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

    // Generate Input & solution arrays
    // If this is a grid-boundary solve, we have 1 extra grid point
    int NXM = NX, NYM = NY, NZM = NZ;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) {NXM++; NYM++; NZM++;}
    int NT = NXM*NYM*NZM;
    RVector Input = RVector(NT,0);
    RVector Output = RVector(NT,0);
    RVector Solution = RVector(NT,0);
    OpenMPfor
    for (int i=0; i<NXM; i++){
        for (int j=0; j<NYM; j++){
            for (int k=0; k<NZM; k++){
                int id = i*NYM*NZM + j*NZM + k;
                Real d1, d2;
                UTest_Omega_Hejlesen(   XGrid[i],YGrid[j],ZGrid[k],0.5, d1, Input[id], d2);
                UTest_Phi_Hejlesen(     XGrid[i],YGrid[j],ZGrid[k],0.5, d1, Solution[id], d2);
            }
        }
    }

    // Scale input/output for Linf
    Real EFac = 1.0/exp(-Cbf);
    for (auto& i : Input)    i *= EFac;
    for (auto& i : Solution) i *= EFac;
    Status = Solver->Set_Input_Unbounded_3D(Input);
    // Status = Solver->Set_Input_Unbounded(Input);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Solver->Get_Output_Unbounded_3D(Output);                    // Retrieve solution
    Solver->Get_Output_Unbounded(Output);                    // Retrieve solution
    Real Error = Error_LInf(Output,Solution,Hx*Hy*Hz);          // Calculate error
    unsigned int t5 = stopwatch();  // Timer
    Real tTot = Real(t2+t3+t4+t5);  // Sum times

    std::cout << "Trial Calculation: Solution of the unbounded 3D Poisson equation." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<" , "<< NY <<" , "<< NZ <<"] cells. " << std::endl;
    std::cout << "E_inf Error =" csp std::scientific << Error << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;
    std::cout << std::scientific;

    if (ExportVTI)  Solver->Create_vtk();

    delete Solver;
}

#endif // TEST_3D_H
