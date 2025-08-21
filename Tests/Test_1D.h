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

    -> 1D Test cases for each of the solver types.

*****************************************************************************/

#ifndef TEST_1D_H
#define TEST_1D_H

#include "../src/SailFFish.h"
#include "Test_Functions.h"

//---------------------
//--- Bounded solvers
//---------------------

void Test_Dirichlet_1D(int NX)
{
    // Test case for 1D bounded Poisson solver with Dirichlet BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Dirichlet_1D *Solver = new SailFFish::Poisson_Dirichlet_1D();
    Status = Solver->Setup(UnitX,NX);
    // Status = Solver->Setup(HUnitX,NX);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Generate Input & Solution Arrays
    RVector Input, Output, Solution;
    int IS = 0;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) IS = 1;
    for (int i=IS; i<NX; i++){
        Real xs = XGrid[i]-UnitX[0];
        Input.push_back(    STest_Omega(xs,Lx));    // Input field
        Solution.push_back( STest_Phi(xs,Lx));      // Solution field
    //     Real xs = XGrid[i]-HUnitX[0];
    //     Input.push_back(    PNTest_Omega(xs));    // Input field
    //     Solution.push_back( PNTest_Phi(xs));      // Solution field
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

    // for (int i=0; i<NX; i++) std::cout << XGrid[i] csp Input[i] csp Output[i] csp Solution[i] csp Output[i]/Solution[i] << std::endl;  // Output do we return the correct result FFt+iFFT

    std::cout << "Trial Calculation: Solution of the 1D Poisson equation with Dirichlet boundary conditions." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<" , "<< 1 <<" , "<< 1 <<"] cells. " << std::endl;
    std::cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;

    delete Solver;
}

void Test_Dirichlet_1D_IHBC(int NX)
{
    // Test case for 1D bounded Poisson solver with Dirichlet BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Dirichlet_1D *Solver = new SailFFish::Poisson_Dirichlet_1D(SailFFish::REGULAR,SailFFish::FD2);
    Status = Solver->Setup(UnitX,NX);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Boundary conditions (these are arbitrary and refer to the boundary a (left) and b (right) of the domain)
    Real a = 4.0, b = -1.0;

    // Generate Input & Solution Arrays
    RVector Input, Output, Solution;
    int IS = 0;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) IS = 1;
    for (int i=IS; i<NX; i++){
        Real xs = XGrid[i]-UnitX[0];
        // std::cout << i csp xs << std::endl;
        Input.push_back(    IH_BC_Omega(xs,Lx,a,b));    // Input field
        Solution.push_back( IH_BC_Phi(xs,Lx,a,b));      // Solution field

        // std::cout << XGrid[i] csp xs csp IH_BC_Omega(xs,Lx,a,b) csp IH_BC_Phi(xs,Lx,a,b) << std::endl;
    }
    Status = Solver->Set_Input(Input);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Set BC
    Solver->Set_BC(a, b);

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output(Output);
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    // for (int i=0; i<NX; i++) std::cout << XGrid[i] csp Input[i] csp Output[i] csp Solution[i] csp Output[i]/Solution[i] << std::endl;  // Output do we return the correct result FFt+iFFT

    std::cout << "Trial Calculation: Solution of the 1D Poisson equation with inhomogeneous Dirichlet boundary conditions." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<" , "<< 1 <<" , "<< 1 <<"] cells. " << std::endl;
    std::cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;

    delete Solver;
}

void Test_Neumann_1D(int NX)
{
    // Test case for 1D bounded Poisson solver with Neumann BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Neumann_1D *Solver = new SailFFish::Poisson_Neumann_1D();
    Status = Solver->Setup(UnitX,NX);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Boundary conditions (if carrying out inhomogeneous test)
//    Real a = -7.0, b = -6.0;

    // Generate Input & Solution Arrays
    RVector Input, Output, Solution;
    int IS = 0;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) IS = 1;
    for (int i=0; i<NX+IS; i++){                     // With the Neumann solver the boundary values are included.
        Real xs = XGrid[i]-UnitX[0];
        Input.push_back(    CTest_Omega(xs,Lx));    // Input field
        Solution.push_back( CTest_Phi(xs,Lx));      // Solution field
//        Input.push_back(    IH_BC_Omega(xs,Lx,a,b));    // Input field (if carrying out inhomogeneous test)
//        Solution.push_back( IH_BC_Phi(xs,Lx,a,b));      // Solution field (if carrying out inhomogeneous test)
    }
    Status = Solver->Set_Input(Input);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Set BC
//    Solver->Set_BC(IH_DTest_GradPhi(0,Lx,a,b),-IH_DTest_GradPhi(Lx,Lx,a,b));

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    Solver->Get_Output(Output);                         // Retrieve solution

//    // If using nonhomogenous BCs, shift solution to align
//    Real MaxSol = *std::max_element(Solution.begin(), Solution.end());
//    Real MaxOut = *std::max_element(Output.begin(), Output.end());
//    for (int i=0; i<Output.size(); i++) Output[i] += (MaxSol-MaxOut);

    // Retrieve solution & collect final timings
    Solver->Get_Output(Output);
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    std::cout << "Trial Calculation: Solution of the 1D Poisson equation with Neumann boundary conditions." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<" , "<< 1 <<" , "<< 1 <<"] cells. " << std::endl;
    std::cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;

    delete Solver;
}

void Test_Periodic_1D(int NX)
{
    // Test case for 1D bounded Poisson solver with periodic BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Periodic_1D *Solver = new SailFFish::Poisson_Periodic_1D();
    Status = Solver->Setup(UnitX,NX);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Generate Input & Solution Arrays
    RVector Input, Output, Solution;
    for (int i=0; i<NX; i++){                     // Shifted: with the Periodic (regular) solver we have 1 less unknown than grid nodes
        Real xs = XGrid[i];                         // For periodic solve, this doesn't need to be shifted
        Input.push_back(    CTest_Omega(xs,Lx));    // Input field
        Solution.push_back( CTest_Phi(xs,Lx));      // Solution field
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

    std::cout << "Trial Calculation: Solution of the 1D Poisson equation with Periodic boundary conditions." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<" , "<< 1 <<" , "<< 1 <<"] cells. " << std::endl;
    std::cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;

    delete Solver;
}

//---------------------
//--- Unbounded solvers
//---------------------

void Test_Unbounded_1D(int NX)
{
    // Test case for 1D unbounded Poisson solver

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Unbounded_Solver_1D *Solver = new SailFFish::Unbounded_Solver_1D();
    Status = Solver->Setup(UnitX,NX);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Generate Input & solution arrays
    RVector Input, Output, Solution;
    int NXM = XGrid.size();
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) NXM++;
    for (int i=0; i<NXM; i++){
        Input.push_back(    UTest_Omega(fabs(XGrid[i])));       // Input field
        Solution.push_back( UTest_Phi(fabs(XGrid[i])));         // Solution field
    }
    Status = Solver->Set_Input_Unbounded_1D(Input);
    if (Status!=SailFFish::NoError)   {std::cout << "Solver exiting." << std::endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output_Unbounded_1D(Output);
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    std::cout << "Trial Calculation: Solution of the unbounded 1D Poisson equation." << std::endl;
    std::cout << std::scientific;
    std::cout << "The grid was resolved with [" << NX <<" , "<< 1 <<" , "<< 1 <<"] cells. " << std::endl;
    std::cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Execution Time:" <<  std::endl;
    std::cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << std::endl;
    std::cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << std::endl;
    std::cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << std::endl;
    std::cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << std::endl;

    delete Solver;
}

#endif // TEST_1D_H
