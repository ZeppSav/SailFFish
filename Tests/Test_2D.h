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

    -> 2D Test cases for each of the solver types.

*****************************************************************************/

#ifndef TEST_2D_H
#define TEST_2D_H

#include "../src/Solvers/Solvers.h"
#include "Test_Functions.h"

//--- Grid vars

//Real X[2] = {-1.0, 1.0};
//Real Y[2] = {-1.0, 1.0};

//---------------------
//--- Bounded solvers
//---------------------

void Test_Dirichlet_2D(int NX, int NY, bool ExportVTI = false)
{
    // Test case for 2D bounded Poisson solver with Dirichlet BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    SailFFish::Poisson_Dirichlet_2D *Solver = new SailFFish::Poisson_Dirichlet_2D();
    Status = Solver->Setup(UnitX,UnitY,NX,NY);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Generate Input & Solution Arrays
    RVector Input, Output, Solution;
    int IS = 0;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) IS = 1;
    for (int i=IS; i<NX; i++){                       // Shifted: with the dirichlet solver the boundary values are ignored.
        for (int j=IS; j<NY; j++){                       // Shifted: with the dirichlet solver the boundary values are ignored.
            Real xs = XGrid[i]-UnitX[0];
            Real ys = YGrid[j]-UnitY[0];
            Input.push_back(    STest_Omega(xs,Lx,ys,Ly));    // Input field
            Solution.push_back( STest_Phi(xs,Lx,ys,Ly));      // Solution field
        }
    }
    Status = Solver->Set_Input(Input);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
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

    cout << "Trial Calculation: Solution of the 2D Poisson equation with Dirichlet boundary conditions." << endl;
    cout << std::scientific;
    cout << "The grid was resolved with [" << NX <<" , "<< NY <<" , "<< 1 <<"] cells. " << endl;
    cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << endl;
    cout << std::fixed << std::setprecision(1);
    cout << "Execution Time:" <<  endl;
    cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << endl;
    cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << endl;
    cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << endl;
    cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << endl;
    cout << std::scientific;

    // If selected, export grid as VTI
    if (ExportVTI) Solver->Create_vti();

    delete Solver;
}

void Test_Dirichlet_2D_IHBC(int NX, int NY, bool ExportVTI = false)
{
    // Test case for 2D bounded Poisson solver with Dirichlet BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Dirichlet_2D *Solver = new SailFFish::Poisson_Dirichlet_2D(SailFFish::REGULAR, SailFFish::FD2);
    Status = Solver->Setup(UnitX,UnitY,NX,NY);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Boundary conditions (these are arbitrary and refer to the boundary a (left) and b (right) of the domain)
    Real a = 2.0, b = -1.0;     // X BCs
    Real c = 1.0, d = -2.0;     // Y BCs

    // Generate Input & Solution Arrays
    RVector Input, Output, Solution;
    int IS = 0;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) IS = 1;
    for (int i=IS; i<NX; i++){                       // Shifted: with the dirichlet solver the boundary values are ignored.
        for (int j=IS; j<NY; j++){                       // Shifted: with the dirichlet solver the boundary values are ignored.
            Real xs = XGrid[i]-UnitX[0];
            Real ys = YGrid[j]-UnitY[0];
            Input.push_back(    IH_BC_Omega(xs,Lx,a,b, ys,Ly,c,d)+ STest_Omega(xs,Lx,ys,Ly)  );     // Input field
            Solution.push_back( IH_BC_Phi(xs,Lx,a,b, ys,Ly,c,d)  + STest_Phi(xs,Lx,ys,Ly)    );     // Solution field
        }
    }
    Status = Solver->Set_Input(Input);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Set BC
    std::vector<Real> AX, BX, AY, BY;
    for (int j=0; j<NY; j++) AX.push_back(IH_BC_Phi(0,Lx,a,b,YGrid[j+IS]-UnitY[0],Ly,c,d));
    for (int j=0; j<NY; j++) BX.push_back(IH_BC_Phi(Lx,Lx,a,b,YGrid[j+IS]-UnitY[0],Ly,c,d));
    for (int i=0; i<NX; i++) AY.push_back(IH_BC_Phi(XGrid[i+IS]-UnitX[0],Lx,a,b,0,Ly,c,d));
    for (int i=0; i<NX; i++) BY.push_back(IH_BC_Phi(XGrid[i+IS]-UnitX[0],Lx,a,b,Ly,Ly,c,d));
    Solver->Set_BC(AX, BX, AY, BY);

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output(Output);
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    cout << "Trial Calculation: Solution of the 2D Poisson equation with Dirichlet boundary conditions." << endl;
    cout << std::scientific;
    cout << "The grid was resolved with [" << NX <<" , "<< NY <<" , "<< 1 <<"] cells. " << endl;
    cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << endl;
    cout << std::fixed << std::setprecision(1);
    cout << "Execution Time:" <<  endl;
    cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << endl;
    cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << endl;
    cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << endl;
    cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << endl;
    cout << std::scientific;

    // If selected, export grid as VTI
    if (ExportVTI) Solver->Create_vti();

    delete Solver;
}

void Test_Neumann_2D(int NX, int NY, bool ExportVTI = false)
{
    // Test case for 2D bounded Poisson solver with Neumann BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Neumann_2D *Solver = new SailFFish::Poisson_Neumann_2D();
    Status = Solver->Setup(UnitX,UnitY,NX,NY);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Generate Input & Solution Arrays
    RVector Input, Output, Solution;
    int IS = 0;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) IS = 1;
    for (int i=0; i<NX+IS; i++){                     // With the Neumann solver the boundary values are included.
        for (int j=0; j<NY+IS; j++){                 // With the Neumann solver the boundary values are included.
            Real xs = XGrid[i]-UnitX[0];
            Real ys = YGrid[j]-UnitY[0];
            Input.push_back(    CTest_Omega(xs,Lx,ys,Ly));    // Input field
            Solution.push_back( CTest_Phi(xs,Lx,ys,Ly));      // Solution field
        }
    }
    Status = Solver->Set_Input(Input);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
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

    cout << "Trial Calculation: Solution of the 2D Poisson equation with Neumann boundary conditions." << endl;
    cout << std::scientific;
    cout << "The grid was resolved with [" << NX <<" , "<< NY <<" , "<< 1 <<"] cells. " << endl;
    cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << endl;
    cout << std::fixed << std::setprecision(1);
    cout << "Execution Time:" <<  endl;
    cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << endl;
    cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << endl;
    cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << endl;
    cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << endl;
    cout << std::scientific;

    // If selected, export grid as VTI
    if (ExportVTI) Solver->Create_vti();

    delete Solver;
}

void Test_Neumann_2D_IHBC(int NX, int NY, bool ExportVTI = false)
{
    // Test case for 2D bounded Poisson solver with Neumann BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Neumann_2D *Solver = new SailFFish::Poisson_Neumann_2D(SailFFish::REGULAR, SailFFish::FD2);
    Status = Solver->Setup(UnitX,UnitY,NX,NY);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Boundary conditions (these are arbitrary and refer to the boundary a (left) and b (right) of the domain)
    Real a = 2.0, b = -1.0;     // X BCs
    Real c = 1.0, d = -2.0;     // Y BCs

    // Generate Input & Solution Arrays
    RVector Input, Output, Solution;
    int IS = 0;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) IS = 1;
    for (int i=0; i<NX+IS; i++){                     // With the Neumann solver the boundary values are included.
        for (int j=0; j<NY+IS; j++){                 // With the Neumann solver the boundary values are included.
            Real xs = XGrid[i]-UnitX[0];
            Real ys = YGrid[j]-UnitY[0];
            Input.push_back(    IH_BC_Omega(xs,Lx,a,b, ys,Ly,c,d) + CTest_Omega(xs,Lx,ys,Ly)    );     // Input field
            Solution.push_back( IH_BC_Phi(  xs,Lx,a,b, ys,Ly,c,d) + CTest_Phi(xs,Lx,ys,Ly)   );     // Solution field
        }
    }
    Status = Solver->Set_Input(Input);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Set BC
    std::vector<Real> AX, BX, AY, BY;
    for (int j=0; j<NY+IS; j++) AX.push_back(IH_BC_GradPhi(0,Lx,a,b));
    for (int j=0; j<NY+IS; j++) BX.push_back(-IH_BC_GradPhi(Lx,Lx,a,b));
    for (int i=0; i<NX+IS; i++) AY.push_back(IH_BC_GradPhi(0,Ly,c,d));
    for (int i=0; i<NX+IS; i++) BY.push_back(-IH_BC_GradPhi(Ly,Ly,c,d));
    Solver->Set_BC(AX, BX, AY, BY);

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output(Output);
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    cout << "Trial Calculation: Solution of the 2D Poisson equation with inhomogeneous Neumann boundary conditions." << endl;
    cout << std::scientific;
    cout << "The grid was resolved with [" << NX <<" , "<< NY <<" , "<< 1 <<"] cells. " << endl;
    cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << endl;
    cout << std::fixed << std::setprecision(1);
    cout << "Execution Time:" <<  endl;
    cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << endl;
    cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << endl;
    cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << endl;
    cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << endl;
    cout << std::scientific;

    // If selected, export grid as VTI
    if (ExportVTI) Solver->Create_vti();

    delete Solver;
}

void Test_Periodic_2D(int NX, int NY, bool ExportVTI = false)
{
    // Test case for 2D bounded Poisson solver with periodic BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Periodic_2D *Solver = new SailFFish::Poisson_Periodic_2D();
    Status = Solver->Setup(UnitX,UnitY,NX,NY);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Generate Input & Solution Arrays
    RVector Input, Output, Solution;
    for (int i=0; i<NX; i++){                           // Shifted: with the Periodic (regular) solver we have 1 less unknown than grid nodes
        for (int j=0; j<NY; j++){                       // Shifted: with the Periodic (regular) solver we have 1 less unknown than grid nodes
            Real xs = XGrid[i];                         // For periodic solve, this doesn't need to be shifted
            Real ys = YGrid[j];                         // For periodic solve, this doesn't need to be shifted
            Input.push_back(    CTest_Omega(xs,Lx,ys,Ly));    // Input field
            Solution.push_back( CTest_Phi(xs,Lx,ys,Ly));      // Solution field
        }
    }
    Status = Solver->Set_Input(Input);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
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

    cout << "Trial Calculation: Solution of the 2D Poisson equation with Periodic boundary conditions." << endl;
    cout << std::scientific;
    cout << "The grid was resolved with [" << NX <<" , "<< NY <<" , "<< 1 <<"] cells. " << endl;
    cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << endl;
    cout << std::fixed << std::setprecision(1);
    cout << "Execution Time:" <<  endl;
    cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << endl;
    cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << endl;
    cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << endl;
    cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << endl;
    cout << std::scientific;

//    cout << NX csp std::scientific << E_Inf(Output,Solution) << endl;

    // If selected, export grid as VTI
    if (ExportVTI) Solver->Create_vti();

    delete Solver;
}

//---------------------
//--- Unbounded solvers
//---------------------

void Test_Unbounded_2D(int NX, int NY, bool ExportVTI = false)
{
    // Test case for 2D unbounded Poisson solver

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Unbounded_Solver_2D *Solver = new SailFFish::Unbounded_Solver_2D();
    Status = Solver->Setup(UnitX,UnitY,NX,NY);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);

    // Generate Input & solution arrays
    RVector Input, Output, Solution;
    // If this is a grid-boundary solve, we have 1 extra grid point
    int NXM = NX, NYM = NY;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) {NXM++; NYM++;}
    for (int i=0; i<NXM; i++){
        for (int j=0; j<NYM; j++){
            Input.push_back(    UTest_Omega_Hejlesen(XGrid[i],YGrid[j],1.0));   // Input field
            Solution.push_back( UTest_Phi(XGrid[i],YGrid[j]));                  // Solution field
        }
    }
    Status = Solver->Set_Input_Unbounded_2D(Input);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output_Unbounded_2D(Output);
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    cout << "Trial Calculation: Solution of the unbounded 2D Poisson equation." << endl;
    cout << std::scientific;
    cout << "The grid was resolved with [" << NX <<" , "<< NY <<" , "<< 1 <<"] cells. " << endl;
    cout << "E_inf Error =" csp std::scientific << E_Inf(Output,Solution) << endl;
    cout << std::fixed << std::setprecision(1);
    cout << "Execution Time:" <<  endl;
    cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << endl;
    cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << endl;
    cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << endl;
    cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << endl;
    cout << std::scientific;

    // If selected, export grid as VTI
    if (ExportVTI) Solver->Create_vti();

    delete Solver;
}

void Test_Unbounded_2D_Grad(int NX, int NY, bool ExportVTI = false)
{
    // Test case for 2D unbounded Poisson solver

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Unbounded_Solver_2D *Solver = new SailFFish::Unbounded_Solver_2D(SailFFish::STAGGERED, SailFFish::HEJ_S0);
    Solver->Specify_Operator(SailFFish::GRAD);
    Status = Solver->Setup(UnitX,UnitY,NX,NY);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
    unsigned int t2 = stopwatch();  // Timer

    // Extract grid values
    RVector XGrid, YGrid, ZGrid;
    Real Lx, Ly, Lz;
    Real Hx, Hy, Hz;
    Solver->Get_XGrid(XGrid);
    Solver->Get_YGrid(YGrid);
    Solver->Get_Grid_Dims(Lx, Ly, Lz);
    Solver->Get_Grid_Res(Hx, Hy, Hz);

    // Generate Input & solution arrays
    RVector Input, Output1, Output2, Output3, Solution1, Solution2;
    int NXM = NX, NYM = NY;
    if (Solver->Get_Grid_Type()==SailFFish::REGULAR) {NXM++; NYM++;}  // Issue here Joe
    for (int i=0; i<NXM; i++){
        for (int j=0; j<NYM; j++){
            Input.push_back(    UTest_Omega_Hejlesen(XGrid[i],YGrid[j],1.0));   // Input field
            Solution1.push_back(UTest_XGrad_Hejlesen(XGrid[i],YGrid[j],1.0));   // Solution field x gradient
            Solution2.push_back(UTest_YGrad_Hejlesen(XGrid[i],YGrid[j],1.0));   // Solution field y gradient
        }
    }
    Status = Solver->Set_Input_Unbounded_2D(Input);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output_Unbounded_2D(Output1,Output2);
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    cout << "Trial Calculation: Solution of the unbounded 2D Poisson equation." << endl;
    cout << std::scientific;
    cout << "The grid was resolved with [" << NX <<" , "<< NY <<" , "<< 1 <<"] cells. " << endl;
    cout << "E_inf Error X component =" csp std::scientific << E_Inf(Output1,Solution1) << endl;
    cout << "E_inf Error X component =" csp std::scientific << E_Inf(Output2,Solution2) << endl;
    cout << std::fixed << std::setprecision(1);
    cout << "Execution Time:" <<  endl;
    cout << "Solver setup "     << std::setw(10) << t2 csp "ms. [" << 100.0*t2/tTot << " %]" << endl;
    cout << "Input spec.  "     << std::setw(10) << t3 csp "ms. [" << 100.0*t3/tTot << " %]" << endl;
    cout << "Execution    "     << std::setw(10) << t4 csp "ms. [" << 100.0*t4/tTot << " %]" << endl;
    cout << "Output spec. "     << std::setw(10) << t5 csp "ms. [" << 100.0*t5/tTot << " %]" << endl;
    cout << std::scientific;

    // If selected, export grid as VTI
    if (ExportVTI) Solver->Create_vti();

    delete Solver;
}

#endif // TEST_2D_H
