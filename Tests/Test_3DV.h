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

#include "../src/SailFFish.h"
#include "Test_Functions.h"

//---------------------
//--- Bounded solvers
//---------------------

void Test_Dirichlet_3DV(int NX, int NY, int NZ, bool ExportVTI = false)
{
    // Test case for 3D (vector field) bounded Poisson solver with Dirichlet BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Dirichlet_3DV *Solver = new SailFFish::Poisson_Dirichlet_3DV();
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
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
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
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
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    cout << "Trial Calculation: Solution of the 3D Poisson equation (vector field) with Dirichlet boundary conditions." << endl;
    cout << std::scientific;
    cout << "The grid was resolved with [" << NX <<","<< NY <<","<< NZ <<"] cells." << endl;
    cout << "E_inf Error (x component) =" csp std::scientific << E_Inf(Output1,Solution) << endl;
    cout << "E_inf Error (y component) =" csp std::scientific << E_Inf(Output2,Solution) << endl;
    cout << "E_inf Error (z component) =" csp std::scientific << E_Inf(Output3,Solution) << endl;
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

void Test_Neumann_3DV(int NX, int NY, int NZ, bool ExportVTI = false)
{
    // Test case for 3D (vector field) bounded Poisson solver with Neumann BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Neumann_3DV *Solver = new SailFFish::Poisson_Neumann_3DV(SailFFish::REGULAR);
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
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
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output(Output1,Output2,Output3);
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    cout << "Trial Calculation: Solution of the 3D Poisson equation (vector field) with Neumann boundary conditions." << endl;
    cout << std::scientific;
    cout << "The grid was resolved with [" << NX <<","<< NY <<","<< NZ <<"] cells." << endl;
    cout << "E_inf Error (x component) =" csp std::scientific << E_Inf(Output1,Solution) << endl;
    cout << "E_inf Error (y component) =" csp std::scientific << E_Inf(Output2,Solution) << endl;
    cout << "E_inf Error (z component) =" csp std::scientific << E_Inf(Output3,Solution) << endl;
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

void Test_Periodic_3DV(int NX, int NY, int NZ, bool ExportVTI = false)
{
    // Test case for 3D (vector field) bounded Poisson solver with periodic BCs

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Poisson_Periodic_3DV *Solver = new SailFFish::Poisson_Periodic_3DV();
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
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
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output(Output1,Output2,Output3);
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    cout << "Trial Calculation: Solution of the 3D Poisson equation (vector field) with periodic boundary conditions." << endl;
    cout << std::scientific;
    cout << "The grid was resolved with [" << NX <<","<< NY <<","<< NZ <<"] cells." << endl;
    cout << "E_inf Error (x component) =" csp std::scientific << E_Inf(Output1,Solution) << endl;
    cout << "E_inf Error (y component) =" csp std::scientific << E_Inf(Output2,Solution) << endl;
    cout << "E_inf Error (z component) =" csp std::scientific << E_Inf(Output3,Solution) << endl;
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

//---------------------
//--- Unbounded solvers
//---------------------

void Test_Unbounded_3DV_Curl(int NX, int NY, int NZ, bool ExportVTI = false)
{
    // Test case for 2D unbounded Poisson solver

    SailFFish::SFStatus Status = SailFFish::NoError;    // Status of execution
    stopwatch();                                        // Begin timer for profiling

    // Define solver
    SailFFish::Unbounded_Solver_3DV *Solver = new SailFFish::Unbounded_Solver_3DV();
    Solver->Specify_Operator(SailFFish::CURL);
    Status = Solver->Setup(UnitX,UnitY,UnitZ,NX,NY,NZ);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
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
                RVector Inp = UTest_Omega_Hejlesen(     XGrid[i],YGrid[j],ZGrid[k],0.5);
                RVector Sol = UTest_Velocity_Hejlesen(  XGrid[i],YGrid[j],ZGrid[k],0.5);
                Input1[id] = Inp[0];        // X Vorticity
                Input2[id] = Inp[1];        // Y Vorticity
                Input3[id] = Inp[2];        // Z Vorticity
                Solution1[id] = Sol[0];     // X Velocity
                Solution2[id] = Sol[1];     // Y Velocity
                Solution3[id] = Sol[2];     // Z Velocity
            }
        }
    }
    Status = Solver->Set_Input_Unbounded_3D(Input1,Input2,Input3);
    if (Status!=SailFFish::NoError)   {cout << "Solver exiting." << endl; return;}
    unsigned int t3 = stopwatch();  // Timer

    // Carry out execution
    Solver->Forward_Transform();
    Solver->Convolution();
    Solver->Backward_Transform();
    unsigned int t4 = stopwatch();

    // Retrieve solution & collect final timings
    Solver->Get_Output_Unbounded_3D(Output1,Output2,Output3);
    unsigned int t5 = stopwatch();
    Real tTot = Real(t2+t3+t4+t5);

    cout << "Trial Calculation: Solution of the unbounded 3D Poisson equation (vector field)." << endl;
    cout << std::scientific;
    cout << "The grid was resolved with [" << NX <<" , "<< NY <<" , "<< NZ <<"] cells. " << endl;
    cout << "E_inf Error (x component) =" csp std::scientific << E_Inf(Output1,Solution1) << endl;
    cout << "E_inf Error (y component) =" csp std::scientific << E_Inf(Output2,Solution2) << endl;
    cout << "E_inf Error (z component) =" csp std::scientific << E_Inf(Output3,Solution3) << endl;
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

#endif // TEST_3DV_H
