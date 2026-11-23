/****************************************************************************
    SailFFish Library
    Copyright (C) 2025 Joseph Saverin j.saverin@tu-berlin.de

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

    -> Test cases for vortex ring simulations

*****************************************************************************/

#ifndef TEST_RINGS_H
#define TEST_RINGS_H

#include <gtest/gtest.h>
#include "SailFFish_Math_Types.h"
#ifdef FFTW
    #include "VPM_Solver/VPM3D_cpu.h"
#elif CUFFT
    #include "VPM_Solver/VPM3D_cuda.h"
#elif VKFFT
    #include "VPM_Solver/VPM3D_ocl.h"
#endif

using namespace SailFFish;

static const int NStep_VPM = 500;       // Number of time steps for vortex ring tests

TEST(SailFFish_VPM_Tests, Vortex_Ring_Evolution)
{
    // This is a test case to check the evolution of the vortex rings

    // Note: Ensure the rings are initiated with the same coresize sigma as is specified below
    // Note: Make sure the Saffman centroid is being calculated in the diagnostics calculations

    // --- VPM solver
    #ifdef FFTW
        auto VPM = std::make_unique<SailFFish::VPM3D_cpu>(SailFFish::STAGGERED, SailFFish::HEJ_G8);
    #elif CUFFT
        auto VPM = std::make_unique<SailFFish::VPM3D_cuda>(SailFFish::STAGGERED, SailFFish::HEJ_G8);
    #elif VKFFT
        auto VPM = std::make_unique<SailFFish::VPM3D_ocl>(SailFFish::STAGGERED, SailFFish::HEJ_G8);
    #endif

    //--- Initialize with VPM_Input
    SailFFish::VPM_Input I;

    // Both unsteady vortex ring cases have been taken from the following publication:
    // "Combining the vortex-in-cell and parallel fast multipole methods for efficient domain decomposition simulations."
    // R. Cocle, G. Winckelmans, G. Daeninck.-> https://doi.org/10.1016/j.jcp.2007.10.010

    //--------------------------------------------------------------------------------
    //------------ Cocle case 1

    // Specify grid parameters (Test case 1- Cocle et al. doi 10.1016/j.jcp.2007.10.010 )
    I.GridDef = SailFFish::BLOCKS;      // Block-style structuring is necessary for GPU tests
    I.H_Grid = 0.05;                    // Grid size
    I.BX = 4;                           // Block grid dims
    I.BY = 4;
    I.BZ = 4;
    I.iBX0 = -8;                        // Initial block indices
    I.iBY0 = -14;
    I.iBZ0 = -14;
    I.NBX = 64;                          // Number of blocks
    I.NBY = 28;
    I.NBZ = 28;
    dim3 G;

    // Simulation parameters
    I.SolverMap = SailFFish::M4D;       // Mapping scheme used within solver for mapping Lagrangian-Eulerian grids
    I.RemeshMap = SailFFish::M4D;       // Mapping scheme used within solver for remeshing
    I.FDOrder = SailFFish::CD8;         // Order of finite difference calculations on the grid
    I.NRemesh = 5;                      // Remeshing frequency
    I.NReproject = 5;                   // Reprojection frequency
    // I.MagFiltFac = 1.0e-5;
    I.MagFiltFac = 0;                   // Magnitude filtering factor
    I.DivFilt = true;                   // Is divergence filtering being carried out?
    // I.Turb = SailFFish::LAM;            // Turbulence model
    I.Turb = SailFFish::RVM2;           // Turbulence model
    I.C_smag = 2.5e-2;                  // Smagorisnky parameter
    I.KinVisc = 1.0;                    // Kinematic viscosity of fluid
    I.Rho = 1.0;                        // Density of fluid
    I.Integrator = SailFFish::LSRK4;    // Time integration scheme

    // Parameters of vortex ring
    Real R = 1.0;                       // Ring radius
    Real Rey = 5500;                    // Ring vorticity Reynolds number
    Real Coresize = 0.4131;             // Ring coresize
    Real Eps = 0.0002;                  // Epsilon of ring perturbation
    Real t0 = R*R/(Rey*I.KinVisc);      // Ring charactersistic time
    I.dT = 0.01*t0;                     // Timestep size
    RVector PhaseShift;                 // Random phase shifts of different modes
    for (int i=0; i<24; i++) PhaseShift.push_back(rand()*1.0/RAND_MAX*M_2PI);

    // Output parameters
    I.NExp = 20;                  // Frequency of export of visualisation
    I.Debug = true;                 // Debugging output of solver
    I.Log = true;                   // Logging output
    I.OutputFolder = "SailFFish_VPM_VortexRing1"; // Output directory (will be created)

    //--------------------------------------------------------------------------------

    // //--------------------------------------------------------------------------------
    // //------------ Cocle case 2

    // I.GridDef = SailFFish::BLOCKS;      // Block-style structuring is necessary for GPU tests
    // I.H_Grid = 0.02;                    // Grid size
    // I.BX = 8;                           // Block grid dims
    // I.BY = 8;
    // I.BZ = 8;
    // I.iBX0 = -6;                        // Initial block indices
    // I.iBY0 = -10;
    // I.iBZ0 = -10;
    // I.NBX = 12;                          // Number of blocks
    // I.NBY = 20;
    // I.NBZ = 20;

    // // Simulation parameters
    // I.SolverMap = SailFFish::M4D;       // Mapping scheme used within solver for mapping Lagrangian-Eulerian grids
    // I.RemeshMap = SailFFish::M4D;       // Mapping scheme used within solver for remeshing
    // I.FDOrder = SailFFish::CD8;         // Order of finite difference calculations on the grid
    // I.NRemesh = 5;                      // Remeshing frequency
    // I.NReproject = 5;                   // Reprojection frequency
    // // I.MagFiltFac = 1.0e-5;
    // I.MagFiltFac = 0;                   // Magnitude filtering factor
    // I.DivFilt = true;                   // Is divergence filtering being carried out?
    // I.Turb = SailFFish::RVM2;           // Turbulence model
    // I.C_smag = pow(0.3,3)*1.39;         // Smagorisnky parameter
    // I.KinVisc = 1.0;                    // Kinematic viscosity of fluid
    // I.Rho = 1.0;                        // Density of fluid
    // I.Integrator = SailFFish::LSRK4;    // Time integration scheme

    // // Parameters of vortex ring
    // Real R = 1.0;                       // Ring radius
    // Real Rey = 25000;                    // Ring vorticity Reynolds number
    // Real Coresize = 0.2;             // Ring coresize
    // Real Eps = 0.0001;                  // Epsilon of ring perturbation
    // Real t0 = R*R/(Rey*I.KinVisc);      // Ring charactersistic time
    // I.dT = 0.0025*t0;                     // Timestep size
    // RVector PhaseShift;                 // Random phase shifts of different modes
    // for (int i=0; i<24; i++) PhaseShift.push_back(rand()*1.0/RAND_MAX*M_2PI);

    // // Output parameters
    // I.NExp = 1;                      // Frequency of export of visualisation
    // I.Debug = true;                  // Debugging output of solver
    // I.Log = true;                    // Logging output
    // I.OutputFolder = "SailFFish_VPM_VortexRing2"; // Output directory (will be created)

    //--------------------------------------------------------------------------------

    //--- Setup VPM solver
    SFStatus Status = VPM->Setup_VPM(&I);
    ASSERT_EQ(Status, NoError);

    //--- Initialize vorticity field (using Analytical function)
    int NNT;
    VPM->Retrieve_Grid_Size(NNT);
    RVector Px(NNT), Py(NNT), Pz(NNT), Ox(NNT), Oy(NNT), Oz(NNT);
    ASSERT_GT(NNT, 0);
    VPM->Retrieve_Grid_Positions(Px,Py,Pz);
    Parallel_Kernel(NNT) {KER_Perturbed_Vortex_Ring(Px[i],Py[i],Pz[i],Ox[i],Oy[i],Oz[i],PhaseShift,R,Coresize,Rey,I.KinVisc,Eps);}
    VPM->Set_Input_Arrays(Ox,Oy,Oz);

    std::cout << "Min component Ox " << *std::min_element(Ox.begin(), Ox.end()) << std::endl;
    std::cout << "Max component Ox " << *std::max_element(Ox.begin(), Ox.end()) << std::endl;
    std::cout << "Min component Oy " << *std::min_element(Oy.begin(), Oy.end()) << std::endl;
    std::cout << "Max component Oy " << *std::max_element(Oy.begin(), Oy.end()) << std::endl;
    std::cout << "Min component Oz " << *std::min_element(Oz.begin(), Oz.end()) << std::endl;
    std::cout << "Max component Oz " << *std::max_element(Oz.begin(), Oz.end()) << std::endl;

    // Sanity check of the initial field: finite and nonzero
    Real OMax = 0;
    for (int i=0; i<NNT; i++){
        ASSERT_TRUE(std::isfinite(Ox[i]) && std::isfinite(Oy[i]) && std::isfinite(Oz[i]));
        OMax = std::max(OMax, std::max(std::fabs(Ox[i]), std::max(std::fabs(Oy[i]), std::fabs(Oz[i]))));
    }
    EXPECT_GT(OMax, 0);

    // //--- Initialize vorticity field (using Imported data) (The file Mesh_3DV.vtk must exist)
    // VPM->Import_Field();

    //--- Execute simulation
    // int NStep = 1000;
    for (int i=0; i<NStep_VPM; i++) VPM->Advance_Particle_Set();

    //--- Finalise simulation
    VPM->Generate_Summary_End();
    VPM->Generate_VTK();
}

#endif // TEST_RINGS_H
