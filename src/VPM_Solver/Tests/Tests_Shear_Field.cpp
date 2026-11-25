/****************************************************************************
    SailFFish Library
    Copyright (C) 2026 Joseph Saverin j.saverin@tu-berlin.de

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

    -> Baseline test case for implementation of shear fields

*****************************************************************************/

#ifndef TEST_SHEAR_FIELD_H
#define TEST_SHEAR_FIELD_H

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

TEST(SailFFish_VPM_Tests, Shear_Field_Baseline)
{
    // Background shear field without input vorticity.
    // The grid velocity must equal the prescribed profile U(z) (+ constant freestream) at every node.

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

    // Specify grid parameters: z in [0, 5.6]
    I.GridDef = SailFFish::BLOCKS;      // Block-style structuring is necessary for GPU tests
    I.H_Grid = 0.05;                    // Grid size
    I.BX = 4;                           // Block grid dims
    I.BY = 4;
    I.BZ = 4;
    I.iBX0 = -8;                        // Initial block indices
    I.iBY0 = -14;
    I.iBZ0 = 0;
    I.NBX = 16;                         // Number of blocks
    I.NBY = 28;
    I.NBZ = 28;

    // Simulation parameters
    I.SolverMap = SailFFish::M4D;       // Mapping scheme used within solver for mapping Lagrangian-Eulerian grids
    I.RemeshMap = SailFFish::M4D;       // Mapping scheme used within solver for remeshing
    I.FDOrder = SailFFish::CD8;         // Order of finite difference calculations on the grid
    I.NRemesh = 5;                      // Remeshing frequency
    I.NReproject = 5;                   // Reprojection frequency
    I.MagFiltFac = 0;                   // Magnitude filtering factor
    I.DivFilt = true;                   // Is divergence filtering being carried out?
    I.Turb = SailFFish::RVM2;           // Turbulence model
    I.C_smag = 2.5e-2;                  // Smagorisnky parameter
    I.KinVisc = 1.0;                    // Kinematic viscosity of fluid
    I.Rho = 1.0;                        // Density of fluid
    I.Integrator = SailFFish::LSRK4;    // Time integration scheme
    I.dT = 0.001;                        // Timestep size (no vorticity: no CFL restriction)

    //--- Background shear field: linear shear U(z) = G*(z - Z_Ref), zero at mid-height of domain
    I.Ux = 1.0;

    // // Shear linear
    // I.Shear.Profile = SailFFish::SHEAR_LINEAR;
    // I.Shear.U_Ref = 0.0;                // U at Z_Ref
    // I.Shear.Z_Ref = 2.8;                // Reference height (mid-height of domain)
    // I.Shear.Z_Ground = 0.0;             // Ground plane (no effect for linear profile)
    // I.Shear.Gradient = 1.0;             // dU/dz
    // I.Shear.Dir = 0.0;                  // Shear velocity along x
    // I.Shear.Initialise();               // Required for reference evaluation below (also called in Setup_VPM)

    // // Shear power
    // I.Shear.Profile = SailFFish::SHEAR_POWER;
    // I.Shear.U_Ref = 2.0;                // U at Z_Ref
    // I.Shear.Z_Ref = 2.8;                // Reference height (mid-height of domain)
    // I.Shear.Alpha = 0.14;               // Power-law exponent
    // I.Shear.Z_Ground = 0.0;             // Ground plane
    // I.Shear.Z_Cut = 0.1;                // Linear ramp below 2H (default of 1.0 is too large for this domain)
    // I.Shear.Dir = 0.0;                  // Shear velocity along x
    // I.Shear.Initialise();

    // // Shear log with veer
    // I.Shear.Profile = SailFFish::SHEAR_LOG;
    // I.Shear.U_Ref = 2.0;                // Speed at Z_Ref
    // I.Shear.Z_Ref = 2.8;                // Reference height
    // I.Shear.Z0 = 0.01;                  // Roughness length (scaled to domain)
    // I.Shear.Inv_L = 0.0;                // Neutral stability
    // I.Shear.Z_Ground = 0.0;
    // I.Shear.Z_Cut = 0.1;
    // I.Shear.Veer = SailFFish::VEER_LINEAR;
    // I.Shear.Veer_Z1 = 0.5;              // Veer region
    // I.Shear.Veer_Z2 = 5.0;
    // I.Shear.Veer_Angle = 20.0*M_PI/180.0;   // 20 deg total direction change
    // I.Shear.Dir = 0.0;
    // I.Shear.Initialise();

    // Output parameters
    I.NExp = 20;                        // Frequency of export of visualisation
    I.Debug = true;                     // Debugging output of solver
    I.Log = true;                       // Logging output
    I.OutputFolder = "SailFFish_VPM_ShearField";  // Output directory (will be created)

    //--- Setup VPM solver
    SFStatus Status = VPM->Setup_VPM(&I);
    ASSERT_EQ(Status, NoError);

    //--- Debug shear field for testing
    VPM->Get_Shear_Field().Print_Grid(8);

    //--- Execute single step (no particles: grid velocity is background field only)
    VPM->Advance_Particle_Set();

    //--- Compare grid velocity with prescribed profile
    TensorGrid *Vel = VPM->Get_gArray();
    ASSERT_NE(Vel, nullptr);
    RVector Z;
    VPM->Get_ZGrid(Z);
    int NNT, NNZ = int(Z.size());
    VPM->Retrieve_Grid_Size(NNT);
    ASSERT_EQ(NNT%NNZ, 0);

    Real Umax = 0.;
    for (int k=0; k<NNZ; k++) Umax = std::max(Umax, std::fabs(I.Ux + I.Shear.Eval(Z[k]).U));
    Real Tol = 16*std::numeric_limits<Real>::epsilon()*std::max(Umax, Real(1.));

    Real ErrMax = 0.;
    for (int i=0; i<NNT; i++){
        int k = i%NNZ;                  // C-style ordering: k fastest (GID)
        Shear_Point P = I.Shear.Eval(Z[k]);
        ErrMax = std::max(ErrMax, std::fabs((*Vel)[0][i] - (I.Ux + P.U)));
        ErrMax = std::max(ErrMax, std::fabs((*Vel)[1][i] - (I.Uy + P.V)));
        ErrMax = std::max(ErrMax, std::fabs((*Vel)[2][i] - I.Uz));
    }
    std::cout << "Shear field: max |U_grid - U_prescribed| = " << ErrMax << " (Umax = " << Umax << ")" << std::endl;
    EXPECT_LT(ErrMax, Tol);

    //--- Finalise simulation
    VPM->Generate_Summary_End();
    VPM->Generate_VTK();
}

#endif // TEST_SHEAR_FIELD_H
