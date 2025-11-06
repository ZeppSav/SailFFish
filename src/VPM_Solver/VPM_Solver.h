/****************************************************************************
    SailFFish Library
    Copyright (C) 2023 Joseph Saverin j.saverin@tu-berlin.de

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

    -> Vortex Particle-Mesh (VPM) method solver
    -> This solver carried out all calculations on the CPU.

*****************************************************************************/

#ifndef VPM_SOLVER_H
#define VPM_SOLVER_H

#include "../Solvers/Unbounded_Solver.h"
#include "VPM3D_kernels_cpu.h"

namespace SailFFish
{

//--- Identifiers

enum TimeIntegrator     {NOTI, EF, EM, RK2, AB2LF, RK3, RK4, LSRK3, LSRK4};
enum FiniteDiff         {CD2, CD4, CD6, CD8};
enum GridDefinition     {NODES, BLOCKS, ADAPTIVE};
enum Turbulence         {LAM, HYP, RVM1, RVM2, RVM3, RVM2_DGC};

static int const NBlock3 = 1024;
typedef Real (*BS_Kernel)(const Real &r, const Real &sigma);
typedef void (*Map_Kernel)(const Real &d, Real &f);

struct ParticleMap
{
    Vector3 Vort = Vector3::Zero(); // Vorticity vector
    // Note on indices:
    // glob_ (monolithic system)
    // block_ (block system)

    // Cartesian ids.
    dim3 cartid;                    // Global cartesian id
    dim3 blockid;                   // Block id
    dim3 interblockid;              // Cartesian id WITHIN block

    int globid;                     // Global monolithic id
    int globblid;                   // Block cartesian id
    // int globblid;                   // Block cartesian id
};

struct CellMap
{
    int id;                         // Global id
    int idin;                       // Position in input array
    dim3 id3;                       // Cartesian id
    Vector3 Pos;                    // Cell Weight
    Vector3 Weight;                 // Cell Weight
    Matrix Coeff;                   // Mapping Coefficients
    std::vector<Vector3> Weights;   // Cell Weights (multiple nodes)
    std::vector<Matrix> Coeffs;     // Mapping Coefficients
    bool Valid = false;             // Check if cell is within appropriate region
};

struct VPM_Input
{
    // A simple input struct which contains all of the important simulation settings.
    // This is a way to encapsulate the variables necessary for the solver in one place.

    //--- Grid variables

    GridDefinition GridDef = NODES;
    Grid_Type Grid = STAGGERED;         // Grid type
    Real H_Grid = 0;                    // Grid resolution (isotropic)
    Real L = 1.0;                       // Characteristic dimension of problem
    Real dX = 0;                        // Grid resolution (x)
    Real dY = 0;                        // Grid resolution (y)
    Real dZ = 0;                        // Grid resolution (z)
    Real Cx = 0;                        // Domain lower limit X
    Real Cy = 0;                        // Domain lower limit Y
    Real Cz = 0;                        // Domain lower limit Z
    int NX = 0;                         // Number of cells in X-direction
    int NY = 0;                         // Number of cells in Y-direction
    int NZ = 0;                         // Number of cells in Z-direction
    int BX = 0;                         // Block dimensions direction (x)
    int BY = 0;                         // Block dimensions direction (y)
    int BZ = 0;                         // Block dimensions direction (z)
    int NBX = 0;                        // Number of blocks in domain (x)
    int NBY = 0;                        // Number of blocks in domain (y)
    int NBZ = 0;                        // Number of blocks in domain (z)
    int iBX0 = 0;                       // Integer index of first box (x)
    int iBY0 = 0;                       // Integer index of first box (y)
    int iBZ0 = 0;                       // Integer index of first box (z)

    // Options for automatic grid limits
    Real AdaptMinX = 0;                 // What is the minimum value of x which should be resolved?
    Real AdaptMaxX = 0;                 // What is the maximum value of x which should be resolved?
    Real AdaptMinY = 0;                 // What is the minimum value of y which should be resolved?
    Real AdaptMaxY = 0;                 // What is the maximum value of y which should be resolved?
    Real AdaptMinZ = 0;                 // What is the minimum value of z which should be resolved?
    Real AdaptMaxZ = 0;                 // What is the maximum value of z which should be resolved?

    //--- Memory vars
    bool BufferFlag = false;               // Is memory stored.inplace

    //--- Simulation parameters
    // SolverType Type = VPM;
    Bounded_Kernel Kernelbnd = FD2;         // Kernel for bounded problems
    Unbounded_Kernel KernelUnbnd = HEJ_G8;  // Kernel for unbounded problems

    //--- Finite difference setting
    FiniteDiff FDOrder = CD4;

    //--- Output parameters
    std::string OutputFolder;           // Where are results exported to?

    //--- Timestepping parameters
    Real dT = 0.0;                      // Timestep size
    TimeIntegrator Integrator = NOTI;   // Timestepping routine

    //--- Simulation parameters
    int NRemesh = 1;                    // Remeshing frequency
    int NReproject = 10;                // Reprojection frequency
    Real MagFiltFac = 0.0;              // Magnitude filtering factor
    Real ReProjFac = 0.0;               // Reprojection factor
    bool DivFilt = false;

    //--- Mapping Parameters
    Mapping SolverMap = M2;
    Mapping RemeshMap = M2;
    Mapping SourceMap = M2;

    //--- Environmental vars
    Real Ux = 0.0, Uy = 0.0, Uz = 0.0;  // Freestream vals
    Real KinVisc = 0.0;                 // Kinematic viscosity
    Real Rho = 1.172;                   // Fluid density

    //--- Output settings
    bool Debug = false;
    int NExp = 0;                       // How often will we export the volume grid?
    int ExpTB = 0;                      // From whcih timestep will we export volume grid?
    bool Log = false;                   // Are we logging solver/field diagnostics?
    std::string Outputdir;              // Output directory

    // Turbulence modelling parameters
    Turbulence Turb = LAM;              // Turbulence modelling scheme
    Real C_smag = 0.0;                  // Smagorinski constant

    //--- Constructor
    VPM_Input() {}

    VPM_Input(std::string &Filepath)
    {
        // This reads a text-based file for setup of the input

        std::string line;
        std::ifstream File;

        File.open(Filepath);
        if (File.is_open())
        {
            while ( std::getline(File,line) )
            {
                std::vector<std::string> Fields = SplitUp(line);    // Split line into segments
                if (Fields.empty())         continue;                   // Empty line
                if (int(Fields.size())==1) continue;                // Incorrect array length

                if (Fields[1] == "BCTYPE"){
                    if (Fields[0] == "BOUND")   {
                        std::cout << "VPM_3D_Solver::Setup_VPM: Input setup not yet configured for bound BC types." << std::endl;
                        //                        return SetupError;
                    }
                }

                //--- Sim params

                if (Fields[1] == "KERNEL"){
                    if (Fields[0] == "HEJ_S0")  KernelUnbnd = HEJ_S0;
                    if (Fields[0] == "HEJ_G2")  KernelUnbnd = HEJ_G2;
                    if (Fields[0] == "HEJ_G4")  KernelUnbnd = HEJ_G4;
                    if (Fields[0] == "HEJ_G6")  KernelUnbnd = HEJ_G6;
                    if (Fields[0] == "HEJ_G8")  KernelUnbnd = HEJ_G8;
                    if (Fields[0] == "HEJ_G10") KernelUnbnd = HEJ_G10;
                }

                if (Fields[1] == "TURB"){
                    if (Fields[0] == "LAM")     {Turb = LAM;    C_smag = 0.0;                   }
                    if (Fields[0] == "HYP")     {Turb = HYP;    C_smag = 2.5e-2;                }   // C_inf, theoretical = pow(0.3,3) -> Smagoriski theoretical
                    if (Fields[0] == "RVM1")    {Turb = RVM1;   C_smag = 0.036;                 }   // Compare: Theoretical =      pow(0.3,3)*1.39 = 0.03753  -> Cocle 2007;
                    if (Fields[0] == "RVM2")    {Turb = RVM2;   C_smag = 0.047663;              }   //          Theoretical = 1.27*pow(0.3,3)*1.39 = 0.047663 -> Cocle 2007;
                    if (Fields[0] == "RVM3")    {Turb = RVM3;   C_smag = 0.060;                 }   // Compare: Theoretical = 1.40*pow(0.3,3)*1.39 = 0.052542 -> Cocle 2007;
                    // if (Fields[0] == "RVMDGC")  {Turb = RVM2_DGC;   C_smag = 1.27*pow(0.3,3)*1.39;  }       //?!?!?
                    if (Fields[0] == "RVM2DTU") {Turb = RVM2;   C_smag = 0.121;  std::cout << "DTU RVM2 Specs" << std::endl;}       //Trial
                }

                // In the paper by Cocle (2009) a set of numerical investigations are carried out wiht a range of different turbulence models.
                // The results are seen to align well along a best-fit curve  log_10(C/C_inf) = -a*exp(-b delta/eta) where a = 10 and b = 0.3;
                // The value of C chosen is therefore be related to the cutoff length (delta-characteristic grid size) and the kolmogorov scale (eta).
                // I should assume that my ratio delta/eta is very large, and hence C/C_inf = 1 for each model and therefore choose C=C_inf.

                if (Fields[1] == "REPROJECT"){
                    if (Fields[0] == "TRUE")    DivFilt = true;
                    else                        DivFilt = false;
                }

                if (Fields[1] == "MAGFILT")     MagFiltFac = std::stod(Fields[0]);

                if (Fields[1] == "NEXP")        NExp = std::stoi(Fields[0]);

                if (Fields[1] == "EXPTB")       ExpTB = std::stoi(Fields[0]);

                if (Fields[1] == "FINDIFF"){
                    if (Fields[0] == "CD2")   FDOrder = CD2;
                    if (Fields[0] == "CD4")   FDOrder = CD4;
                    if (Fields[0] == "CD6")   FDOrder = CD6;
                    if (Fields[0] == "CD8")   FDOrder = CD8;
                }

                if (Fields[1] == "SOLVERMAP"){
                    if (Fields[0] == "M2")  SolverMap = M2;
                    if (Fields[0] == "M4")  SolverMap = M4;
                    if (Fields[0] == "M4D") SolverMap = M4D;
                    if (Fields[0] == "M6D") SolverMap = M6D;
                }

                if (Fields[1] == "REMESHMAP"){
                    if (Fields[0] == "M2")  RemeshMap = M2;
                    if (Fields[0] == "M4")  RemeshMap = M4;
                    if (Fields[0] == "M4D") RemeshMap = M4D;
                    if (Fields[0] == "M6D") RemeshMap = M6D;
                }

                if (Fields[1] == "SOURCEMAP"){
                    if (Fields[0] == "M2")  SourceMap = M2;
                    if (Fields[0] == "M4")  SourceMap = M4;
                    if (Fields[0] == "M4D") SourceMap = M4D;
                    if (Fields[0] == "M6D") SourceMap = M6D;
                }

                if (Fields[1] == "DEBUG"){
                    if (Fields[0] == "TRUE")    Debug = true;
                }

                if (Fields[1] == "LOG"){
                    if (Fields[0] == "TRUE")    Log = true;
                }

                if (Fields[1] == "UINFX")   Ux = std::stod(Fields[0]);
                if (Fields[1] == "UINFY")   Uy = std::stod(Fields[0]);
                if (Fields[1] == "UINFZ")   Uz = std::stod(Fields[0]);
                if (Fields[1] == "KINVISC") KinVisc = std::stod(Fields[0]);
                if (Fields[1] == "DENSITY") Rho = std::stod(Fields[0]);

                //--- Timestep params

                if (Fields[1] == "TIMESTEP")   dT = std::stod(Fields[0]);

                if (Fields[1] == "SCHEME"){
                    if (Fields[0] == "NONE")    Integrator = NOTI;
                    if (Fields[0] == "EF")      Integrator = EF;
                    if (Fields[0] == "RK2")     Integrator = RK2;
                    if (Fields[0] == "AB2LF")   Integrator = AB2LF;
                    if (Fields[0] == "RK3")     Integrator = RK3;
                    if (Fields[0] == "RK4")     Integrator = RK4;
                    if (Fields[0] == "LSRK3")   Integrator = LSRK3;
                    if (Fields[0] == "LSRK4")   Integrator = LSRK4;
                }

                if (Fields[1] == "REMESH")          NRemesh = std::stoi(Fields[0]);
                if (Fields[1] == "REPROJECTFREQ")   NReproject = std::stoi(Fields[0]);

                //--- Grid params (inner grid)

                if (Fields[1] == "GRIDDEF"){
                    if (Fields[0] == "BLOCK")       GridDef = BLOCKS;
                    if (Fields[0] == "NODE")        GridDef = NODES;
                    if (Fields[0] == "ADAPTIVE")    GridDef = ADAPTIVE;
                }

                if (Fields[1] == "GRID"){
                    if (Fields[0] == "STAGGERED")   Grid = STAGGERED;
                    if (Fields[0] == "REGULAR")     Grid = REGULAR;
                }

                if (Fields[1] == "CHARDIM") L = std::stod(Fields[0]);

                if (Fields[1] == "CORNERX") Cx = std::stod(Fields[0])*L;
                if (Fields[1] == "CORNERY") Cy = std::stod(Fields[0])*L;
                if (Fields[1] == "CORNERZ") Cz = std::stod(Fields[0])*L;
                if (Fields[1] == "DX")      dX = std::stod(Fields[0])*L;
                if (Fields[1] == "DY")      dY = std::stod(Fields[0])*L;
                if (Fields[1] == "DZ")      dZ = std::stod(Fields[0])*L;

                if (Fields[1] == "NX")      NX = std::stoi(Fields[0]);
                if (Fields[1] == "NY")      NY = std::stoi(Fields[0]);
                if (Fields[1] == "NZ")      NZ = std::stoi(Fields[0]);

                if (Fields[1] == "BX")          BX = std::stoi(Fields[0]);
                if (Fields[1] == "BY")          BY = std::stoi(Fields[0]);
                if (Fields[1] == "BZ")          BZ = std::stoi(Fields[0]);
                if (Fields[1] == "NBX")         NBX = std::stoi(Fields[0]);
                if (Fields[1] == "NBY")         NBY = std::stoi(Fields[0]);
                if (Fields[1] == "NBZ")         NBZ = std::stoi(Fields[0]);
                if (Fields[1] == "BCORNERX")    iBX0 = std::stoi(Fields[0]);
                if (Fields[1] == "BCORNERY")    iBY0 = std::stoi(Fields[0]);
                if (Fields[1] == "BCORNERZ")    iBZ0 = std::stoi(Fields[0]);

                if (Fields[1] == "ADAPTMINX")   AdaptMinX = std::stod(Fields[0])*L;
                if (Fields[1] == "ADAPTMAXX")   AdaptMaxX = std::stod(Fields[0])*L;
                if (Fields[1] == "ADAPTMINY")   AdaptMinY = std::stod(Fields[0])*L;
                if (Fields[1] == "ADAPTMAXY")   AdaptMaxY = std::stod(Fields[0])*L;
                if (Fields[1] == "ADAPTMINZ")   AdaptMinZ = std::stod(Fields[0])*L;
                if (Fields[1] == "ADAPTMAXZ")   AdaptMaxZ = std::stod(Fields[0])*L;
            }
        }
        File.close();

        // Grid check: Isotropic?
        if ((dX==dY)&&(dX==dZ)) H_Grid = dX;

        Set_Blocks_Adaptive();
    }

    //--- Grid calculations
    void Set_Blocks_Adaptive()
    {
        // In the case that the grids are set up to be adaptive,
        if (GridDef!=ADAPTIVE) return;

        // Specify block inputs (main grid)
        Real HLx = BX*H_Grid;
        Real HLy = BY*H_Grid;
        Real HLz = BZ*H_Grid;

        iBX0 = floor(AdaptMinX/HLx);                    if (iBX0 >0)   iBX0--;
        iBY0 = floor(AdaptMinY/HLy);                    if (iBY0 >0)   iBY0--;
        iBZ0 = floor(AdaptMinZ/HLz);                    if (iBZ0 >0)   iBZ0--;
        int highblockx = ceil(AdaptMaxX/HLx);           if (highblockx>0)   highblockx--;
        int highblocky = ceil(AdaptMaxY/HLy);           if (highblocky>0)   highblocky--;
        int highblockz = ceil(AdaptMaxZ/HLz);           if (highblockz>0)   highblockz--;
        NBX = highblockx - iBX0 + 1;
        NBY = highblocky - iBY0 + 1;
        NBZ = highblockz - iBZ0 + 1;

        // Reset grid def for initialization
        GridDef = BLOCKS;
    }

};

class VPM_3D_Solver : public Unbounded_Solver_3DV
{
protected:

    //--- Grid params
    Real H_Grid = 0;        // Grid size
    Real X0;                // Corner coordinate of grid (x)
    Real Y0;                // Corner coordinate of grid (y)
    Real Z0;                // Corner coordinate of grid (z)
    Real XN1;               // Coordinate of first node in grid (x)
    Real YN1;               // Coordinate of first node in grid (y)
    Real ZN1;               // Coordinate of first node in grid (z)

    int NNX;                // Number of grid nodes (x)
    int NNY;                // Number of grid nodes (y)
    int NNZ;                // Number of grid nodes (z)
    int NNT;                // Total number of grid nodes

    //--- Grid parameters (Block grid)
    int BX = 0;             // Block dimensions direction (x)
    int BY = 0;             // Block dimensions direction (y)
    int BZ = 0;             // Block dimensions direction (z)
    int BT = 0;             // Block dimensions
    int NBX = 0;            // Number of blocks in domain (x)
    int NBY = 0;            // Number of blocks in domain (y)
    int NBZ = 0;            // Number of blocks in domain (z)
    int NBT = 0;            // Total number of blocks in domain
    int iBX0 = 0;           // Integer index of first box (x)
    int iBY0 = 0;           // Integer index of first box (y)
    int iBZ0 = 0;           // Integer index of first box (z)
    Real HLx;               // Box sidelength
    Real HLy;
    Real HLz;

    //--- Grid Arrays
    TensorGrid eu_d;
    TensorGrid MapFactors;

    //--- Mapping parameters
    int Set_Map_Shift(Mapping Map);
    int Set_Map_Stencil_Width(Mapping Map);

    //--- External forcing
    BS_Kernel BS;
    // Real Sigma;

    //--- Finite different objects & parameters
    FiniteDiff FDOrder = CD4;

    //--- Simulation parameters
    TimeIntegrator Integrator = NOTI;       // Time stepping scheme
    bool Remeshing = false;                 // Remeshing flag
    bool DivFilt = false;                   // Divergence filtering flag
    Real T=0.;                              // Simulation real time
    Real dT=0.;                              // Timestep size
    int NStep = 0;                          // Current timestep
    int NInit = 0;                          // Initial timestep
    int NRemesh = 1;                        // Remeshing frequency
    int NReproject = 10;                    // Reprojection frequency
    Real OmMax = 0.0;                       // Maximum vorticity
    Real SMax = 0.0;                        // Maximum strethcing metric
    Real MagFiltFac = 0.0;                  // Magnitue filering factor
    Real Ux = 0.0, Uy = 0.0, Uz = 0.0;      // Freestream vals
    Real KinVisc = 0.0;                     // Kinematic viscosity
    Real Rho = 0.0;                         // Fluid density
    Mapping SolverMap = M4D;                // Mapping scheme between Lagrangian and Eulerian grids
    Mapping RemeshMap = M4D;                // Mapping scheme for remeshing
    Turbulence Turb = LAM;                  // Turbulence model
    Real C_smag = 0.;                       // Hyperviscosity constant

    //--- External forcing
    std::vector<ParticleMap> Ext_Forcing;

    //--- Debugging & output parameters
    int NExp = 0;                           // Output visualisation frequency
    int ExpTB = 0;                          // Exporting from which timestep?
    bool Debug = false;                     // Are we debugging into the console at each timestep?
    bool Log = false;                       // Are we writing the diagnostics to a log file?
    std::chrono::steady_clock::time_point Sim_begin, Sim_end;   // Start/stop time of sims--- total timing

    //--- Output vti/k names
    std::string vtk_Prefix = "Mesh_3DV_";

public:

    //--- Constructor
    VPM_3D_Solver(Grid_Type G, Unbounded_Kernel B);

    //--- Solver setup
    virtual SFStatus Setup_VPM(VPM_Input &I)            {}
    virtual SFStatus Allocate_Data()                    {}

    //--- Grid Setup
    virtual void Set_Grid_Positions();
    virtual SFStatus Define_Grid_Nodes(VPM_Input *I);
    virtual SFStatus Define_Grid_Blocks(VPM_Input *I);
    virtual SFStatus Define_Grid_Blocks_Adaptive(VPM_Input *I);
    void Retrieve_Grid_Size(int &N) {N = NNX*NNY*NNZ;}

    //--- External sources
    void Get_External_Forcing_Nodes(std::vector<Vector3> &Om, std::vector<Vector3> &Pos);
    virtual void Interpolate_Ext_Sources(Mapping M)     {}

    //--- Timestepping
    void Set_Timestep(int N)                        {NStep = N;}
    virtual void Advance_Particle_Set()             {}
    virtual void Update_Particle_Field()            {}
    virtual void Grid_Shear_Stresses()              {}
    virtual void Grid_Turb_Shear_Stresses()         {}
    virtual void Add_Freestream_Velocity()          {}
    virtual void Calc_Grid_SpectralRatesof_Change() {}
    virtual void Calc_Grid_FDRatesof_Change()       {}
    virtual void Increment_Time()                   {NStep++; T += dT;}

    //--- Grid utilities
    virtual void Remesh_Particle_Set()              {}
    virtual void Reproject_Particle_Set()           {}
    virtual void Reproject_Particle_Set_Spectral()  {}
    virtual void Magnitude_Filtering()              {}

    void Domain_Bounds(std::vector<CellMap> &CD, dim3 &Lower, dim3 &Upper);

    void Process_Cells(const RVector &Px, const RVector &Py, const RVector &Pz,
                       const RVector &Ox, const RVector &Oy, const RVector &Oz,
                       std::vector<CellMap> &IDs, Mapping Map);

    void Store_Grid_Sources(const RVector &Px, const RVector &Py, const RVector &Pz, const RVector &Ox, const RVector &Oy, const RVector &Oz, Mapping Map);

    virtual void Map_from_Grid( const RVector &Px, const RVector &Py, const RVector &Pz,
                               const RVector &Gx, const RVector &Gy, const RVector &Gz,
                               RVector &uX, RVector &uY, RVector &uZ, Mapping Map);

    virtual void Map_to_Grid(   const RVector &Px, const RVector &Py, const RVector &Pz,
                             const RVector &Ox, const RVector &Oy, const RVector &Oz,
                             RVector &gX, RVector &gY, RVector &gZ, Mapping Map);

    virtual void Add_Grid_Sources(const RVector &Px, const RVector &Py, const RVector &Pz, const RVector &Ox, const RVector &Oy, const RVector &Oz, Mapping Map) {}
    virtual void Extract_Sol_Values(const RVector &Px, const RVector &Py, const RVector &Pz, RVector &Ugx, RVector &Ugy, RVector &Ugz, Mapping Map) {}
    virtual void Extract_Source_Values(const RVector &Px, const RVector &Py, const RVector &Pz, RVector &Ugx, RVector &Ugy, RVector &Ugz, Mapping Map) {}
    virtual void Store_Grid_Node_Sources(const RVector &Px, const RVector &Py, const RVector &Pz, const RVector &Ox, const RVector &Oy, const RVector &Oz, Mapping Map);

    void Map_Source_Nodes(const RVector &Px, const RVector &Py, const RVector &Pz,
                          const RVector &Ox, const RVector &Oy, const RVector &Oz, std::vector<ParticleMap> &GP, Mapping Map);

    void Get_Ext_Velocity(const RVector &Px, const RVector &Py, const RVector &Pz,
                          RVector &Ux, RVector &Uy, RVector &Uz, Mapping Map);

    //--- Grid operations
    virtual void Clear_Source_Grid()        {}
    virtual void Clear_Solution_Grid()      {}
    virtual TensorGrid *Get_pArray()    {return nullptr;}
    virtual TensorGrid *Get_gArray()    {return nullptr;}
    virtual Real* Get_Vort_Array()      {return nullptr;}
    virtual Real* Get_Vel_Array()       {return nullptr;}
    Real GetXgridMax()  {RVector XG; Get_XGrid(XG); return XG[XG.size()-1];}

    //--- Grid statistics
    virtual void Calc_Grid_Diagnostics()            {}

    //--- Auxiliary grid
    virtual void Map_from_Auxiliary_Grid()          {}
    virtual void Map_to_Auxiliary_Grid()            {}

    //--- Visualisation
    virtual void Generate_Plane(RVector &U)        {}

    //--- Output grid
    virtual void Generate_VTK()                     {}
    virtual void Generate_VTK_Scalar()              {}
    void Set_VTK_Prefix(std::string S)              {vtk_Prefix = S;}
    virtual void Generate_Traverse(int XP, RVector &U, RVector &V, RVector &W)      {}

    //--- Output summary
    void Generate_Summary(std::string Filename);
    void Generate_Summary_End();
    std::string Output_Filename = "";

    //--- Testing
    virtual void Benchmarking()                     {}

    //--- Destructor
    ~VPM_3D_Solver()    {}
};

}

#endif // VPM_SOLVER_H
