//-----------------------------------------------------------------------
//----------------------- VPM Solver Functions (CPU) --------------------
//-----------------------------------------------------------------------

#include "VPM3D_cpu.h"

namespace SailFFish
{

//---------------------------------
//------ Initialization -----------
//---------------------------------

//--- Solver setup

SFStatus VPM3D_cpu::Setup_VPM(VPM_Input *I)
{
    // This sets up the problem for the case that a full VPM solver is being executed.
    // This includes generation of all of the templates arguments and additional arrays for field preparation etc.

    SFStatus Stat = NoError;

    //------------------------------
    //--- Set parameters
    //------------------------------

    // Specify sim parameters
    dT = I->dT;
    Integrator = I->Integrator;

    //    PotFlow = I->Potential_Flow;
    SolverMap = I->SolverMap;
    RemeshMap= I->RemeshMap;
    if (RemeshMap>SolverMap) SolverMap = RemeshMap;     // Carry out check to avoid mixed fidelity mappings.
    if (SolverMap>RemeshMap) RemeshMap = SolverMap;     // Carry out check to avoid mixed fidelity mappings.
    // SourceMap= I->SourceMap;
    FDOrder = I->FDOrder;
    NRemesh = I->NRemesh;
    NReproject = I->NReproject;
    MagFiltFac = I->MagFiltFac;
    DivFilt = I->DivFilt;

    Ux = I->Ux;
    Uy = I->Uy;
    Uz = I->Uz;
    KinVisc = I->KinVisc;
    Rho = I->Rho;

    // Turbulence modelling parameters
    Turb = I->Turb;
    C_smag = I->C_smag;

    // Export params
    NExp = I->NExp;
    ExpTB = I->ExpTB;
    Debug = I->Debug;
    Log = I->Log;
    OutputFolder = I->OutputFolder;

    //-------------------------------
    //--- Do SailFFish solver setup
    //-------------------------------

    if (I->GridDef==NODES)    Stat = Define_Grid_Nodes(I);
    if (I->GridDef==BLOCKS)   Stat = Define_Grid_Blocks(I);

    //--- Prepare dim id array
    loc_ID = (dim3*)malloc(NNT*sizeof(dim3));
    OpenMPfor
    for (int i=0; i<NNX; i++){
        for (int j=0; j<NNY; j++){
            for (int k=0; k<NNZ; k++) loc_ID[GID(i,j,k,NNX,NNY,NNZ)] = dim3(i,j,k);
        }
    }
    dom_size = dim3(NNX,NNY,NNZ);
    padded_dom_size = dim3(NX,NY,NZ);


    Stat = Allocate_Data();                   // Allocate memory arrays
    if (Stat != NoError)    return Stat;
    Set_Grid_Positions();                   // Specify positions of grid (for diagnostics and/or initialisation
    Prepare_Exclusion_Zones();              // Mark exlusion zones for finite differences

    //     if (DivFilt)                            // Generate Dirichlet solver for particle set reprojection
    //     {
    //         Dirichlet_Solver = new Poisson_Dirichlet_3D();
    //         Stat = Dirichlet_Solver->Setup(Xg, Yg, Zg, NNX, NNY, NNZ);
    // //        for (int i=0; i<10; i++) Reproject_Particle_Set();          // Initial reprojection
    //     }

    // Prepare outputs
    std::string OutputDirectory = "Output";
    Create_Directory(OutputDirectory);
    Generate_Summary("Summary.dat");

    return Stat;
}

SFStatus VPM3D_cpu::Allocate_Data()
{
    // The memory arrays of the grids

    try{
        ConstructTensorGrid(lg_d,NNT,3);          // State array
        ConstructTensorGrid(lg_o,NNT,3);          // State array
        ConstructTensorGrid(eu_d,NNT,3);          // State array
        ConstructTensorGrid(eu_o,NNT,3);          // State array
        ConstructTensorGrid(lg_dddt,NNT,3);       // Rate of change
        ConstructTensorGrid(lg_dodt,NNT,3);       // Rate of change
        ConstructTensorGrid(eu_dddt,NNT,6);       // Grid Rate of change
        ConstructTensorGrid(eu_dodt,NNT,6);       // Grid Rate of change

        ConstructTensorGrid(Laplacian,NNT,3);           // Laplacian
        ConstructTensorGrid(GradU,NNT,9);               // Velocity gradient
        int nf;
        if (SolverMap==M2) nf = 9;
        if (SolverMap==M4||SolverMap==M4D) nf = 15;
        if (SolverMap==M6D) nf = 21;
        ConstructTensorGrid(MapFactors,NNT,nf);         // Mapping factors for updating field

        if (Integrator == EM){
            ConstructTensorGrid(int_lg_d,NNT,3);   // Intermediate state array
            ConstructTensorGrid(int_lg_o,NNT,3);   // Intermediate state array
            ConstructTensorGrid(kd_d,NNT,3);      // Intermediate state array
            ConstructTensorGrid(k2_o,NNT,3);      // Intermediate state array
        }

        if (Integrator == RK2){
            ConstructTensorGrid(int_lg_d,NNT,3);     // Intermediate state array
            ConstructTensorGrid(int_lg_o,NNT,3);     // Intermediate state array
            ConstructTensorGrid(kd_d,NNT,3);      // Intermediate state array
            ConstructTensorGrid(k2_o,NNT,3);      // Intermediate state array
        }

        if (Integrator == AB2LF){
            ConstructTensorGrid(int_lg_d,NNT,3);     // Intermediate state array
            ConstructTensorGrid(int_lg_o,NNT,3);     // Intermediate state array
            ConstructTensorGrid(tm1_d,NNT,3);     // Intermediate state array
            ConstructTensorGrid(tm1_o,NNT,3);     // Intermediate state array
            ConstructTensorGrid(tm1_dddt,NNT,3);     // Intermediate state array
            ConstructTensorGrid(tm1_dodt,NNT,3);     // Intermediate state array
        }

        if (Integrator == RK3){
            ConstructTensorGrid(int_lg_d,NNT,3);     // Intermediate state array
            ConstructTensorGrid(int_lg_o,NNT,3);     // Intermediate state array
            ConstructTensorGrid(kd_d,NNT,3);      // Intermediate state array
            ConstructTensorGrid(k2_o,NNT,3);      // Intermediate state array
            ConstructTensorGrid(k3_d,NNT,3);      // Intermediate state array
            ConstructTensorGrid(k3_o,NNT,3);      // Intermediate state array
        }

        if (Integrator == RK4){
            ConstructTensorGrid(int_lg_d,NNT,3);     // Intermediate state array
            ConstructTensorGrid(int_lg_o,NNT,3);     // Intermediate state array
            ConstructTensorGrid(kd_d,NNT,3);      // Intermediate state array
            ConstructTensorGrid(k2_o,NNT,3);      // Intermediate state array
            ConstructTensorGrid(k3_d,NNT,3);      // Intermediate state array
            ConstructTensorGrid(k3_o,NNT,3);      // Intermediate state array
            ConstructTensorGrid(k4_d,NNT,3);      // Intermediate state array
            ConstructTensorGrid(k4_o,NNT,3);      // Intermediate state array
        }

        if (Integrator == LSRK3){
            ConstructTensorGrid(int_lg_d,NNT,3);     // Intermediate state array
            ConstructTensorGrid(int_lg_o,NNT,3);     // Intermediate state array
        }

        if (Integrator == LSRK4){
            ConstructTensorGrid(int_lg_d,NNT,3);     // Intermediate state array
            ConstructTensorGrid(int_lg_o,NNT,3);     // Intermediate state array
        }

        if (Turb==RVM1 || Turb==RVM2){
            SGS = RVector(NNT,0.0);                      // Subgrid scale
            ConstructTensorGrid(gfilt_Array1,NNT,3);     // Small scale vorticity
            ConstructTensorGrid(gfilt_Array2,NNT,3);     // Small scale vorticity
        }
    }
    catch (std::bad_alloc& ex){
        std::cout << "VPM3D_cpu::Allocate_Data(): Insufficient memory for allocation of solver arrays." << std::endl;
        return MemError;
    }

    return NoError;
}

void VPM3D_cpu::Prepare_Exclusion_Zones()
{
    // In order to avoid Ker execution at certain grid pionts (e.g. near boundaries, inlets etc)
    // Exclusion zones will be prepared.

    FD_Flag = std::vector<bool>(NNT,true);
    Map_Flag = std::vector<bool>(NNT,true);
    Remesh_Flag = std::vector<bool>(NNT,true);

    OpenMPfor
    for (int n=0; n<NNT; n++){

        int i = loc_ID[n].x;
        int j = loc_ID[n].y;
        int k = loc_ID[n].z;

        // FD exclusion zones
        if (FDOrder==CD2){
            if (i<1 || i>(NNX-2))      FD_Flag[n] = false;
            if (j<1 || j>(NNY-2))      FD_Flag[n] = false;
            if (k<1 || k>(NNZ-2))      FD_Flag[n] = false;
        }
        if (FDOrder==CD4){
            if (i<2 || i>(NNX-3))       FD_Flag[n] = false;
            if (j<2 || j>(NNY-3))       FD_Flag[n] = false;
            if (k<2 || k>(NNZ-3))       FD_Flag[n] = false;
        }
        if (FDOrder==CD6){
            if (i<3 || i>(NNX-4))       FD_Flag[n] = false;
            if (j<3 || j>(NNY-4))       FD_Flag[n] = false;
            if (k<3 || k>(NNZ-4))       FD_Flag[n] = false;
        }
        if (FDOrder==CD8){
            if (i<4 || i>(NNX-5))       FD_Flag[n] = false;
            if (j<4 || j>(NNY-5))       FD_Flag[n] = false;
            if (k<4 || k>(NNZ-5))       FD_Flag[n] = false;
        }

        // Mapping exclusion zones
        if (SolverMap==M2){
            if (i<1 || i>(NNX-2))     Map_Flag[n] = false;
            if (j<1 || j>(NNY-2))     Map_Flag[n] = false;
            if (k<1 || k>(NNZ-2))     Map_Flag[n] = false;
        }
        if (SolverMap==M4 || SolverMap==M4D){
            if (i<2 || i>(NNX-3))     Map_Flag[n] = false;
            if (j<2 || j>(NNY-3))     Map_Flag[n] = false;
            if (k<2 || k>(NNZ-3))     Map_Flag[n] = false;
        }
        if (SolverMap==M6D){
            if (i<3 || i>(NNX-4))     Map_Flag[n] = false;
            if (j<3 || j>(NNY-4))     Map_Flag[n] = false;
            if (k<3 || k>(NNZ-4))     Map_Flag[n] = false;
        }

        // Remeshing exclusion zones
        if (RemeshMap==M2){
            if (i<1 || i>(NNX-2))     Remesh_Flag[n] = false;
            if (j<1 || j>(NNY-2))     Remesh_Flag[n] = false;
            if (k<1 || k>(NNZ-2))     Remesh_Flag[n] = false;
        }
        if (RemeshMap==M4 || RemeshMap==M4D){
            if (i<2 || i>(NNX-3))     Remesh_Flag[n] = false;
            if (j<2 || j>(NNY-3))     Remesh_Flag[n] = false;
            if (k<2 || k>(NNZ-3))     Remesh_Flag[n] = false;
        }
        if (RemeshMap==M6D){
            if (i<3 || i>(NNX-4))     Remesh_Flag[n] = false;
            if (j<3 || j>(NNY-4))     Remesh_Flag[n] = false;
            if (k<3 || k>(NNZ-4))     Remesh_Flag[n] = false;
        }
    }

    int NFD= 0, NMap = 0, NRmsh = 0;
    for (int i=0; i<NNT; i++){
        if (FD_Flag[i]) NFD++;
        if (Map_Flag[i]) NMap++;
        if (Remesh_Flag[i]) NRmsh++;
    }

    std::cout << "NFD = " << NFD << " , NMap = " << NMap << " , NRmsh = " << NRmsh << " , NNT = " << NNT << std::endl;
}

//-----------------------------------
//--- Initial vorticity distribution
//-----------------------------------

void VPM3D_cpu::Retrieve_Grid_Positions(RVector &xc, RVector &yc, RVector &zc)
{
    // Returns the global grid positions for intiail specification of vorticity
    // I assume the arrays have already been allocated
    OpenMPfor
    for (int i=0; i<NNT; i++){
        xc[i] = XN1 + H_Grid*loc_ID[i].x;
        yc[i] = YN1 + H_Grid*loc_ID[i].y;
        zc[i] = ZN1 + H_Grid*loc_ID[i].z;
    }
}

void VPM3D_cpu::Set_Input_Arrays(RVector &xo, RVector &yo, RVector &zo)
{
    // Returns the global grid positions for intiail specification of vorticity
    // I assume the arrays have already been allocated
    std::memcpy(lg_o[0].data(), xo.data(), NNT*sizeof(Real));
    std::memcpy(lg_o[1].data(), yo.data(), NNT*sizeof(Real));
    std::memcpy(lg_o[2].data(), zo.data(), NNT*sizeof(Real));
}

//---------------------------------
//----- Time integration ----------
//---------------------------------

void VPM3D_cpu::Advance_Particle_Set()
{
    // This is a stepping function which updates the particle field and carried out the desired
    // updates of the field.
    if (NStep%NRemesh==0 && NStep!=NInit)   Remeshing = true;
    else                                    Remeshing = false;
    if (Remeshing)                    Remesh_Particle_Set();          // Remesh
    if (Remeshing && MagFiltFac>0)    Magnitude_Filtering();          // Filter magnitude
    if (DivFilt && Remeshing && NStep%NReproject==0)   Reproject_Particle_Set_Spectral();   // Reproject vorticity field
    Update_Particle_Field();                                            // Update vorticity field
    // Filter_Boundary();
    if (NExp>0 && NStep%NExp==0 && NStep>0 && NStep>=ExpTB) Generate_VTK();             // Export grid if desired
    Increment_Time();
}

void VPM3D_cpu::Update_Particle_Field()
{
    if (Integrator == EF)       // Eulerian forward
    {
        // Calc_Particle_RateofChange(p_Array, dpdt_Array);
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        Parallel_Kernel(NNT) {KER_Update(lg_d, lg_o, lg_dddt, lg_dodt, i, dT);}
    }

    if (Integrator == EM)       // Explicit midpoint
    {
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        Parallel_Kernel(NNT) {KER_Copy(lg_d, lg_o, int_lg_d, int_lg_o, i);}
        Parallel_Kernel(NNT) {KER_Update(int_lg_d, int_lg_o, lg_dddt, lg_dodt, i, 0.5*dT);}
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, kd_d, k2_o);
        Parallel_Kernel(NNT) {KER_Update(lg_d, lg_o, kd_d, k2_o, i, dT);}
    }

    if (Integrator == AB2LF)
    {
        if (Remeshing)
        {
            Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
            Calc_Grid_Diagnostics();
            Parallel_Kernel(NNT) {KER_Copy(lg_d, lg_o, tm1_d, tm1_o, i);}         // Store for next time step
            Parallel_Kernel(NNT) {KER_Copy(lg_dddt, lg_dodt, tm1_dddt, tm1_dodt, i);}         // Store for next time step
            Parallel_Kernel(NNT) {KER_Copy(lg_d, lg_o, int_lg_d, int_lg_o, i);}
            Parallel_Kernel(NNT) {KER_Update(int_lg_d, int_lg_o, lg_dddt, lg_dodt, i, 0.5*dT);}
            Calc_Particle_RateofChange(int_lg_d, int_lg_o, lg_dddt, lg_dodt);
            Parallel_Kernel(NNT) {KER_Update(lg_d, lg_o, lg_dddt, lg_dodt, i, dT);}
        }
        else
        {
            Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
            Calc_Grid_Diagnostics();
            Parallel_Kernel(NNT) {KER_Copy(lg_d, lg_o, int_lg_d, int_lg_o, i);}         // Store for next time step
            Parallel_Kernel(NNT) {KER_Update_AB2LF(lg_d, lg_o, lg_dddt, lg_dodt, tm1_d, tm1_dodt, i, dT);}         // Store for next time step
            Parallel_Kernel(NNT) {KER_Copy(int_lg_d, int_lg_o, tm1_d, tm1_o, i);}         // Store for next time step
            Parallel_Kernel(NNT) {KER_Copy(lg_dddt, lg_dodt, tm1_dddt, tm1_dodt, i);}   // Store for next time step
        }
    }

    if (Integrator == RK2)      // Runge-Kutta second order
    {
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        Parallel_Kernel(NNT)    {KER_Update_RK(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, i, dT);}
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, kd_d, k2_o);
        Parallel_Kernel(NNT)    {KER_Update_RK2(lg_d, lg_o, lg_dddt, lg_dodt, kd_d, k2_o, i, dT);}
    }

    if (Integrator == RK3)      // Runge-Kutta third order
    {
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        Parallel_Kernel(NNT)    {KER_Update_RK(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, i, 0.5*dT);}
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, kd_d, k2_o);
        Parallel_Kernel(NNT)    {KER_Update_RK(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, i, -1.0*dT);}
        Parallel_Kernel(NNT)    {KER_Update(int_lg_d, int_lg_o, kd_d, k2_o, i, 2.0*dT);}
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k3_d, k3_o);
        Parallel_Kernel(NNT)    {KER_Update_RK3(lg_d, lg_o, lg_dddt, lg_dodt, kd_d, k2_o, k3_d, k3_o, i, dT);}
    }

    if (Integrator == RK4)      // Runge-Kutta fourth order
    {
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        Parallel_Kernel(NNT)    {KER_Update_RK(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, i, 0.5*dT);}
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, kd_d, k2_o);
        Parallel_Kernel(NNT)    {KER_Update_RK(lg_d, lg_o, kd_d, k2_o, int_lg_d, int_lg_o, i, 0.5*dT);}
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k3_d, k3_o);
        Parallel_Kernel(NNT)    {KER_Update_RK(lg_d, lg_o, k3_d, k3_o, int_lg_d, int_lg_o, i, dT);}
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k4_d, k4_o);
        Parallel_Kernel(NNT)    {KER_Update_RK4(lg_d, lg_o, lg_dddt, lg_dodt, kd_d, k2_o, k3_d, k3_o, k4_d, k4_o, i, dT);}
    }

    if (Integrator == LSRK3)    // Runge-Kutta third order low-storage (4-stage)
    {
        Real RK3A[4] =  {0.0, -756391.0/934407.0, -36441873.0/15625000.0, -1953125.0/1085297.0};
        Real RK3B[4] =  {8.0/141.0, 6627.0/2000.0, 609375.0/1085297.0, 198961.0/526283.0};
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        Parallel_Kernel(NNT)    {KER_Update_RKLS(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, RK3A[0], RK3B[0], i, dT);}
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Parallel_Kernel(NNT)    {KER_Update_RKLS(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, RK3A[1], RK3B[1], i, dT);}
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Parallel_Kernel(NNT)    {KER_Update_RKLS(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, RK3A[2], RK3B[2], i, dT);}
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Parallel_Kernel(NNT)    {KER_Update_RKLS(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, RK3A[3], RK3B[3], i, dT);}
    }

    if (Integrator == LSRK4)    // Runge-Kutta fourth order low-storage (5-stage)
    {
        Real RK4A[5] =  {0.0,-567301805773.0/1357537059087.0,-2404267990393.0/2016746695238.0,-3550918686646.0/2091501179385.0,-1275806237668.0/842570457699.0};
        Real RK4B[5] =  {1432997174477.0/9575080441755.0,5161836677717.0/13612068292357.0,1720146321549.0/2090206949498.0,3134564353537.0/4481467310338.0,2277821191437.0/14882151754819.0};
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        Parallel_Kernel(NNT)    {KER_Update_RKLS(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, RK4A[0], RK4B[0], i, dT);}
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Parallel_Kernel(NNT)    {KER_Update_RKLS(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, RK4A[1], RK4B[1], i, dT);}
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Parallel_Kernel(NNT)    {KER_Update_RKLS(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, RK4A[2], RK4B[2], i, dT);}
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Parallel_Kernel(NNT)    {KER_Update_RKLS(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, RK4A[3], RK4B[3], i, dT);}
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Parallel_Kernel(NNT)    {KER_Update_RKLS(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, RK4A[4], RK4B[4], i, dT);}
    }
}

void VPM3D_cpu::Calc_Particle_RateofChange(const TensorGrid &p_d_Vals,
                                           const TensorGrid &p_o_Vals,
                                           TensorGrid &dpdt_d_Vals,
                                           TensorGrid &dpdt_o_Vals)
{
    // This is the general rate of change scheme which calculates the rates of change at the partice positions

    // Clear grid (vorticity) arrays
    Parallel_Kernel(NNT) {KER_Clear(eu_o,i); }          // Clear grid (vorticity) arrays
    Parallel_Kernel(NNT) {KER_Clear(lg_dddt,i); }       // Clear grid (rate of change) arrays
    Parallel_Kernel(NNT) {KER_Clear(lg_dodt,i); }       // Clear grid (rate of change) arrays
    Parallel_Kernel(NNT) {KER_Clear(eu_dddt,i); }       // Clear grid (rate of change) arrays
    Parallel_Kernel(NNT) {KER_Clear(eu_dodt,i); }       // Clear grid (rate of change) arrays

    // Calculate mapping coefficients between particles and grid
    switch (SolverMap)
    {
        case (M2):      {Parallel_Kernel(NNT) {KER_M2_Map_Coeffs(p_d_Vals, MapFactors, i, H_Grid);}  break;}
        case (M4):      {Parallel_Kernel(NNT) {KER_M4_Map_Coeffs(p_d_Vals, MapFactors, i, H_Grid);}  break;}
        case (M4D):     {Parallel_Kernel(NNT) {KER_M4D_Map_Coeffs(p_d_Vals, MapFactors, i, H_Grid);} break;}
        case (M6D):     {Parallel_Kernel(NNT) {KER_M6D_Map_Coeffs(p_d_Vals, MapFactors, i, H_Grid);} break;}
        default:        {std::cout << "VPM3D_cpu::Calc_Particle_RateofChange 1. Mapping undefined."; return;}
    }

    // Map vorticity from particles to grid
    Parallel_Kernel(NNT) {if (Map_Flag[i]) KER_Map(p_o_Vals, MapFactors, eu_o, loc_ID[i], dom_size, SolverMap);}

    //    Calc_Grid_SpectralRatesof_Change();         // The rates of change are calculated on the fixed grid
    Calc_Grid_FDRatesof_Change();                 // The rates of change are calculated on the fixed grid

    // Map grid values onto particles
    Real C[4] = {XN1, YN1, ZN1, H_Grid};
    Parallel_Kernel(NNT)   {if (Map_Flag[i]) KER_Interpolation(eu_dddt, eu_dodt, p_d_Vals, eu_d, C, dpdt_d_Vals, dpdt_o_Vals, loc_ID[i], dom_size, SolverMap);}
}

void VPM3D_cpu::Calc_Grid_SpectralRatesof_Change()
{
    // This function calculates the desired field quantities necessary for the VPM on the grid.
    // These quantities must be mapped later to the particles for the Lagrangian update.

    // SailFFish requires these to be passed as standard vector arrays.
    // If we are using GPU arrays we therefore need to copy the arrays into or out of the memory buffers

    //    //--- Add sources to grid if required
    //    if (PotFlow){Parallel_Kernel(NNT) {KER_AddSource(pf_Array, g_Array, i);}}

    //    //--- Specify input field
    //    SFStatus Stat = Set_Input_Unbounded_3D(g_Array[3],g_Array[4],g_Array[5]);
    //    Stat = Transfer_Data_Device();

    //    //--- Calculate and extract velocity
    //    Forward_Transform();
    //    Convolution();
    //    Spectral_Gradients_3DV_Curl();  // Note: the velocity array is stored prior to later steps
    //    Backward_Transform();
    //    Get_Output_Unbounded_3D(dgdt_Array[0], dgdt_Array[1], dgdt_Array[2]);  //--- Extract velocity
    //    Add_Freestream_Velocity();
    //    Add_PotFlow_Velocity();

    //    //--- Calculate and extract velocity gradient terms.
    //    Spectral_Gradients_3DV_Grad(XComp);
    //    Backward_Transform();   // dUxdx   dUxdy     dUxdz
    //    Get_Output_Unbounded_3D(GradU[0], GradU[3], GradU[6]);
    //    Spectral_Gradients_3DV_Grad(YComp);
    //    Backward_Transform();   // dUydx   dUydy     dUydz
    //    Get_Output_Unbounded_3D(GradU[1], GradU[4], GradU[7]);
    //    Spectral_Gradients_3DV_Grad(ZComp);
    //    Backward_Transform();   // dUzdx   dUzdy     dUzdz
    //    Get_Output_Unbounded_3D(GradU[2], GradU[5], GradU[8]);

    //    //--- Calculate and extract Laplacian (this can be sped up by precalcing the second spatial multiplication)
    //    Transfer_FTInOut_Comp();
    //    Spectral_Gradients_3DV_Nabla();
    //    Backward_Transform();
    //    Get_Output_Unbounded_3D(Laplacian[0], Laplacian[1], Laplacian[2]);   //--- Extract Laplacian

    //    //--- Calculate stretching terms based on velocity gradient and vorticity
}

void VPM3D_cpu::Calc_Grid_FDRatesof_Change()
{
    // This function calculates the desired field quantities necessary for the VPM on the grid.
    // These quantities must be mapped later to the particles for the Lagrangian update.
    // Unlike the Calc_Grid_SpectralRatesof_Change() function, the gradients on the grid are calculated using
    // second order finite differences.

    //--- Add sources to grid if required
    SFStatus Stat;
    Stat = Set_Input_Unbounded_3D(eu_o[0],eu_o[1],eu_o[2]);
    Map_from_Auxiliary_Grid();
    // Stat = Transfer_Data_Device();

    //--- Calculate and extract velocity
    Forward_Transform();
    Convolution();
    Spectral_Gradients_3DV_Curl();  // Note: the velocity array is stored prior to later steps
    Backward_Transform();
    Get_Output_Unbounded_3D(eu_dddt[0], eu_dddt[1], eu_dddt[2]);  // Extract velocity

    //--- Calculate shear stresses on grid
    Grid_Shear_Stresses();
    Grid_Turb_Shear_Stresses();

    //--- Add external velocity components
    Add_Freestream_Velocity();
}

void VPM3D_cpu::Grid_Shear_Stresses()
{
    // Shear stresses on the grid are added (Vortex stretching and molecular diffusion)

    switch (FDOrder)
    {
        case (CD2): {Parallel_Kernel(NNT) {if (FD_Flag[i])    KER_Stretch_FD2(eu_o, eu_dddt, eu_dodt, GradU, Laplacian, loc_ID[i], dom_size, H_Grid, KinVisc);}         break;}
        case (CD4): {Parallel_Kernel(NNT) {if (FD_Flag[i])    KER_Stretch_FD4(eu_o, eu_dddt, eu_dodt, GradU, Laplacian, loc_ID[i], dom_size, H_Grid, KinVisc);}         break;}
        case (CD6): {Parallel_Kernel(NNT) {if (FD_Flag[i])    KER_Stretch_FD6(eu_o, eu_dddt, eu_dodt, GradU, Laplacian, loc_ID[i], dom_size, H_Grid, KinVisc);}         break;}
        case (CD8): {Parallel_Kernel(NNT) {if (FD_Flag[i])    KER_Stretch_FD8(eu_o, eu_dddt, eu_dodt, GradU, Laplacian, loc_ID[i], dom_size, H_Grid, KinVisc);}         break;}
        default:    {std::cout << "VPM3D_cpu::Grid_Shear_Stresses(). FDOrder undefined.";   return;}
    }

    // Generate_VTK(g_Array[0], g_Array[1], g_Array[2], dgdt_Array[3],dgdt_Array[4],dgdt_Array[5]);        // Omega, velocity on Eulerian grid
}

void VPM3D_cpu::Grid_Turb_Shear_Stresses()
{
    // Turbulent shear stresses are calculated on the grid

    if (Turb==LAM) return;

    // Hyper viscosity model

    if (Turb==HYP)
    {
        switch (FDOrder)
        {
            case (CD2): {Parallel_Kernel(NNT)   {if (FD_Flag[i])    KER_HypVisc_FD2(Laplacian, eu_dodt, loc_ID[i], dom_size, H_Grid, C_smag);} break; }
            case (CD4): {Parallel_Kernel(NNT)   {if (FD_Flag[i])    KER_HypVisc_FD4(Laplacian, eu_dodt, loc_ID[i], dom_size, H_Grid, C_smag);} break; }
            case (CD6): {Parallel_Kernel(NNT)   {if (FD_Flag[i])    KER_HypVisc_FD6(Laplacian, eu_dodt, loc_ID[i], dom_size, H_Grid, C_smag);} break; }
            case (CD8): {Parallel_Kernel(NNT)   {if (FD_Flag[i])    KER_HypVisc_FD8(Laplacian, eu_dodt, loc_ID[i], dom_size, H_Grid, C_smag);} break; }
            default:    {std::cout << "VPM3D_cpu::Grid_Turb_Shear_Stresses. FDOrder undefined.";  break; }
        }
    }

    //--- Regularized variational multiscale models

    // Prepare discrete sub-grid scale filter

    if (Turb==RVM1)                     // RVM first order
    {
        Parallel_Kernel(NNT)    {KER_Clear(gfilt_Array2,i);}
        Parallel_Kernel(NNT)    {KER_Copy(eu_o,gfilt_Array1,i);}        // Copy Omega array into temp array.
        Parallel_Kernel(NNT)    {if (FD_Flag[i]) KER_SG_Disc_Filter(gfilt_Array1, gfilt_Array2, loc_ID[i], dom_size, Ez);}     // Z sweep
        Parallel_Kernel(NNT)    {if (FD_Flag[i]) KER_SG_Disc_Filter(gfilt_Array2, gfilt_Array1, loc_ID[i], dom_size, Ey);}     // Y sweep
        Parallel_Kernel(NNT)    {if (FD_Flag[i]) KER_SG_Disc_Filter(gfilt_Array1, gfilt_Array2, loc_ID[i], dom_size, Ex);}     // X sweep

        Parallel_Kernel(NNT)    {KER_RVM_SS(eu_o, gfilt_Array2, gfilt_Array1, i);}          // Set small-scale
        Parallel_Kernel(NNT)    {KER_RVM_SGS(GradU, SGS,i, H_Grid, C_smag);}                // Calculate SGS viscosity on grid
        switch (FDOrder)                                                                    // Calculate SGS shear stress
        {
        case (CD2): {Parallel_Kernel(NNT)   {if (FD_Flag[i]) KER_RVM_FD2(gfilt_Array1, SGS, eu_dodt, loc_ID[i], dom_size, H_Grid);}  break;}
        case (CD4): {Parallel_Kernel(NNT)   {if (FD_Flag[i]) KER_RVM_FD4(gfilt_Array1, SGS, eu_dodt, loc_ID[i], dom_size, H_Grid);}  break;}
        case (CD6): {Parallel_Kernel(NNT)   {if (FD_Flag[i]) KER_RVM_FD6(gfilt_Array1, SGS, eu_dodt, loc_ID[i], dom_size, H_Grid);}  break;}
        case (CD8): {Parallel_Kernel(NNT)   {if (FD_Flag[i]) KER_RVM_FD8(gfilt_Array1, SGS, eu_dodt, loc_ID[i], dom_size, H_Grid);}  break;}
        default:    {std::cout << "VPM3D_cpu::Grid_Turb_Shear_Stresses. FDOrder undefined for SG disc filter."; return;}
        }
     }

    if (Turb==RVM2)   // RVM second order
    {
        Parallel_Kernel(NNT)    {KER_Clear(gfilt_Array1,i);}
        Parallel_Kernel(NNT)    {KER_Copy(eu_o,gfilt_Array2,i);}        // Copy Omega array into temp array.
        Parallel_Kernel(NNT)    {if (FD_Flag[i]) KER_SG_Disc_Filter(gfilt_Array2, gfilt_Array1, loc_ID[i], dom_size, Ez);}     // Z sweep
        Parallel_Kernel(NNT)    {if (FD_Flag[i]) KER_SG_Disc_Filter(gfilt_Array1, gfilt_Array2, loc_ID[i], dom_size, Ez);}     // Z sweep 2
        Parallel_Kernel(NNT)    {if (FD_Flag[i]) KER_SG_Disc_Filter(gfilt_Array2, gfilt_Array1, loc_ID[i], dom_size, Ey);}     // Y sweep
        Parallel_Kernel(NNT)    {if (FD_Flag[i]) KER_SG_Disc_Filter(gfilt_Array1, gfilt_Array2, loc_ID[i], dom_size, Ey);}     // Y sweep 2
        Parallel_Kernel(NNT)    {if (FD_Flag[i]) KER_SG_Disc_Filter(gfilt_Array2, gfilt_Array1, loc_ID[i], dom_size, Ex);}     // X sweep
        Parallel_Kernel(NNT)    {if (FD_Flag[i]) KER_SG_Disc_Filter(gfilt_Array1, gfilt_Array2, loc_ID[i], dom_size, Ex);}     // X sweep 2

        Parallel_Kernel(NNT)    {KER_RVM_SS(eu_o, gfilt_Array2, gfilt_Array1, i);}          // Set small-scale
        Parallel_Kernel(NNT)    {KER_RVM_SGS(GradU, SGS,i, H_Grid, C_smag);}                // Calculate SGS viscosity on grid
        switch (FDOrder)                                                                    // Calculate SGS shear stress
        {
            case (CD2): {Parallel_Kernel(NNT)   {if (FD_Flag[i]) KER_RVM_FD2(gfilt_Array1, SGS, eu_dodt, loc_ID[i], dom_size, H_Grid);}  break;}
            case (CD4): {Parallel_Kernel(NNT)   {if (FD_Flag[i]) KER_RVM_FD4(gfilt_Array1, SGS, eu_dodt, loc_ID[i], dom_size, H_Grid);}  break;}
            case (CD6): {Parallel_Kernel(NNT)   {if (FD_Flag[i]) KER_RVM_FD6(gfilt_Array1, SGS, eu_dodt, loc_ID[i], dom_size, H_Grid);}  break;}
            case (CD8): {Parallel_Kernel(NNT)   {if (FD_Flag[i]) KER_RVM_FD8(gfilt_Array1, SGS, eu_dodt, loc_ID[i], dom_size, H_Grid);}  break;}
            default:    {std::cout << "VPM3D_cpu::Grid_Turb_Shear_Stresses. FDOrder undefined for SG disc filter."; return;}
        }
    }

}

void VPM3D_cpu::Add_Freestream_Velocity()
{
    // The freestream velocity of superposed onto the grid
    // Currently only configured for constant freestream velocity

    //--- Constant
    Real U_inf = sqrt(Ux*Ux + Uy*Uy + Uz*Uz);
    if (U_inf>0.0)
    {
        Parallel_Kernel(NNT) {
            eu_dddt[0][i] += Ux;
            eu_dddt[1][i] += Uy;
            eu_dddt[2][i] += Uz;
        }
    }
}

void VPM3D_cpu::Filter_Boundary()
{
    // This ensures that the boundary region has zero vorticity so that boundary effects do not affect the solution
    Parallel_Kernel(NNT) {
        if (!FD_Flag[i]) {
            lg_o[0][i] = 0.0;
            lg_o[1][i] = 0.0;
            lg_o[2][i] = 0.0;
        }
    }
}

//---------------------------------
//----- Field Calculations --------
//---------------------------------

void VPM3D_cpu::Calc_Grid_Diagnostics()
{
    // Diagnostics are calculated in blocks of size BS

    RVector ED = RVector(NBT,0.0);
    TensorGrid t_C = TensorGrid(3,ED), t_L = TensorGrid(3,ED), t_A = TensorGrid(3,ED);
    RVector t_K1 = RVector(NBT,0.0), t_K2 = RVector(NBT,0.0), t_E = RVector(NBT,0.0), t_H = RVector(NBT,0.0);
    RVector t_Om = RVector(NBT,0.0), t_UMax = RVector(NBT,0.0), t_SMax = RVector(NBT,0.0);
    int BS = NBlock3;

    OpenMPfor
    for (int i=0; i<NBT; i++){
        for (int n=0; n<BS; n++){
            int id = i*BS+n;
            if (id<NNT) KER_Diagnostics(eu_d, eu_o, eu_dddt, GradU, t_C, t_L, t_A, t_K1, t_K2, t_E, t_H, t_Om, t_UMax, t_SMax, id, i);
            //            if (id<NNT) KER_Diagnostics(g_Array, dgdt_Array, t_C, t_L, t_A, t_K1, t_K2, t_E, t_H, t_Om, t_UMax, t_SMax, id, i);
        }
    }

    // Now sum blocks
    Real dV = Hx*Hy*Hz;
    RVector C = DataBlock_Contraction3(t_C,NBT,dV);
    RVector L = DataBlock_Contraction3(t_L,NBT,dV);
    RVector A = DataBlock_Contraction3(t_A,NBT,dV);
    Real K1 = std::accumulate(t_K1.begin(), t_K1.end(), 0.0)*dV;
    Real K2 = std::accumulate(t_K2.begin(), t_K2.end(), 0.0)*dV;
    Real E = std::accumulate(t_E.begin(), t_E.end(), 0.0)*dV;
    Real H = std::accumulate(t_H.begin(), t_H.end(), 0.0)*dV;
    Real UMax =  *std::max_element(t_UMax.begin(), t_UMax.end());
    Real CFL = UMax*dT/H_Grid;
    OmMax =  *std::max_element(t_Om.begin(), t_Om.end());
    SMax =  *std::max_element(t_SMax.begin(), t_SMax.end());

    //    //--------------------- Vorticity centroid
    //    TensorGrid t_Cent = TensorGrid(3,ED);
    //    OpenMPfor
    //    for (int i=0; i<NBT; i++){
    //        for (int n=0; n<BS; n++){
    //            int id = i*BS+n;
    //            if (id<NNT) KER_Centroid(eu_o,L,t_Cent,id,i);
    //        }
    //    }
    //    RVector Cent = DataBlock_Contraction3(t_Cent,NBT);
    //    Real CMag = sqrt(C[0]*C[0] + C[1]*C[1] + C[2]*C[2]);
    //    std::cout << CMag << std::endl;
    //    std::cout << "Centroid: " << Cent[0]*dV/CMag csp Cent[1]*dV/CMag csp Cent[2]*dV/CMag << std::endl;
    //    //---------------------

    //    //--------------------- Saffman (impulse) centroid (Saffman vortex ring tests)
    //    TensorGrid t_SaffCent = TensorGrid(3,ED);
    //    OpenMPfor
    //    for (int i=0; i<NBT; i++){
    //        for (int n=0; n<BS; n++){
    //            int id = i*BS+n;
    //            if (id<NNT) KER_Saffman_Centroid(eu_o,L,t_SaffCent,id,i);
    //        }
    //    }
    //    RVector SC = DataBlock_Contraction3(t_SaffCent,NBT,dV);
    //    Real SaffCentre = SC[0];
    //    if (NStep==0) SCVals[0] = SaffCentre;
    //    if (NStep==1) SCVals[1] = SaffCentre;
    //    if (NStep==2) SCVals[2] = SaffCentre;
    //    if (NStep>2)
    //    {
    //        SCVals[0] = SCVals[1];
    //        SCVals[1] = SCVals[2];
    //        SCVals[2] = SaffCentre;
    //    }

    //    if (NStep>=2)
    //    {
    //        Real t_gamma = (NStep-1.0)*dT*Kin_Visc/R/R + a2/R/R/4.0;
    //        std::cout << (NStep-1.0)*dT csp t_gamma csp (SCVals[2]-SCVals[0])/(2.0*dT)*(R/Gamma0)  csp CFL << std::endl;
    //    }
    //    //---------------------

    //--------------------- Output these values into a file in the output called "Diagnostics.dat"
    //--------------------- Create the file & directory if they don't yet exist

    if (Log)
    {
        // File target location
        std::string OutputDirectory = "Output/" + OutputFolder;
        std::string FilePath = OutputDirectory + "/Diagnostics.dat";

        if (NStep==NInit)
        {
            Create_Directory(OutputDirectory);
            std::ofstream file;
            file.open(FilePath, std::ofstream::out | std::ofstream::trunc); // Clear!
            file.close();
        }

        // Print diagnostics to file
        std::ofstream file;
        file.open(FilePath, std::ios_base::app);
        if (file.is_open())
        {           //1      //2            //3     //4     //5       //6       //7     //8     //9
            file << NStep csp NStep*dT csp CFL csp C[0] csp C[1] csp C[2] csp L[0] csp L[1] csp L[2] csp
                            //10      //11    //12     //13  //14    //15  //16  //17
                            A[0] csp A[1] csp A[2] csp K1 csp K2 csp E csp H csp OmMax << std::endl;
        }
    }

    //    std::cout << "Step " << NStep << ", CFL: "  << UMax csp dT csp H_Grid csp UMax*dT/H_Grid << std::endl;
    if (Debug) std::cout     << "Step "          << NStep
                  << ", UMax: "       << UMax
                  << ", OmegaMax: "   << OmMax
                  << ", CFL: "        << CFL
                  << ", Max Om*dt (<=0.4): "<< OmMax*dT
                  << ", Max S*dt (<=0.2): "    << SMax*dT << std::endl;

    // std::cout << C[0]   << std::endl;
    // std::cout << C[1]   << std::endl;
    // std::cout << C[2]   << std::endl;
    // std::cout << L[0]   << std::endl;
    // std::cout << L[1]   << std::endl;
    // std::cout << L[2]   << std::endl;
    // std::cout << A[0]   << std::endl;
    // std::cout << A[1]   << std::endl;
    // std::cout << A[2]   << std::endl;
    // std::cout << K1     << std::endl;
    // std::cout << K2     << std::endl;
    // std::cout << E      << std::endl;
    // std::cout << H      << std::endl;
    // std::cout << OmMax  << std::endl;


    // Winckelmans: |S|*dt <= 0.2   , |omega|*dt <= 0.4

    // Saffman Centroid
    //    Real Saffman_Cent_X = SaffCent[0];
    //    Real dUxdT = (Saffman_Cent_X-Saffman_Cent_X_prev)/dT;
    //    Saffman_Cent_X_prev = Saffman_Cent_X;
    //    std::cout << "Saffman Centroid dUx/dt: " << dUxdT << std::endl;

}

void VPM3D_cpu::Remesh_Particle_Set()
{
    // This is the general rate of change scheme which calculates the rates of change at the particle positions

    // Parallel_Kernel(NNT) {KER_Clear3_2(g_Array,i); }            // Clear grid (vorticity) arrays
    Parallel_Kernel(NNT) {KER_Clear(eu_o,i); }            // Clear grid (vorticity) arrays

    // Set mapping factors
    switch (RemeshMap)
    {
        case (M2):  {Parallel_Kernel(NNT) {KER_M2_Map_Coeffs( lg_d, MapFactors, i, H_Grid);}         break;}
        case (M4):  {Parallel_Kernel(NNT) {KER_M4_Map_Coeffs( lg_d, MapFactors, i, H_Grid);}         break;}
        case (M4D): {Parallel_Kernel(NNT) {KER_M4D_Map_Coeffs(lg_d, MapFactors, i, H_Grid);}         break;}
        case (M6D): {Parallel_Kernel(NNT) {KER_M6D_Map_Coeffs(lg_d, MapFactors, i, H_Grid);}         break;}
        default:    {std::cout << "VPM3D_cpu::Remesh_Particle_Set(). Mapping undefined.";   return;}
    }

    // Map to new grid
    Parallel_Kernel(NNT) {if (Map_Flag[i]) KER_Map(lg_o, MapFactors, eu_o, loc_ID[i], dom_size, RemeshMap);}

    // Swap over arrays and clear
    Parallel_Kernel(NNT) {KER_Remesh(lg_d, lg_o, eu_o, i); }  // Based on displacement create new updated vorticity map

    if (Debug) std::cout << "Particle set has been remeshed. Timestep " << NStep << std::endl;
}

void VPM3D_cpu::Magnitude_Filtering()
{
    // This function loops over the positions and if the mgniutde of vorticity is bleow the threshold, it will be removed.
    // The sum removed vorticity is added to the remaining (nonzero) values

    //--- Remove lower values
    int BS = NBlock3;
    std::vector<Real> rem_OmX = std::vector<Real>(NBT,0.0);
    std::vector<Real> rem_OmY = std::vector<Real>(NBT,0.0);
    std::vector<Real> rem_OmZ = std::vector<Real>(NBT,0.0);
    std::vector<int>  rem_Count = std::vector<int>(NBT,0);
    Real Om_Thresh = OmMax*MagFiltFac;
    std::vector<bool> NonZero = std::vector<bool>(NNT,false);

    OpenMPfor
        for (int i=0; i<NBT; i++){
        for (uint n=0; n<BS; n++)
        {
            int id = i*BS+n;
            if (id>NNT) continue;
            Real mag_Om = sqrt( lg_o[0][id]*lg_o[0][id] +
                                lg_o[1][id]*lg_o[1][id] +
                                lg_o[2][id]*lg_o[2][id]);

            if (mag_Om==0.0)    continue;       // If mag == 0, ignore.
            else if (mag_Om<Om_Thresh)  {       // If mag < threshold, remove
                rem_OmX[i] += lg_o[0][id];
                rem_OmY[i] += lg_o[1][id];
                rem_OmZ[i] += lg_o[2][id];
                lg_o[0][id] = 0.0;
                lg_o[1][id] = 0.0;
                lg_o[2][id] = 0.0;
            }
            else    {                           // Mag > threshold, keep!
                NonZero[id] = true;
                rem_Count[i]++;
            }
        }
    }

    //--- Sum remaining terms
    Real grem_Ox = 0.0, grem_Oy = 0.0, grem_Oz = 0.0;
    int NP = 0;
    for (int i=0; i<NBT; i++){
        grem_Ox += rem_OmX[i];
        grem_Oy += rem_OmY[i];
        grem_Oz += rem_OmZ[i];
        NP += rem_Count[i];
    }

    //--- Normalize
    grem_Ox *= 1.0/NP;
    grem_Oy *= 1.0/NP;
    grem_Oz *= 1.0/NP;

    //--- Now pass back to nonzero terms
    Parallel_Kernel (NNT) {
        if (NonZero[i]) {
            lg_o[0][i] += grem_Ox;
            lg_o[1][i] += grem_Oy;
            lg_o[2][i] += grem_Oz;
        }
    }

    if (Debug)  std::cout << "Particle set has undergone magnitude filtering. Timestep " << NStep << std::endl;
}

void VPM3D_cpu::Reproject_Particle_Set()
{
    // Calc divergence using FD. This has been checked and is functioning correctly.
    RVector DivOm = RVector(NNT,0);
    switch (FDOrder)
    {
        case (CD2):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Calc_DivOmega2(lg_o, DivOm, loc_ID[i], dom_size, H_Grid);} break;}
        case (CD4):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Calc_DivOmega4(lg_o, DivOm, loc_ID[i], dom_size, H_Grid);} break;}
        case (CD6):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Calc_DivOmega6(lg_o, DivOm, loc_ID[i], dom_size, H_Grid);} break;}
        case (CD8):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Calc_DivOmega8(lg_o, DivOm, loc_ID[i], dom_size, H_Grid);} break;}
        default:        {std::cout << "VPM3D_cpu::Reproject_Particle_Set(): Incorrect specification of FD order"; return;}
    }

    Real MaxVortDiv = *std::max_element(DivOm.begin(), DivOm.end());
    Real MinVortDiv = *std::min_element(DivOm.begin(), DivOm.end());
    Real VortDivFac = std::max(fabs(MinVortDiv),fabs(MaxVortDiv))*H_Grid/OmMax;

    if (Debug) std::cout << "MinVortDiv = " << MinVortDiv << " MaxVortDiv = " << MaxVortDiv << " VortDivFactor = " << VortDivFac << std::endl;
    // Real ReProjFac = 5.0e-4;
    // if (VortDivFac<ReProjFac) return;

    std::cout << "MinVortDiv = " << MinVortDiv << " MaxVortDiv = " << MaxVortDiv << " VortDivFactor = " << VortDivFac << std::endl;

    // Solver using dirichlet solver with zero BC. This has been checked and is functioning correctly.
    // Dirichlet_Solver->Set_Input(DivOm);
    // Dirichlet_Solver->Forward_Transform();
    // Dirichlet_Solver->Convolution();
    // Dirichlet_Solver->Backward_Transform();
    // RVector F = RVector(NNT);
    // Dirichlet_Solver->Get_Output(F);

    // Reproject
    // switch (FDOrder)
    // {
    // case (CD2):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Reproject2(p_Array, F, i, FDTemplate, H_Grid);} break;}
    // case (CD4):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Reproject4(p_Array, F, i, FDTemplate, H_Grid);} break;}
    // case (CD6):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Reproject6(p_Array, F, i, FDTemplate, H_Grid);} break;}
    // case (CD8):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Reproject8(p_Array, F, i, FDTemplate, H_Grid);} break;}
    // }
}

void VPM3D_cpu::Reproject_Particle_Set_Spectral()
{
    // This procedures carries out the reprojection of the field in the spectral space.

    // Step 1: Check current divergence
    RVector DivOm = RVector(NNT,0);
    switch (FDOrder)
    {
        case (CD2):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Calc_DivOmega2(lg_o, DivOm, loc_ID[i], dom_size, H_Grid);} break;}
        case (CD4):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Calc_DivOmega4(lg_o, DivOm, loc_ID[i], dom_size, H_Grid);} break;}
        case (CD6):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Calc_DivOmega6(lg_o, DivOm, loc_ID[i], dom_size, H_Grid);} break;}
        case (CD8):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Calc_DivOmega8(lg_o, DivOm, loc_ID[i], dom_size, H_Grid);} break;}
        default:        {std::cout << "VPM3D_cpu::Reproject_Particle_Set_Spectral(): Incorrect specification of FD order"; return;}
    }

    Real MaxVortDiv = *std::max_element(DivOm.begin(), DivOm.end());
    Real MinVortDiv = *std::min_element(DivOm.begin(), DivOm.end());
    Real VortDivFac = std::max(fabs(MinVortDiv),fabs(MaxVortDiv))*H_Grid/OmMax;
    std::cout << "MinVortDiv = " << MinVortDiv << " MaxVortDiv = " << MaxVortDiv << " VortDivFactor = " << VortDivFac << std::endl;

    // Solve reprojection
    SFStatus Stat;
    Stat = Set_Input_Unbounded_3D(lg_o[0],lg_o[1],lg_o[2]);
    Stat = Transfer_Data_Device();

    //--- Calculate and extract velocity
    Forward_Transform();
    Spectral_Gradients_3DV_Reprojection();  // Note: the velocity array is stored prior to later steps
    Backward_Transform();
    Get_Output_Unbounded_3D(lg_o[0],lg_o[1],lg_o[2]);  // Extract updated field

    // Check new vorticity divergence
    RVector DivOm2 = RVector(NNT,0);
    switch (FDOrder)
    {
        case (CD2):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Calc_DivOmega2(lg_o, DivOm2, loc_ID[i], dom_size, H_Grid);} break;}
        case (CD4):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Calc_DivOmega4(lg_o, DivOm2, loc_ID[i], dom_size, H_Grid);} break;}
        case (CD6):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Calc_DivOmega6(lg_o, DivOm2, loc_ID[i], dom_size, H_Grid);} break;}
        case (CD8):     {Parallel_Kernel(NNT) {if (FD_Flag[i]) KER_Calc_DivOmega8(lg_o, DivOm2, loc_ID[i], dom_size, H_Grid);} break;}
        default:        {std::cout << "VPM3D_cpu::Reproject_Particle_Set_Spectral(): Incorrect specification of FD order"; return;}
    }
    Real MaxVortDivpost = *std::max_element(DivOm2.begin(), DivOm2.end());
    Real MinVortDivpost = *std::min_element(DivOm2.begin(), DivOm2.end());
    Real VortDivFacpost = std::max(fabs(MaxVortDivpost),fabs(MaxVortDivpost))*H_Grid/OmMax;
    std::cout << "MinVortDiv Post = " << MinVortDivpost << " MaxVortDiv Post = " << MaxVortDivpost << " VortDivFactor Post = " << VortDivFacpost << std::endl;

    std::cout << "Vorticity field reprojection completed with spectral method." << std::endl;
}

//-------------------------------------------
//----- Auxiliary grid operations  ----------
//-------------------------------------------

void VPM3D_cpu::Map_from_Auxiliary_Grid()
{
    // The vorticity grid of the auxiliary grid is mapped to the input array for the Fast Poisson solver.
    // This ensure that the vorticity field of the main domain is decoupled from the lifting body.
    if (Ext_Forcing.empty()) return;

    // Alternate approach: External sources
    // std::cout << "Alternate auxiliary grid mapping. NNodes = " << size(Ext_Forcing) << std::endl;
    OpenMPfor
    for (size_t i=0; i<Ext_Forcing.size(); i++){
        dim3 did = Ext_Forcing[i].cartid;       //std::get<0>(Ext_Forcing[i]);
        int gid = GID(did.x,did.y,did.z,NX,NY,NZ);
        r_Input1[gid] += Ext_Forcing[i].Vort(0); // std::get<1>(Ext_Forcing[i])(0);
        r_Input2[gid] += Ext_Forcing[i].Vort(1); //std::get<1>(Ext_Forcing[i])(1);
        r_Input3[gid] += Ext_Forcing[i].Vort(2); //std::get<1>(Ext_Forcing[i])(2);
    }
}

void VPM3D_cpu::Interpolate_Ext_Sources(Mapping M)
{
    // Prior to executing this function, a source field has been defined on the ource grid of the auxiliary grid.
    // In this step, the source field of the lagrangian grid p_Array are appended with this source field

    // External forcing stored in Ext_Forcing array.
    // In this step we loop over particles and interpolate to Lagrangian positions

    // Set mapping parameters
    int idsh = Set_Map_Shift(M) - 1;
    int nc = Set_Map_Stencil_Width(M) + 1;
    Map_Kernel Map;
    switch (M)
    {
    case (M2):  {Map = &mapM2;  break;}
    case (M4):  {Map = &mapM4;  break;}
    case (M4D): {Map = &mapM4D; break;}
    case (M6D): {Map = &mapM6D; break;}
    default:   break;
    }

    // Loop over external nodes
    for (size_t p=0; p<Ext_Forcing.size(); p++){
        dim3 sid = Ext_Forcing[p].cartid;   //std::get<0>(Ext_Forcing[p]);
        Vector3 Omega = Ext_Forcing[p].Vort;  //std::get<1>(Ext_Forcing[p]);

        // Loop over surrounding nodes
        // #pragma omp parallel for collapse(3)
        for (int i=0; i<nc; i++){
            for (int j=0; j<nc; j++){
                for (int k=0; k<nc; k++){

                    dim3 rid = dim3(sid.x+idsh+i, sid.y+idsh+j, sid.z+idsh+k);
                    int idr = GID(rid,dom_size);

                    // Calc mapping factors for this receiver node
                    Real fx, fy, fz;
                    Map(fabs((idsh+i)*1.0 + lg_d[0][idr]/Hx), fx);
                    Map(fabs((idsh+j)*1.0 + lg_d[1][idr]/Hy), fy);
                    Map(fabs((idsh+k)*1.0 + lg_d[2][idr]/Hz), fz);
                    Real mfac = fx*fy*fz;

                    // Add contribution
                    lg_o[0][idr] += Omega(0)*mfac;
                    lg_o[1][idr] += Omega(1)*mfac;
                    lg_o[2][idr] += Omega(2)*mfac;
                }
            }
        }
    }
}

//-------------------------------------------
//----- Generate Output grid for vis --------
//-------------------------------------------

void VPM3D_cpu::Get_Laplacian(RVector &xo, RVector &yo, RVector &zo)
{
    // For testing: Returns references to the particle vorticity field
    xo = Laplacian[0];    // Reference x grid
    yo = Laplacian[1];    // Reference y grid
    zo = Laplacian[2];    // Reference z grid
}

void VPM3D_cpu::Set_Laplacian(RVector &xo, RVector &yo, RVector &zo)
{
    // For testing: Overwrites
    for (int i=0; i<NNT; i++){
        Laplacian[0][i] = xo[i];
        Laplacian[1][i] = yo[i];
        Laplacian[2][i] = zo[i];
    }
}

void VPM3D_cpu::Get_Stretching( RVector &dgidx, RVector &dgidy, RVector &dgidz,
                                RVector &dgjdx, RVector &dgjdy, RVector &dgjdz,
                                RVector &dgkdx, RVector &dgkdy, RVector &dgkdz)
{
    // For testing: Returns references to the particle vorticity field

    // First calculate the stretching        
    switch (FDOrder)
    {
        case (CD2): {Parallel_Kernel(NNT) {if (FD_Flag[i])    KER_Stretch_FD2(eu_o, eu_o, eu_dodt, GradU, Laplacian, loc_ID[i], dom_size, H_Grid, KinVisc);}         break;}
        case (CD4): {Parallel_Kernel(NNT) {if (FD_Flag[i])    KER_Stretch_FD4(eu_o, eu_o, eu_dodt, GradU, Laplacian, loc_ID[i], dom_size, H_Grid, KinVisc);}         break;}
        case (CD6): {Parallel_Kernel(NNT) {if (FD_Flag[i])    KER_Stretch_FD6(eu_o, eu_o, eu_dodt, GradU, Laplacian, loc_ID[i], dom_size, H_Grid, KinVisc);}         break;}
        case (CD8): {Parallel_Kernel(NNT) {if (FD_Flag[i])    KER_Stretch_FD8(eu_o, eu_o, eu_dodt, GradU, Laplacian, loc_ID[i], dom_size, H_Grid, KinVisc);}         break;}
        default:    {std::cout << "VPM3D_cpu::Grid_Shear_Stresses(). FDOrder undefined.";   return;}
    }

    dgidx = GradU[0];
    dgidy = GradU[1];
    dgidz = GradU[2];
    dgjdx = GradU[3];
    dgjdy = GradU[4];
    dgjdz = GradU[5];
    dgkdx = GradU[6];
    dgkdy = GradU[7];
    dgkdz = GradU[8];
}

void VPM3D_cpu::Set_Stretching( RVector &dgidx, RVector &dgidy, RVector &dgidz,
                                RVector &dgjdx, RVector &dgjdy, RVector &dgjdz,
                                RVector &dgkdx, RVector &dgkdy, RVector &dgkdz)
{
    // For testing: Overwrites
    for (int i=0; i<NNT; i++){
        GradU[0][i] = dgidx[i];
        GradU[1][i] = dgidy[i];
        GradU[2][i] = dgidz[i];
        GradU[3][i] = dgjdx[i];
        GradU[4][i] = dgjdy[i];
        GradU[5][i] = dgjdz[i];
        GradU[6][i] = dgkdx[i];
        GradU[7][i] = dgkdy[i];
        GradU[8][i] = dgkdz[i];
    }
}

void VPM3D_cpu::Generate_VTK()
{
    // Specifies a specific output and then produces a vtk file for this
    Generate_VTK(eu_o[0], eu_o[1], eu_o[2], eu_dddt[0],eu_dddt[1],eu_dddt[2]);        // Omega, velocity on Eulerian grid
}

void VPM3D_cpu::Generate_VTK(const RVector &A1, const RVector &A2, const RVector &A3, const RVector &B1, const RVector &B2, const RVector &B3)
{
    // Specifies a specific output and then produces a vtk file for this

    // std::cout << dom_size.x csp dom_size.y csp dom_size.z << std::endl;
    // std::cout << padded_dom_size.x csp padded_dom_size.y csp padded_dom_size.z << std::endl;

    // Fill in output vector
    OpenMPfor
    for (int i=0; i<NNT; i++){
        int gid = GID(loc_ID[i],dom_size);
        int lid = GID(loc_ID[i],padded_dom_size);
        r_Input1[ lid] = A1[gid];
        r_Input2[ lid] = A2[gid];
        r_Input3[ lid] = A3[gid];
        r_Output1[lid] = B1[gid];
        r_Output2[lid] = B2[gid];
        r_Output3[lid] = B3[gid];
    }

    // Specify current filename.
    vtk_Name = vtk_Prefix + std::to_string(NStep) + ".vtk";

    Create_vtk();
}

void VPM3D_cpu::Generate_Plane(RVector &Uvoid)
{
    // Specifies a specific output and then produces a vtk file for this

    // I specify the y-value which shall be taken.
    int Y = NY/4;
    // Fill in output vector

    // TensorGrid *Src = &g_Array;
    TensorGrid *Src = &eu_o;

    OpenMPfor
    for (int i=0; i<NNX; i++){
        //        for (int j=0; j<NNY; j++){
        for (int k=0; k<NNZ; k++){
            int gid = i*NNZ+k;
            int lid = GID(i,Y,k,NNX,NNY,NNZ);
            Real ux = Src->at(0)[lid];
            Real uy = Src->at(1)[lid];
            Real uz = Src->at(2)[lid];
            Uvoid[gid] = sqrt(ux*ux + uy*uy + uz*uz);
        }
    }

}

//--- Destructor

VPM3D_cpu::~VPM3D_cpu()
{
    // Clears all memory allocated. The parent class destructors will also be called.
    DestructTensorGrid(lg_d);
    DestructTensorGrid(lg_o);
    DestructTensorGrid(int_lg_d);
    DestructTensorGrid(int_lg_o);
    DestructTensorGrid(kd_d);
    DestructTensorGrid(k2_o);
    DestructTensorGrid(k3_d);
    DestructTensorGrid(k3_o);
    DestructTensorGrid(k4_d);
    DestructTensorGrid(k4_o);
    DestructTensorGrid(tm1_d);
    DestructTensorGrid(tm1_o);
    DestructTensorGrid(tm1_dddt);
    DestructTensorGrid(tm1_dodt);

    DestructTensorGrid(Laplacian);
    DestructTensorGrid(GradU);
    DestructTensorGrid(gfilt_Array1);
    DestructTensorGrid(gfilt_Array1);

    //--- Exclusion zones
    FD_Flag.clear();
    Map_Flag.clear();
    Remesh_Flag.clear();
    SGS.clear();
}

}
