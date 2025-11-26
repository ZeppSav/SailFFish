#include "VPM3D_ocl.h"
#include "VPM3D_kernels_ocl.h"

#ifdef VKFFT

namespace SailFFish
{

SFStatus VPM3D_ocl::Setup_VPM(VPM_Input *I)
{
    // This sets up the problem for the case that a full VPM solver is being executed.
    // This includes generation of all of the templates arguments and additional arrays for field preparation etc.

    SFStatus Stat = NoError;

    //------------------------------
    //--- Set parameters
    //------------------------------

    // Specify sim parameters
    // Type = VPM;
    dT = I->dT;
    Integrator = I->Integrator;

    SolverMap = I->SolverMap;
    RemeshMap= I->RemeshMap;
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

    // // Set flag for auxiliary grid
    // Auxiliary = I->AuxGrid;

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

    // This is accomplished in one of two ways: Either we define the grid based on "Blocks" or we define the
    // grid based purely on bounding limits

    if (I->GridDef==NODES)      Stat = Define_Grid_Nodes(I);
    if (I->GridDef==BLOCKS)     Stat = Define_Grid_Blocks(I);

    //--- Worksize parameters to be called during kernel execution:

    // Block architecture is for 3D workgroups
    BlockArch.Dim = 3;
    BlockArch.global[0] = (size_t)NNX;
    BlockArch.global[1] = (size_t)NNY;
    BlockArch.global[2] = (size_t)NNZ;
    BlockArch.local[0] = (size_t)BX;
    BlockArch.local[1] = (size_t)BY;
    BlockArch.local[2] = (size_t)BZ;

    // List architecture for 1D workgroups
    ListArch.Dim = 1;
    ListArch.global[0] = (size_t)NNT;
    ListArch.local[0] = (size_t)BT;

    // List architecture for 1D workgroups
    ConvArch.Dim = 1;
    ConvArch.global[0] = (size_t)NTM;
    ConvArch.local[0] = (size_t)BT;

    // 1D Block architecture for loading external sources
    ExtArch.Dim = 3;
    ExtArch.global[1] = (size_t)BY;
    ExtArch.global[2] = (size_t)BZ;
    ExtArch.local[0] = (size_t)BX;
    ExtArch.local[1] = (size_t)BY;
    ExtArch.local[2] = (size_t)BZ;

    //--- Carry out sanity check here
    if (Hx!=H_Grid || Hy!=H_Grid || Hy!=H_Grid)
    {
        std::cout << "VPM3D_ocl::Setup_VPM: Mismatch between Input grid size H_Grid and calculated H_Grid. H_Grid =" csp H_Grid csp "Hx =" csp Hx csp "Hy =" csp Hy csp "Hz =" csp Hz << std::endl;
        // return GridError;
    }

    //--- Initialize kernels
    Stat = Initialize_Kernels();
    if (Stat != NoError)    return Stat;

    //--- Allocate and initialize data
    Stat = Allocate_Data();
    if (Stat != NoError)    return Stat;
    Initialize_Data();                          // Initialize halo data for FD calcs
    Initialize_Halo_Data();
    Set_Grid_Positions();                       // Specify positions of grid (for diagnostics and/or initialisation)

    //--- Prepare outputs
//    create_directory(std::filesystem::path("Output"));  // Generate output folder if not existing
    Create_Directory("Output");
    Generate_Summary("Summary.dat");
    Sim_begin = std::chrono::steady_clock::now();    // Begin clock

    // Carry out memory checks
    // size_t free_mem = 0;
    // size_t total_mem = 0;
    // CUresult result = cuMemGetInfo(&free_mem, &total_mem);
    // std::cout << "Total GPU memory: " << total_mem / (1024.0 * 1024.0) << " MB" << std::endl;
    // std::cout << "Free GPU memory: " << free_mem / (1024.0 * 1024.0) << " MB" << std::endl;
    // std::cout << "Used GPU memory: " << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB" << std::endl;

    return Stat;
}

SFStatus VPM3D_ocl::Allocate_Data()
{
    // Memory for the execution of the solver on the gpu is allocated

    try{

        Allocate_Buffer(lg_d, 3*NNT*sizeof(Real));
        Allocate_Buffer(lg_o, 3*NNT*sizeof(Real));
        Allocate_Buffer(lg_dddt, 3*NNT*sizeof(Real));
        Allocate_Buffer(lg_dodt, 3*NNT*sizeof(Real));
        // Allocate_Buffer(eu_d, 3*NNT*sizeof(Real));          // Not required... remove this!
        Allocate_Buffer(eu_o, 3*NNT*sizeof(Real));

        // Standard memory allocation
        // Allocate_Buffer(eu_dddt, 3*NNT*sizeof(Real));
        // Allocate_Buffer(eu_dodt, 3*NNT*sizeof(Real));
        // Allocate_Buffer(sgs, NNT*sizeof(Real));          // This variable is defined here, even if we are not employing a turbulence model
        // Allocate_Buffer(dumbuffer, 3*NNT*sizeof(Real));

        // Exploiting the "empty" space in the cl_r_Input/Output arrays
        // The data storage required for the cuFFT input arrays occupies approximately half of the input/output array(2*NT = 4*NNT)
        // When in-place data arrays are used, then cl_r_Inputx = cl_r_Outputx
        Allocate_SubBuffer(eu_dddt, cl_r_Output1, (uint64_t)4*NNT*sizeof(cl_real), (uint64_t)3*NNT*sizeof(cl_real));
        Allocate_SubBuffer(eu_dodt, cl_r_Output2, (uint64_t)4*NNT*sizeof(cl_real), (uint64_t)3*NNT*sizeof(cl_real));
        // dumbuffer = cl_r_Output3 + 4*NNT;
        Allocate_SubBuffer(sgs, cl_r_Output1, (uint64_t)7*NNT*sizeof(cl_real), (uint64_t)NNT*sizeof(cl_real));
        Allocate_SubBuffer(qcrit, cl_r_Output2, (uint64_t)7*NNT*sizeof(cl_real), (uint64_t)NNT*sizeof(cl_real));

        Allocate_Buffer(diagnostic_reduced, NDiags*NBT*sizeof(Real));
        Allocate_Buffer(magfilt_count, NBT*sizeof(int));
        // Allocate_Buffer(vis_plane, NBX*BX*NBZ*BZ*sizeof(Real));
        // Allocate_Buffer(travx, NBY*BY*NBZ*BZ*sizeof(Real));
        // Allocate_Buffer(travy, NBY*BY*NBZ*BZ*sizeof(Real));
        // Allocate_Buffer(travz, NBY*BY*NBZ*BZ*sizeof(Real));

        if (Integrator == EF){
            Allocate_Buffer(int_lg_d, 3*NNT*sizeof(Real));     // Spare buffer for generating .vtk files
            Allocate_Buffer(int_lg_o, 3*NNT*sizeof(Real));
        }

        if (Integrator == EM){
            Allocate_Buffer(int_lg_d, 3*NNT*sizeof(Real));
            Allocate_Buffer(int_lg_o, 3*NNT*sizeof(Real));
            Allocate_Buffer(k2_d, 3*NNT*sizeof(Real));
            Allocate_Buffer(k2_o, 3*NNT*sizeof(Real));
        }

        if (Integrator == RK2){
            Allocate_Buffer(int_lg_d, 3*NNT*sizeof(Real));
            Allocate_Buffer(int_lg_o, 3*NNT*sizeof(Real));
            Allocate_Buffer(k2_d, 3*NNT*sizeof(Real));
            Allocate_Buffer(k2_o, 3*NNT*sizeof(Real));
        }

        if (Integrator == AB2LF){
            Allocate_Buffer(int_lg_d, 3*NNT*sizeof(Real));
            Allocate_Buffer(int_lg_o, 3*NNT*sizeof(Real));
            Allocate_Buffer(tm1_d, 3*NNT*sizeof(Real));
            Allocate_Buffer(tm1_o, 3*NNT*sizeof(Real));
            Allocate_Buffer(tm1_dddt, 3*NNT*sizeof(Real));
            Allocate_Buffer(tm1_dodt, 3*NNT*sizeof(Real));
        }

        if (Integrator == RK3){
            Allocate_Buffer(int_lg_d, 3*NNT*sizeof(Real));
            Allocate_Buffer(int_lg_o, 3*NNT*sizeof(Real));
            Allocate_Buffer(k2_d, 3*NNT*sizeof(Real));
            Allocate_Buffer(k2_o, 3*NNT*sizeof(Real));
            Allocate_Buffer(k3_d, 3*NNT*sizeof(Real));
            Allocate_Buffer(k3_o, 3*NNT*sizeof(Real));
        }

        if (Integrator == LSRK3){
            // Specify dummy vectors
            Allocate_Buffer(int_lg_d, 3*NNT*sizeof(Real));
            Allocate_Buffer(int_lg_o, 3*NNT*sizeof(Real));
        }

        if (Integrator == RK4){
            Allocate_Buffer(int_lg_d, 3*NNT*sizeof(Real));
            Allocate_Buffer(int_lg_o, 3*NNT*sizeof(Real));
            Allocate_Buffer(k2_d, 3*NNT*sizeof(Real));
            Allocate_Buffer(k2_o, 3*NNT*sizeof(Real));
            Allocate_Buffer(k3_d, 3*NNT*sizeof(Real));
            Allocate_Buffer(k3_o, 3*NNT*sizeof(Real));
            Allocate_Buffer(k4_d, 3*NNT*sizeof(Real));
            Allocate_Buffer(k4_o, 3*NNT*sizeof(Real));
        }

        if (Integrator == LSRK4){
            // Specify dummy vectors
            Allocate_Buffer(int_lg_d, 3*NNT*sizeof(Real));
            Allocate_Buffer(int_lg_o, 3*NNT*sizeof(Real));
        }

        if (Turb==HYP){
            Allocate_Buffer(Laplacian, 3*NNT*sizeof(Real));
        }

        if (Turb==RVM1 || Turb==RVM2){

            // Allocate_Buffer(gfilt_Array1, 3*NNT*sizeof(Real));
            // Allocate_Buffer(gfilt_Array2, 3*NNT*sizeof(Real));

            // Exploiting the "empty" space in the cl_r_Input/Output arrays
            gfilt_Array1 = cl_r_Output1;
            gfilt_Array2 = cl_r_Output2;
        }
    }
    catch (std::bad_alloc& ex){
        std::cout << "VPM3D_ocl::Allocate_Data(): Insufficient memory for allocation of solver arrays." << std::endl;
        return MemError;
    }

    std::cout << "VPM3D_ocl::Allocate_Data: Memory allocated." << std::endl;

    return NoError;
}

void VPM3D_ocl::Initialize_Data()
{
    // Memory for the execution of the solver on the gpu is allocated

    if (lg_d   )        Zero_FloatBuffer(lg_d,      3*NNT*sizeof(Real));
    if (lg_o   )        Zero_FloatBuffer(lg_o,      3*NNT*sizeof(Real));
    if (lg_dddt)        Zero_FloatBuffer(lg_dddt,   3*NNT*sizeof(Real));
    if (lg_dodt)        Zero_FloatBuffer(lg_dodt,   3*NNT*sizeof(Real));
    if (eu_o   )        Zero_FloatBuffer(eu_o,      3*NNT*sizeof(Real));
    if (eu_dddt)        Zero_FloatBuffer(eu_dddt,   3*NNT*sizeof(Real));
    if (eu_dodt)        Zero_FloatBuffer(eu_dodt,   3*NNT*sizeof(Real));

    if (cl_r_Input1)    Zero_FloatBuffer(cl_r_Input1,   NT*sizeof(Real));
    if (cl_r_Input2)    Zero_FloatBuffer(cl_r_Input2,   NT*sizeof(Real));
    if (cl_r_Input3)    Zero_FloatBuffer(cl_r_Input3,   NT*sizeof(Real));
    if (cl_r_Output1)   Zero_FloatBuffer(cl_r_Output1,  NT*sizeof(Real));
    if (cl_r_Output2)   Zero_FloatBuffer(cl_r_Output2,  NT*sizeof(Real));
    if (cl_r_Output3)   Zero_FloatBuffer(cl_r_Output3,  NT*sizeof(Real));

    // ConstructTensorGrid(Laplacian,NNT,3);        // Laplacian
    // ConstructTensorGrid(GradU,NNT,9);            // Velocity gradient
    // ConstructTensorGrid(MapFactors,NNT,15);      // Mapping factors for updating field

    if (Integrator == EF){
        Zero_FloatBuffer(int_lg_d, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(int_lg_o, 3*NNT*sizeof(Real));
    }

    if (Integrator == EM){
        Zero_FloatBuffer(int_lg_d, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(int_lg_o, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(k2_d, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(k2_o, 3*NNT*sizeof(Real));
    }

    if (Integrator == RK2){
        Zero_FloatBuffer(int_lg_d, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(int_lg_o, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(k2_d, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(k2_o, 3*NNT*sizeof(Real));
    }

    if (Integrator == AB2LF){
        Zero_FloatBuffer(int_lg_d, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(int_lg_o, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(tm1_d, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(tm1_o, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(tm1_dddt, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(tm1_dodt, 3*NNT*sizeof(Real));
    }

    if (Integrator == RK3){
        Zero_FloatBuffer(int_lg_d, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(int_lg_o, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(k2_d, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(k2_o, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(k3_d, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(k3_o, 3*NNT*sizeof(Real));
    }

    if (Integrator == RK4){
        Zero_FloatBuffer(int_lg_d, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(int_lg_o, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(k2_d, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(k2_o, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(k3_d, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(k3_o, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(k4_d, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(k4_o, 3*NNT*sizeof(Real));
    }

    if (Turb==RVM1 || Turb==RVM2){
        // SGS = RVector(NNT,Real(0.0));                      // Subgrid scale
        // ConstructTensorGrid(gfilt_Array1,NNT,3);     // Small scale vorticity
        // ConstructTensorGrid(gfilt_Array2,NNT,3);     // Small scale vorticity
    }
}

void VPM3D_ocl::Retrieve_Grid_Positions(RVector &xc, RVector &yc, RVector &zc)
{
    //--- Return grid positions for vorticity field initialisation
    OpenMPfor
    for (int i=0; i<NNX; i++){
        for (int j=0; j<NNY; j++){
            for (int k=0; k<NNZ; k++)
            {
                int id_dest = GID(i,j,k,NNX,NNY,NNZ);
                xc[id_dest] = gX[i];
                yc[id_dest] = gY[j];
                zc[id_dest] = gZ[k];
            }
        }
    }

    std::cout << "VPM3D_ocl::Retrieve_Grid_Positions: Grid positions retrieved." << std::endl;
}

void VPM3D_ocl::Set_Input_Arrays(RVector &x0, RVector &y0, RVector &z0)
{
    // This input array is set externally. This is imported here, flattened and passed to the ocl buffer.
    Set_Input_Unbounded(x0,y0,z0);

    // Transfer to ocl buffer
    if (Architecture==BLOCK){
        // SFStatus stat = Execute_Kernel(ocl_map_fromUnbounded, BlockArch, {cl_r_Input1, cl_r_Input2, cl_r_Input3, eu_o}); // Testing initial grid
        SFStatus stat = Execute_Kernel(ocl_map_fromUnbounded, BlockArch, {cl_r_Input1, cl_r_Input2, cl_r_Input3, lg_o});
    }
}

//-------------------------------------------
//------------- Kernel setup ----------------
//-------------------------------------------

//--- Set up halo data

void VPM3D_ocl::Initialize_Halo_Data()
{
    // When data is loaded into shared memory during  Kernel execution for either mapping routines or FD routines, a halo around the block must also be loaded.
    // These parameters are prepared once as they will be accessed numeous time during execution.

    // Note: I tried loading this as function for the array, but something went wrong

    cl_int res;

    int NBuff, NHA, NHIT, nx, ny, nz;
    std::vector<int> xhs,yhs,zhs,hs;

    // Halo 1 padding
    xhs.clear();
    yhs.clear();
    zhs.clear();
    hs.clear();
    NBuff = 1;
    nx = BX+2*NBuff;
    ny = BY+2*NBuff;
    nz = BZ+2*NBuff;
    for (int i=0; i<nx; i++){
        for (int j=0; j<ny; j++){
            for (int k=0; k<nz; k++){
                bool inx = (i>=NBuff) && (i<=nz-NBuff-1);
                bool iny = (j>=NBuff) && (j<=ny-NBuff-1);
                bool inz = (k>=NBuff) && (k<=nz-NBuff-1);
                if (!(inx && iny && inz)){
                    xhs.push_back(i);
                    yhs.push_back(j);
                    zhs.push_back(k);
                }
            }
        }
    }
    NHA = int(xhs.size());
    NHIT = int(std::ceil(Real(NHA)/Real(BT)));
    StdAppend(hs,xhs);
    for (int i=NHA; i<NHIT*BT; i++) hs.push_back(nx);            // Dummy value catches the halo-reading loop
    StdAppend(hs,yhs);
    for (int i=NHA; i<NHIT*BT; i++) hs.push_back(0);
    StdAppend(hs,zhs);
    for (int i=NHA; i<NHIT*BT; i++) hs.push_back(0);
    NHA = NHIT*BT;
    // cudaMalloc((void**)&Halo1data, 3*NHA*sizeof(int));
    // Copy_Buffer(Halo1data, hs.data(), 3*NHA*sizeof(int), cudaMemcpyHostToDevice);
    Allocate_Buffer(Halo1data, 3*NHA*sizeof(int));
    res = clEnqueueWriteBuffer(vkGPU->commandQueue, Halo1data, CL_TRUE, 0, 3*NHA*sizeof(int), hs.data(), 0, NULL, NULL);
    std::cout << "Halo 1 array padded to: " << int(xhs.size()) csp int(yhs.size()) csp int(zhs.size()) csp "Total Halo buffer size: " << hs.size()
              << ". Each thread will be required to pull halo data N Times: " << NHIT <<  std::endl;

    // Halo 2 padding
    xhs.clear();
    yhs.clear();
    zhs.clear();
    hs.clear();
    NBuff = 2;
    nx = BX+2*NBuff;
    ny = BY+2*NBuff;
    nz = BZ+2*NBuff;
    for (int i=0; i<nx; i++){
        for (int j=0; j<ny; j++){
            for (int k=0; k<nz; k++){
                bool inx = (i>=NBuff) && (i<=nz-NBuff-1);
                bool iny = (j>=NBuff) && (j<=ny-NBuff-1);
                bool inz = (k>=NBuff) && (k<=nz-NBuff-1);
                if (!(inx && iny && inz)){
                    xhs.push_back(i);
                    yhs.push_back(j);
                    zhs.push_back(k);
                }
            }
        }
    }
    NHA = int(xhs.size());
    NHIT = int(std::ceil(Real(NHA)/Real(BT)));
    StdAppend(hs,xhs);
    for (int i=NHA; i<NHIT*BT; i++) hs.push_back(nx);            // Dummy value catches the halo-reading loop
    StdAppend(hs,yhs);
    for (int i=NHA; i<NHIT*BT; i++) hs.push_back(0);
    StdAppend(hs,zhs);
    for (int i=NHA; i<NHIT*BT; i++) hs.push_back(0);
    NHA = NHIT*BT;
    // cudaMalloc((void**)&Halo2data, 3*NHA*sizeof(int));
    // Copy_Buffer(Halo2data, hs.data(), 3*NHA*sizeof(int), cudaMemcpyHostToDevice);
    Allocate_Buffer(Halo2data, 3*NHA*sizeof(int));
    res = clEnqueueWriteBuffer(vkGPU->commandQueue, Halo2data, CL_TRUE, 0, 3*NHA*sizeof(int), hs.data(), 0, NULL, NULL);
    std::cout << "Halo 2 array padded to: " << int(xhs.size()) csp int(yhs.size()) csp int(zhs.size()) csp "Total Halo buffer size: " << hs.size()
              << ". Each thread will be required to pull halo data N Times: " << NHIT <<  std::endl;

    // Halo 3 padding
    xhs.clear();
    yhs.clear();
    zhs.clear();
    hs.clear();
    NBuff = 3;
    nx = BX+2*NBuff;
    ny = BY+2*NBuff;
    nz = BZ+2*NBuff;
    for (int i=0; i<nx; i++){
        for (int j=0; j<ny; j++){
            for (int k=0; k<nz; k++){
                bool inx = (i>=NBuff) && (i<=nz-NBuff-1);
                bool iny = (j>=NBuff) && (j<=ny-NBuff-1);
                bool inz = (k>=NBuff) && (k<=nz-NBuff-1);
                if (!(inx && iny && inz)){
                    xhs.push_back(i);
                    yhs.push_back(j);
                    zhs.push_back(k);
                }
            }
        }
    }
    NHA = int(xhs.size());
    NHIT = int(std::ceil(Real(NHA)/Real(BT)));
    StdAppend(hs,xhs);
    for (int i=NHA; i<NHIT*BT; i++) hs.push_back(nx);            // Dummy value catches the halo-reading loop
    StdAppend(hs,yhs);
    for (int i=NHA; i<NHIT*BT; i++) hs.push_back(0);
    StdAppend(hs,zhs);
    for (int i=NHA; i<NHIT*BT; i++) hs.push_back(0);
    NHA = NHIT*BT;
    Allocate_Buffer(Halo3data, 3*NHA*sizeof(int));
    res = clEnqueueWriteBuffer(vkGPU->commandQueue, Halo3data, CL_TRUE, 0, 3*NHA*sizeof(int), hs.data(), 0, NULL, NULL);
    std::cout << "Halo 3 array padded to: " << int(xhs.size()) csp int(yhs.size()) csp int(zhs.size()) csp "Total Halo buffer size: " << hs.size()
              << ". Each thread will be required to pull halo data N Times: " << NHIT <<  std::endl;

    // Halo 4 padding
    xhs.clear();
    yhs.clear();
    zhs.clear();
    hs.clear();
    NBuff = 4;
    nx = BX+2*NBuff;
    ny = BY+2*NBuff;
    nz = BZ+2*NBuff;
    for (int i=0; i<nx; i++){
        for (int j=0; j<ny; j++){
            for (int k=0; k<nz; k++){
                bool inx = (i>=NBuff) && (i<=nz-NBuff-1);
                bool iny = (j>=NBuff) && (j<=ny-NBuff-1);
                bool inz = (k>=NBuff) && (k<=nz-NBuff-1);
                if (!(inx && iny && inz)){
                    xhs.push_back(i);
                    yhs.push_back(j);
                    zhs.push_back(k);
                }
            }
        }
    }
    NHA = int(xhs.size());
    NHIT = int(std::ceil(Real(NHA)/Real(BT)));
    StdAppend(hs,xhs);
    for (int i=NHA; i<NHIT*BT; i++) hs.push_back(nx);            // Dummy value catches the halo-reading loop
    StdAppend(hs,yhs);
    for (int i=NHA; i<NHIT*BT; i++) hs.push_back(0);
    StdAppend(hs,zhs);
    for (int i=NHA; i<NHIT*BT; i++) hs.push_back(0);
    NHA = NHIT*BT;
    Allocate_Buffer(Halo4data, 3*NHA*sizeof(int));
    res = clEnqueueWriteBuffer(vkGPU->commandQueue, Halo4data, CL_TRUE, 0, 3*NHA*sizeof(int), hs.data(), 0, NULL, NULL);
    std::cout << "Halo 4 array padded to: " << int(xhs.size()) csp int(yhs.size()) csp int(zhs.size()) csp "Total Halo buffer size: " << hs.size()
              << ". Each thread will be required to pull halo data N Times: " << NHIT <<  std::endl;
}

//--- Prepare kernels

cl_kernel VPM3D_ocl::Generate_Kernel(const std::string &Body,       // Body of the kernel
                                     const std::string &Tag,        // Identifier of the kernel function
                                     int Halo,                      // If a halo kernel is being used, how large is the halo?
                                     int Map,                       // If a mapping procedure is being used, what type of mapping?
                                     int NHT,                       // What is the size of the shared memory array?
                                     bool Print)                    // Should the kernel be printed?
{
    cl_int err;
    std::string Source;

    // Add required types
    if (std::is_same<Real,float>::value)    Source.append(VPM3D_ocl_kernels_float);
    if (std::is_same<Real,double>::value)   Source.append(VPM3D_ocl_kernels_double);

    // Add grid constants (#defines- will be substituted in (or ignored) during compilation)
    Source.append("#define NX " + std::to_string(NNX) + "\n");
    Source.append("#define NY " + std::to_string(NNY) + "\n");
    Source.append("#define NZ " + std::to_string(NNZ) + "\n");
    Source.append("#define NT " + std::to_string(NNT) + "\n");
    Source.append("#define BX " + std::to_string(BX) + "\n");
    Source.append("#define BY " + std::to_string(BY) + "\n");
    Source.append("#define BZ " + std::to_string(BZ) + "\n");
    Source.append("#define BT " + std::to_string(BT) + "\n");
    Source.append("#define NBX " + std::to_string(NBX) + "\n");
    Source.append("#define NBY " + std::to_string(NBY) + "\n");
    Source.append("#define NBZ " + std::to_string(NBZ) + "\n");
    Source.append("__constant Real hx = " + std::to_string(Hx) + "; \n");
    Source.append("__constant Real hy = " + std::to_string(Hy) + "; \n");
    Source.append("__constant Real hz = " + std::to_string(Hz) + "; \n");
    Source.append("__constant Real XN1 = " + std::to_string(XN1) + "; \n");
    Source.append("__constant Real YN1 = " + std::to_string(YN1) + "; \n");
    Source.append("__constant Real ZN1 = " + std::to_string(ZN1) + "; \n");
    Source.append("__constant Real KinVisc = " + std::to_string(KinVisc) + "; \n");     // Only used for FD kernels

    // Add mapping and halo parameters
    if (Map!=0){
        Source.append("#define Map " + std::to_string(Map) + "\n");
        Source.append(VPM3D_ocl_kernels_mapping_functions);
    }
    if (Halo!=0){
        int NFX = BX+2*Halo;
        int NFY = BY+2*Halo;
        int NFZ = BZ+2*Halo;
        Source.append("#define Halo " + std::to_string(Halo) + "\n");
        Source.append("#define NFDX " + std::to_string(NFX) + "\n");
        Source.append("#define NFDY " + std::to_string(NFY) + "\n");
        Source.append("#define NFDZ " + std::to_string(NFZ) + "\n");
        int NHIT = std::ceil(Real(NFX*NFY*NFZ-BT)/Real(BT));
        Source.append("#define NHIT " + std::to_string(NHIT) + "\n");
    }
    Source.append("#define NHT " + std::to_string(NHT) + "\n");

    // Add grid id helper functions
    Source.append(ocl_GID_functions);       // Add gid functions (they are usually required but will be optimised out by compiler if not necessary)

    // Finally, add body of kernel
    Source.append(Body);

    if (Print) std::cout << Source << std::endl;

    // Compile kernel
    const char* source_str = Source.c_str();
    cl_program program = clCreateProgramWithSource(vkGPU->context, 1, &source_str, NULL, &err);
    err = clBuildProgram(program, 1, &vkGPU->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, vkGPU->device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "Build error:\n%s\n", buffer);
    }

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, Tag.c_str(), &err);

    std::cout << "Kernel with identifier " << Tag << " successfully compiled." << std::endl;
    return kernel;
}

SFStatus VPM3D_ocl::Initialize_Kernels()
{
    // This function compiles the ocl kernels on the device.

    std::string Kernel;

    try{
        // Rules:
        // __shared__ -> __local
        // threadIdx.x -> get_local_id(0)
        // threadIdx.y -> get_local_id(1)
        // threadIdx.z -> get_local_id(2)
        // blockIdx.x -> get_group_id(0)
        // blockIdx.y -> get_group_id(1)
        // blockIdx.z -> get_group_id(2)
        // Real(x) -> (Real)x
        // __syncthreads() -> barrier(CLK_LOCAL_MEM_FENCE);
        // gridDim.x -> get_num_groups(0)

        // Compile kernels
        ocl_VPM_convolution  = Generate_Kernel(ocl_curl,"curl",0,0,0);
        ocl_VPM_reprojection  = Generate_Kernel(VPM3D_ocl_kernels_reprojection,"vpm_reprojection",0,0,0);
        // ocl_monolith_to_block_arch = Generate_Kernel(VPM3D_ocl_kernels_monolith_to_block,"Monolith_to_Block",0,0,0);    // Obsolete
        ocl_block_to_monolith_arch  = Generate_Kernel(VPM3D_ocl_kernels_block_to_monolith,"Block_to_Monolith",0,0,0);   // Used for export to vtk
        // ocl_block_to_monolith_single  = new cudaKernel(Source,"Block_to_Monolith_Single");
        ocl_map_toUnbounded  = Generate_Kernel(VPM3D_ocl_kernels_map_to_unbounded,"Map_toUnbounded",0,0,0);
        ocl_map_fromUnbounded  = Generate_Kernel(VPM3D_ocl_kernels_map_from_unbounded,"Map_fromUnbounded",0,0,0);       // Used to import grid
        ocl_mapM2  = Generate_Kernel(VPM3D_ocl_kernels_map,"MapKernel",1,2,(BX+2)*(BY+2)*(BZ+2));
        ocl_mapM4  = Generate_Kernel(VPM3D_ocl_kernels_map,"MapKernel",2,4,(BX+4)*(BY+4)*(BZ+4));
        ocl_mapM4D  = Generate_Kernel(VPM3D_ocl_kernels_map,"MapKernel",2,42,(BX+4)*(BY+4)*(BZ+4));
        ocl_mapM6D  = Generate_Kernel(VPM3D_ocl_kernels_map,"MapKernel",3,6,(BX+6)*(BY+6)*(BZ+6));
        ocl_interpM2 = Generate_Kernel(VPM3D_ocl_kernels_interp,"InterpKernel",1,2,(BX+2)*(BY+2)*(BZ+2));
        ocl_interpM4 = Generate_Kernel(VPM3D_ocl_kernels_interp,"InterpKernel",2,4,(BX+4)*(BY+4)*(BZ+4));
        ocl_interpM4D = Generate_Kernel(VPM3D_ocl_kernels_interp,"InterpKernel",2,42,(BX+4)*(BY+4)*(BZ+4));
        ocl_interpM6D = Generate_Kernel(VPM3D_ocl_kernels_interp,"InterpKernel",3,6,(BX+6)*(BY+6)*(BZ+6));
        ocl_update = Generate_Kernel(VPM3D_ocl_kernels_update,"update",0,0,0);
        ocl_updateRK = Generate_Kernel(VPM3D_ocl_kernels_updateRK,"updateRK",0,0,0);
        ocl_updateRK2 = Generate_Kernel(VPM3D_ocl_kernels_updateRK2,"updateRK2",0,0,0);
        ocl_updateRK3 = Generate_Kernel(VPM3D_ocl_kernels_updateRK3,"updateRK3",0,0,0);
        ocl_updateRK4 = Generate_Kernel(VPM3D_ocl_kernels_updateRK4,"updateRK4",0,0,0);
        ocl_updateRKLS = Generate_Kernel(VPM3D_ocl_kernels_updateRKLS,"updateRK_LS",0,0,0);
        ocl_stretch_FD2 = Generate_Kernel(VPM3D_ocl_kernels_ShearStress,"Shear_Stress", 1, 2, (BX+2)*(BY+2)*(BZ+2));
        ocl_stretch_FD4 = Generate_Kernel(VPM3D_ocl_kernels_ShearStress,"Shear_Stress", 2, 4, (BX+4)*(BY+4)*(BZ+4));
        ocl_stretch_FD6 = Generate_Kernel(VPM3D_ocl_kernels_ShearStress,"Shear_Stress", 3, 6, (BX+6)*(BY+6)*(BZ+6));
        ocl_stretch_FD8 = Generate_Kernel(VPM3D_ocl_kernels_ShearStress,"Shear_Stress", 4, 8, (BX+8)*(BY+8)*(BZ+8));
        ocl_Diagnostics = Generate_Kernel(VPM3D_ocl_kernels_Diagnostics,"DiagnosticsKernel",0,0,0);
        // ocl_MagFilt1 = new cudaKernel(Source,"MagnitudeFiltering_Step1",BT);
        // ocl_MagFilt2 = new cudaKernel(Source,"MagnitudeFiltering_Step2",BT);
        // ocl_MagFilt3 = new cudaKernel(Source,"MagnitudeFiltering_Step3");
        ocl_freestream = Generate_Kernel(VPM3D_ocl_kernels_freestream, "AddFreestream",0,0,0);

        ocl_interpM2_block =  Generate_Kernel(VPM3D_ocl_kernels_InterpBlock,"Interp_Block",1,2,(BX+2)*(BY+2)*(BZ+2));
        ocl_interpM4_block =  Generate_Kernel(VPM3D_ocl_kernels_InterpBlock,"Interp_Block",2,4,(BX+4)*(BY+4)*(BZ+4));
        ocl_interpM4D_block = Generate_Kernel(VPM3D_ocl_kernels_InterpBlock,"Interp_Block",2,42,(BX+4)*(BY+4)*(BZ+4));
        ocl_interpM6D_block = Generate_Kernel(VPM3D_ocl_kernels_InterpBlock,"Interp_Block",3,6,(BX+6)*(BY+6)*(BZ+6));

        ocl_interpM2_ext =  Generate_Kernel(VPM3D_ocl_kernels_ExtSourceInterp,"Interp_Block_Ext",1,2,(BX+2)*(BY+2)*(BZ+2));
        ocl_interpM4_ext  = Generate_Kernel(VPM3D_ocl_kernels_ExtSourceInterp,"Interp_Block_Ext",2,4,(BX+4)*(BY+4)*(BZ+4));
        ocl_interpM4D_ext = Generate_Kernel(VPM3D_ocl_kernels_ExtSourceInterp,"Interp_Block_Ext",2,42,(BX+4)*(BY+4)*(BZ+4));
        ocl_interpM6D_ext = Generate_Kernel(VPM3D_ocl_kernels_ExtSourceInterp,"Interp_Block_Ext",3,6,(BX+6)*(BY+6)*(BZ+6));

        // External source operations
        Map_Ext = Generate_Kernel(VPM3D_ocl_kernels_MapExt,"Map_Ext_Bounded",0,0,0);
        Map_Ext_Unbounded = Generate_Kernel(VPM3D_ocl_kernels_MapExtUnb,"Map_Ext_Unbounded",0,0,0);

        // // Turbulence kernels
        // ocl_Laplacian_FD2 = new cudaKernel(Source,"Laplacian_Operator",1,2,(BX+2)*(BY+2)*(BZ+2));
        // ocl_Laplacian_FD4 = new cudaKernel(Source,"Laplacian_Operator",2,4,(BX+4)*(BY+4)*(BZ+4));
        // ocl_Laplacian_FD6 = new cudaKernel(Source,"Laplacian_Operator",3,6,(BX+6)*(BY+6)*(BZ+6));
        // ocl_Laplacian_FD8 = new cudaKernel(Source,"Laplacian_Operator",4,8,(BX+8)*(BY+8)*(BZ+8));
        // ocl_Turb_Hyp_FD2 = new cudaKernel(Source,"Hyperviscosity_Operator",1,2,(BX+2)*(BY+2)*(BZ+2));
        // ocl_Turb_Hyp_FD4 = new cudaKernel(Source,"Hyperviscosity_Operator",2,4,(BX+4)*(BY+4)*(BZ+4));
        // ocl_Turb_Hyp_FD6 = new cudaKernel(Source,"Hyperviscosity_Operator",3,6,(BX+6)*(BY+6)*(BZ+6));
        // ocl_Turb_Hyp_FD8 = new cudaKernel(Source,"Hyperviscosity_Operator",4,8,(BX+8)*(BY+8)*(BZ+8));
        ocl_sg_discfiltx  = Generate_Kernel(VPM3D_ocl_kernels_subgrid_discfilter,"SubGrid_DiscFilter",1,0,(BX+2)*(BY+2)*(BZ+2));
        ocl_sg_discfilty  = Generate_Kernel(VPM3D_ocl_kernels_subgrid_discfilter,"SubGrid_DiscFilter",1,1,(BX+2)*(BY+2)*(BZ+2));
        ocl_sg_discfiltz  = Generate_Kernel(VPM3D_ocl_kernels_subgrid_discfilter,"SubGrid_DiscFilter",1,2,(BX+2)*(BY+2)*(BZ+2));
        ocl_sg_discfiltss = Generate_Kernel(VPM3D_ocl_kernels_subgrid_discfilter,"SubGrid_DiscFilter",1,3,(BX+2)*(BY+2)*(BZ+2));
        ocl_Turb_RVM_FD2 = Generate_Kernel(VPM3D_ocl_kernels_RVM,"RVM_turbulentstress",1,2,(BX+2)*(BY+2)*(BZ+2));
        ocl_Turb_RVM_FD4 = Generate_Kernel(VPM3D_ocl_kernels_RVM,"RVM_turbulentstress",2,4,(BX+4)*(BY+4)*(BZ+4));
        ocl_Turb_RVM_FD6 = Generate_Kernel(VPM3D_ocl_kernels_RVM,"RVM_turbulentstress",3,6,(BX+6)*(BY+6)*(BZ+6));
        ocl_Turb_RVM_FD8 = Generate_Kernel(VPM3D_ocl_kernels_RVM,"RVM_turbulentstress",4,8,(BX+8)*(BY+8)*(BZ+8));
        // ocl_Turb_RVM_DGC_FD2 = new cudaKernel(Source,"RVM_DGC_turbulentstress",1,2,(BX+2)*(BY+2)*(BZ+2));
        // ocl_Turb_RVM_DGC_FD4 = new cudaKernel(Source,"RVM_DGC_turbulentstress",2,4,(BX+4)*(BY+4)*(BZ+4));
        // ocl_Turb_RVM_DGC_FD6 = new cudaKernel(Source,"RVM_DGC_turbulentstress",3,6,(BX+6)*(BY+6)*(BZ+6));
        // ocl_Turb_RVM_DGC_FD8 = new cudaKernel(Source,"RVM_DGC_turbulentstress",4,8,(BX+8)*(BY+8)*(BZ+8));

        //--- Superfluous kernels from previous implementations
        // ocl_ExtractPlaneX = new cudaKernel(Source,"ExtractPlaneX");
        // ocl_ExtractPlaneY = new cudaKernel(Source,"ExtractPlaneY");

        // std::cout << "-----------Shared Memory Analysis-----------" << std::endl;
        // // // // 1) How much dynamic memory is available?
        // // CUfunction_attribute attr = CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
        // // int sharedMemPerSM = ocl_stretch_FD8->Get_Instance()->get_func_attribute(attr);
        // // std::cout << "Max shared memory ocl_stretch_FD8 : " << sharedMemPerSM << " bytes" << std::endl;
        // // std::cout << "adapting this..." << std::endl;
        // // ocl_stretch_FD8->Get_Instance()->set_func_attribute(attr,96*1024);                  // Crashing!!!!
        // // int sharedMemPerSM2 = ocl_stretch_FD8->Get_Instance()->get_func_attribute(attr);
        // // std::cout << "Max shared memory ocl_stretch_FD8 : " << sharedMemPerSM2 << " bytes" << std::endl;

        // int ns;
        // if (FDOrder==CD2) ns = 6*(BX+2)*(BY+2)*(BZ+2);
        // if (FDOrder==CD4) ns = 6*(BX+4)*(BY+4)*(BZ+4);
        // if (FDOrder==CD6) ns = 6*(BX+6)*(BY+6)*(BZ+6);
        // if (FDOrder==CD8) ns = 6*(BX+8)*(BY+8)*(BZ+8);
        // std::cout << "Size of shared memory required for stretching kernel: " << ns*sizeof(Real) << " bytes." << std::endl;

        // int nm;
        // if (SolverMap==M2)  nm = 6*(BX+2)*(BY+2)*(BZ+2);
        // if (SolverMap==M4)  nm = 6*(BX+4)*(BY+4)*(BZ+4);
        // if (SolverMap==M4D) nm = 6*(BX+4)*(BY+4)*(BZ+4);
        // if (SolverMap==M6D) nm = 6*(BX+6)*(BY+6)*(BZ+6);
        // std::cout << "Size of shared memory required for mapping kernel: " << nm*sizeof(Real) << " bytes." << std::endl;

        // std::cout << "---------------------------------------------" << std::endl;

    }
    catch (std::bad_alloc& ex){
        std::cout << "VPM3D_ocl::Initialize_Kernels(): Problem with initializing kernels. Error code: " << ex.what() << std::endl;
        return SetupError;
    }

    std::cout << "VPM3D_ocl::Initialize_Kernels: Kernel initialization successful." << std::endl;
    return NoError;

}

SFStatus VPM3D_ocl::Execute_Kernel(cl_kernel kernel, OpenCLWorkSize &Worksize, const std::vector<cl_mem> &buffers, const std::vector<Real> &params)
{
    // This is a generalized function for executing any of the kernels in this solver.

    // The required buffers are passed to the kernel
    for (size_t i = 0; i<buffers.size(); i++)
    {
        cl_int err = clSetKernelArg(kernel, static_cast<cl_uint>(i), sizeof(cl_mem), &buffers[i]);
        if (err != CL_SUCCESS){
            std::cout << "VPM3D_ocl::Execute_Kernel: Setting arguments failed." << std::endl;
            return ExecError;
        }
    }

    // The required parameters are passed to the kernel
    for (size_t i = 0; i<params.size(); i++)
    {
        cl_int err = clSetKernelArg(kernel, static_cast<cl_uint>(i+buffers.size()), sizeof(cl_real), &params[i]);
        if (err != CL_SUCCESS){
            std::cout << "VPM3D_ocl::Execute_Kernel: Setting parameters failed." << std::endl;
            return ExecError;
        }
    }

    // Execute kernel with the appropriate work group size.
    cl_int err = clEnqueueNDRangeKernel(vkGPU->commandQueue, kernel, Worksize.Dim, NULL, Worksize.global, Worksize.local, 0, NULL, NULL);
    err = clFinish(vkGPU->commandQueue);
    return ConvertClError(err);
}

//-------------------------------------------
//------------- Timestepping ----------------
//-------------------------------------------

void VPM3D_ocl::Advance_Particle_Set()
{
    // This is a stepping function which updates the particle field and carried out the desired
    // updates of the field.
    if (NStep%NRemesh==0 && NStep!=NInit)   Remeshing = true;
    else                                    Remeshing = false;
    if (Remeshing)                    Remesh_Particle_Set();                                // Remesh
    if (Remeshing && MagFiltFac>0)    Magnitude_Filtering();                                // Filter magnitude
    if (DivFilt && Remeshing && NStep%NReproject==0)   Reproject_Particle_Set_Spectral();   // Reproject vorticity field
    Update_Particle_Field();                                                                // Update vorticity field
    if (NExp>0 && NStep%NExp==0 && NStep>0 && NStep>=ExpTB) Generate_VTK();                 // Export grid if desired
    Increment_Time();
}

void VPM3D_ocl::Update_Particle_Field()
{
    SFStatus Stat = NoError;

    if (Integrator == EF)       // Eulerian forward
    {
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        Stat = Execute_Kernel(ocl_update, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt}, {cl_real(dT)});
    }

    if (Integrator == EM)       // Explicit midpoint
    {
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        Copy_Buffer(int_lg_d, lg_d, 3*NNT*sizeof(Real));
        Copy_Buffer(int_lg_o, lg_o, 3*NNT*sizeof(Real));

        Stat = Execute_Kernel(ocl_update, ListArch, {int_lg_d, int_lg_o, lg_dddt, lg_dodt}, {cl_real(0.5*dT)});
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k2_d, k2_o);
        Stat = Execute_Kernel(ocl_update, ListArch, {lg_d, lg_o, k2_d, k2_o}, {cl_real(dT)});
    }

    if (Integrator == AB2LF)
    {
        if (Remeshing)
        {

            std::cout << "AB2LF not yet implemented" << std::endl;

            // Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
            // Calc_Grid_Diagnostics();
            // Parallel_Kernel(NNT) {KER_Copy(lg_d, lg_o, tm1_d, tm1_o, i);}         // Store for next time step
            // Parallel_Kernel(NNT) {KER_Copy(lg_dddt, lg_dodt, tm1_dddt, tm1_dodt, i);}   // Store for next time step
            // Parallel_Kernel(NNT) {KER_Copy(lg_d, lg_o, int_lg_d, int_lg_o, i);}
            // Parallel_Kernel(NNT) {KER_Update(int_lg_d, int_lg_o, lg_dddt, lg_dodt, i, 0.5*dT);}
            // Calc_Particle_RateofChange(int_lg_d, int_lg_o, lg_dddt, lg_dodt);
            // Parallel_Kernel(NNT) {KER_Update(lg_d, lg_o, lg_dddt, lg_dodt, i, dT);}
        }
        else
        {
            std::cout << "AB2LF not yet implemented" << std::endl;

            // Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
            // Calc_Grid_Diagnostics();
            // Parallel_Kernel(NNT) {KER_Copy(lg_d, lg_o, int_lg_d, int_lg_o, i);}         // Temporarily store before updating
            // Parallel_Kernel(NNT) {KER_Update_AB2LF(lg_d, lg_o, lg_dddt, lg_dodt, tm1_d, tm1_o, tm1_dddt, tm1_dodt, i, dT);}
            // Parallel_Kernel(NNT) {KER_Copy(int_lg_d, int_lg_o, tm1_d, tm1_o, i);}     // Store for next time step
            // Parallel_Kernel(NNT) {KER_Copy(lg_dddt, lg_dodt, tm1_dddt, tm1_dodt, i);}   // Store for next time step
        }
    }

    if (Integrator == RK2)      // Runge-Kutta second order
    {
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        Stat = Execute_Kernel(ocl_updateRK, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o}, {cl_real(dT)});
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k2_d, k2_o);
        Stat = Execute_Kernel(ocl_updateRK2, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, k2_d, k2_o}, {cl_real(dT)});
    }

    if (Integrator == RK3)      // Runge-Kutta third order
    {
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        Stat = Execute_Kernel(ocl_updateRK, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o}, {cl_real(0.5*dT)});
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k2_d, k2_o);
        Stat = Execute_Kernel(ocl_updateRK, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o}, {cl_real(-dT)});
        Stat = Execute_Kernel(ocl_update, ListArch, {int_lg_d, int_lg_o, k2_d, k2_o}, {cl_real(2.0*dT)});
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k3_d, k3_o);
        Stat = Execute_Kernel(ocl_updateRK3, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, k2_d, k2_o, k3_d, k3_o}, {cl_real(dT)});
    }

    if (Integrator == RK4)      // Runge-Kutta fourth order
    {
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        Stat = Execute_Kernel(ocl_updateRK, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o}, {cl_real(0.5*dT)});
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k2_d, k2_o);
        Stat = Execute_Kernel(ocl_updateRK, ListArch, {lg_d, lg_o, k2_d, k2_o, int_lg_d, int_lg_o}, {cl_real(0.5*dT)});
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k3_d, k3_o);
        Stat = Execute_Kernel(ocl_updateRK, ListArch, {lg_d, lg_o, k3_d, k3_o, int_lg_d, int_lg_o}, {cl_real(dT)});
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k4_d, k4_o);
        Stat = Execute_Kernel(ocl_updateRK4, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, k2_d, k2_o, k3_d, k3_o, k4_d, k4_o}, {cl_real(dT)});
    }

    if (Integrator == LSRK3)    // Runge-Kutta third order low-storage (4-stage)
    {
        // Zero_FloatBuffer(int_lg_d,    0, 3*NNT*sizeof(Real));       // Clear intermediate arrays (Necessary?)
        // Zero_FloatBuffer(int_lg_o,    0, 3*NNT*sizeof(Real));       // Clear intermediate arrays (Necessary?)
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        Stat = Execute_Kernel(ocl_updateRKLS, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o}, {cl_real(RK3A[0]), cl_real(RK3B[0]), cl_real(dT)});
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Stat = Execute_Kernel(ocl_updateRKLS, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o}, {cl_real(RK3A[1]), cl_real(RK3B[1]), cl_real(dT)});
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Stat = Execute_Kernel(ocl_updateRKLS, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o}, {cl_real(RK3A[2]), cl_real(RK3B[2]), cl_real(dT)});
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Stat = Execute_Kernel(ocl_updateRKLS, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o}, {cl_real(RK3A[3]), cl_real(RK3B[3]), cl_real(dT)});
    }

    if (Integrator == LSRK4)    // Runge-Kutta fourth order low-storage (5-stage)
    {
        // Zero_FloatBuffer(int_lg_d, 3*NNT*sizeof(Real));       // Clear intermediate arrays (Necessary?)
        // Zero_FloatBuffer(int_lg_o, 3*NNT*sizeof(Real));       // Clear intermediate arrays (Necessary?)
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        Stat = Execute_Kernel(ocl_updateRKLS, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o}, {cl_real(RK4A[0]), cl_real(RK4B[0]), cl_real(dT)});
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Stat = Execute_Kernel(ocl_updateRKLS, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o}, {cl_real(RK4A[1]), cl_real(RK4B[1]), cl_real(dT)});
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Stat = Execute_Kernel(ocl_updateRKLS, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o}, {cl_real(RK4A[2]), cl_real(RK4B[2]), cl_real(dT)});
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Stat = Execute_Kernel(ocl_updateRKLS, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o}, {cl_real(RK4A[3]), cl_real(RK4B[3]), cl_real(dT)});
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Stat = Execute_Kernel(ocl_updateRKLS, ListArch, {lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o}, {cl_real(RK4A[4]), cl_real(RK4B[4]), cl_real(dT)});
    }
}

void VPM3D_ocl::Calc_Particle_RateofChange(const cl_mem pd, const cl_mem po, cl_mem dpddt, cl_mem dpodt)
{
    // This calculates for the given input field the rate of change arrays

    // Clear grid and lagrangian input derivative arrays
    Zero_FloatBuffer(eu_o,    3*NNT*sizeof(Real));       // Clear grid (vorticity) arrays
    Zero_FloatBuffer(dpddt,   3*NNT*sizeof(Real));       // Clear Lagrangian grid rate of change arrays
    Zero_FloatBuffer(dpodt,   3*NNT*sizeof(Real));       // Clear Lagrangian grid rate of change arrays
    Zero_FloatBuffer(eu_dddt, 3*NNT*sizeof(Real));       // Clear Eulerian grid rate of change arrays
    Zero_FloatBuffer(eu_dodt, 3*NNT*sizeof(Real));       // Clear Eulerian grid rate of change arrays

    // Map vorticity from Lagrangian grid to Eulerian grid
    SFStatus Stat = NoError;
    switch (SolverMap)
    {
        case (M2):  {Stat = Execute_Kernel(ocl_mapM2 , BlockArch, {po, pd, Halo1data, eu_o});    break;}
        case (M4):  {Stat = Execute_Kernel(ocl_mapM4 , BlockArch, {po, pd, Halo2data, eu_o});    break;}
        case (M4D): {Stat = Execute_Kernel(ocl_mapM4D, BlockArch, {po, pd, Halo2data, eu_o});    break;}
        case (M6D): {Stat = Execute_Kernel(ocl_mapM6D, BlockArch, {po, pd, Halo3data, eu_o});    break;}
        default:    {std::cout << "VPM3D_ocl::Calc_Particle_RateofChange(). Mapping undefined.";   return;}
    }

    // // Calc_Grid_SpectralRatesof_Change();         // The rates of change are calculated on the fixed grid
    Calc_Grid_FDRatesof_Change();                 // The rates of change are calculated on the fixed grid

    // Map grid values to particles
    switch (SolverMap)
    {
        case (M2):  {Stat = Execute_Kernel(ocl_interpM2 , BlockArch, {eu_dddt, eu_dodt, pd, Halo1data, dpddt, dpodt});    break;}
        case (M4):  {Stat = Execute_Kernel(ocl_interpM4 , BlockArch, {eu_dddt, eu_dodt, pd, Halo2data, dpddt, dpodt});    break;}
        case (M4D): {Stat = Execute_Kernel(ocl_interpM4D, BlockArch, {eu_dddt, eu_dodt, pd, Halo2data, dpddt, dpodt});    break;}
        case (M6D): {Stat = Execute_Kernel(ocl_interpM6D, BlockArch, {eu_dddt, eu_dodt, pd, Halo3data, dpddt, dpodt});    break;}
        default:    {std::cout << "VPM3D_ocl::Calc_Particle_RateofChange(). Mapping undefined.";   return;}
    }
}

void VPM3D_ocl::Calc_Grid_FDRatesof_Change()
{
    // The rates of change on the grid (eu_dddt, eu_dodt) are calculated using SailFFish and finite differences

    // Calculate velocity on the grid- the method due to eastwood requires an expanded domain, so this must be mapped
    // NOTE: We are skipping mapping external sources for now... will include this later

    // Reset input arrays for FFT solver
    Zero_FloatBuffer(cl_r_Input1, NT*sizeof(Real));
    Zero_FloatBuffer(cl_r_Input2, NT*sizeof(Real));
    Zero_FloatBuffer(cl_r_Input3, NT*sizeof(Real));

    // Stat = Execute_Kernel(Kernel, Arch, {});

    // Specify input arrays for FFT solver
    SFStatus Stat = NoError;
    Stat = Execute_Kernel(ocl_map_toUnbounded, BlockArch, {eu_o, cl_r_Input1, cl_r_Input2, cl_r_Input3});
    Map_External_Sources();
    Forward_Transform();
    Stat = Execute_Kernel(ocl_VPM_convolution, ConvArch, {c_FTInput1, c_FTInput2, c_FTInput3, c_FG, c_FGi, c_FGj, c_FGk, c_FTOutput1, c_FTOutput2, c_FTOutput3});
    Backward_Transform();
    Stat = Execute_Kernel(ocl_map_fromUnbounded, BlockArch, {cl_r_Output1, cl_r_Output2, cl_r_Output3, eu_dddt});

    //--- Calculate shear stresses on grid
    Grid_Shear_Stresses();
    Grid_Turb_Shear_Stresses();

    //--- Add freestream velocity
    Add_Freestream_Velocity();
}

void VPM3D_ocl::Grid_Shear_Stresses()
{
    // We shall execute the different fd depending on the order of the FD
    SFStatus Stat = NoError;
    switch (FDOrder)
    {
        case (CD2): {Stat = Execute_Kernel(ocl_stretch_FD2, BlockArch, {eu_o, eu_dddt, Halo1data, eu_dodt, sgs, qcrit});   break;}
        case (CD4): {Stat = Execute_Kernel(ocl_stretch_FD4, BlockArch, {eu_o, eu_dddt, Halo2data, eu_dodt, sgs, qcrit});   break;}
        case (CD6): {Stat = Execute_Kernel(ocl_stretch_FD6, BlockArch, {eu_o, eu_dddt, Halo3data, eu_dodt, sgs, qcrit});   break;}
        case (CD8): {Stat = Execute_Kernel(ocl_stretch_FD8, BlockArch, {eu_o, eu_dddt, Halo4data, eu_dodt, sgs, qcrit});   break;}
        default:    {std::cout << "VPM3D_ocl::Grid_Shear_Stresses(). FDOrder undefined.";   break;}
    }
    return;
}

void VPM3D_ocl::Grid_Turb_Shear_Stresses()
{
    SFStatus Stat = NoError;

    // Turbulent shear stresses are calculated on the Eulerian grid
    if (Turb==LAM) return;

    if (Turb==HYP)
    {
        // As the storage of the laplacian operator incurs a lot of additional overhead, unlike for the CPU implementation,
        // I will here recalculate it.

        std::cout << "VPM3D_ocl::Grid_Turb_Shear_Stresses(): Hyperviscosity style Turbulence kernels not yet implemented." << std::endl;

        // Zero_FloatBuffer(Laplacian, Real(0.0), 3*NNT*sizeof(Real));

        // switch (FDOrder)
        // {
        // case (CD2): {ocl_Laplacian_FD2->Execute(eu_o, Halo1data, Laplacian);     break; }
        // case (CD4): {ocl_Laplacian_FD4->Execute(eu_o, Halo2data, Laplacian);     break; }
        // case (CD6): {ocl_Laplacian_FD6->Execute(eu_o, Halo3data, Laplacian);     break; }
        // case (CD8): {ocl_Laplacian_FD8->Execute(eu_o, Halo4data, Laplacian);     break; }
        // default:    {std::cout << "VPM3D_ocl::Grid_Turb_Shear_Stresses. FDOrder undefined.";  break; }
        // }

        // switch (FDOrder)
        // {
        // case (CD2): {ocl_Turb_Hyp_FD2->Execute(Laplacian, Halo1data, C_smag, eu_dodt);     break; }
        // case (CD4): {ocl_Turb_Hyp_FD4->Execute(Laplacian, Halo2data, C_smag, eu_dodt);     break; }
        // case (CD6): {ocl_Turb_Hyp_FD6->Execute(Laplacian, Halo3data, C_smag, eu_dodt);     break; }
        // case (CD8): {ocl_Turb_Hyp_FD8->Execute(Laplacian, Halo4data, C_smag, eu_dodt);     break; }
        // default:    {std::cout << "VPM3D_ocl::Grid_Turb_Shear_Stresses. FDOrder undefined.";  break; }
        // }
    }

    // Regularized variational multiscale models

    if (Turb==RVM1)         // RVM first order
    {
        // Copy Omega array into temp array.
        Zero_FloatBuffer(gfilt_Array1, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(gfilt_Array2, 3*NNT*sizeof(Real));
        Copy_Buffer(gfilt_Array2, eu_o, 3*NNT*sizeof(Real));
        Stat = Execute_Kernel(ocl_sg_discfiltz, BlockArch, {eu_o, Halo1data, gfilt_Array2, gfilt_Array1});   // Z sweep
        Stat = Execute_Kernel(ocl_sg_discfilty, BlockArch, {eu_o, Halo1data, gfilt_Array1, gfilt_Array2});   // Y sweep
        Stat = Execute_Kernel(ocl_sg_discfiltx, BlockArch, {eu_o, Halo1data, gfilt_Array2, gfilt_Array1});   // X sweep
        Stat = Execute_Kernel(ocl_sg_discfiltss, BlockArch, {eu_o, Halo1data, gfilt_Array1, gfilt_Array2});  // set small scale

        switch (FDOrder)
        {
            case (CD2): {Stat = Execute_Kernel(ocl_Turb_RVM_FD2, BlockArch, {gfilt_Array2, sgs, Halo1data, eu_dodt}, {C_smag});    break; }
            case (CD4): {Stat = Execute_Kernel(ocl_Turb_RVM_FD4, BlockArch, {gfilt_Array2, sgs, Halo2data, eu_dodt}, {C_smag});    break; }
            case (CD6): {Stat = Execute_Kernel(ocl_Turb_RVM_FD6, BlockArch, {gfilt_Array2, sgs, Halo3data, eu_dodt}, {C_smag});    break; }
            case (CD8): {Stat = Execute_Kernel(ocl_Turb_RVM_FD8, BlockArch, {gfilt_Array2, sgs, Halo4data, eu_dodt}, {C_smag});    break; }
            default:    {std::cout << "VPM3D_ocl::Grid_Turb_Shear_Stresses. FDOrder undefined for RVM2.";  break; }
        }
    }

    if (Turb==RVM2)         // RVM second order
    {
        // Copy Omega array into temp array.
        Zero_FloatBuffer(gfilt_Array1, 3*NNT*sizeof(Real));
        Zero_FloatBuffer(gfilt_Array2, 3*NNT*sizeof(Real));
        Copy_Buffer(gfilt_Array1, eu_o, 3*NNT*sizeof(Real));
        Stat = Execute_Kernel(ocl_sg_discfiltz, BlockArch, {eu_o, Halo1data, gfilt_Array1, gfilt_Array2});  // Z sweep 1
        Stat = Execute_Kernel(ocl_sg_discfiltz, BlockArch, {eu_o, Halo1data, gfilt_Array2, gfilt_Array1});  // Z sweep 2
        Stat = Execute_Kernel(ocl_sg_discfilty, BlockArch, {eu_o, Halo1data, gfilt_Array1, gfilt_Array2});  // Y sweep 1
        Stat = Execute_Kernel(ocl_sg_discfilty, BlockArch, {eu_o, Halo1data, gfilt_Array2, gfilt_Array1});  // Y sweep 2
        Stat = Execute_Kernel(ocl_sg_discfiltx, BlockArch, {eu_o, Halo1data, gfilt_Array1, gfilt_Array2});  // X sweep 1
        Stat = Execute_Kernel(ocl_sg_discfiltx, BlockArch, {eu_o, Halo1data, gfilt_Array2, gfilt_Array1});  // X sweep 2
        Stat = Execute_Kernel(ocl_sg_discfiltss, BlockArch, {eu_o, Halo1data, gfilt_Array1, gfilt_Array2}); // set small scale

        switch (FDOrder)
        {
            case (CD2): {Stat = Execute_Kernel(ocl_Turb_RVM_FD2, BlockArch, {gfilt_Array2, sgs, Halo1data, eu_dodt}, {C_smag});    break; }
            case (CD4): {Stat = Execute_Kernel(ocl_Turb_RVM_FD4, BlockArch, {gfilt_Array2, sgs, Halo2data, eu_dodt}, {C_smag});    break; }
            case (CD6): {Stat = Execute_Kernel(ocl_Turb_RVM_FD6, BlockArch, {gfilt_Array2, sgs, Halo3data, eu_dodt}, {C_smag});    break; }
            case (CD8): {Stat = Execute_Kernel(ocl_Turb_RVM_FD8, BlockArch, {gfilt_Array2, sgs, Halo4data, eu_dodt}, {C_smag});    break; }
            default:    {std::cout << "VPM3D_ocl::Grid_Turb_Shear_Stresses. FDOrder undefined for RVM2.";  break; }
        }
    }

}

void VPM3D_ocl::Remesh_Particle_Set()
{
    // Clear current array
    Zero_FloatBuffer(eu_o, 3*NNT*sizeof(cl_real));

    // Map to new grid
    switch (RemeshMap)
    {
        case (M2):  {Execute_Kernel(ocl_mapM2 ,BlockArch,{lg_o, lg_d, Halo1data, eu_o});    break;}
        case (M4):  {Execute_Kernel(ocl_mapM4 ,BlockArch,{lg_o, lg_d, Halo2data, eu_o});    break;}
        case (M4D): {Execute_Kernel(ocl_mapM4D,BlockArch,{lg_o, lg_d, Halo2data, eu_o});    break;}
        case (M6D): {Execute_Kernel(ocl_mapM6D,BlockArch,{lg_o, lg_d, Halo3data, eu_o});    break;}
        default:    {std::cout << "VPM3D_ocl::Remesh_Particle_Set(). Mapping undefined.";   return;}
    }

    // Transfer data from Eulerian grid to Lagrangian grid and clear temp vars
    Copy_Buffer(lg_o, eu_o, 3*NNT*sizeof(cl_real));
    Zero_FloatBuffer(eu_o, 3*NNT*sizeof(cl_real));
    Zero_FloatBuffer(lg_d, 3*NNT*sizeof(cl_real));             // Reset Lagrangian grid displacement

    std::cout << "Particle Set Remeshed" << std::endl;
}

// void VPM3D_ocl::Magnitude_Filtering()
// {
//     if (MagFiltFac==0.0) return;

//     // Step 1: Check maximum vorticity with reduction kernel.
//     Zero_FloatBuffer(diagnostic_reduced,     Real(0.0), NDiags*NBT*sizeof(Real));
//     ocl_MagFilt1->Execute(lg_o, diagnostic_reduced);
//     RVector dred(NBT);
//     Copy_Buffer(dred.data(), diagnostic_reduced, NBT*sizeof(Real), cudaMemcpyDeviceToHost);
//     Real OmMax = *std::max_element(dred.begin(), dred.begin()+NBT);
//     Real TargetOmMax = OmMax*MagFiltFac;

//     // Step 2: Sweep over field and remove particles with low strength and count remaining (active) particles
//     Zero_FloatBuffer(diagnostic_reduced,  Real(0.0), NDiags*NBT*sizeof(Real));
//     Zero_FloatBuffer(magfilt_count,    int(0), NBT*sizeof(int));
//     ocl_MagFilt2->Execute(lg_o, diagnostic_reduced, magfilt_count, TargetOmMax);
//     RVector VorticityRemoved(3*NBT,0);
//     Copy_Buffer(VorticityRemoved.data(), diagnostic_reduced, 3*NBT*sizeof(Real), cudaMemcpyDeviceToHost);
//     std::vector<int> CountActive(NBT,0);
//     Copy_Buffer(CountActive.data(), magfilt_count, NBT*sizeof(int), cudaMemcpyDeviceToHost);

//     // Now sum over values
//     Real MagRem[3] = {0};
//     int Counted;
//     OpenMPfor
//         for (int i=0; i<4; i++){
//         if (i<3)    MagRem[i] = std::accumulate(VorticityRemoved.begin()+i*NBT,VorticityRemoved.begin()+(i+1)*NBT,0.0);
//         else        Counted = std::accumulate(CountActive.begin(), CountActive.end(), 0);
//     }

//     // Step 3: Add removed vorticity to remaining particles with nonzero strength in order to conserve total circulation
//     Real Magincx = MagRem[0]/Real(Counted);
//     Real Magincy = MagRem[1]/Real(Counted);
//     Real Magincz = MagRem[2]/Real(Counted);
//     ocl_MagFilt3->Execute(lg_o, Magincx, Magincy, Magincz);

//     // std::cout << "Magnitude filtering has been carried out. Number of remaining particles with nonzero vorticity: " << Counted << std::endl;
// }

void VPM3D_ocl::Reproject_Particle_Set_Spectral()
{
    // The particle field is reprojected in the spectral space.
    Zero_FloatBuffer(cl_r_Input1, NT*sizeof(Real));
    Zero_FloatBuffer(cl_r_Input2, NT*sizeof(Real));
    Zero_FloatBuffer(cl_r_Input3, NT*sizeof(Real));

    // Stat = Execute_Kernel(Kernel, Arch, {Inputs})

    SFStatus Stat = NoError;
    Stat = Execute_Kernel(ocl_map_toUnbounded, BlockArch, {lg_o, cl_r_Input1, cl_r_Input2, cl_r_Input3});
    Forward_Transform();
    Stat = Execute_Kernel(ocl_VPM_reprojection, ConvArch, {c_FTInput1, c_FTInput2, c_FTInput3,
                                                           c_FG, c_FGi, c_FGj, c_FGk,
                                                           c_FTOutput1, c_FTOutput2, c_FTOutput3});
    Backward_Transform();
    Stat = Execute_Kernel(ocl_map_fromUnbounded, BlockArch, {cl_r_Output1, cl_r_Output2, cl_r_Output3, lg_o});
    std::cout << "Vorticity field reprojection completed with spectral method." << std::endl;
}

void VPM3D_ocl::Add_Freestream_Velocity()
{
    // This adds the local freestream to the grid
    if (sqrt(Ux*Ux + Uy*Uy + Uz*Uz)==0.0) return;
    Execute_Kernel(ocl_freestream,ListArch,{eu_dddt},{Real(Ux), Real(Uy), Real(Uz)});
}

void VPM3D_ocl::Calc_Grid_Diagnostics()
{
    // Diagnostics are calculated in blocks of size BS This is transferred back to the CPU and summed there for simplicity
    Zero_FloatBuffer(diagnostic_reduced, NDiags*NBT*sizeof(Real));
    SFStatus Stat = Execute_Kernel(ocl_Diagnostics, ListArch, {eu_o, eu_dddt, diagnostic_reduced});


    // Transfer back to host
    RVector dred(NDiags*NBT);
    VkFFTResult res = transferDataToCPU(vkGPU, dred.data(), &diagnostic_reduced, (uint64_t)NDiags*NBT*sizeof(Real));

    // Carry out sum on host (could be optimised, but for now this probably isn't a bottleneck
    RVector Sums(15,0);
    Parallel_Kernel(15) {
        if (i<13)   Sums[i] = std::accumulate(dred.begin()+i*NBT,dred.begin()+(i+1)*NBT,0.0);
        else        Sums[i] = *std::max_element(dred.begin()+i*NBT, dred.begin()+(i+1)*NBT);
    }

    Real dV = Hx*Hy*Hz;
    Real C[3] = {Sums[0 ]*dV, Sums[1 ]*dV, Sums[2 ]*dV};
    Real L[3] = {Sums[3 ]*dV, Sums[4 ]*dV, Sums[5 ]*dV};
    Real A[3] = {Sums[6 ]*dV, Sums[7 ]*dV, Sums[8 ]*dV};
    Real K1 = Sums[9]*dV;
    Real K2 = Sums[10]*dV;
    Real E = Sums[11]*dV;
    Real H = Sums[12]*dV;
    Real OmMax = Sums[13];
    Real UMax = Sums[14];
    Real CFL = UMax*dT/H_Grid;
    // Real SMax =  *std::max_element(t_SMax.begin(), t_SMax.end());

    //--------------------- Output these values into a file in the output called "Diagnostics.dat"
    //--------------------- Create the file & directory if they don't yet exist

    if (Log)
    {
        // File target location
        std::string OutputDirectory = "Output/" + OutputFolder;
        std::string FilePath = OutputDirectory + "/Diagnostics.dat";

        if (NStep==NInit)
        {
//            create_directory(std::filesystem::path(OutputDirectory));   // Generate directory
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

// //-------------------------------------------
// //----------- Grid functions ----------------
// //-------------------------------------------

void VPM3D_ocl::Extract_Field(const cl_mem Field, const RVector &Px, const RVector &Py, const RVector &Pz, RVector &Ux, RVector &Uy, RVector &Uz, Mapping M)
{
    // Values are extracted from the grid using a local interpolation. This is carried out by loading a block into memory and then
    // interpolating this in shared memory
    int NP = Px.size();

    // Step 1: Bin interpolation positions
    std::vector<std::vector<Vector3>> IDB(NBT);
    std::vector<std::vector<int>> IDPart(NBT);
    for (int i=0; i<NP; i++){
        Real rx = Px[i] - X0;           // Specify relative position of node.
        Real ry = Py[i] - Y0;
        Real rz = Pz[i] - Z0;
        int nbx = int(rx/HLx);
        int nby = int(ry/HLy);
        int nbz = int(rz/HLz);
        int nbt = nbx*NBY*NBZ + nby*NBZ + nbz;
        IDB[nbt].push_back(Vector3(rx-HLx*nbx, ry-HLy*nby, rz-HLz*nbz));
        IDPart[nbt].push_back(i);

        // std::cout << rx csp ry csp rz csp nbx csp nby csp nbz csp nbt << " GLOB Boxes " << NBX csp NBY csp NBZ << std::endl;
    }

    // Sort into blocks of size NT <= BT
    std::vector<Vector3> NDS;
    std::vector<int> map_blx, map_bly, map_blz, idout(NP);
    int count = 0;
    for (int b=0; b<NBT; b++){
        int n = IDB[b].size();                               // # particles in this box
        if (n==0)         continue;                         // No particles in this box
        int nbx = int(b/(NBY*NBZ)), nbs = b-nbx*NBY*NBZ;    // Box x index
        int nby = int(nbs/NBZ);                             // Box y index
        int nbz = nbs - nby*NBZ;                            // Box z index
        int nb = int(n/BT);                                 // # blocks filled
        if ((n-nb*BT)>0)   nb++;                            // Last box is not full

        // Fill global particle array with particle corresponding to this box
        StdAppend(NDS,IDB[b]);
        std::vector<Vector3> rem(nb*BT-n,Vector3::Zero());  // Remaining zeros
        StdAppend(NDS,rem);                                 // Pad arrays

        // Fill block array with corresponding block data
        for (int i=0; i<nb; i++){
            map_blx.push_back(nbx);
            map_bly.push_back(nby);
            map_blz.push_back(nbz);
        }

        // Identify the id in NDS corresponding to node id Input arrays
        for (int i=0; i<n; i++) idout[IDPart[b][i]] = count++;
        // count += nb*BT-n;
        count = int(NDS.size());
    }

    // NDS now contains the padded evaluation positions (in relative CS)
    // map_bl_i contains the block id of the boxes to be evaluated
    // map_nP is the number of evaluation points for this block
    // Sort into SOA
    int sND = NDS.size();
    RVector map_X(sND), map_Y(sND), map_Z(sND);
    Parallel_Kernel(sND) {
     // Serial_Kernel(sND) {
        map_X[i] = NDS[i](0);
        map_Y[i] = NDS[i](1);
        map_Z[i] = NDS[i](2);
    }

    // Allocate temporary buffers
    cl_mem px, py, pz, ux, uy, uz;
    Allocate_Buffer(px, sND*sizeof(cl_real));
    Allocate_Buffer(py, sND*sizeof(cl_real));
    Allocate_Buffer(pz, sND*sizeof(cl_real));
    Allocate_Buffer(ux, sND*sizeof(cl_real));
    Allocate_Buffer(uy, sND*sizeof(cl_real));
    Allocate_Buffer(uz, sND*sizeof(cl_real));

    int NBlock = map_blx.size();
    cl_mem bidx, bidy, bidz;
    Allocate_Buffer(bidx, NBlock*sizeof(cl_int));
    Allocate_Buffer(bidy, NBlock*sizeof(cl_int));
    Allocate_Buffer(bidz, NBlock*sizeof(cl_int));

    // for (int i=0; i<NBlock; i++) std::cout << map_blx[i] csp map_bly[i] csp map_blz[i] << std::endl;

    // Pass data to device
    VkFFTResult Stat = VKFFT_SUCCESS;
    Stat = transferDataFromCPU(vkGPU, map_X.data(),     &px,     sND*sizeof(cl_real));
    Stat = transferDataFromCPU(vkGPU, map_Y.data(),     &py,     sND*sizeof(cl_real));
    Stat = transferDataFromCPU(vkGPU, map_Z.data(),     &pz,     sND*sizeof(cl_real));
    Stat = transferDataFromCPU(vkGPU, map_blx.data(),   &bidx,   NBlock*sizeof(cl_int));
    Stat = transferDataFromCPU(vkGPU, map_bly.data(),   &bidy,   NBlock*sizeof(cl_int));
    Stat = transferDataFromCPU(vkGPU, map_blz.data(),   &bidz,   NBlock*sizeof(cl_int));

    // Note: JOE
    // When simulating with the UBerT Turbine, the following call crashes.
    // SFStat = Execute_Kernel(ocl_interpM4_block , ExtArch, {Field,px,py,pz,bidx,bidy,bidz,Halo2data,ux,uy,uz});
    // wiht the error: CL_INVALID_COMMAND_QUEUE if Map >= M4 (works for M2?!?!?).
    // I have looked at the kernel and the work sizes appear to be specified correctly--- the only conceptual difference
    // is that UBeRT has a prescribed circulation distribution..

    // Execute kernel
    ExtArch.global[0] = (size_t)BX*NBlock;
    SFStatus SFStat = NoError;
    switch (M)
    {
        case (M2):      {SFStat = Execute_Kernel(ocl_interpM2_block , ExtArch, {Field,px,py,pz,bidx,bidy,bidz,Halo1data,ux,uy,uz});    break;}
        case (M4):      {SFStat = Execute_Kernel(ocl_interpM4_block , ExtArch, {Field,px,py,pz,bidx,bidy,bidz,Halo2data,ux,uy,uz});    break;}
        case (M4D):     {SFStat = Execute_Kernel(ocl_interpM4D_block, ExtArch, {Field,px,py,pz,bidx,bidy,bidz,Halo2data,ux,uy,uz});    break;}
        case (M6D):     {SFStat = Execute_Kernel(ocl_interpM6D_block, ExtArch, {Field,px,py,pz,bidx,bidy,bidz,Halo3data,ux,uy,uz});    break;}
        default:        {std::cout << "VPM3D_ocl::Extract Field: Mapping undefined."; return;}
    }

    // Extract out data
    RVector rux(sND), ruy(sND), ruz(sND);
    Stat = transferDataToCPU(vkGPU, rux.data(), &ux, sND*sizeof(cl_real));
    Stat = transferDataToCPU(vkGPU, ruy.data(), &uy, sND*sizeof(cl_real));
    Stat = transferDataToCPU(vkGPU, ruz.data(), &uz, sND*sizeof(cl_real));

    // Now cycle through and pass back values at correct positions
    for (int i=0; i<NP; i++){
        // std::cout << rux[idout[i]] csp ruy[idout[i]] csp ruz[idout[i]] << std::endl;
        Ux[i] = rux[idout[i]];
        Uy[i] = ruy[idout[i]];
        Uz[i] = ruz[idout[i]];
    }

    // Clear temporary arrays
    clReleaseMemObject(px);
    clReleaseMemObject(py);
    clReleaseMemObject(pz);
    clReleaseMemObject(ux);
    clReleaseMemObject(uy);
    clReleaseMemObject(uz);
    clReleaseMemObject(bidx);
    clReleaseMemObject(bidy);
    clReleaseMemObject(bidz);
}

//-------------------------------------------
//------- Auxiliary grid functions ----------
//-------------------------------------------

//-----------------------------------
//------- External sources ----------
//-----------------------------------

void VPM3D_ocl::Interpolate_Ext_Sources(Mapping M)
{
    // Elements are being permanently mapped from the auxiliary grid to the current Lagrangian grid.
    // In order to ensure the strengths are mapped to the current particle positions, an interpolation of the vorticity on
    // the auxiliary grid to the lagrangian positions has to take place.

    // We map the new grid to an intermediate (full) grid which has been zeroed.
    if (NBExt==0) return;

    SFStatus Stat = NoError;

    // Map to full grid with Map_Ext kernel.
    ExtArch.global[0] = (size_t)BX*NBExt;
    Zero_FloatBuffer(eu_dddt, 3*NNT*sizeof(Real));      // Dummy grid (currently unused)
    Stat = Execute_Kernel(Map_Ext, ExtArch, {ExtVortX,ExtVortY,ExtVortZ,blX,blY,blZ,eu_dddt});

    // Now execute interp block onto lg_o grid
    switch (M)
    {
        case (M2):      {Stat = Execute_Kernel(ocl_interpM2_ext,    ExtArch, {eu_dddt,blX,blY,blZ,lg_d,Halo1data,lg_o});    break;}
        case (M4):      {Stat = Execute_Kernel(ocl_interpM4_ext,    ExtArch, {eu_dddt,blX,blY,blZ,lg_d,Halo2data,lg_o});    break;}
        case (M4D):     {Stat = Execute_Kernel(ocl_interpM4D_ext,   ExtArch, {eu_dddt,blX,blY,blZ,lg_d,Halo2data,lg_o});    break;}
        case (M6D):     {Stat = Execute_Kernel(ocl_interpM6D_ext,   ExtArch, {eu_dddt,blX,blY,blZ,lg_d,Halo3data,lg_o});    break;}
        default:        {std::cout << "VPM3D_ocl::Extract Field: Mapping undefined."; return;}
    }

    // reset vorticity grid
    Zero_FloatBuffer(eu_dddt, 3*NNT*sizeof(Real));                 // Dummy grid (currently unused)
}

void VPM3D_ocl::Store_Grid_Node_Sources(const RVector &Px, const RVector &Py, const RVector &Pz, const RVector &Ox, const RVector &Oy, const RVector &Oz, Mapping Map)
{
    // This is a modified version of the function within VPM_Solver. Passing data to and from the GPU requires it to be packaged in
    // a block-style architecture for efficiency. This is done here.

    if (Px.empty() || Py.empty() || Pz.empty()) return;

    Ext_Forcing.clear();
    Map_Source_Nodes(Px, Py, Pz, Ox, Oy, Oz, Ext_Forcing, Map);

    // Specify block IDs
    std::vector<ParticleMap> BlockParts, OrdBlockParts;
    for (ParticleMap p : Ext_Forcing){
        dim3 pid =  p.cartid;
        dim3 bid = dim3(pid.x/BX, pid.y/BY, pid.z/BZ);
        ParticleMap bp;
        bp.cartid = pid;
        bp.blockid = bid;
        bp.globblid = GID(bid.x,bid.y,bid.z,NBX,NBY,NBZ);
        bp.interblockid = dim3(pid.x-bid.x*BX, pid.y-bid.y*BY, pid.z-bid.z*BZ);
        bp.Vort = p.Vort;
        BlockParts.push_back(bp);
    }

    // Sort based on box id
    // std::vector<ParticleMap*> pBlockParts;
    // for (ParticleMap p : BlockParts) pBlockParts.push_back(&p);
    // std::sort(pBlockParts.begin(), pBlockParts.end(), [](const ParticleMap* a, const ParticleMap* b) { return a->globblid < b->globblid; });
    // for (ParticleMap *p : pBlockParts) OrdBlockParts.push_back(*p);
    std::sort(BlockParts.begin(), BlockParts.end(), [](const ParticleMap a, const ParticleMap b) { return a.globblid < b.globblid; });
    // for (ParticleMap p : BlockParts) OrdBlockParts.push_back(p);

    // We now have the block parts ordered correctly. We now group them together.
    RVector Obx, Oby, Obz;
    std::vector<int> sIDX, sIDY, sIDZ, sID;     // X,Y,Z indices of active blocks
    int idb = -1, bgid = 0;
    for (ParticleMap p : BlockParts){
        if (p.globblid != idb){                 // New block

            // Add new block to array.
            bgid = BT*sID.size();
            Obx.insert(Obx.end(), BT, Real(0.));    // Append onto x array
            Oby.insert(Oby.end(), BT, Real(0.));    // Append onto y array
            Obz.insert(Obz.end(), BT, Real(0.));    // Append onto z array

            // Set block ids
            idb = p.globblid;                   // Update current block id
            sID.push_back(idb);                 // Update current block number
            sIDX.push_back(p.blockid.x);        // Add cartesian block ids
            sIDY.push_back(p.blockid.y);        // (necessary for mapping temp elements to vort array)
            sIDZ.push_back(p.blockid.z);
        }

        // Now add source to block array
        int ids = GID(p.interblockid.x, p.interblockid.y, p.interblockid.z, BX, BY, BZ);
        Obx[bgid+ids] = p.Vort(0);
        Oby[bgid+ids] = p.Vort(1);
        Obz[bgid+ids] = p.Vort(2);
    }

    // These should now be passed to the corresponding OpenCl buffers

    // Allocate external arrays if necessary
    NBExt = sID.size();
    if (NBufferExt < NBExt){

        // Reset size of external array
        NBufferExt = NBExt;

        // Clear arrays if necessary
        if (ExtVortX) clReleaseMemObject(ExtVortX);   ExtVortX = nullptr;
        if (ExtVortY) clReleaseMemObject(ExtVortY);   ExtVortY = nullptr;
        if (ExtVortZ) clReleaseMemObject(ExtVortZ);   ExtVortZ = nullptr;
        if (blX) clReleaseMemObject(blX);             blX = nullptr;
        if (blY) clReleaseMemObject(blY);             blY = nullptr;
        if (blZ) clReleaseMemObject(blZ);             blZ = nullptr;

        // Allocate OpenlCL Buffers
        SFStatus AllStat = NoError;
        AllStat = Allocate_Buffer(ExtVortX, NBufferExt*BT*sizeof(cl_real));
        AllStat = Allocate_Buffer(ExtVortY, NBufferExt*BT*sizeof(cl_real));
        AllStat = Allocate_Buffer(ExtVortZ, NBufferExt*BT*sizeof(cl_real));
        AllStat = Allocate_Buffer(blX     , NBufferExt*sizeof(cl_int));
        AllStat = Allocate_Buffer(blY     , NBufferExt*sizeof(cl_int));
        AllStat = Allocate_Buffer(blZ     , NBufferExt*sizeof(cl_int));
    }

    // Transfer data to arrays
    VkFFTResult Stat = VKFFT_SUCCESS;
    Stat = transferDataFromCPU(vkGPU, Obx.data(),  &ExtVortX, NBExt*BT*sizeof(cl_real));
    Stat = transferDataFromCPU(vkGPU, Oby.data(),  &ExtVortY, NBExt*BT*sizeof(cl_real));
    Stat = transferDataFromCPU(vkGPU, Obz.data(),  &ExtVortZ, NBExt*BT*sizeof(cl_real));
    Stat = transferDataFromCPU(vkGPU, sIDX.data(), &blX,      NBExt*sizeof(cl_int));
    Stat = transferDataFromCPU(vkGPU, sIDY.data(), &blY,      NBExt*sizeof(cl_int));
    Stat = transferDataFromCPU(vkGPU, sIDZ.data(), &blZ,      NBExt*sizeof(cl_int));

    ConvertVkFFTError(Stat);
    // std::cout << Obx.size() csp Oby.size() csp Obz.size() csp sIDX.size() csp sIDY.size() csp sIDZ.size() csp NBExt csp BT << std::endl;
    // std::cout << "VPM3D_ocl::Store_Grid_Node_Sources: Successfully stored. Size of external forcing array: " << Ext_Forcing.size() << std::endl;
}

void VPM3D_ocl::Map_External_Sources()
{
    // In the case that the external sources should be included in the vorticity (e.g. from a lifting line),
    // rather than adding these to the full vorticity field,they are only added to the unbounded array
    if (NBExt==0) return;
    ExtArch.global[0] = (size_t)BX*NBExt;
    SFStatus Stat = Execute_Kernel(Map_Ext_Unbounded, ExtArch, {ExtVortX,ExtVortY,ExtVortZ,blX,blY,blZ,cl_r_Input1,cl_r_Input2,cl_r_Input3});
    // Map_Ext_Unbounded->Instantiate(Ext_block_extent, blockarch_block);
    // Map_Ext_Unbounded->Execute(ExtVortX,ExtVortY,ExtVortZ,blX,blY,blZ,cl_r_Input1,cl_r_Input2,cl_r_Input3);
}

// //-------------------------------------------
// //----- Generate Output grid for vis --------
// //-------------------------------------------

void VPM3D_ocl::Generate_VTK()
{
    Generate_VTK(eu_o, eu_dddt);
    // Generate_VTK(eu_dodt, eu_dddt);
    // Generate_VTK(lg_o, lg_dddt);
}

void VPM3D_ocl::Generate_VTK(cl_mem vtkoutput1, cl_mem vtkoutput2, const std::string& Name)
{
    // Specifies a specific output and then produces a vtk file for this
    // The arrays int_lg_d & int_p_o are stored as dummy arrays

    if (vtkoutput1==nullptr) return;
    if (vtkoutput2==nullptr) return;
    SFStatus Stat = NoError;

    // Convert to correct data ordering if necessary
    if (Architecture==BLOCK){
        Stat = Execute_Kernel(ocl_block_to_monolith_arch, BlockArch, {vtkoutput1, int_lg_d});
        Stat = Execute_Kernel(ocl_block_to_monolith_arch, BlockArch, {vtkoutput2, int_lg_o});
    }
    if (Architecture==MONO){
        // Copy_Buffer(int_lg_d, vtkoutput1, 3*NNT*sizeof(Real), cudaMemcpyDeviceToHost);        // Lagrangian grid vorticity field
        // Copy_Buffer(int_lg_o, vtkoutput2, 3*NNT*sizeof(Real), cudaMemcpyDeviceToHost);        // Lagrangian grid vorticity field
    }

    // Clear arrays (I don't think this is necessary)
    if (r_in1)      memset(r_Input1, 0., NT*sizeof(Real));
    if (r_in2)      memset(r_Input2, 0., NT*sizeof(Real));
    if (r_in3)      memset(r_Input3, 0., NT*sizeof(Real));
    if (r_out_1)    memset(r_Output1, 0., NT*sizeof(Real));
    if (r_out_2)    memset(r_Output2, 0., NT*sizeof(Real));
    if (r_out_3)    memset(r_Output3, 0., NT*sizeof(Real));

    // Transfer data from device to host
    Real *hostout1 = (Real*)malloc(3*NNT*sizeof(Real));
    Real *hostout2 = (Real*)malloc(3*NNT*sizeof(Real));
    VkFFTResult res = VKFFT_SUCCESS;
    res = transferDataToCPU(vkGPU, hostout1, &int_lg_d, (uint64_t)3*NNT*sizeof(Real));
    res = transferDataToCPU(vkGPU, hostout2, &int_lg_o, (uint64_t)3*NNT*sizeof(Real));

    // Fill in output vector
    OpenMPfor
    for (int i=0; i<NNX; i++){
        for (int j=0; j<NNY; j++){
            for (int k=0; k<NNZ; k++){
                // int gid = GID(i,j,k,NX,NY,NZ);
                // int lid = GID(i,j,k,NNX,NNY,NNZ);
                int gid = GF_GID3(i,j,k,NX,NY,NZ);
                int lid = GF_GID3(i,j,k,NNX,NNY,NNZ);
                r_Input1[gid] = hostout1[lid      ];
                r_Input2[gid] = hostout1[lid+1*NNT];
                r_Input3[gid] = hostout1[lid+2*NNT];
                r_Output1[gid] = hostout2[lid      ];
                r_Output2[gid] = hostout2[lid+1*NNT];
                r_Output3[gid] = hostout2[lid+2*NNT];
            }
        }
    }

    // Specify current filename.
    vtk_Name = vtk_Prefix + std::to_string(NStep) + Name + ".vtk";

    Create_vtk();

    Zero_FloatBuffer(int_lg_d, 3*NNT*sizeof(Real));
    Zero_FloatBuffer(int_lg_o, 3*NNT*sizeof(Real));
    free(hostout1);
    free(hostout2);

    // // HACK to export qcrit
    // Generate_VTK_Scalar();
}

// void VPM3D_ocl::Generate_Plane(RVector &U)
// {
//     // This function retrieves the wake plane from the data for visualisation

//     // std::cout << "Check Prev " << *std::min_element(U.begin(), U.end()) csp  *std::max_element(U.begin(), U.end()) << std::endl;
//     int NPlane = NBX*BX*NBZ*BZ;
//     int YPlane = NNY/2;          // Approximately middle plane
//     ocl_ExtractPlaneY->Execute(eu_o, vis_plane, YPlane);
//     Copy_Buffer(U.data(), vis_plane, sizeof(Real)*NPlane, cudaMemcpyDeviceToHost);
//     // std::cout << "Check Post " << *std::min_element(U.begin(), U.end()) csp  *std::max_element(U.begin(), U.end()) << std::endl;
// }

// void VPM3D_ocl::Generate_Traverse(int XP, RVector &U, RVector &V, RVector &W)
// {
//     // This function retrieves the wake plane from the data for visualisation

//     // // std::cout << "Check Prev " << *std::min_element(U.begin(), U.end()) csp  *std::max_element(U.begin(), U.end()) << std::endl;
//     int NPlane = NBY*BY*NBZ*BZ;
//     ocl_ExtractPlaneX->Execute(eu_dddt, travx, travy, travz, XP);
//     Copy_Buffer(U.data(), travx, sizeof(Real)*NPlane, cudaMemcpyDeviceToHost);
//     Copy_Buffer(V.data(), travy, sizeof(Real)*NPlane, cudaMemcpyDeviceToHost);
//     Copy_Buffer(W.data(), travz, sizeof(Real)*NPlane, cudaMemcpyDeviceToHost);
// }

//--- Destructor

VPM3D_ocl::~VPM3D_ocl()
{
    // Deallocate data

    //--- Grid Arrays
    clReleaseMemObject(lg_d   );
    clReleaseMemObject(lg_o   );
    clReleaseMemObject(lg_dddt);
    clReleaseMemObject(lg_dodt);
    // cudaFree(eu_d   );
    clReleaseMemObject(eu_o   );
    clReleaseMemObject(eu_dddt);
    clReleaseMemObject(eu_dodt);
    // clReleaseMemObject(dumbuffer);

    clReleaseMemObject(diagnostic_reduced);   // Reduced diagnostics arrays
    // clReleaseMemObject(vis_plane);   // Reduced diagnostics arrays
    clReleaseMemObject(magfilt_count);         // Count of particle which have non-zero strength after magnitude filtering

    // Indices for halo data
    clReleaseMemObject(Halo1data);
    clReleaseMemObject(Halo2data);
    clReleaseMemObject(Halo3data);
    clReleaseMemObject(Halo4data);
    // clReleaseMemObject(AuxGridData);

    //--- Timestepping (temporary) arrays
    clReleaseMemObject(int_lg_d);
    clReleaseMemObject(int_lg_o);
    clReleaseMemObject(k2_d);
    clReleaseMemObject(k2_o);
    clReleaseMemObject(k3_d);
    clReleaseMemObject(k3_o);
    clReleaseMemObject(k4_d);
    clReleaseMemObject(k4_o);
    clReleaseMemObject(tm1_d);
    clReleaseMemObject(tm1_o);
    clReleaseMemObject(tm1_dddt);
    clReleaseMemObject(tm1_dodt);

    //--- Turbulence arrays
    clReleaseMemObject(Laplacian);
}

}

#endif
