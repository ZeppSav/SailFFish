#include "VPM3D_cuda.h"
#include "VPM3D_kernels_cuda.h"

#ifdef CUFFT

namespace SailFFish
{

SFStatus VPM3D_cuda::Setup_VPM(VPM_Input *I)
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

    // Set block size for Kernel definition
    blockarch_grid = dim3(NBX,NBY,NBZ);
    blockarch_block = dim3(BX,BY,BZ);

    //--- Carry out sanity check here
    if (Hx!=H_Grid || Hy!=H_Grid || Hy!=H_Grid)
    {
        std::cout << "VPM3D_cuda::Setup_VPM: Mismatch between Input grid size H_Grid and calculated H_Grid. H_Grid =" csp H_Grid csp "Hx =" csp Hx csp "Hy =" csp Hy csp "Hz =" csp Hz << std::endl;
        // return GridError;
    }

    //--- Initialize kernels
    Stat = Initialize_Kernels();
    if (Stat != NoError)    return Stat;

    //--- Allocate and initialize data
    // if (Auxiliary)  Stat = Allocate_Auxiliary_Data();
    // else            Stat = Allocate_Data();
    Stat = Allocate_Data();
    if (Stat != NoError)    return Stat;
    Initialize_Data();                          // Initialize halo data for FD calcs
    Initialize_Halo_Data();
    Set_Grid_Positions();                       // Specify positions of grid (for diagnostics and/or initialisation)

    // Prepare outputs
    create_directory(std::filesystem::path("Output"));  // Generate output folder if not existing
    Generate_Summary("Summary.dat");
    Sim_begin = std::chrono::steady_clock::now();    // Begin clock

    // Carry out memory checks
    size_t free_mem = 0;
    size_t total_mem = 0;
    CUresult result = cuMemGetInfo(&free_mem, &total_mem);
    std::cout << "Total GPU memory: " << total_mem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Free GPU memory: " << free_mem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Used GPU memory: " << (total_mem - free_mem) / (1024.0 * 1024.0) << " MB" << std::endl;

    return Stat;
}

SFStatus VPM3D_cuda::Allocate_Data()
{
    // Memory for the execution of the solver on the gpu is allocated

    try{

        cudaMalloc((void**)&lg_d, 3*NNT*sizeof(Real));
        cudaMalloc((void**)&lg_o, 3*NNT*sizeof(Real));
        cudaMalloc((void**)&lg_dddt, 3*NNT*sizeof(Real));
        cudaMalloc((void**)&lg_dodt, 3*NNT*sizeof(Real));
        // cudaMalloc((void**)&eu_d, 3*NNT*sizeof(Real));          // Not required... remove this!
        cudaMalloc((void**)&eu_o, 3*NNT*sizeof(Real));

        // Standard memory allocation
        // cudaMalloc((void**)&eu_dddt, 3*NNT*sizeof(Real));
        // cudaMalloc((void**)&eu_dodt, 3*NNT*sizeof(Real));
        // cudaMalloc((void**)&sgs, NNT*sizeof(Real));          // This variable is defined here, even if we are not employing a turbulence model
        // cudaMalloc((void**)&dumbuffer, 3*NNT*sizeof(Real));

        // Exploiting the "empty" space in the cuda_r_Input/Output arrays
        // The data storage required for the cuFFT input arrays occupies approximately half of the input/output array(2*NT = 4*NNT)
        // When in-place data arrays are used, then cuda_r_Inputx = cuda_r_Outputx
        eu_dddt = cuda_r_Output1 + 4*NNT;
        eu_dodt = cuda_r_Output2 + 4*NNT;
        dumbuffer = cuda_r_Output3 + 4*NNT;
        sgs = cuda_r_Output1 + 7*NNT;
        qcrit = cuda_r_Output2 + 7*NNT;

        cudaMalloc((void**)&diagnostic_reduced, NDiags*NBT*sizeof(Real));
        cudaMalloc((void**)&magfilt_count, NBT*sizeof(int));
        cudaMalloc((void**)&vis_plane, NBX*BX*NBZ*BZ*sizeof(Real));
        cudaMalloc((void**)&travx, NBY*BY*NBZ*BZ*sizeof(Real));
        cudaMalloc((void**)&travy, NBY*BY*NBZ*BZ*sizeof(Real));
        cudaMalloc((void**)&travz, NBY*BY*NBZ*BZ*sizeof(Real));

        if (Integrator == EF){
            cudaMalloc((void**)&int_lg_d, 3*NNT*sizeof(Real));     // Spare buffer for generating .vtk files
            cudaMalloc((void**)&int_lg_o, 3*NNT*sizeof(Real));
        }

        if (Integrator == EM){
            cudaMalloc((void**)&int_lg_d, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&int_lg_o, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&k2_d, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&k2_o, 3*NNT*sizeof(Real));
        }

        if (Integrator == RK2){
            cudaMalloc((void**)&int_lg_d, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&int_lg_o, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&k2_d, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&k2_o, 3*NNT*sizeof(Real));
        }

        if (Integrator == AB2LF){
            cudaMalloc((void**)&int_lg_d, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&int_lg_o, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&tm1_d, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&tm1_o, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&tm1_dddt, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&tm1_dodt, 3*NNT*sizeof(Real));
        }

        if (Integrator == RK3){
            cudaMalloc((void**)&int_lg_d, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&int_lg_o, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&k2_d, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&k2_o, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&k3_d, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&k3_o, 3*NNT*sizeof(Real));
        }

        if (Integrator == LSRK3){
            // Specify dummy vectors
            cudaMalloc((void**)&int_lg_d, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&int_lg_o, 3*NNT*sizeof(Real));
        }

        if (Integrator == RK4){
            cudaMalloc((void**)&int_lg_d, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&int_lg_o, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&k2_d, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&k2_o, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&k3_d, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&k3_o, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&k4_d, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&k4_o, 3*NNT*sizeof(Real));
        }

        if (Integrator == LSRK4){
            // Specify dummy vectors
            cudaMalloc((void**)&int_lg_d, 3*NNT*sizeof(Real));
            cudaMalloc((void**)&int_lg_o, 3*NNT*sizeof(Real));
        }

        if (Turb==HYP){
            cudaMalloc((void**)&Laplacian, 3*NNT*sizeof(Real));
        }

        if (Turb==RVM1 || Turb==RVM2){

            // cudaMalloc((void**)&gfilt_Array1, 3*NNT*sizeof(Real));
            // cudaMalloc((void**)&gfilt_Array2, 3*NNT*sizeof(Real));

            // Exploiting the "empty" space in the cuda_r_Input/Output arrays
            gfilt_Array1 = cuda_r_Output1;
            gfilt_Array2 = cuda_r_Output2;
        }
    }
    catch (std::bad_alloc& ex){
        std::cout << "VPM3D_cuda::Allocate_Data(): Insufficient memory for allocation of solver arrays." << std::endl;
        return MemError;
    }

    std::cout << "VPM3D_cuda::Allocate_Data: Memory allocated." << std::endl;

    return NoError;
}

void VPM3D_cuda::Initialize_Data()
{
    // Memory for the execution of the solver on the gpu is allocated

    if (lg_d   )        cudaMemset(lg_d,      Real(0.0), 3*NNT*sizeof(Real));
    if (lg_o   )        cudaMemset(lg_o,      Real(0.0), 3*NNT*sizeof(Real));
    if (lg_dddt)        cudaMemset(lg_dddt,   Real(0.0), 3*NNT*sizeof(Real));
    if (lg_dodt)        cudaMemset(lg_dodt,   Real(0.0), 3*NNT*sizeof(Real));
    if (eu_o   )        cudaMemset(eu_o,      Real(0.0), 3*NNT*sizeof(Real));
    if (eu_dddt)        cudaMemset(eu_dddt,   Real(0.0), 3*NNT*sizeof(Real));
    if (eu_dodt)        cudaMemset(eu_dodt,   Real(0.0), 3*NNT*sizeof(Real));

    if (cuda_r_Input1)  cudaMemset(cuda_r_Input1,   Real(0.0), NT*sizeof(Real));
    if (cuda_r_Input2)  cudaMemset(cuda_r_Input2,   Real(0.0), NT*sizeof(Real));
    if (cuda_r_Input3)  cudaMemset(cuda_r_Input3,   Real(0.0), NT*sizeof(Real));
    if (cuda_r_Output1) cudaMemset(cuda_r_Output1,  Real(0.0), NT*sizeof(Real));
    if (cuda_r_Output2) cudaMemset(cuda_r_Output2,  Real(0.0), NT*sizeof(Real));
    if (cuda_r_Output3) cudaMemset(cuda_r_Output3,  Real(0.0), NT*sizeof(Real));

    // ConstructTensorGrid(Laplacian,NNT,3);        // Laplacian
    // ConstructTensorGrid(GradU,NNT,9);            // Velocity gradient
    // ConstructTensorGrid(MapFactors,NNT,15);      // Mapping factors for updating field

    if (Integrator == EF){
        cudaMemset(int_lg_d, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(int_lg_o, Real(0.0), 3*NNT*sizeof(Real));
    }

    if (Integrator == EM){
        cudaMemset(int_lg_d, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(int_lg_o, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(k2_d, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(k2_o, Real(0.0), 3*NNT*sizeof(Real));
    }

    if (Integrator == RK2){
        cudaMemset(int_lg_d, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(int_lg_o, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(k2_d, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(k2_o, Real(0.0), 3*NNT*sizeof(Real));
    }

    if (Integrator == AB2LF){
        cudaMemset(int_lg_d, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(int_lg_o, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(tm1_d, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(tm1_o, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(tm1_dddt, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(tm1_dodt, Real(0.0), 3*NNT*sizeof(Real));
    }

    if (Integrator == RK3){
        cudaMemset(int_lg_d, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(int_lg_o, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(k2_d, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(k2_o, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(k3_d, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(k3_o, Real(0.0), 3*NNT*sizeof(Real));
    }

    if (Integrator == RK4){
        cudaMemset(int_lg_d, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(int_lg_o, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(k2_d, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(k2_o, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(k3_d, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(k3_o, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(k4_d, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(k4_o, Real(0.0), 3*NNT*sizeof(Real));
    }

    if (Turb==RVM1 || Turb==RVM2){
        // SGS = RVector(NNT,Real(0.0));                      // Subgrid scale
        // ConstructTensorGrid(gfilt_Array1,NNT,3);     // Small scale vorticity
        // ConstructTensorGrid(gfilt_Array2,NNT,3);     // Small scale vorticity
    }
}

void VPM3D_cuda::Retrieve_Grid_Positions(RVector &xc, RVector &yc, RVector &zc)
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

    std::cout << "VPM3D_cuda::Retrieve_Grid_Positions: Grid positions retrieved." << std::endl;
}

void VPM3D_cuda::Set_Input_Arrays(RVector &xo, RVector &yo, RVector &zo)
{
    // This input array is set externally. This is imported here, flattened and passed to the cuda buffer.

    RVector input;
    StdAppend(input,xo);
    StdAppend(input,yo);
    StdAppend(input,zo);

    // Transfer to cuda buffer
    if (Architecture==BLOCK){
        cudaMemcpy(eu_o, input.data(), 3*NNT*sizeof(Real), cudaMemcpyHostToDevice);
        cuda_monolith_to_block_arch->Execute(eu_o,lg_o);
        cudaMemset(eu_o,      Real(0.0), 3*NNT*sizeof(Real));
    }
}

//------------- Debugging ----------------

void VPM3D_cuda::Output_Max_Components(const Real *A, int N)
{
    // This is a helper function to checks solver inputs and outputs
    RVector AR(3*N);
    cudaMemcpy(AR.data(), A, 3*N*sizeof(Real), cudaMemcpyDeviceToHost);
    std::cout << "Min component 1 " << *std::min_element(AR.begin(),        AR.begin()+N) csp std::endl;
    std::cout << "Min component 2 " << *std::min_element(AR.begin()+N,      AR.begin()+2*N) csp std::endl;
    std::cout << "Min component 3 " << *std::min_element(AR.begin()+2*N,    AR.begin()+3*N) csp std::endl;
    std::cout << "Max component 1 " << *std::max_element(AR.begin(),        AR.begin()+N) csp std::endl;
    std::cout << "Max component 2 " << *std::max_element(AR.begin()+N,      AR.begin()+2*N) csp std::endl;
    std::cout << "Max component 3 " << *std::max_element(AR.begin()+2*N,    AR.begin()+3*N) csp std::endl;
}

void VPM3D_cuda::Output_Max_Components(const Real *A, const Real *B, const Real *C, int N)
{
    // This is a helper function to checks solver inputs and outputs
    RVector AR(N), BR(N), CR(N);
    cudaMemcpy(AR.data(), A, N*sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(BR.data(), B, N*sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(CR.data(), C, N*sizeof(Real), cudaMemcpyDeviceToHost);
    std::cout << "Min component 1 " << *std::min_element(AR.begin(),    AR.end()) csp std::endl;
    std::cout << "Min component 2 " << *std::min_element(BR.begin(),    BR.end()) csp std::endl;
    std::cout << "Min component 3 " << *std::min_element(CR.begin(),    CR.end()) csp std::endl;
    std::cout << "Max component 1 " << *std::max_element(AR.begin(),    AR.end()) csp std::endl;
    std::cout << "Max component 2 " << *std::max_element(BR.begin(),    BR.end()) csp std::endl;
    std::cout << "Max component 3 " << *std::max_element(CR.begin(),    CR.end()) csp std::endl;
}

//-------------------------------------------
//------------- Kernel setup ----------------
//-------------------------------------------

//--- Set up halo data

void VPM3D_cuda::Initialize_Halo_Data()
{
    // When data is loaded into shared memory during  Kernel execution for either mapping routines or FD routines, a halo around the block must also be loaded.
    // These parameters are prepared once as they will be accessed numeous time during execution.

    // Note: I tried loading this as function for the array, but something went wrong

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
    cudaMalloc((void**)&Halo1data, 3*NHA*sizeof(int));
    cudaMemcpy(Halo1data, hs.data(), 3*NHA*sizeof(int), cudaMemcpyHostToDevice);
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
    cudaMalloc((void**)&Halo2data, 3*NHA*sizeof(int));
    cudaMemcpy(Halo2data, hs.data(), 3*NHA*sizeof(int), cudaMemcpyHostToDevice);
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
    cudaMalloc((void**)&Halo3data, 3*NHA*sizeof(int));
    cudaMemcpy(Halo3data, hs.data(), 3*NHA*sizeof(int), cudaMemcpyHostToDevice);
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
    cudaMalloc((void**)&Halo4data, 3*NHA*sizeof(int));
    cudaMemcpy(Halo4data, hs.data(), 3*NHA*sizeof(int), cudaMemcpyHostToDevice);
    std::cout << "Halo 4 array padded to: " << int(xhs.size()) csp int(yhs.size()) csp int(zhs.size()) csp "Total Halo buffer size: " << hs.size()
              << ". Each thread will be required to pull halo data N Times: " << NHIT <<  std::endl;
}

//--- Prepare kernels

void VPM3D_cuda::Set_Kernel_Constants(jitify::KernelInstantiation *KI, int Halo)
{
    // Kernel constants are set here.
    // Set constant grid parameters of kernel
    cuMemcpyHtoD(KI->get_constant_ptr("NX"), &NNX, sizeof(int));
    cuMemcpyHtoD(KI->get_constant_ptr("NY"), &NNY, sizeof(int));
    cuMemcpyHtoD(KI->get_constant_ptr("NZ"), &NNZ, sizeof(int));
    cuMemcpyHtoD(KI->get_constant_ptr("NT"), &NNT, sizeof(int));

    cuMemcpyHtoD(KI->get_constant_ptr("BX"), &BX, sizeof(int));
    cuMemcpyHtoD(KI->get_constant_ptr("BY"), &BY, sizeof(int));
    cuMemcpyHtoD(KI->get_constant_ptr("BZ"), &BZ, sizeof(int));
    cuMemcpyHtoD(KI->get_constant_ptr("BT"), &BT, sizeof(int));

    cuMemcpyHtoD(KI->get_constant_ptr("NBX"), &NBX, sizeof(int));
    cuMemcpyHtoD(KI->get_constant_ptr("NBY"), &NBY, sizeof(int));
    cuMemcpyHtoD(KI->get_constant_ptr("NBZ"), &NBZ, sizeof(int));

    // Set constant type-dependent parameters
    cuMemcpyHtoD(KI->get_constant_ptr("hx"), &H_Grid, sizeof(Real));
    cuMemcpyHtoD(KI->get_constant_ptr("hy"), &H_Grid, sizeof(Real));
    cuMemcpyHtoD(KI->get_constant_ptr("hz"), &H_Grid, sizeof(Real));

    // Set constant environmental parameters
    cuMemcpyHtoD(KI->get_constant_ptr("KinVisc"), &KinVisc, sizeof(Real));

    // Set halo parameters if required
    if (Halo>0){
        int NFX = BX+2*Halo;
        int NFY = BY+2*Halo;
        int NFZ = BZ+2*Halo;

        cuMemcpyHtoD(KI->get_constant_ptr("NFDX"), &NFX, sizeof(int));
        cuMemcpyHtoD(KI->get_constant_ptr("NFDY"), &NFY, sizeof(int));
        cuMemcpyHtoD(KI->get_constant_ptr("NFDZ"), &NFZ, sizeof(int));
        // cuMemcpyHtoD(KI->get_constant_ptr("Halo"), &Halo, sizeof(int));
        int NHTOT = NFX*NFY*NFZ-BT;
        int NHIT = std::ceil(Real(NHTOT)/Real(BT));
        // std::cout << "NHTOT = " << NHTOT << ", NHIT = " << NHIT << std::endl;
        cuMemcpyHtoD(KI->get_constant_ptr("NHIT"), &NHIT, sizeof(int));
    }
}

SFStatus VPM3D_cuda::Initialize_Kernels()
{
    // This function compiles the cuda kernels on the device.

    std::string Kernel;

    try{

        // Generate full source code
        std::string Source = "my_program\n";
        if (std::is_same<Real,float>::value)    Source.append(VPM3D_cuda_kernels_float);
        if (std::is_same<Real,double>::value)   Source.append(VPM3D_cuda_kernels_double);
        Source.append(VPM3D_cuda_kernels_source);

        // Compile kernels
        using jitify::reflection::type_of;
        CUDAComplex ComplexType = CUDAComplex(0,0);
        cuda_VPM_convolution = new cudaKernel(Source,"vpm_convolution",KID, type_of(ComplexType));
        cuda_VPM_reprojection = new cudaKernel(Source,"vpm_reprojection",KID, type_of(ComplexType));
        cuda_monolith_to_block_arch = new cudaKernel(Source,"Monolith_to_Block",KID);
        cuda_block_to_monolith_arch  = new cudaKernel(Source,"Block_to_Monolith",KID);
        cuda_block_to_monolith_single  = new cudaKernel(Source,"Block_to_Monolith_Single",KID);
        cuda_map_toUnbounded = new cudaKernel(Source,"Map_toUnbounded",KID);
        cuda_map_fromUnbounded = new cudaKernel(Source,"Map_fromUnbounded",KID);
        cuda_mapM2 = new cudaKernel(Source,"MapKernel",KID, 1,2,(BX+2)*(BY+2)*(BZ+2));
        cuda_mapM4 = new cudaKernel(Source,"MapKernel",KID, 2,4,(BX+4)*(BY+4)*(BZ+4));
        cuda_mapM4D = new cudaKernel(Source,"MapKernel",KID, 2,42,(BX+4)*(BY+4)*(BZ+4));
        cuda_mapM6D = new cudaKernel(Source,"MapKernel",KID,3,6, (BX+6)*(BY+6)*(BZ+6));
        cuda_interpM2 = new cudaKernel(Source,"InterpKernel",KID,1,2,(BX+2)*(BY+2)*(BZ+2));
        cuda_interpM4 = new cudaKernel(Source,"InterpKernel",KID,2,4,(BX+4)*(BY+4)*(BZ+4));
        cuda_interpM4D = new cudaKernel(Source,"InterpKernel",KID,2,42,(BX+4)*(BY+4)*(BZ+4));
        cuda_interpM6D = new cudaKernel(Source,"InterpKernel",KID,3,6,(BX+6)*(BY+6)*(BZ+6));
        cuda_update = new cudaKernel(Source,"update",KID);
        cuda_updateRK = new cudaKernel(Source,"updateRK",KID);
        cuda_updateRK2 = new cudaKernel(Source,"updateRK2",KID);
        cuda_updateRK3 = new cudaKernel(Source,"updateRK3",KID);
        cuda_updateRK4 = new cudaKernel(Source,"updateRK4",KID);
        cuda_updateRKLS = new cudaKernel(Source,"updateRK_LS",KID);
        cuda_stretch_FD2 = new cudaKernel(Source,"Shear_Stress",KID, 1 ,2,(BX+2)*(BY+2)*(BZ+2));
        cuda_stretch_FD4 = new cudaKernel(Source,"Shear_Stress",KID, 2 ,4,(BX+4)*(BY+4)*(BZ+4));
        cuda_stretch_FD6 = new cudaKernel(Source,"Shear_Stress",KID, 3 ,6,(BX+6)*(BY+6)*(BZ+6));
        cuda_stretch_FD8 = new cudaKernel(Source,"Shear_Stress",KID, 4 ,8,(BX+8)*(BY+8)*(BZ+8));
        cuda_Diagnostics = new cudaKernel(Source,"DiagnosticsKernel",KID,BT);
        cuda_MagFilt1 = new cudaKernel(Source,"MagnitudeFiltering_Step1",KID,BT);
        cuda_MagFilt2 = new cudaKernel(Source,"MagnitudeFiltering_Step2",KID,BT);
        cuda_MagFilt3 = new cudaKernel(Source,"MagnitudeFiltering_Step3",KID);
        cuda_freestream = new cudaKernel(Source, "AddFreestream",KID);
        cuda_Airywave = new cudaKernel(Source, "AddAiryWave", KID);      // Addition for Airy wave component

        // External source kernels
        Map_Ext = new cudaKernel(Source,"Map_Ext_Bounded",KID);
        Map_Ext_Unbounded = new cudaKernel(Source,"Map_Ext_Unbounded",KID);

        cuda_ExtractPlaneX = new cudaKernel(Source,"ExtractPlaneX",KID);
        cuda_ExtractPlaneY = new cudaKernel(Source,"ExtractPlaneY",KID);

        cuda_interpM2_block = new cudaKernel(Source,"Interp_Block",KID,1,2,(BX+2)*(BY+2)*(BZ+2));
        cuda_interpM4_block = new cudaKernel(Source,"Interp_Block",KID,2,4,(BX+4)*(BY+4)*(BZ+4));
        cuda_interpM4D_block = new cudaKernel(Source,"Interp_Block",KID,2,42,(BX+4)*(BY+4)*(BZ+4));
        cuda_interpM6D_block = new cudaKernel(Source,"Interp_Block",KID,3,6,(BX+6)*(BY+6)*(BZ+6));

        cuda_interpM2_block2 = new cudaKernel(Source,"Interp_Block2",KID,1,2,(BX+2)*(BY+2)*(BZ+2));
        cuda_interpM4_block2 = new cudaKernel(Source,"Interp_Block2",KID,2,4,(BX+4)*(BY+4)*(BZ+4));
        cuda_interpM4D_block2 = new cudaKernel(Source,"Interp_Block2",KID,2,42,(BX+4)*(BY+4)*(BZ+4));
        cuda_interpM6D_block2 = new cudaKernel(Source,"Interp_Block2",KID,3,6,(BX+6)*(BY+6)*(BZ+6));

        // Turbulence kernels
        cuda_Laplacian_FD2 = new cudaKernel(Source,"Laplacian_Operator",KID,1,2,(BX+2)*(BY+2)*(BZ+2));
        cuda_Laplacian_FD4 = new cudaKernel(Source,"Laplacian_Operator",KID,2,4,(BX+4)*(BY+4)*(BZ+4));
        cuda_Laplacian_FD6 = new cudaKernel(Source,"Laplacian_Operator",KID,3,6,(BX+6)*(BY+6)*(BZ+6));
        cuda_Laplacian_FD8 = new cudaKernel(Source,"Laplacian_Operator",KID,4,8,(BX+8)*(BY+8)*(BZ+8));
        cuda_Turb_Hyp_FD2 = new cudaKernel(Source,"Hyperviscosity_Operator",KID,1,2,(BX+2)*(BY+2)*(BZ+2));
        cuda_Turb_Hyp_FD4 = new cudaKernel(Source,"Hyperviscosity_Operator",KID,2,4,(BX+4)*(BY+4)*(BZ+4));
        cuda_Turb_Hyp_FD6 = new cudaKernel(Source,"Hyperviscosity_Operator",KID,3,6,(BX+6)*(BY+6)*(BZ+6));
        cuda_Turb_Hyp_FD8 = new cudaKernel(Source,"Hyperviscosity_Operator",KID,4,8,(BX+8)*(BY+8)*(BZ+8));
        cuda_sg_discfilt = new cudaKernel(Source,"SubGrid_DiscFilter",KID,1,(BX+2)*(BY+2)*(BZ+2));
        // cuda_sg_discfilt2 = new cudaKernel(Source,"SubGrid_DiscFilter2",KID,1,(BX+2)*(BY+2)*(BZ+2));
        cuda_Turb_RVM_FD2 = new cudaKernel(Source,"RVM_turbulentstress",KID,1,2,(BX+2)*(BY+2)*(BZ+2));
        cuda_Turb_RVM_FD4 = new cudaKernel(Source,"RVM_turbulentstress",KID,2,4,(BX+4)*(BY+4)*(BZ+4));
        cuda_Turb_RVM_FD6 = new cudaKernel(Source,"RVM_turbulentstress",KID,3,6,(BX+6)*(BY+6)*(BZ+6));
        cuda_Turb_RVM_FD8 = new cudaKernel(Source,"RVM_turbulentstress",KID,4,8,(BX+8)*(BY+8)*(BZ+8));
        cuda_Turb_RVM_DGC_FD2 = new cudaKernel(Source,"RVM_DGC_turbulentstress",KID,1,2,(BX+2)*(BY+2)*(BZ+2));
        cuda_Turb_RVM_DGC_FD4 = new cudaKernel(Source,"RVM_DGC_turbulentstress",KID,2,4,(BX+4)*(BY+4)*(BZ+4));
        cuda_Turb_RVM_DGC_FD6 = new cudaKernel(Source,"RVM_DGC_turbulentstress",KID,3,6,(BX+6)*(BY+6)*(BZ+6));
        cuda_Turb_RVM_DGC_FD8 = new cudaKernel(Source,"RVM_DGC_turbulentstress",KID,4,8,(BX+8)*(BY+8)*(BZ+8));

        //--- Specify grid constants
        Set_Kernel_Constants(cuda_monolith_to_block_arch->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_block_to_monolith_arch->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_block_to_monolith_single->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_map_toUnbounded->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_map_fromUnbounded->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_mapM2->Get_Instance(), 1);
        Set_Kernel_Constants(cuda_mapM4->Get_Instance(), 2);
        Set_Kernel_Constants(cuda_mapM4D->Get_Instance(), 2);
        Set_Kernel_Constants(cuda_mapM6D->Get_Instance(), 3);
        Set_Kernel_Constants(cuda_interpM2->Get_Instance(), 1);
        Set_Kernel_Constants(cuda_interpM4->Get_Instance(), 2);
        Set_Kernel_Constants(cuda_interpM4D->Get_Instance(), 2);
        Set_Kernel_Constants(cuda_interpM6D->Get_Instance(), 3);
        Set_Kernel_Constants(cuda_update->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_updateRK->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_updateRK2->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_updateRK3->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_updateRK4->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_updateRKLS->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_stretch_FD2->Get_Instance(),1);
        Set_Kernel_Constants(cuda_stretch_FD4->Get_Instance(),2);
        Set_Kernel_Constants(cuda_stretch_FD6->Get_Instance(),3);
        Set_Kernel_Constants(cuda_stretch_FD8->Get_Instance(),4);
        Set_Kernel_Constants(cuda_Diagnostics->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_MagFilt1->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_MagFilt2->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_MagFilt3->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_freestream->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_Airywave->Get_Instance(), 0);         // Addition for Airy wave component

        Set_Kernel_Constants(cuda_interpM2_block->Get_Instance(), 1);
        Set_Kernel_Constants(cuda_interpM4_block->Get_Instance(), 2);
        Set_Kernel_Constants(cuda_interpM4D_block->Get_Instance(), 2);
        Set_Kernel_Constants(cuda_interpM6D_block->Get_Instance(), 3);

        Set_Kernel_Constants(cuda_interpM2_block2->Get_Instance(), 1);
        Set_Kernel_Constants(cuda_interpM4_block2->Get_Instance(), 2);
        Set_Kernel_Constants(cuda_interpM4D_block2->Get_Instance(), 2);
        Set_Kernel_Constants(cuda_interpM6D_block2->Get_Instance(), 3);

        // External source operations
        Set_Kernel_Constants(Map_Ext_Unbounded->Get_Instance(), 0);
        Set_Kernel_Constants(Map_Ext->Get_Instance(), 0);

        Set_Kernel_Constants(cuda_ExtractPlaneX->Get_Instance(), 0);
        Set_Kernel_Constants(cuda_ExtractPlaneY->Get_Instance(), 0);

        // Turbulence kernels
        Set_Kernel_Constants(cuda_Laplacian_FD2->Get_Instance(),1);
        Set_Kernel_Constants(cuda_Laplacian_FD4->Get_Instance(),2);
        Set_Kernel_Constants(cuda_Laplacian_FD6->Get_Instance(),3);
        Set_Kernel_Constants(cuda_Laplacian_FD8->Get_Instance(),4);
        Set_Kernel_Constants(cuda_Turb_Hyp_FD2->Get_Instance(),1);
        Set_Kernel_Constants(cuda_Turb_Hyp_FD4->Get_Instance(),2);
        Set_Kernel_Constants(cuda_Turb_Hyp_FD6->Get_Instance(),3);
        Set_Kernel_Constants(cuda_Turb_Hyp_FD8->Get_Instance(),4);
        Set_Kernel_Constants(cuda_sg_discfilt->Get_Instance(), 1);
        // Set_Kernel_Constants(cuda_sg_discfilt2->Get_Instance(), 1);
        Set_Kernel_Constants(cuda_Turb_RVM_FD2->Get_Instance(),1);
        Set_Kernel_Constants(cuda_Turb_RVM_FD4->Get_Instance(),2);
        Set_Kernel_Constants(cuda_Turb_RVM_FD6->Get_Instance(),3);
        Set_Kernel_Constants(cuda_Turb_RVM_FD8->Get_Instance(),4);
        Set_Kernel_Constants(cuda_Turb_RVM_DGC_FD2->Get_Instance(),1);
        Set_Kernel_Constants(cuda_Turb_RVM_DGC_FD4->Get_Instance(),2);
        Set_Kernel_Constants(cuda_Turb_RVM_DGC_FD6->Get_Instance(),3);
        Set_Kernel_Constants(cuda_Turb_RVM_DGC_FD8->Get_Instance(),4);

        if (Architecture==BLOCK){

            //--- Configure grid
            cuda_monolith_to_block_arch->Instantiate(blockarch_grid, blockarch_block);
            cuda_block_to_monolith_arch->Instantiate(blockarch_grid, blockarch_block);
            cuda_block_to_monolith_single->Instantiate(blockarch_grid, blockarch_block);
            cuda_mapM2->Instantiate(blockarch_grid, blockarch_block);
            cuda_mapM4->Instantiate(blockarch_grid, blockarch_block);
            cuda_mapM4D->Instantiate(blockarch_grid, blockarch_block);
            cuda_mapM6D->Instantiate(blockarch_grid, blockarch_block);
            cuda_interpM2->Instantiate(blockarch_grid, blockarch_block);
            cuda_interpM4->Instantiate(blockarch_grid, blockarch_block);
            cuda_interpM4D->Instantiate(blockarch_grid, blockarch_block);
            cuda_interpM6D->Instantiate(blockarch_grid, blockarch_block);

            cuda_map_toUnbounded->Instantiate(blockarch_grid, blockarch_block);
            cuda_map_fromUnbounded->Instantiate(blockarch_grid, blockarch_block);

            cuda_stretch_FD2->Instantiate(blockarch_grid, blockarch_block);
            cuda_stretch_FD4->Instantiate(blockarch_grid, blockarch_block);
            cuda_stretch_FD6->Instantiate(blockarch_grid, blockarch_block);
            cuda_stretch_FD8->Instantiate(blockarch_grid, blockarch_block);

            // Turbulence kernels
            cuda_Laplacian_FD2->Instantiate(blockarch_grid, blockarch_block);
            cuda_Laplacian_FD4->Instantiate(blockarch_grid, blockarch_block);
            cuda_Laplacian_FD6->Instantiate(blockarch_grid, blockarch_block);
            cuda_Laplacian_FD8->Instantiate(blockarch_grid, blockarch_block);
            cuda_Turb_Hyp_FD2->Instantiate(blockarch_grid, blockarch_block);
            cuda_Turb_Hyp_FD4->Instantiate(blockarch_grid, blockarch_block);
            cuda_Turb_Hyp_FD6->Instantiate(blockarch_grid, blockarch_block);
            cuda_Turb_Hyp_FD8->Instantiate(blockarch_grid, blockarch_block);
            cuda_sg_discfilt->Instantiate(blockarch_grid, blockarch_block);
            // cuda_sg_discfilt2->Instantiate(blockarch_grid, blockarch_block);
            cuda_Turb_RVM_FD2->Instantiate(blockarch_grid, blockarch_block);
            cuda_Turb_RVM_FD4->Instantiate(blockarch_grid, blockarch_block);
            cuda_Turb_RVM_FD6->Instantiate(blockarch_grid, blockarch_block);
            cuda_Turb_RVM_FD8->Instantiate(blockarch_grid, blockarch_block);
            cuda_Turb_RVM_DGC_FD2->Instantiate(blockarch_grid, blockarch_block);
            cuda_Turb_RVM_DGC_FD4->Instantiate(blockarch_grid, blockarch_block);
            cuda_Turb_RVM_DGC_FD6->Instantiate(blockarch_grid, blockarch_block);
            cuda_Turb_RVM_DGC_FD8->Instantiate(blockarch_grid, blockarch_block);

            dim3 lingrid(NNT/BT), linblock(BT);
            cuda_update->Instantiate(lingrid, linblock);
            cuda_updateRK->Instantiate(lingrid, linblock);
            cuda_updateRK2->Instantiate(lingrid, linblock);
            cuda_updateRK3->Instantiate(lingrid, linblock);
            cuda_updateRK4->Instantiate(lingrid, linblock);
            cuda_updateRKLS->Instantiate(lingrid, linblock);
            cuda_Diagnostics->Instantiate(lingrid, linblock);
            cuda_freestream->Instantiate(lingrid, linblock);
            cuda_MagFilt1->Instantiate(lingrid, linblock);
            cuda_MagFilt2->Instantiate(lingrid, linblock);
            cuda_MagFilt3->Instantiate(lingrid, linblock);
            cuda_Airywave->Instantiate(lingrid, linblock);      // Additions for Airy wave component

            dim3 linplanex(NBY,NBZ), linblockplanex(BY,BZ);
            cuda_ExtractPlaneX->Instantiate(linplanex,linblockplanex);

            dim3 linplaney(NBX,NBZ), linblockplaney(BX,BZ);
            cuda_ExtractPlaneY->Instantiate(linplaney,linblockplaney);

            dim3 ConvGrid(NTM/BT), ConvBlockSF(BT);
            cuda_VPM_convolution->Instantiate(ConvGrid,ConvBlockSF);
            cuda_VPM_reprojection->Instantiate(ConvGrid,ConvBlockSF);
        }

        std::cout << "-----------Shared Memory Analysis-----------" << std::endl;
        // // // 1) How much dynamic memory is available?
        // CUfunction_attribute attr = CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
        // int sharedMemPerSM = cuda_stretch_FD8->Get_Instance()->get_func_attribute(attr);
        // std::cout << "Max shared memory cuda_stretch_FD8 : " << sharedMemPerSM << " bytes" << std::endl;
        // std::cout << "adapting this..." << std::endl;
        // cuda_stretch_FD8->Get_Instance()->set_func_attribute(attr,96*1024);                  // Crashing!!!!
        // int sharedMemPerSM2 = cuda_stretch_FD8->Get_Instance()->get_func_attribute(attr);
        // std::cout << "Max shared memory cuda_stretch_FD8 : " << sharedMemPerSM2 << " bytes" << std::endl;

        int ns;
        if (FDOrder==CD2) ns = 6*(BX+2)*(BY+2)*(BZ+2);
        if (FDOrder==CD4) ns = 6*(BX+4)*(BY+4)*(BZ+4);
        if (FDOrder==CD6) ns = 6*(BX+6)*(BY+6)*(BZ+6);
        if (FDOrder==CD8) ns = 6*(BX+8)*(BY+8)*(BZ+8);
        std::cout << "Size of shared memory required for stretching kernel: " << ns*sizeof(Real) << " bytes." << std::endl;

        int nm;
        if (SolverMap==M2)  nm = 6*(BX+2)*(BY+2)*(BZ+2);
        if (SolverMap==M4)  nm = 6*(BX+4)*(BY+4)*(BZ+4);
        if (SolverMap==M4D) nm = 6*(BX+4)*(BY+4)*(BZ+4);
        if (SolverMap==M6D) nm = 6*(BX+6)*(BY+6)*(BZ+6);
        std::cout << "Size of shared memory required for mapping kernel: " << nm*sizeof(Real) << " bytes." << std::endl;

        std::cout << "---------------------------------------------" << std::endl;

    }
    catch (std::bad_alloc& ex){
        std::cout << "VPM3D_cuda::Initialize_Kernels(): Problem with initializing kernels. Error code: " << ex.what() << std::endl;
        return SetupError;
    }

    std::cout << "VPM3D_cuda::Initialize_Kernels: Kernel initialization successful." << std::endl;
    return NoError;

}

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

//-------------------------------------------
//------------- Timestepping ----------------
//-------------------------------------------

void VPM3D_cuda::Advance_Particle_Set()
{
    // This is a stepping function which updates the particle field and carried out the desired
    // updates of the field.
    if (NStep%NRemesh==0 && NStep!=NInit)   Remeshing = true;
    else                                    Remeshing = false;
    if (Remeshing)                    Remesh_Particle_Set();          // Remesh
    if (Remeshing && MagFiltFac>0)    Magnitude_Filtering();          // Filter magnitude
    if (DivFilt && Remeshing && NStep%NReproject==0)   Reproject_Particle_Set_Spectral();   // Reproject vorticity field
    Update_Particle_Field();                                            // Update vorticity field                    // HERR !
    if (NExp>0 && NStep%NExp==0 && NStep>0 && NStep>=ExpTB) Generate_VTK();                 // Export grid if desired
    Increment_Time();

    // RVector dgdtcheck(NNT);     // Something is wrong!
    // cudaMemcpy(dgdtcheck.data(), eu_dddt, NTAux*sizeof(Real), cudaMemcpyDeviceToHost);
    // std::cout << "Coming out: Max eu_dddt 0 " << *std::max_element(dgdtcheck.begin(),          dgdtcheck.begin()+NTAux) csp std::endl;
    // std::cout << "Coming out: Max eu_dddt 1 " << *std::max_element(dgdtcheck.begin()+NTAux,    dgdtcheck.begin()+2*NTAux) csp std::endl;
    // std::cout << "Coming out: Max eu_dddt 2 " << *std::max_element(dgdtcheck.begin()+2*NTAux,  dgdtcheck.begin()+3*NTAux) csp std::endl;
}

void VPM3D_cuda::Update_Particle_Field()
{
    if (Integrator == EF)       // Eulerian forward
    {
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        // if (AuxGrid) Map_to_Auxiliary_Grid();
        cuda_update->Execute(lg_d, lg_o, lg_dddt, lg_dodt, Real(dT));
    }

    if (Integrator == EM)       // Explicit midpoint
    {
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        // if (AuxGrid) Map_to_Auxiliary_Grid();
        cudaMemcpy(int_lg_d, lg_d, 3*NNT*sizeof(Real), cudaMemcpyDeviceToDevice);
        cudaMemcpy(int_lg_o, lg_o, 3*NNT*sizeof(Real), cudaMemcpyDeviceToDevice);
        cuda_update->Execute(int_lg_d, int_lg_o, lg_dddt, lg_dodt, Real(0.5*dT));
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k2_d, k2_o);
        cuda_update->Execute(lg_d, lg_o, k2_d, k2_o, Real(dT));
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
        // if (AuxGrid) Map_to_Auxiliary_Grid();
        cuda_updateRK->Execute(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, Real(dT));
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k2_d, k2_o);
        cuda_updateRK2->Execute(lg_d, lg_o, lg_dddt, lg_dodt, k2_d, k2_o, Real(dT));
    }

    if (Integrator == RK3)      // Runge-Kutta third order
    {
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        // if (AuxGrid) Map_to_Auxiliary_Grid();
        cuda_updateRK->Execute(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, Real(0.5*dT));
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k2_d, k2_o);
        cuda_updateRK->Execute(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, Real(-dT));
        cuda_update->Execute(int_lg_d, int_lg_o, k2_d, k2_o, Real(2.0*dT));
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k3_d, k3_o);
        cuda_updateRK3->Execute(lg_d, lg_o, lg_dddt, lg_dodt, k2_d, k2_o, k3_d, k3_o, Real(dT));
    }

    if (Integrator == RK4)      // Runge-Kutta fourth order
    {
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        // if (AuxGrid) Map_to_Auxiliary_Grid();
        cuda_updateRK->Execute(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, Real(0.5*dT));
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k2_d, k2_o);
        cuda_updateRK->Execute(lg_d, lg_o, k2_d, k2_o, int_lg_d, int_lg_o, Real(0.5*dT));
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k3_d, k3_o);
        cuda_updateRK->Execute(lg_d, lg_o, k3_d, k3_o, int_lg_d, int_lg_o, Real(dT));
        Calc_Particle_RateofChange(int_lg_d, int_lg_o, k4_d, k4_o);
        cuda_updateRK4->Execute(lg_d, lg_o, lg_dddt, lg_dodt, k2_d, k2_o, k3_d, k3_o, k4_d, k4_o, Real(dT));
    }

    if (Integrator == LSRK3)    // Runge-Kutta third order low-storage (4-stage)
    {
        Real RK3A[4] =  {0.0, -756391.0/934407.0, -36441873.0/15625000.0, -1953125.0/1085297.0};
        Real RK3B[4] =  {8.0/141.0, 6627.0/2000.0, 609375.0/1085297.0, 198961.0/526283.0};
        // cudaMemset(int_lg_d,    0, 3*NNT*sizeof(Real));       // Clear intermediate arrays
        // cudaMemset(int_lg_o,    0, 3*NNT*sizeof(Real));       // Clear intermediate arrays

        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        // if (AuxGrid) Map_to_Auxiliary_Grid();        // Commenting this out causes code to not crash!!!!

        cuda_updateRKLS->Execute(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, Real(RK3A[0]), Real(RK3B[0]), Real(dT));
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        cuda_updateRKLS->Execute(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, Real(RK3A[1]), Real(RK3B[1]), Real(dT));
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        cuda_updateRKLS->Execute(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, Real(RK3A[2]), Real(RK3B[2]), Real(dT));
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        cuda_updateRKLS->Execute(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, Real(RK3A[3]), Real(RK3B[3]), Real(dT));
    }

    if (Integrator == LSRK4)    // Runge-Kutta fourth order low-storage (5-stage)
    {
        Real RK4A[5] =  {0.0,-567301805773.0/1357537059087.0,-2404267990393.0/2016746695238.0,-3550918686646.0/2091501179385.0,-1275806237668.0/842570457699.0};
        Real RK4B[5] =  {1432997174477.0/9575080441755.0,5161836677717.0/13612068292357.0,1720146321549.0/2090206949498.0,3134564353537.0/4481467310338.0,2277821191437.0/14882151754819.0};
        cudaMemset(int_lg_d,    0, 3*NNT*sizeof(Real));       // Clear intermediate arrays
        cudaMemset(int_lg_o,    0, 3*NNT*sizeof(Real));       // Clear intermediate arrays
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        Calc_Grid_Diagnostics();
        // if (AuxGrid) Map_to_Auxiliary_Grid();
        cuda_updateRKLS->Execute(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, Real(RK4A[0]), Real(RK4B[0]), Real(dT));
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        cuda_updateRKLS->Execute(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, Real(RK4A[1]), Real(RK4B[1]), Real(dT));
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        cuda_updateRKLS->Execute(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, Real(RK4A[2]), Real(RK4B[2]), Real(dT));
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        cuda_updateRKLS->Execute(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, Real(RK4A[3]), Real(RK4B[3]), Real(dT));
        Calc_Particle_RateofChange(lg_d, lg_o, lg_dddt, lg_dodt);
        cuda_updateRKLS->Execute(lg_d, lg_o, lg_dddt, lg_dodt, int_lg_d, int_lg_o, Real(RK4A[4]), Real(RK4B[4]), Real(dT));
    }
}

void VPM3D_cuda::Calc_Particle_RateofChange(const Real *pd, const Real *po, Real *dpddt, Real *dpodt)
{
    // This calculates for the given input field the rate of change arrays

    // Clear grid and lagrangian input derivative arrays
    cudaMemset(eu_o,    0, 3*NNT*sizeof(Real));       // Clear grid (vorticity) arrays
    cudaMemset(dpddt,   0, 3*NNT*sizeof(Real));       // Clear Lagrangian grid rate of change arrays
    cudaMemset(dpodt,   0, 3*NNT*sizeof(Real));       // Clear Lagrangian grid rate of change arrays
    cudaMemset(eu_dddt, 0, 3*NNT*sizeof(Real));       // Clear Eulerian grid rate of change arrays
    cudaMemset(eu_dodt, 0, 3*NNT*sizeof(Real));       // Clear Eulerian grid rate of change arrays

    // Map vorticity from Lagrangian grid to Eulerian grid
    switch (SolverMap)
    {
    case (M2):      {cuda_mapM2->Execute(po, pd, Halo1data, eu_o);     break;}
    case (M4):      {cuda_mapM4->Execute(po, pd, Halo2data, eu_o);     break;}
    case (M4D):     {cuda_mapM4D->Execute(po, pd, Halo2data, eu_o);    break;}
    case (M6D):     {cuda_mapM6D->Execute(po, pd, Halo3data, eu_o);    break;}
    // case (D2):      {Parallel_Kernel(NNT) {KER_MapFac3_D2(p_Vals, MapFactors, i, H_Grid);} break;}
    // case (D3):      {Parallel_Kernel(NNT) {KER_MapFac3_D3(p_Vals, MapFactors, i, H_Grid);} break;}
    default:        {std::cout << "VPM3D_cuda::Calc_Particle_RateofChange. Mapping undefined."; return;}
    }

    // Calc_Grid_SpectralRatesof_Change();         // The rates of change are calculated on the fixed grid
    Calc_Grid_FDRatesof_Change();                 // The rates of change are calculated on the fixed grid

    // Map grid values to particles
    switch (SolverMap)
    {
    case (M2):      {cuda_interpM2->Execute(eu_dddt, eu_dodt, pd, Halo1data, dpddt, dpodt);   break;}
    case (M4):      {cuda_interpM4->Execute(eu_dddt, eu_dodt, pd, Halo2data, dpddt, dpodt);   break;}
    case (M4D):     {cuda_interpM4D->Execute(eu_dddt, eu_dodt, pd, Halo2data, dpddt, dpodt);  break;}
    case (M6D):     {cuda_interpM6D->Execute(eu_dddt, eu_dodt, pd, Halo3data, dpddt, dpodt);  break;}
    // case (D2):      {Parallel_Kernel(NNT) {KER_MapFac3_D2(p_Vals, MapFactors, i, H_Grid);} break;}
    // case (D3):      {Parallel_Kernel(NNT) {KER_MapFac3_D3(p_Vals, MapFactors, i, H_Grid);} break;}
    default:        {std::cout << "VPM3D_cuda::Calc_Particle_RateofChange. Interpolation undefined."; return;}
    }
}

void VPM3D_cuda::Calc_Grid_FDRatesof_Change()
{
    // The rates of change on the grid (eu_dddt, eu_dodt) are calculated using SailFFish and finite differences

    // Calculate velocity on the grid- the method due to eastwood requires an expanded domain, so this must be mapped
    // NOTE: We are skipping mapping external sources for now... will include this later

    // Reset input arrays for FFT solver
    cudaMemset(cuda_r_Input1,   Real(0.0), NT*sizeof(Real));
    cudaMemset(cuda_r_Input2,   Real(0.0), NT*sizeof(Real));
    cudaMemset(cuda_r_Input3,   Real(0.0), NT*sizeof(Real));

    // Specify input arrays for FFT solver
    cuda_map_toUnbounded->Execute(eu_o, cuda_r_Input1, cuda_r_Input2, cuda_r_Input3);
    // if (AuxGrid) cuda_map_aux_toUnboundedVPM->Execute(AuxGrid->Get_Vort_Array(), cuda_r_Input1, cuda_r_Input2, cuda_r_Input3);
    Map_from_Auxiliary_Grid();
    Forward_Transform();
    cuda_VPM_convolution->Execute(c_FTInput1, c_FTInput2, c_FTInput3, c_FG, c_FGi, c_FGj, c_FGk, c_FTOutput1, c_FTOutput2, c_FTOutput3);
    Backward_Transform();
    cuda_map_fromUnbounded->Execute(cuda_r_Output1, cuda_r_Output2, cuda_r_Output3, eu_dddt);

    //--- Calculate shear stresses on grid
    Grid_Shear_Stresses();
    Grid_Turb_Shear_Stresses();

    //--- Add freestream velocity
    Add_Freestream_Velocity();
}

void VPM3D_cuda::Grid_Shear_Stresses()
{
    // We shall execute the different fd depending on the order of the FD
    switch (FDOrder)
    {
    case (CD2): {cuda_stretch_FD2->Execute(eu_o, eu_dddt, Halo1data, eu_dodt, sgs, qcrit);   break;}
    case (CD4): {cuda_stretch_FD4->Execute(eu_o, eu_dddt, Halo2data, eu_dodt, sgs, qcrit);   break;}
    case (CD6): {cuda_stretch_FD6->Execute(eu_o, eu_dddt, Halo3data, eu_dodt, sgs, qcrit);   break;}
    case (CD8): {cuda_stretch_FD8->Execute(eu_o, eu_dddt, Halo4data, eu_dodt, sgs, qcrit);   break;}
    default:    {std::cout << "VPM3D_cuda::Grid_Shear_Stresses(). FDOrder undefined.";   break;}
    }
    return;
}

void VPM3D_cuda::Grid_Turb_Shear_Stresses()
{
    // Turbulent shear stresses are calculated on the Eulerian grid
    if (Turb==LAM) return;

    if (Turb==HYP)
    {
        // As the storage of the laplacian operator incurs a lot of additional overhead, unlike for the CPU implementation,
        // I will here recalculate it.

        cudaMemset(Laplacian, Real(0.0), 3*NNT*sizeof(Real));

        switch (FDOrder)
        {
        case (CD2): {cuda_Laplacian_FD2->Execute(eu_o, Halo1data, Laplacian);     break; }
        case (CD4): {cuda_Laplacian_FD4->Execute(eu_o, Halo2data, Laplacian);     break; }
        case (CD6): {cuda_Laplacian_FD6->Execute(eu_o, Halo3data, Laplacian);     break; }
        case (CD8): {cuda_Laplacian_FD8->Execute(eu_o, Halo4data, Laplacian);     break; }
        default:    {std::cout << "VPM3D_cpu::Grid_Turb_Shear_Stresses. FDOrder undefined.";  break; }
        }

        switch (FDOrder)
        {
        case (CD2): {cuda_Turb_Hyp_FD2->Execute(Laplacian, Halo1data, C_smag, eu_dodt);     break; }
        case (CD4): {cuda_Turb_Hyp_FD4->Execute(Laplacian, Halo2data, C_smag, eu_dodt);     break; }
        case (CD6): {cuda_Turb_Hyp_FD6->Execute(Laplacian, Halo3data, C_smag, eu_dodt);     break; }
        case (CD8): {cuda_Turb_Hyp_FD8->Execute(Laplacian, Halo4data, C_smag, eu_dodt);     break; }
        default:    {std::cout << "VPM3D_cpu::Grid_Turb_Shear_Stresses. FDOrder undefined.";  break; }
        }
    }

    // Regularized variational multiscale models

    if (Turb==RVM1)         // RVM first order
    {
        // Copy Omega array into temp array.
        cudaMemset(gfilt_Array1, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(gfilt_Array2, Real(0.0), 3*NNT*sizeof(Real));

        cudaMemcpy(gfilt_Array2, eu_o, 3*NNT*sizeof(Real), cudaMemcpyDeviceToDevice);
        cuda_sg_discfilt->Execute(eu_o, Halo1data, gfilt_Array2, gfilt_Array1, int(2));   // Z sweep
        cuda_sg_discfilt->Execute(eu_o, Halo1data, gfilt_Array1, gfilt_Array2, int(1));   // Y sweep
        cuda_sg_discfilt->Execute(eu_o, Halo1data, gfilt_Array2, gfilt_Array1, int(0));   // X sweep
        cuda_sg_discfilt->Execute(eu_o, Halo1data, gfilt_Array1, gfilt_Array2, int(3));   // set small scale

        // cuda_sg_discfilt2->Execute(eu_o, Halo1data, gfilt_Array2, gfilt_Array1, int(0));   // Full sweep
        // cuda_sg_discfilt2->Execute(eu_o, Halo1data, gfilt_Array1, gfilt_Array2, int(1));   // set small scale

        switch (FDOrder)
        {
        case (CD2): {cuda_Turb_RVM_FD2->Execute(gfilt_Array2, sgs, Halo1data, C_smag , eu_dodt);     break; }
        case (CD4): {cuda_Turb_RVM_FD4->Execute(gfilt_Array2, sgs, Halo2data, C_smag , eu_dodt);     break; }
        case (CD6): {cuda_Turb_RVM_FD6->Execute(gfilt_Array2, sgs, Halo3data, C_smag , eu_dodt);     break; }
        case (CD8): {cuda_Turb_RVM_FD8->Execute(gfilt_Array2, sgs, Halo4data, C_smag , eu_dodt);     break; }
        default:    {std::cout << "VPM3D_cpu::Grid_Turb_Shear_Stresses. FDOrder undefined for RVM2.";  break; }
        }
    }

    if (Turb==RVM2)         // RVM second order
    {
        // Copy Omega array into temp array.
        cudaMemset(gfilt_Array1, Real(0.0), 3*NNT*sizeof(Real));
        cudaMemset(gfilt_Array2, Real(0.0), 3*NNT*sizeof(Real));

        cudaMemcpy(gfilt_Array1, eu_o, 3*NNT*sizeof(Real), cudaMemcpyDeviceToDevice);
        cuda_sg_discfilt->Execute(eu_o, Halo1data, gfilt_Array1, gfilt_Array2, int(2));   // Z sweep 1
        cuda_sg_discfilt->Execute(eu_o, Halo1data, gfilt_Array2, gfilt_Array1, int(2));   // Z sweep 2
        cuda_sg_discfilt->Execute(eu_o, Halo1data, gfilt_Array1, gfilt_Array2, int(1));   // Y sweep 1
        cuda_sg_discfilt->Execute(eu_o, Halo1data, gfilt_Array2, gfilt_Array1, int(1));   // Y sweep 2
        cuda_sg_discfilt->Execute(eu_o, Halo1data, gfilt_Array1, gfilt_Array2, int(0));   // X sweep 1
        cuda_sg_discfilt->Execute(eu_o, Halo1data, gfilt_Array2, gfilt_Array1, int(0));   // X sweep 2
        cuda_sg_discfilt->Execute(eu_o, Halo1data, gfilt_Array1, gfilt_Array2, int(3));   // set small scale

        switch (FDOrder)
        {
        case (CD2): {cuda_Turb_RVM_FD2->Execute(gfilt_Array2, sgs, Halo1data, C_smag , eu_dodt);     break; }
        case (CD4): {cuda_Turb_RVM_FD4->Execute(gfilt_Array2, sgs, Halo2data, C_smag , eu_dodt);     break; }
        case (CD6): {cuda_Turb_RVM_FD6->Execute(gfilt_Array2, sgs, Halo3data, C_smag , eu_dodt);     break; }
        case (CD8): {cuda_Turb_RVM_FD8->Execute(gfilt_Array2, sgs, Halo4data, C_smag , eu_dodt);     break; }
        default:    {std::cout << "VPM3D_cpu::Grid_Turb_Shear_Stresses. FDOrder undefined for RVM2.";  break; }
        }
    }

}

void VPM3D_cuda::Remesh_Particle_Set()
{
    // Clear current array
    cudaMemset(eu_o,      Real(0.0), 3*NNT*sizeof(Real));

    // Map to new grid
    switch (RemeshMap)
    {
    case (M2):  {cuda_mapM2->Execute(lg_o, lg_d, Halo1data, eu_o);      break;}
    case (M4):  {cuda_mapM4->Execute(lg_o, lg_d, Halo2data, eu_o);      break;}
    case (M4D): {cuda_mapM4D->Execute(lg_o, lg_d, Halo2data, eu_o);     break;}
    case (M6D): {cuda_mapM6D->Execute(lg_o, lg_d, Halo3data, eu_o);     break;}
    default:    {std::cout << "VPM_3D_Solver::Remesh_Particle_Set(). Mapping undefined.";   return;}
    }

    // Transfer data from Eulerian grid to Lagrangian grid and clear temp vars
    cudaMemcpy(lg_o, eu_o, 3*NNT*sizeof(Real), cudaMemcpyDeviceToDevice);
    cudaMemset(eu_o,      Real(0.0), 3*NNT*sizeof(Real));               // Reset Eulerian grid vorticity
    cudaMemset(lg_d,      Real(0.0), 3*NNT*sizeof(Real));               // Reset Lagrangian grid displacement

    std::cout << "Particle Set Remeshed" << std::endl;
}

void VPM3D_cuda::Magnitude_Filtering()
{
    if (MagFiltFac==0.0) return;

    // Step 1: Check maximum vorticity with reduction kernel.
    cudaMemset(diagnostic_reduced,     Real(0.0), NDiags*NBT*sizeof(Real));
    cuda_MagFilt1->Execute(lg_o, diagnostic_reduced);
    RVector dred(NBT);
    cudaMemcpy(dred.data(), diagnostic_reduced, NBT*sizeof(Real), cudaMemcpyDeviceToHost);
    Real OmMax = *std::max_element(dred.begin(), dred.begin()+NBT);
    Real TargetOmMax = OmMax*MagFiltFac;

    // Step 2: Sweep over field and remove particles with low strength and count remaining (active) particles
    cudaMemset(diagnostic_reduced,  Real(0.0), NDiags*NBT*sizeof(Real));
    cudaMemset(magfilt_count,    int(0), NBT*sizeof(int));
    cuda_MagFilt2->Execute(lg_o, diagnostic_reduced, magfilt_count, TargetOmMax);
    RVector VorticityRemoved(3*NBT,0);
    cudaMemcpy(VorticityRemoved.data(), diagnostic_reduced, 3*NBT*sizeof(Real), cudaMemcpyDeviceToHost);
    std::vector<int> CountActive(NBT,0);
    cudaMemcpy(CountActive.data(), magfilt_count, NBT*sizeof(int), cudaMemcpyDeviceToHost);

    // Now sum over values
    Real MagRem[3] = {0};
    int Counted;
    OpenMPfor
        for (int i=0; i<4; i++){
        if (i<3)    MagRem[i] = std::accumulate(VorticityRemoved.begin()+i*NBT,VorticityRemoved.begin()+(i+1)*NBT,0.0);
        else        Counted = std::accumulate(CountActive.begin(), CountActive.end(), 0);
    }

    // Step 3: Add removed vorticity to remaining particles with nonzero strength in order to conserve total circulation
    Real Magincx = MagRem[0]/Real(Counted);
    Real Magincy = MagRem[1]/Real(Counted);
    Real Magincz = MagRem[2]/Real(Counted);
    cuda_MagFilt3->Execute(lg_o, Magincx, Magincy, Magincz);

    // std::cout << "Magnitude filtering has been carried out. Number of remaining particles with nonzero vorticity: " << Counted << std::endl;
}

void VPM3D_cuda::Reproject_Particle_Set_Spectral()
{
    // The particle field is reprojected in the spectral space.
    cudaMemset(cuda_r_Input1,   Real(0.0), NT*sizeof(Real));
    cudaMemset(cuda_r_Input2,   Real(0.0), NT*sizeof(Real));
    cudaMemset(cuda_r_Input3,   Real(0.0), NT*sizeof(Real));
    cuda_map_toUnbounded->Execute(lg_o, cuda_r_Input1, cuda_r_Input2, cuda_r_Input3);
    Forward_Transform();
    cuda_VPM_reprojection->Execute(c_FTInput1, c_FTInput2, c_FTInput3, c_FG, c_FGi, c_FGj, c_FGk, BFac, c_FTOutput1, c_FTOutput2, c_FTOutput3);
    Backward_Transform();
    cuda_map_fromUnbounded->Execute(cuda_r_Output1, cuda_r_Output2, cuda_r_Output3, lg_o);
    std::cout << "Vorticity field reprojection completed with spectral method." << std::endl;
}

inline Real Calc_Root_Finite_Depth(Real O, Real H)
{
    // This function makes use of formulas (25) and (27) from:
    // Newman, J. N. 1990 “Numerical solutions of the water-wave dispersion relation,” Applied Ocean Research, 12, 14-18.
    // In order to calculate the roots of the equation κ tanh κh = ω^2/g
    Real Gravity = 9.81;
    Real x = O*O*H/Gravity;
    //    Real An[9] = {0.03355458, 0.03262249, -0.00088239, 0.00004620, -0.00000303, 0.00000034, -0.00000007, 0.00000003, -0.00000001};
    Real Bn[6] = {0.000000122, 0.073250017, -0.009899981, 0.002640863, 0.000829239, -0.000176411};
    Real Cn[9] = {1.0, -0.33333372, -0.01109668, 0.01726435, 0.01325580, -0.00116594, 0.00829006, -0.01252603, 0.00404923};

    Real y = 0.0;
    if (x <= 2.0){
        Real Den = 0.0;
        for (int i=0; i<=8; i++) Den += Cn[i]*pow( 0.5*x , i);
        y = sqrt(x)/Den;                                                        // Maximum error e = 1.0 x 10^-8.
    }
    else {
        y += x;
        for (int i=0; i<=5; i++) y += Bn[i]*pow( 0.5*x*exp(4.0-2.0*x) , i);     // Maximum error e = 1.2 x 10^-7.
    }
    return y/H;
}

void VPM3D_cuda::Add_Freestream_Velocity()
{
    // This adds the local freestream to the grid
    if (sqrt(Ux*Ux + Uy*Uy + Uz*Uz)==0.0) return;
    cuda_freestream->Execute(eu_dddt, Real(Ux), Real(Uy), Real(Uz));

    // Additions for Airy wave term

    // NumBeRT model is being applied, so the turbine has a radius of 2.59960008m->Length scale L = 4 (compared to UBert)
    // Values from Sascha:
    // Real Amp = 2.0, Omega = 0.1*2.0*M_PI;       // Config 1: UBert Scale: Amp 0.5m, Freq = 0.2Hz ->Numbert: Amp = 2.0, Freq = 0.2/(sqrt(L)) = 0.1;
    // Real Amp = 2.0, Omega = 1.0*2.0*M_PI;       // Config 1: UBert Scale: Amp 0.5m, Freq = 2.0Hz ->Numbert: Amp = 2.0, Freq = 2.0/(sqrt(L)) = 1.0;
    // Real TurbineDepth = 10.0;   // Attempt for extreme influence
    // Real BeckenDepth = 20.0;    // Actual tank depth-> 5m-> Scaled = 20m
    // Real k = Calc_Root_Finite_Depth(Omega,BeckenDepth);
    // cuda_Airywave->Execute( eu_dddt,
    //                         Real(NStep*dT),
    //                         Real(XN1),
    //                         Real(ZN1),
    //                         Real(Amp),
    //                         Real(-TurbineDepth),
    //                         Real(BeckenDepth),
    //                         Real(k),
    //                         Real(Omega)); // Current time
}

void VPM3D_cuda::Calc_Grid_Diagnostics()
{
    // Diagnostics are calculated in blocks of size BS This is transferred back to the CPU and summed there for simplicity
    cudaMemset(diagnostic_reduced,     Real(0.0), NDiags*NBT*sizeof(Real));
    cuda_Diagnostics->Execute(Real(XN1), Real(YN1), Real(ZN1), eu_o, eu_dddt, diagnostic_reduced);

    // Transfer back to host
    RVector dred(NDiags*NBT);
    cudaMemcpy(dred.data(), diagnostic_reduced, NDiags*NBT*sizeof(Real), cudaMemcpyDeviceToHost);

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
            create_directory(std::filesystem::path(OutputDirectory));   // Generate directory
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

//-------------------------------------------
//----------- Grid functions ----------------
//-------------------------------------------

void VPM3D_cuda::Extract_Field(const Real *Field, const RVector &Px, const RVector &Py, const RVector &Pz, RVector &Ux, RVector &Uy, RVector &Uz, Mapping M)
{
    // Values are extracted from the grid using a local interpolation. This is carried out by loading a block into memory and then
    // interpolating this in shared memory
    int NP = size(Px);

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
    }

    // Sort into blocks of size NT <= BT
    std::vector<Vector3> NDS;
    std::vector<int> map_blx, map_bly, map_blz, idout(NP);
    int count = 0;
    for (int b=0; b<NBT; b++){
        int n = size(IDB[b]);                               // # particles in this box
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

    // std::cout << "Checking mapping " << map_blx.size() csp map_bly.size() csp map_blz.size() csp map_blx.size()*BT csp NDS.size() << " NNodes: " << NP << std::endl;
    // std::cout << "Testing node numbering: " << std::endl;
    // for (int i=0; i<NP; i++)  std::cout << "P Orig: " << Px[i] csp Py[i] csp Pz[i] csp " P Test: "<<  NDS[idout[i]](0) + X0 csp  NDS[idout[i]](1) + Y0 csp NDS[idout[i]](2) + Z0 << std::endl;

    // NDS now contains the padded evaluation positions (in relative CS)
    // map_bl_i contains the block id of the boxes to be evaluated
    // map_nP is the number of evaluation points for this block
    // Sort into SOA
    int sND = size(NDS);
    RVector map_X(sND), map_Y(sND), map_Z(sND);
    Parallel_Kernel(sND) {
        map_X[i] = NDS[i](0);
        map_Y[i] = NDS[i](1);
        map_Z[i] = NDS[i](2);
    }

    // Transfer data to device
    Real *px, *py, *pz, *ux, *uy, *uz;
    cudaMalloc((void**)&px,         sND*sizeof(Real));
    cudaMalloc((void**)&py,         sND*sizeof(Real));
    cudaMalloc((void**)&pz,         sND*sizeof(Real));
    cudaMalloc((void**)&ux,         sND*sizeof(Real));
    cudaMalloc((void**)&uy,         sND*sizeof(Real));
    cudaMalloc((void**)&uz,         sND*sizeof(Real));
    cudaMemcpy(px,  map_X.data(),   sND*sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(py,  map_Y.data(),   sND*sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(pz,  map_Z.data(),   sND*sizeof(Real), cudaMemcpyHostToDevice);

    // int NBlock = map_nP.size();
    int NBlock = map_blx.size();
    // std::cout << "NBlocks = " << NBlock csp map_blx.size() csp map_bly.size() csp map_blz.size() << std::endl;
    int *bidx, *bidy, *bidz;
    cudaMalloc((void**)&bidx,           NBlock*sizeof(int));
    cudaMalloc((void**)&bidy,           NBlock*sizeof(int));
    cudaMalloc((void**)&bidz,           NBlock*sizeof(int));
    cudaMemcpy(bidx, map_blx.data(),    NBlock*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bidy, map_bly.data(),    NBlock*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bidz, map_blz.data(),    NBlock*sizeof(int), cudaMemcpyHostToDevice);

    // Configure kernel
    dim3 Grid(NBlock,1,1);
    switch (M)
    {
    case (M2):      {cuda_interpM2_block->Instantiate(Grid,blockarch_block);   break;}
    case (M4):      {cuda_interpM4_block->Instantiate(Grid,blockarch_block);   break;}
    case (M4D):     {cuda_interpM4D_block->Instantiate(Grid,blockarch_block);  break;}
    case (M6D):     {cuda_interpM6D_block->Instantiate(Grid,blockarch_block);  break;}
    default:        {std::cout << "VPM3D_cuda::Extract Field: Mapping undefined."; return;}
    }

    // Execute kernel
    switch (M)
    {
    case (M2):      {cuda_interpM2_block->Execute(Field, px,py,pz,bidx,bidy,bidz,Halo1data,ux,uy,uz);    break;}
    case (M4):      {cuda_interpM4_block->Execute(Field, px,py,pz,bidx,bidy,bidz,Halo2data,ux,uy,uz);    break;}
    case (M4D):     {cuda_interpM4D_block->Execute(Field,px,py,pz,bidx,bidy,bidz,Halo2data,ux,uy,uz);    break;}
    case (M6D):     {cuda_interpM6D_block->Execute(Field,px,py,pz,bidx,bidy,bidz,Halo3data,ux,uy,uz);    break;}
    default:        {std::cout << "VPM3D_cuda::Extract Field: Mapping undefined."; return;}
    }

    // Extract out data
    RVector rux(sND), ruy(sND), ruz(sND);
    cudaMemcpy(rux.data(), ux, sND*sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(ruy.data(), uy, sND*sizeof(Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(ruz.data(), uz, sND*sizeof(Real), cudaMemcpyDeviceToHost);

    // Now cycle through and pass back values at correct positions
    for (int i=0; i<NP; i++){
        // std::cout << rux[idout[i]] csp ruy[idout[i]] csp ruz[idout[i]] << std::endl;
        Ux[i] = rux[idout[i]];
        Uy[i] = ruy[idout[i]];
        Uz[i] = ruz[idout[i]];
    }

    // Clear temporary arrays
    cudaFree(px);
    cudaFree(py);
    cudaFree(pz);
    cudaFree(ux);
    cudaFree(uy);
    cudaFree(uz);
    cudaFree(bidx);
    cudaFree(bidy);
    cudaFree(bidz);
}

void VPM3D_cuda::Add_Grid_Sources(const RVector &Px, const RVector &Py, const RVector &Pz,const RVector &Ox, const RVector &Oy, const RVector &Oz, Mapping M)
{
    // The process of adding sources block-wise to the GPU is so inefficient, I will take a different appraoch here.
    // I will carry out the interpolation onto a grid defined on the host, and map this host to the device data.
    // This should work fine as long as the grid is not too large.

    RVector gx_cpu(NNT,0);
    RVector gy_cpu(NNT,0);
    RVector gz_cpu(NNT,0);

    // std::fill(gx_cpu.begin(), gx_cpu.end(), 0.);
    // std::fill(gy_cpu.begin(), gy_cpu.end(), 0.);
    // std::fill(gz_cpu.begin(), gz_cpu.end(), 0.);

    // Map source to source grid
    Map_to_Grid(Px, Py, Pz, Ox, Oy, Oz, gx_cpu, gy_cpu, gz_cpu, M);
    // switch (M)
    // {
    //     case (M2):  {Map_Grid_Sources(Px,Py,Pz,Ox,Oy,Oz,gx_cpu,gy_cpu,gz_cpu, M2);       break;}
    //     case (M4):  {Map_Grid_Sources(Px,Py,Pz,Ox,Oy,Oz,gx_cpu,gy_cpu,gz_cpu, M4);       break;}
    //     case (M4D): {Map_Grid_Sources(Px,Py,Pz,Ox,Oy,Oz,gx_cpu,gy_cpu,gz_cpu, M4D);      break;}
    //     default:    {std::cout << "VPM3D_cuda::Add_Grid_Sources. Interpolation undefined."; return;}
    // }

    // Append to array on device
    Real *gh2d = int_lg_d;                                      // Specify dummy array
    cudaMemset(gh2d,        Real(0.0), 3*NNT*sizeof(Real));     // Clear dummy array
    cudaMemcpy(gh2d,        gx_cpu.data(), NNT*sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(gh2d+NNT,    gy_cpu.data(), NNT*sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(gh2d+2*NNT,  gz_cpu.data(), NNT*sizeof(Real), cudaMemcpyHostToDevice);

    // Add to eulerian grid
    cuda_monolith_to_block_arch->Execute(gh2d, eu_o);
}

//-------------------------------------------
//------- External source functions ---------
//-------------------------------------------

void VPM3D_cuda::Interpolate_Ext_Sources(Mapping M)
{
    // Elements are being permanently mapped from the auxiliary grid to the current Lagrangian grid.
    // In order to ensure the strengths are mapped to the current particle positions, an interpolation of the vorticity on
    // the auxiliary grid to the lagrangian positions has to take place.

    // // Map auxiliary vorticity grid to eu_o
    // cudaMemset(eu_o, 0, 3*NNT*sizeof(Real));
    // cuda_map_from_AuxiliaryVPM->Execute(AuxGrid->Get_Vort_Array(), eu_o);
    // switch (M)
    // {
    // case (M2):      {cuda_interp_auxM2->Execute(eu_o, lg_d,  Halo1data, lg_o);  break;}
    // case (M4):      {cuda_interp_auxM4->Execute(eu_o, lg_d,  Halo2data, lg_o);  break;}
    // case (M4D):     {cuda_interp_auxM4D->Execute(eu_o, lg_d, Halo2data, lg_o);  break;}
    // case (M6D):     {cuda_interp_auxM6D->Execute(eu_o, lg_d, Halo3data, lg_o);  break;}
    // default:        {std::cout << "VPM3D_cuda::Interpolate_Ext_Sources. Interpolation undefined."; return;}
    // }

    // New approach: The sources are already define in blocks in ExtVortX,ExtVortY,ExtVortZ,blX,blY,blZ,
    // carry out block-wise interpolation to the given current node positions

    // We map the new grid to an intermediate (full) grid which has been zeroed.
    if (NBExt==0) return;

    // Map to full grid with Map_Ext kernel.
    dim3 Ext_block_extent = dim3(NBExt,1,1);
    Map_Ext->Instantiate(Ext_block_extent, blockarch_block);
    cudaMemset(eu_dddt, 0, 3*NNT*sizeof(Real));                 // Dummy grid (currently unused)
    Map_Ext->Execute(ExtVortX,ExtVortY,ExtVortZ,blX,blY,blZ,eu_dddt);

    // Now execute interp block on new grid

    // Instantiate size
    switch (M)
    {
    case (M2):      {cuda_interpM2_block2->Instantiate(Ext_block_extent,blockarch_block);   break;}
    case (M4):      {cuda_interpM4_block2->Instantiate(Ext_block_extent,blockarch_block);   break;}
    case (M4D):     {cuda_interpM4D_block2->Instantiate(Ext_block_extent,blockarch_block);  break;}
    case (M6D):     {cuda_interpM6D_block2->Instantiate(Ext_block_extent,blockarch_block);  break;}
    default:        {std::cout << "VPM3D_cuda::Extract Field: Mapping undefined."; return;}
    }

    // Execute kernel
    switch (M)
    {
    case (M2):      {cuda_interpM2_block2->Execute(eu_dddt,blX,blY,blZ,lg_d,Halo1data,lg_o);    break;}
    case (M4):      {cuda_interpM4_block2->Execute(eu_dddt,blX,blY,blZ,lg_d,Halo2data,lg_o);    break;}
    case (M4D):     {cuda_interpM4D_block2->Execute(eu_dddt,blX,blY,blZ,lg_d,Halo2data,lg_o);    break;}
    case (M6D):     {cuda_interpM6D_block2->Execute(eu_dddt,blX,blY,blZ,lg_d,Halo3data,lg_o);    break;}
    default:        {std::cout << "VPM3D_cuda::Extract Field: Mapping undefined."; return;}
    }

    // reset vorticity grid
    cudaMemset(eu_dddt, 0, 3*NNT*sizeof(Real));                 // Dummy grid (currently unused)


    // __global__ void Interp_Block2(  const Real* src,                                // Source grid values (permanently mapped particles)
    //                                 const int *blX, const int *blY, const int *blZ, // Block indices
    //                                 const Real* disp,                               // Particle displacement
    //                                 const int* hs,                                  // Halo indices
    //                                 Real* dest)                                     // Destination grid (vorticity)

}

//-----------------------------------
//------- External sources ----------
//-----------------------------------

void VPM3D_cuda::Store_Grid_Node_Sources(const RVector &Px, const RVector &Py, const RVector &Pz, const RVector &Ox, const RVector &Oy, const RVector &Oz, Mapping Map)
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
        if (p.globblid != idb){                   // New block

            // Add new block to array.
            bgid = BT*size(sID);
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

    // These should now be passed to the corresponding cuda arrays

    // Allocate external arrays if necessary
    NBExt = size(sID);
    if (NBufferExt < NBExt){

        // Reset size of external array
        NBufferExt = NBExt;

        // Clear arrays if necessary
        if (ExtVortX) cudaFree(ExtVortX);   ExtVortX = nullptr;
        if (ExtVortY) cudaFree(ExtVortY);   ExtVortY = nullptr;
        if (ExtVortZ) cudaFree(ExtVortZ);   ExtVortZ = nullptr;
        if (blX) cudaFree(blX);             blX = nullptr;
        if (blY) cudaFree(blY);             blY = nullptr;
        if (blZ) cudaFree(blZ);             blZ = nullptr;

        // Cuda allocate
        cudaMalloc((void**)&ExtVortX, NBufferExt*BT*sizeof(Real));
        cudaMalloc((void**)&ExtVortY, NBufferExt*BT*sizeof(Real));
        cudaMalloc((void**)&ExtVortZ, NBufferExt*BT*sizeof(Real));
        cudaMalloc((void**)&blX, NBufferExt*sizeof(int));
        cudaMalloc((void**)&blY, NBufferExt*sizeof(int));
        cudaMalloc((void**)&blZ, NBufferExt*sizeof(int));
    }

    // Transfer data to arrays
    cudaMemcpy(ExtVortX,    Obx.data(),     NBExt*BT*sizeof(Real),  cudaMemcpyHostToDevice);
    cudaMemcpy(ExtVortY,    Oby.data(),     NBExt*BT*sizeof(Real),  cudaMemcpyHostToDevice);
    cudaMemcpy(ExtVortZ,    Obz.data(),     NBExt*BT*sizeof(Real),  cudaMemcpyHostToDevice);
    cudaMemcpy(blX,         sIDX.data(),    NBExt*sizeof(int),      cudaMemcpyHostToDevice);
    cudaMemcpy(blY,         sIDY.data(),    NBExt*sizeof(int),      cudaMemcpyHostToDevice);
    cudaMemcpy(blZ,         sIDZ.data(),    NBExt*sizeof(int),      cudaMemcpyHostToDevice);

    // std::cout << "VPM3D_cuda::Store_Grid_Node_Sources: Successfully stored." << std::endl;
}

void VPM3D_cuda::Map_from_Auxiliary_Grid()
{
    // In the case that the external sources should be included in the vorticity (e.g. from a lifting line),
    // rather than adding these to the full vorticity field,they are only added to the unbounded array
    if (NBExt==0) return;
    dim3 Ext_block_extent = dim3(NBExt,1,1);
    Map_Ext_Unbounded->Instantiate(Ext_block_extent, blockarch_block);
    Map_Ext_Unbounded->Execute(ExtVortX,ExtVortY,ExtVortZ,blX,blY,blZ,cuda_r_Input1,cuda_r_Input2,cuda_r_Input3);
}

//-------------------------------------------
//----- Generate Output grid for vis --------
//-------------------------------------------

void VPM3D_cuda::Generate_VTK()
{
    Generate_VTK(eu_o, eu_dddt);
    // Generate_VTK(eu_dodt, eu_dddt);
    // Generate_VTK(lg_o, lg_dddt);
}

void VPM3D_cuda::Generate_VTK(const Real *vtkoutput1, const Real *vtkoutput2)
{
    // Specifies a specific output and then produces a vtk file for this
    // The arrays int_lg_d & int_p_o are stored as dummy arrays

    if (vtkoutput1==nullptr) return;
    if (vtkoutput2==nullptr) return;

    // Convert to correct data ordering if necessary
    if (Architecture==BLOCK){
        cuda_block_to_monolith_arch->Execute(vtkoutput1,int_lg_d);
        cuda_block_to_monolith_arch->Execute(vtkoutput2,int_lg_o);
        // out1d = int_lg_d;
        // out2d = int_lg_o;
    }
    if (Architecture==MONO){
        cudaMemcpy(int_lg_d, vtkoutput1, 3*NNT*sizeof(Real), cudaMemcpyDeviceToHost);        // Lagrangian grid vorticity field
        cudaMemcpy(int_lg_o, vtkoutput2, 3*NNT*sizeof(Real), cudaMemcpyDeviceToHost);        // Lagrangian grid vorticity field
    }

    // Transfer data from device to host
    Real *hostout1 = (Real*)malloc(3*NNT*sizeof(Real));
    Real *hostout2 = (Real*)malloc(3*NNT*sizeof(Real));
    cudaMemcpy(hostout1, int_lg_d, 3*NNT*sizeof(Real), cudaMemcpyDeviceToHost);        // Lagrangian grid vorticity field
    cudaMemcpy(hostout2, int_lg_o, 3*NNT*sizeof(Real), cudaMemcpyDeviceToHost);        // Eulerian grid velocity field

    // Fill in output vector
    // OpenMPfor
    for (int i=0; i<NNX; i++){
        for (int j=0; j<NNY; j++){
            for (int k=0; k<NNZ; k++){
                int gid = GID(i,j,k,NX,NY,NZ);
                int lid = GID(i,j,k,NNX,NNY,NNZ);
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
    vtk_Name = vtk_Prefix + std::to_string(NStep) + ".vtk";

    Create_vtk();

    cudaMemset(int_lg_d,      Real(0.0), 3*NNT*sizeof(Real));
    cudaMemset(int_lg_o,      Real(0.0), 3*NNT*sizeof(Real));
    free(hostout1);
    free(hostout2);

    // HACK to export qcrit
    // Generate_VTK_Scalar();
}

void VPM3D_cuda::Generate_VTK_Scalar()
{
    // A vtk output is generated for a scalar output array.
    // This function is included for now only for the q-criterion

    // Convert to correct data ordering if necessary
    if (Architecture==BLOCK){
        cuda_block_to_monolith_single->Execute(qcrit,int_lg_d);
    }
    if (Architecture==MONO){
        // cudaMemcpy(int_lg_d, vtkoutput1, 3*NNT*sizeof(Real), cudaMemcpyDeviceToHost);        // Lagrangian grid vorticity field
    }

    // Transfer data from device to host
    Real *hostout1 = (Real*)malloc(NNT*sizeof(Real));
    cudaMemcpy(hostout1, int_lg_d, NNT*sizeof(Real), cudaMemcpyDeviceToHost);        // Lagrangian grid vorticity field

    // Extra hack for q-criterion
    // Real *hostout3 = (Real*)malloc(NNT*sizeof(Real));
    // cudaMemcpy(hostout3, qcrit, NNT*sizeof(Real), cudaMemcpyDeviceToHost);        // Eulerian grid velocity field

    // Fill in output vector
    // OpenMPfor
    for (int i=0; i<NNX; i++){
        for (int j=0; j<NNY; j++){
            for (int k=0; k<NNZ; k++){
                int gid = GID(i,j,k,NX,NY,NZ);
                int lid = GID(i,j,k,NNX,NNY,NNZ);
                r_Input1[gid] = hostout1[lid      ];
                r_Input2[gid] = 0.0;
                r_Input3[gid] = 0.0;
                r_Output1[gid] = 0.0;
                r_Output2[gid] = 0.0;
                r_Output3[gid] = 0.0;
            }
        }
    }

    // Specify current filename.
    vtk_Name = vtk_Prefix + std::to_string(NStep) + "_qcrit.vtk";

    Create_vtk();

    cudaMemset(int_lg_d,      Real(0.0), 3*NNT*sizeof(Real));
    cudaMemset(int_lg_o,      Real(0.0), 3*NNT*sizeof(Real));
    free(hostout1);
}


void VPM3D_cuda::Generate_Plane(RVector &U)
{
    // This function retrieves the wake plane from the data for visualisation

    // std::cout << "Check Prev " << *std::min_element(U.begin(), U.end()) csp  *std::max_element(U.begin(), U.end()) << std::endl;
    int NPlane = NBX*BX*NBZ*BZ;
    int YPlane = NNY/2;          // Approximately middle plane
    cuda_ExtractPlaneY->Execute(eu_o, vis_plane, YPlane);
    cudaMemcpy(U.data(), vis_plane, sizeof(Real)*NPlane, cudaMemcpyDeviceToHost);
    // std::cout << "Check Post " << *std::min_element(U.begin(), U.end()) csp  *std::max_element(U.begin(), U.end()) << std::endl;
}

void VPM3D_cuda::Generate_Traverse(int XP, RVector &U, RVector &V, RVector &W)
{
    // This function retrieves the wake plane from the data for visualisation

    // // std::cout << "Check Prev " << *std::min_element(U.begin(), U.end()) csp  *std::max_element(U.begin(), U.end()) << std::endl;
    int NPlane = NBY*BY*NBZ*BZ;
    cuda_ExtractPlaneX->Execute(eu_dddt, travx, travy, travz, XP);
    cudaMemcpy(U.data(), travx, sizeof(Real)*NPlane, cudaMemcpyDeviceToHost);
    cudaMemcpy(V.data(), travy, sizeof(Real)*NPlane, cudaMemcpyDeviceToHost);
    cudaMemcpy(W.data(), travz, sizeof(Real)*NPlane, cudaMemcpyDeviceToHost);
}

//--- Destructor

VPM3D_cuda::~VPM3D_cuda()
{
    // Deallocate data

    //--- Grid Arrays
    cudaFree(lg_d   );
    cudaFree(lg_o   );
    cudaFree(lg_dddt);
    cudaFree(lg_dodt);
    // cudaFree(eu_d   );
    cudaFree(eu_o   );
    cudaFree(eu_dddt);
    cudaFree(eu_dodt);
    cudaFree(dumbuffer);

    cudaFree(diagnostic_reduced);   // Reduced diagnostics arrays
    cudaFree(vis_plane);   // Reduced diagnostics arrays
    cudaFree(magfilt_count);         // Count of particle which have non-zero strength after magnitude filtering

    // Indices for halo data
    cudaFree(Halo1data);
    cudaFree(Halo2data);
    cudaFree(Halo3data);
    cudaFree(Halo4data);

    //--- Timestepping (temporary) arrays
    cudaFree(int_lg_d);
    cudaFree(int_lg_o);
    cudaFree(k2_d);
    cudaFree(k2_o);
    cudaFree(k3_d);
    cudaFree(k3_o);
    cudaFree(k4_d);
    cudaFree(k4_o);
    cudaFree(tm1_d);
    cudaFree(tm1_o);
    cudaFree(tm1_dddt);
    cudaFree(tm1_dodt);

    //--- Turbulence arrays
    cudaFree(Laplacian);
}

}

#endif
