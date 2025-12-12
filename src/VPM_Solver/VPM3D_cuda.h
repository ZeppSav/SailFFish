#ifndef VPM3D_CUDA_H
#define VPM3D_CUDA_H

#include "VPM_Solver.h"

#ifdef CUFFT

#include "jitify.hpp"           // Just-in-time compiling suite for cuda

namespace SailFFish
{

class cudaKernel
{
    const char *Handle;
    std::string Source;                         // Source code
    jitify::Kernel K;                           // Kernel object
    jitify::KernelInstantiation KI;             // Kernel Instantiation
    jitify::KernelLauncher KL;                  // Kernel Launcher
    std::string Label;                          // Label for execution
    int NBytes = 0;                             // Number of bytes required for execution
    std::string Identifier = "Modified";                     // This is used to ensure uniqeuness between compiled kernels.

public:

    // Constructor
    template <typename... TemplateArgs>
    cudaKernel(std::string &Src,
               const char *FunctionName,
               TemplateArgs... targs){
        // Set inputs
        Handle = FunctionName;

        // // Add unique Kernel identifier (if required) This is no longer required without the Auxiliary grid
        // std::string strgHandle(Handle);
        // std::string newstrgHandle = Handle + Kid;
        // auto StringPos = Src.find(strgHandle);
        // auto StringPosnew = Src.find(newstrgHandle);
        // if (StringPosnew == std::string::npos && StringPos != std::string::npos) Src.replace(StringPos, strgHandle.length(), newstrgHandle);

        // Generate Kernel
        static jitify::JitCache kernel_cache;
        jitify::Program program = kernel_cache.program(Src.c_str(), 0);
        K = program.kernel(Handle);
        // K = program.kernel(newstrgHandle);
        // Handle = newstrgHandle.c_str();

        // Create instance of kernel
        KI = K.instantiate(std::vector<std::string>({jitify::reflection::reflect(targs)...}));

        // std::cout << "Kernel handle " << newstrgHandle << " compiled." << std::endl;
        std::cout << "Kernel handle " << Handle << " compiled." << std::endl;
    }

    // Compilation flags
    bool GridFlag = false;
    bool TypeFlag = false;
    bool FinDiffFlag = false;

    // Retrieve points to set memory

    // Compose Kernel source
    jitify::KernelInstantiation *Get_Instance() {return &KI;}
    void Instantiate(dim3 Grid, dim3 Block) {KL = KI.configure(Grid, Block);}

    // Kernel memory bounds
    // void Maximise_Dynamic_Memory()  {cudaFuncSetAttribute(KI,cudaFuncAttributeMaxDynamicSharedMemorySize,96*1024);}

    template <typename... ArgTypes>
    void Execute(const ArgTypes&... args)
    {
        // Add flags here for sanity & timing checks
        KL.launch(std::vector<void*>({(void*)&args...}));
    }
    // Timer....
};

enum gpuDataArch {MONO,PENCIL,BLOCK};

class VPM3D_cuda : public VPM_3D_Solver //, public cuda_Grid_Data
{

    //--- Grid Arrays
    Real *lg_d;         // Lagrangian grid - particle displacement
    Real *lg_o;         // Lagrangian grid - particle vorticity
    Real *lg_dddt;      // Lagrangian grid - Rate of change particle displacement
    Real *lg_dodt;      // Lagrangian grid - Rate of change of vorticity
    // Real *eu_d;         // Eulerian grid - particle displacement                    // Only necessary in initialisation-- can use a dummy array here
    Real *eu_o;         // Eulerian grid - particle vorticity
    Real *eu_dddt;      // Eulerian grid - Rate of change particle displacement
    Real *eu_dodt;      // Eulerian grid - Rate of change of vorticity

    // TensorGrid egu_cpu
    // RVector g_dpu;      // CPU Buffer for mapping sources to grid
    // RVector gx_cpu;     // CPU buffer for mapping sources to grid.
    // RVector gy_cpu;     // CPU buffer for mapping sources to grid.
    // RVector gz_cpu;     // CPU buffer for mapping sources to grid.

    // Arrays required for turbulence models
    Real *Laplacian;        // Laplacian of vorticity field                         // Only necessary if using hyperviscosity turbulence model
    // Real *NablaU;           // Gradients of velocity field
    Real *sgs;              // sub grid scale
    Real *qcrit;            // qcriterion
    Real *gfilt_Array1;     // Filtered vorticity field
    Real *gfilt_Array2;     // Gradients of velocity field

    //--- Timestepping (temporary) arrays
    Real *int_lg_d;
    Real *int_lg_o;
    Real *k2_d;
    Real *k2_o;
    Real *k3_d;
    Real *k3_o;
    Real *k4_d;
    Real *k4_o;
    Real *tm1_d;
    Real *tm1_o;
    Real *tm1_dddt;
    Real *tm1_dodt;

    Real *diagnostic_reduced;       // Reduced diagnostics arrays
    Real *vis_plane;                // Reduced diagnostics arrays
    Real *travx, *travy, *travz;    // Reduced diagnostics arrays
    int *magfilt_count;         // Count of particle which have non-zero strength after magnitude filtering

    // Arrays for external sources
    size_t NBExt = 0, NBufferExt = 0;
    Real *ExtVortX = nullptr;
    Real *ExtVortY = nullptr;
    Real *ExtVortZ = nullptr;
    Real *blX = nullptr;
    Real *blY = nullptr;
    Real *blZ = nullptr;

    //--- Diagnostics
    static const int NDiags = 15;                             // Number of diagnostics outputs

    // Indices for halo data
    int *Halo1data;
    int *Halo2data;
    int *Halo3data;
    int *Halo4data;

    //---- External grid
    int NBXShift;
    int NBYShift;
    int NBZShift;

    RVector DummyArrayX,DummyArrayY,DummyArrayZ;

    //--- cuda Block & Grid size
    dim3 blockarch_grid, blockarch_block;

    //--- Auxiliary source data
    void Map_External_Sources() override ;

    Real *dumbuffer;            // A buffer allocated for arbitrary tasks and data migration

    //--- cuda Kernels
    cudaKernel *cuda_VPM_convolution;
    cudaKernel *cuda_VPM_reprojection;
    cudaKernel *cuda_monolith_to_block_arch;
    cudaKernel *cuda_block_to_monolith_arch;
    cudaKernel *cuda_block_to_monolith_single;
    cudaKernel *cuda_mapM2;
    cudaKernel *cuda_mapM4;
    cudaKernel *cuda_mapM4D;
    cudaKernel *cuda_mapM6D;
    cudaKernel *cuda_interpM2;
    cudaKernel *cuda_interpM4;
    cudaKernel *cuda_interpM4D;
    cudaKernel *cuda_interpM6D;
    cudaKernel *cuda_map_toUnbounded;
    cudaKernel *cuda_map_fromUnbounded;
    cudaKernel *cuda_update;
    cudaKernel *cuda_updateRK;
    cudaKernel *cuda_updateRK2;
    cudaKernel *cuda_updateRK3;
    cudaKernel *cuda_updateRK4;
    cudaKernel *cuda_updateRKLS;
    cudaKernel *cuda_stretch_FD2;
    cudaKernel *cuda_stretch_FD4;
    cudaKernel *cuda_stretch_FD6;
    cudaKernel *cuda_stretch_FD8;
    cudaKernel *cuda_Diagnostics;
    cudaKernel *cuda_freestream;
    cudaKernel *cuda_MagFilt1;
    cudaKernel *cuda_MagFilt2;
    cudaKernel *cuda_MagFilt3;
    cudaKernel *cuda_ExtractPlaneX, *cuda_ExtractPlaneY;

    cudaKernel *cuda_Airywave;      // Additiona for Airy wave term

    cudaKernel *Map_Ext;
    cudaKernel *Map_Ext_Unbounded;

    cudaKernel *cuda_interpM2_block;
    cudaKernel *cuda_interpM4_block;
    cudaKernel *cuda_interpM4D_block;
    cudaKernel *cuda_interpM6D_block;

    cudaKernel *cuda_interpM2_block2;
    cudaKernel *cuda_interpM4_block2;
    cudaKernel *cuda_interpM4D_block2;
    cudaKernel *cuda_interpM6D_block2;

    // Turbulence Kernels
    cudaKernel *cuda_Laplacian_FD2;
    cudaKernel *cuda_Laplacian_FD4;
    cudaKernel *cuda_Laplacian_FD6;
    cudaKernel *cuda_Laplacian_FD8;
    cudaKernel *cuda_Turb_Hyp_FD2;
    cudaKernel *cuda_Turb_Hyp_FD4;
    cudaKernel *cuda_Turb_Hyp_FD6;
    cudaKernel *cuda_Turb_Hyp_FD8;
    cudaKernel *cuda_sg_discfilt;
    // cudaKernel *cuda_sg_discfilt2;
    cudaKernel *cuda_Turb_RVM_FD2;
    cudaKernel *cuda_Turb_RVM_FD4;
    cudaKernel *cuda_Turb_RVM_FD6;
    cudaKernel *cuda_Turb_RVM_FD8;
    cudaKernel *cuda_Turb_RVM_DGC_FD2;
    cudaKernel *cuda_Turb_RVM_DGC_FD4;
    cudaKernel *cuda_Turb_RVM_DGC_FD6;
    cudaKernel *cuda_Turb_RVM_DGC_FD8;


    //--- Block data format
    gpuDataArch Architecture = BLOCK;

    //--- vtk output
    // Real *vtkoutput1 = nullptr;
    // Real *vtkoutput2 = nullptr;

public:

    //--- Constructor
    VPM3D_cuda(Grid_Type G, Unbounded_Kernel B) : VPM_3D_Solver(G,B)
    {
        // Child constructor
        InPlace = true;        // Specify that transforms should occur either in place of out of place (reduced memory footprint)
        c_dbf_1 = false;        // This dummy buffer is required as I have not included custom Kernels into Datatype_Cuda yet
        // InPlace = true;     // Specify that transforms should occur either in place of out of place (reduced memory footprint)
    }

    //--- Solver setup
    SFStatus Setup_VPM(VPM_Input *I);
    SFStatus Allocate_Data() override;
    SFStatus Allocate_Auxiliary_Data();
    void Initialize_Data();
    void Set_Grid_Positions() override {}  // In cuda, these values are calcualted within the kernel for simplicity
    void Initialize_Halo_Data();

    //--- Kernel setup
    void Set_Kernel_Constants(jitify::KernelInstantiation *KI, int Halo);
    SFStatus Initialize_Kernels();
    //--- Initial vorticity distribution
    void Retrieve_Grid_Positions(RVector &xc, RVector &yc, RVector &zc);
    void Set_Input_Arrays(RVector &xo, RVector &yo, RVector &zo);

    //--- Auxiliary grid operations
    Real *Get_Vorticity_Array() {return eu_o;}
    // void Map_to_Auxiliary_Grid() override;
    void Interpolate_Ext_Sources(Mapping M) override;
    Real* Get_Vort_Array() override {return eu_o;}
    Real* Get_Vel_Array() override {return eu_dddt;}

    //--- Grid routines
    void Clear_Source_Grid()    override    {cudaMemset(eu_o, Real(0.0), 3*NNT*sizeof(Real));}
    void Clear_Solution_Grid()  override    {cudaMemset(eu_dddt, Real(0.0), 3*NNT*sizeof(Real));}
    void Transfer_Source_Grid()             {cudaMemcpy(lg_o, eu_o, 3*NNT*sizeof(Real), cudaMemcpyDeviceToDevice);}

    //--- Grid operations
    void Extract_Field(const Real *Field, const RVector &Px, const RVector &Py, const RVector &Pz, RVector &Ux, RVector &Uy, RVector &Uz, Mapping M);

    void Add_Grid_Sources(const RVector &Px, const RVector &Py, const RVector &Pz,const RVector &Ox, const RVector &Oy, const RVector &Oz, Mapping M) override;

    void Extract_Sol_Values(const RVector &Px, const RVector &Py, const RVector &Pz, RVector &Ugx, RVector &Ugy, RVector &Ugz, Mapping Map) override {
        Extract_Field(eu_dddt, Px, Py, Pz, Ugx, Ugy, Ugz, Map);}

    void Extract_Source_Values(const RVector &Px, const RVector &Py, const RVector &Pz, RVector &Ugx, RVector &Ugy, RVector &Ugz, Mapping Map) override {
        Extract_Field(eu_o, Px, Py, Pz, Ugx, Ugy, Ugz, Map);}

    void Store_Grid_Node_Sources(   const RVector &Px, const RVector &Py, const RVector &Pz,
                                         const RVector &Ox, const RVector &Oy, const RVector &Oz, Mapping Map) override;


    //--- Timestepping
    void Advance_Particle_Set() override;
    void Update_Particle_Field() override;
    void Calc_Particle_RateofChange(const Real *pd, const Real *po, Real *dpddt, Real *dpodt);
    void Calc_Grid_FDRatesof_Change() override;
    void Grid_Shear_Stresses() override;
    void Grid_Turb_Shear_Stresses() override;
    void Add_Freestream_Velocity() override;
    // void Solve_Velocity() override;

    // Debugging
    void Output_Max_Components(const Real *A, int N);
    void Output_Max_Components(const Real *A, const Real *B, const Real *C, int N);

    // //--- Grid utilities
    void Remesh_Particle_Set() override;
    void Reproject_Particle_Set_Spectral() override;
    void Magnitude_Filtering() override;

    //--- Grid statistics
    void Calc_Grid_Diagnostics() override;

    //--- Import grid
    void Import_Field();    // Dummy function for now

    //--- Output grid
    void Generate_VTK() override;
    void Generate_VTK_Scalar()  override;
    void Generate_VTK(const Real *vtkoutput1, const Real *vtkoutput2, const std::string &Name = "");
    void Generate_Plane(RVector &U) override;
    void Generate_Traverse(int XP, RVector &U, RVector &V, RVector &W) override;

    ~VPM3D_cuda() override;
};

}

#endif

#endif // VPM3D_CUDA_H
