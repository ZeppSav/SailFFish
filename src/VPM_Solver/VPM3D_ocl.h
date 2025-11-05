#ifndef VPM3D_OCL_H
#define VPM3D_OCL_H

#include "VPM_Solver.h"

#ifdef VKFFT

namespace SailFFish
{

enum gpuDataArch {MONO,PENCIL,BLOCK};

class VPM3D_ocl : public VPM_3D_Solver
{

    //--- Grid Arrays
    cl_mem lg_d = nullptr;         // Lagrangian grid - particle displacement
    cl_mem lg_o = nullptr;         // Lagrangian grid - particle vorticity
    cl_mem lg_dddt = nullptr;      // Lagrangian grid - Rate of change particle displacement
    cl_mem lg_dodt = nullptr;      // Lagrangian grid - Rate of change of vorticity
    // cl_mem eu_d = nullptr;         // Eulerian grid - particle displacement                    // Only necessary in initialisation-- can use a dummy array here
    cl_mem eu_o = nullptr;         // Eulerian grid - particle vorticity
    cl_mem eu_dddt = nullptr;      // Eulerian grid - Rate of change particle displacement
    cl_mem eu_dodt = nullptr;      // Eulerian grid - Rate of change of vorticity

    // Arrays required for turbulence models
    cl_mem Laplacian = nullptr;        // Laplacian of vorticity field                         // Only necessary if using hyperviscosity turbulence model
    // cl_mem NablaU = nullptr;           // Gradients of velocity field
    cl_mem sgs = nullptr;              // sub grid scale
    cl_mem qcrit = nullptr;            // qcriterion
    cl_mem gfilt_Array1 = nullptr;     // Filtered vorticity field
    cl_mem gfilt_Array2 = nullptr;     // Gradients of velocity field

    //--- Timestepping (temporary) arrays
    cl_mem int_lg_d = nullptr;
    cl_mem int_lg_o = nullptr;
    cl_mem k2_d = nullptr;
    cl_mem k2_o = nullptr;
    cl_mem k3_d = nullptr;
    cl_mem k3_o = nullptr;
    cl_mem k4_d = nullptr;
    cl_mem k4_o = nullptr;
    cl_mem tm1_d = nullptr;
    cl_mem tm1_o = nullptr;
    cl_mem tm1_dddt = nullptr;
    cl_mem tm1_dodt = nullptr;

    cl_mem diagnostic_reduced = nullptr;       // Reduced diagnostics arrays
    // cl_mem vis_plane = nullptr;                // Reduced diagnostics arrays
    // cl_mem travx, *travy, *travz = nullptr;    // Reduced diagnostics arrays
    cl_mem magfilt_count = nullptr;         // Count of particle which have non-zero strength after magnitude filtering

    // Arrays for external sources
    size_t NBExt = 0, NBufferExt = 0;
    cl_mem ExtVortX = nullptr;
    cl_mem ExtVortY = nullptr;
    cl_mem ExtVortZ = nullptr;
    cl_mem blX = nullptr;
    cl_mem blY = nullptr;
    cl_mem blZ = nullptr;

    //--- Diagnostics
    static const int NDiags = 15;                             // Number of diagnostics outputs

    // Indices for halo data
    cl_mem Halo1data = nullptr;
    cl_mem Halo2data = nullptr;
    cl_mem Halo3data = nullptr;
    cl_mem Halo4data = nullptr;

    //--- cuda Block & Grid size
    dim3 blockarch_grid, blockarch_block;

    //--- OpenCL Kernels
    void Add_Grid_Constants(std::string &Source);
    // std::string KID;                        // Unique kernel identifier to ensure to common kernel names compiled on GPU.
    cl_kernel ocl_VPM_convolution;
    cl_kernel ocl_VPM_reprojection;
    cl_kernel ocl_monolith_to_block_arch;
    cl_kernel ocl_block_to_monolith_arch;
    cl_kernel ocl_block_to_monolith_single;
    cl_kernel ocl_mapM2;
    cl_kernel ocl_mapM4;
    cl_kernel ocl_mapM4D;
    cl_kernel ocl_mapM6D;
    cl_kernel ocl_interpM2;
    cl_kernel ocl_interpM4;
    cl_kernel ocl_interpM4D;
    cl_kernel ocl_interpM6D;
    cl_kernel ocl_map_toUnbounded;
    cl_kernel ocl_map_fromUnbounded;
    cl_kernel ocl_update;
    cl_kernel ocl_updateRK;
    cl_kernel ocl_updateRK2;
    cl_kernel ocl_updateRK3;
    cl_kernel ocl_updateRK4;
    cl_kernel ocl_updateRKLS;
    cl_kernel ocl_stretch_FD2;
    cl_kernel ocl_stretch_FD4;
    cl_kernel ocl_stretch_FD6;
    cl_kernel ocl_stretch_FD8;
    cl_kernel ocl_Diagnostics;
    cl_kernel ocl_freestream;
    cl_kernel ocl_MagFilt1;
    cl_kernel ocl_MagFilt2;
    cl_kernel ocl_MagFilt3;
    // cudaKernel *ocl_ExtractPlaneX, *ocl_ExtractPlaneY;

    cl_kernel Map_Ext;
    cl_kernel Map_Ext_Unbounded;

    // cl_kernel ocl_interpM2_block;
    // cl_kernel ocl_interpM4_block;
    // cl_kernel ocl_interpM4D_block;
    // cudaKernel *ocl_interpM6D_block;

    // cudaKernel *ocl_interpM2_block2;
    // cudaKernel *ocl_interpM4_block2;
    // cudaKernel *ocl_interpM4D_block2;
    // cudaKernel *ocl_interpM6D_block2;

    // Turbulence Kernels
    cl_kernel ocl_Laplacian_FD2;
    cl_kernel ocl_Laplacian_FD4;
    cl_kernel ocl_Laplacian_FD6;
    cl_kernel ocl_Laplacian_FD8;
    cl_kernel ocl_Turb_Hyp_FD2;
    cl_kernel ocl_Turb_Hyp_FD4;
    cl_kernel ocl_Turb_Hyp_FD6;
    cl_kernel ocl_Turb_Hyp_FD8;
    cl_kernel ocl_sg_discfilt;
    // cl_kernel ocl_sg_discfilt2;
    cl_kernel ocl_Turb_RVM_FD2;
    cl_kernel ocl_Turb_RVM_FD4;
    cl_kernel ocl_Turb_RVM_FD6;
    cl_kernel ocl_Turb_RVM_FD8;
    cl_kernel ocl_Turb_RVM_DGC_FD2;
    cl_kernel ocl_Turb_RVM_DGC_FD4;
    cl_kernel ocl_Turb_RVM_DGC_FD6;
    cl_kernel ocl_Turb_RVM_DGC_FD8;

    //--- Block data format
    gpuDataArch Architecture = BLOCK;

    // template <typename... ArgTypes>
    // void ExecuteKernel(cl_kernel kernel, const ArgTypes&... args)
    // {
    //     // Add flags here for sanity & timing checks
    //     KL.launch(std::vector<void*>({(void*)&args...}));
    // }

public:

    //--- Constructor
    VPM3D_ocl(Grid_Type G, Unbounded_Kernel B) : VPM_3D_Solver(G,B)
    {
        // Child constructor
        InPlace = true;        // Specify that transforms should occur either in place of out of place (reduced memory footprint)
        c_dbf_1 = false;        // This dummy buffer is required as I have not included custom Kernels into Datatype_Cuda yet
        // InPlace = true;     // Specify that transforms should occur either in place of out of place (reduced memory footprint)
    }

    //--- Solver setup
    SFStatus Setup_VPM(VPM_Input *I);
    SFStatus Allocate_Data() override;
    void Initialize_Data();
    void Set_Grid_Positions() override {}  // In OpenCL, these values are calcualted within the kernel for simplicity
    void Initialize_Halo_Data();

    //--- Kernel setup
//     void Set_Kernel_Constants(jitify::KernelInstantiation *KI, int Halo);
    cl_kernel Generate_Kernel(const std::string &Body, const std::string &Tag, bool Print = false);
    SFStatus Initialize_Kernels();

    //--- Kernel execution
    SFStatus Execute_Block_Kernel(cl_kernel kernel);

//     //--- Initial vorticity distribution
    void Retrieve_Grid_Positions(RVector &xc, RVector &yc, RVector &zc);
    void Set_Input_Arrays(RVector &x0, RVector &y0, RVector &z0);

//     //--- Auxiliary grid operations
//     Real *Get_Vorticity_Array() {return eu_o;}
//     void Set_External_Grid(VPM_3D_Solver *G) override;
//     void Map_to_Auxiliary_Grid() override;
//     void Interpolate_Ext_Sources(Mapping M) override;
//     Real* Get_Vort_Array() override {return eu_o;}
//     Real* Get_Vel_Array() override {return eu_dddt;}

//     //--- Grid routines
//     void Clear_Source_Grid()    override    {cudaMemset(eu_o, Real(0.0), 3*NNT*sizeof(Real));}
//     void Clear_Solution_Grid()  override    {cudaMemset(eu_dddt, Real(0.0), 3*NNT*sizeof(Real));}
//     void Transfer_Source_Grid()             {cudaMemcpy(lg_o, eu_o, 3*NNT*sizeof(Real), cudaMemcpyDeviceToDevice);}

//     //--- Grid operations
//     void Extract_Field(const Real *Field, const RVector &Px, const RVector &Py, const RVector &Pz, RVector &Ux, RVector &Uy, RVector &Uz, Mapping M);

//     void Add_Grid_Sources(const RVector &Px, const RVector &Py, const RVector &Pz,const RVector &Ox, const RVector &Oy, const RVector &Oz, Mapping M) override;

//     void Extract_Sol_Values(const RVector &Px, const RVector &Py, const RVector &Pz, RVector &Ugx, RVector &Ugy, RVector &Ugz, Mapping Map) override {
//         Extract_Field(eu_dddt, Px, Py, Pz, Ugx, Ugy, Ugz, Map);}

//     void Extract_Source_Values(const RVector &Px, const RVector &Py, const RVector &Pz, RVector &Ugx, RVector &Ugy, RVector &Ugz, Mapping Map) override {
//         Extract_Field(eu_o, Px, Py, Pz, Ugx, Ugy, Ugz, Map);}

//     void Store_Grid_Node_Sources(   const RVector &Px, const RVector &Py, const RVector &Pz,
//                                  const RVector &Ox, const RVector &Oy, const RVector &Oz, Mapping Map) override;


    //--- Timestepping
    void Advance_Particle_Set() override;
//     void Update_Particle_Field() override;
//     void Calc_Particle_RateofChange(const Real *pd, const Real *po, Real *dpddt, Real *dpodt);
//     void Calc_Grid_FDRatesof_Change() override;
//     void Grid_Shear_Stresses() override;
//     void Grid_Turb_Shear_Stresses() override;
//     void Add_Freestream_Velocity() override;
//     void Solve_Velocity() override;

//     // Debugging
//     void Output_Max_Components(const Real *A, int N);
//     void Output_Max_Components(const Real *A, const Real *B, const Real *C, int N);

//     // //--- Grid utilities
//     void Remesh_Particle_Set() override;
//     void Reproject_Particle_Set_Spectral() override;
//     void Magnitude_Filtering() override;

//     //--- Grid statistics
//     void Calc_Grid_Diagnostics() override;

    //--- Output grid
    void Generate_VTK() override;
//     void Generate_VTK_Scalar()  override;
    void Generate_VTK(cl_mem vtkoutput1, cl_mem vtkoutput2);
//     void Generate_Plane(RVector &U) override;
//     void Generate_Traverse(int XP, RVector &U, RVector &V, RVector &W) override;

//     ///--- Testing function
//     void MatMultTest();

//     ~VPM3D_cuda() override;
// };

};

}

#endif

#endif // VPM3D_OCL_H
