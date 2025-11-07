#ifndef VPM3D_CPU_H
#define VPM3D_CPU_H

#include "VPM_Solver.h"

namespace SailFFish
{

class VPM3D_cpu : public VPM_3D_Solver
{
protected:

    //--- ID types
    dim3 *loc_ID;
    dim3 dom_size;
    dim3 padded_dom_size;

    //--- Grid Arrays
    TensorGrid Laplacian;
    TensorGrid GradU;

    //--- Grid arrays
    TensorGrid lg_d, lg_o;
    TensorGrid lg_dddt, lg_dodt;
    TensorGrid eu_o;
    TensorGrid eu_dddt, eu_dodt;
    TensorGrid int_lg_d, int_lg_o;
    TensorGrid kd_d, k2_o;
    TensorGrid k3_d, k3_o;
    TensorGrid k4_d, k4_o;
    TensorGrid tm1_d, tm1_o;
    TensorGrid tm1_dddt, tm1_dodt;

    //--- Auxiliary grid data
    bool Auxiliary = false;

    //--- Exclusion zones
    std::vector<bool> FD_Flag, Map_Flag, Remesh_Flag;

    //--- Turbulence models params
    Real C_smag;
    TensorGrid gfilt_Array1, gfilt_Array2;
    RVector SGS;

public:

    //--- Constructor
    VPM3D_cpu(Grid_Type G = STAGGERED, Unbounded_Kernel B = HEJ_G8)  : VPM_3D_Solver(G,B)
    {
        // Basis constructor

        // Specify grid and Ker types
        Grid = G;
        Greens_Kernel = B;

        std::cout << "Calling here" << std::endl;

        // Specify operator
        Specify_Operator(SailFFish::CURL);
        InPlace = false;        // Specify that transforms should occur either in place of out of place (reduced memory footprint)
        // c_dbf_1 = true;         // This dummy buffer is required as I have not included custom Kernels into Datatype_Cuda yet
    }

    //--- Solver setup
    SFStatus Setup_VPM(VPM_Input *I);
    SFStatus Allocate_Data() override;
    void Prepare_Exclusion_Zones();

    //--- Mapping functions
    void Add_Grid_Sources(const RVector &Px, const RVector &Py, const RVector &Pz, const RVector &Ox, const RVector &Oy, const RVector &Oz, Mapping Map) override {
        Map_to_Grid(Px, Py, Pz, Ox, Oy, Oz, eu_o[0], eu_o[1], eu_o[2], Map);}

    void Extract_Sol_Values(const RVector &Px, const RVector &Py, const RVector &Pz, RVector &Ugx, RVector &Ugy, RVector &Ugz, Mapping Map) override {
        Map_from_Grid(Px, Py, Pz, eu_dddt[0], eu_dddt[1], eu_dddt[2], Ugx, Ugy, Ugz, Map);}

    void Extract_Source_Values(const RVector &Px, const RVector &Py, const RVector &Pz, RVector &Ugx, RVector &Ugy, RVector &Ugz, Mapping Map) override {
        Map_from_Grid(Px, Py, Pz, eu_o[0], eu_o[1], eu_o[2], Ugx, Ugy, Ugz, Map);}

    //--- Timestepping
    void Advance_Particle_Set() override;
    void Update_Particle_Field() override;
    void Calc_Particle_RateofChange(const TensorGrid &p_d_Vals, const TensorGrid &p_o_Vals, TensorGrid &dpdt_d_Vals, TensorGrid &dpdt_o_Vals);
    void Grid_Shear_Stresses() override;
    void Grid_Turb_Shear_Stresses() override;
    void Add_Freestream_Velocity() override;
    void Calc_Grid_SpectralRatesof_Change() override;
    void Calc_Grid_FDRatesof_Change() override;

    //--- Grid utilities
    void Clear_Source_Grid() override {Parallel_Kernel(NNT) {KER_Clear(eu_o, i);}}
    void Remesh_Particle_Set() override ;
    void Reproject_Particle_Set() override ;
    void Reproject_Particle_Set_Spectral() override ;
    void Magnitude_Filtering() override ;
    void Transfer_Source_Grid()     {Parallel_Kernel(NNT) {KER_Copy(eu_o, lg_o, i);}}
    void Filter_Boundary();

    //--- Initial vorticity distribution
    void Retrieve_Grid_Positions(RVector &xc, RVector &yc, RVector &zc);
    void Set_Input_Arrays(RVector &xo, RVector &yo, RVector &zo);

    //--- Auxiliary Grid
    void Map_from_Auxiliary_Grid();
    void Interpolate_Ext_Sources(Mapping M) override ;

    //--- Grid operations
    TensorGrid *Get_pArray()  override  {return &eu_o;}
    TensorGrid *Get_gArray()  override  {return &eu_dddt;}

    //--- Grid statistics
    void Calc_Grid_Diagnostics() override;

    //--- Output grid
    void Get_Laplacian(RVector &xo, RVector &yo, RVector &zo);
    void Set_Laplacian(RVector &xo, RVector &yo, RVector &zo);
    void Get_Stretching(RVector &dgidx, RVector &dgidy, RVector &dgidz, RVector &dgjdx, RVector &dgjdy, RVector &dgjdz,RVector &dgkdx, RVector &dgkdy, RVector &dgkdz);
    void Set_Stretching(RVector &dgidx, RVector &dgidy, RVector &dgidz, RVector &dgjdx, RVector &dgjdy, RVector &dgjdz,RVector &dgkdx, RVector &dgkdy, RVector &dgkdz);
    void Generate_VTK() override;
    void Generate_VTK(const RVector &A1, const RVector &A2, const RVector &A3, const RVector &B1, const RVector &B2, const RVector &B3);
    void Generate_Plane(RVector &Uvoid) override;

    //--- Destructor
    ~VPM3D_cpu() override;
};

}

#endif // VPM3D_CPU_H
