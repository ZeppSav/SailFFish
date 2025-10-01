//-----------------------------------------------------------------------------
//------------------- Periodic Poisson Solver Functions -----------------------
//-----------------------------------------------------------------------------

#include "Periodic_Solver.h"
#include "Greens_Functions.h"

namespace SailFFish
{

//---------------------------------
//--- Bounded Poisson Periodic ----
//---------------------------------

//--- Constructors

Poisson_Periodic_1D::Poisson_Periodic_1D(Grid_Type G, Bounded_Kernel B)
{
    // Specify arrays to be allocated
    r_fg = true;
    c_in1 = true;
    c_ft_in1 = true;
    c_out_1 = true;
    c_fg = true;

    // Specify transform  and kernel type
    Grid = G;
    Spect_Kernel = B;
    Transform = DFT_C2C;
}

Poisson_Periodic_2D::Poisson_Periodic_2D(Grid_Type G, Bounded_Kernel B)
{
    // Specify arrays to be allocated
    r_fg = true;
    c_in1 = true;
    c_ft_in1 = true;
    c_out_1 = true;
    c_fg = true;

    // Specify transform  and kernel type
    Grid = G;
    Spect_Kernel = B;
    Transform = DFT_C2C;
}

Poisson_Periodic_3D::Poisson_Periodic_3D(Grid_Type G, Bounded_Kernel B)
{
    // Specify arrays to be allocated
    r_fg = true;
    c_in1 = true;
    c_fg = true;
    c_out_1 = true;
    c_ft_in1 = true;

    // Specify transform  and kernel type
    Grid = G;
    Spect_Kernel = B;
    Transform = DFT_C2C;
}

Poisson_Periodic_3DV::Poisson_Periodic_3DV(Grid_Type G, Bounded_Kernel B)
{
    // Specify arrays to be allocated
    r_fg = true;
    c_in1 = true;       c_in2 = true;       c_in3 = true;
    c_out_1 = true;     c_out_2 = true;     c_out_3 = true;
    c_ft_in1 = true;    c_ft_in2 = true;    c_ft_in3 = true;
    c_fg = true;

    // Specify transform  and kernel type
    Grid = G;
    Spect_Kernel = B;
    Transform = DFT_C2C;
}

//--- Greens function spec

void Poisson_Periodic_1D::Specify_Greens_Function()
{
    // Specify wavenumber components in x direction

    std::vector<Real> fx;
    if (Spect_Kernel==FD2){
            for (int i=0; i<NX; i++)    fx.push_back(EV_FD_Reg_Periodic(i,NX,Hx));
    }
    if (Spect_Kernel==PS){
            for (int i=0; i<NX; i++)    fx.push_back(EV_PS_Reg_Periodic(i,NX,Lx));
        // for (int i=0; i<NX; i++)    fx.push_back(EV_PS_Reg_Periodic(i,NX,2.0*Lx));
    }

    OpenMPfor
    for (int i=0; i<NX; i++) r_FG[i] = BFac/(fx[i]);
    r_FG[0] = 0.0;

    Prep_Greens_Function_C2C();
}

void Poisson_Periodic_2D::Specify_Greens_Function()
{
    // Specify wavenumber components in x,y, directions
    // Cues taken from :
    // https://stackoverflow.com/questions/35173102/fftw3-for-poisson-with-dirichlet-boundary-condition-for-all-side-of-computationa
    // https://kluge.in-chemnitz.de/opensource/poisson_pde/

    std::vector<Real> fx, fy;
    if (Spect_Kernel==FD2){
        for (int i=0; i<NX; i++)  fx.push_back(EV_FD_Reg_Periodic(i,NX,Hx));
        for (int j=0; j<NY; j++)  fy.push_back(EV_FD_Reg_Periodic(j,NY,Hy));
    }
    if (Spect_Kernel==PS){
        for (int i=0; i<NX; i++)  fx.push_back(EV_PS_Reg_Periodic(i,NX,Lx));
        for (int j=0; j<NY; j++)  fy.push_back(EV_PS_Reg_Periodic(j,NY,Ly));
    }

    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++) r_FG[GF_GID2(i,j,NX,NY)] = BFac/(fx[i]+fy[j]);
    }
    r_FG[0] = 0.0;

    Prep_Greens_Function_C2C();
}

void Poisson_Periodic_3D::Specify_Greens_Function()
{
    // Specify wavenumber components in x,y,z directions

    std::vector<Real> fx, fy, fz;
    if (Spect_Kernel==FD2){
        for (int i=0; i<NX; i++)  fx.push_back(EV_FD_Reg_Periodic(i,NX,Hx));
        for (int j=0; j<NY; j++)  fy.push_back(EV_FD_Reg_Periodic(j,NY,Hy));
        for (int k=0; k<NZ; k++)  fz.push_back(EV_FD_Reg_Periodic(k,NZ,Hz));
    }
    if (Spect_Kernel==PS){
        for (int i=0; i<NX; i++)  fx.push_back(EV_PS_Reg_Periodic(i,NX,Lx));
        for (int j=0; j<NY; j++)  fy.push_back(EV_PS_Reg_Periodic(j,NY,Ly));
        for (int k=0; k<NZ; k++)  fz.push_back(EV_PS_Reg_Periodic(k,NZ,Lz));
    }

    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++){
            for (int k=0; k<NZ; k++)    r_FG[GF_GID3(i,j,k,NX,NY,NZ)] = BFac/(fx[i]+fy[j]+fz[k]);
        }
    }
    r_FG[0] = 0.0;

    Prep_Greens_Function_C2C();
}

void Poisson_Periodic_3DV::Specify_Greens_Function()
{
    // Specify wavenumber components in x,y,z directions

    std::vector<Real> fx, fy, fz;
    if (Spect_Kernel==FD2){
        for (int i=0; i<NX; i++)  fx.push_back(EV_FD_Reg_Periodic(i,NX,Hx));
        for (int j=0; j<NY; j++)  fy.push_back(EV_FD_Reg_Periodic(j,NY,Hy));
        for (int k=0; k<NZ; k++)  fz.push_back(EV_FD_Reg_Periodic(k,NZ,Hz));
    }
    if (Spect_Kernel==PS){
        for (int i=0; i<NX; i++)  fx.push_back(EV_PS_Reg_Periodic(i,NX,Lx));
        for (int j=0; j<NY; j++)  fy.push_back(EV_PS_Reg_Periodic(j,NY,Ly));
        for (int k=0; k<NZ; k++)  fz.push_back(EV_PS_Reg_Periodic(k,NZ,Lz));
    }

    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++){
            for (int k=0; k<NZ; k++)    r_FG[GF_GID3(i,j,k,NX,NY,NZ)] = BFac/(fx[i]+fy[j]+fz[k]);
        }
    }
    r_FG[0] = 0.0;

    Prep_Greens_Function_C2C();
}

}
