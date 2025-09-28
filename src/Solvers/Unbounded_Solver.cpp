//-----------------------------------------------------------------------
//------------------- Unbounded Solver Functions ------------------------
//-----------------------------------------------------------------------

#include "Unbounded_Solver.h"
#include "Greens_Functions.h"

namespace SailFFish
{

//------------------------------------
//--- Unbounded Poisson Periodic -----
//------------------------------------

//--- Constructors

Unbounded_Solver_1D::Unbounded_Solver_1D(Grid_Type G, Unbounded_Kernel B)
{
    // Specify arrays to be allocated
    r_in1 = true;
    r_out_1 = true;
    r_fg = true;
    c_fg = true;
    c_ft_in1 = true;

    // Specify transform type
    Transform = DFT_R2C;

    // Specify grid and kernel types
    Grid = G;
    Greens_Kernel = B;
}

Unbounded_Solver_2D::Unbounded_Solver_2D(Grid_Type G, Unbounded_Kernel B)
{
    // Specify arrays to be allocated
    r_in1 = true;
    r_out_1 = true;
    r_fg = true;
    c_fg = true;
    c_ft_in1 = true;

    // Specify transform type
    Transform = DFT_R2C;

    // Specify grid and kernel types
    Grid = G;
    Greens_Kernel = B;
}

Unbounded_Solver_3D::Unbounded_Solver_3D(Grid_Type G, Unbounded_Kernel B)
{
    // Specify arrays to be allocated
    r_in1 = true;
    r_out_1 = true;
    r_fg = true;
    c_fg = true;
    c_ft_in1 = true;

    // Specify transform type
    Transform = DFT_R2C;

    // Specify grid and kernel types
    Grid = G;
    Greens_Kernel = B;
}

Unbounded_Solver_3DV::Unbounded_Solver_3DV(Grid_Type G, Unbounded_Kernel B)
{
    // Specify arrays to be allocated
    r_in1 = true;       r_in2 = true;       r_in3 = true;
    r_out_1 = true;     r_out_2 = true;     r_out_3 = true;
    c_ft_in1 = true;    c_ft_in2 = true;    c_ft_in3 = true;
    r_fg = true;
    c_fg = true;

    // Specify transform type
    Transform = DFT_R2C;

    // Specify grid and kernel types
    Grid = G;
    Greens_Kernel = B;
}

//--- Solver setup

void Unbounded_Solver_2D::Specify_Operator(OperatorType O)
{
    // Specify differential operator. If unspecified, this has the default value NONE
    Operator = O;
    if (Operator==GRAD)
    {
        c_ft_in2 = true;
        r_out_2 = true;
        c_fg_i = true;
        c_fg_j = true;
    }
    if (Operator==CURL)
    {
        c_ft_in2 = true;
        r_out_2 = true;
        c_fg_i = true;
        c_fg_j = true;

        c_dbf_1 = true; // May require a dummy  buffer
    }
    if (Operator==DIV)
    {
        c_fg_i = true;
        c_fg_j = true;
    }
    if (Operator==NABLA)
    {
        c_ft_in2 = true;
        r_out_2 = true;
        c_fg_i = true;
        c_fg_j = true;
    }
}

void Unbounded_Solver_3D::Specify_Operator(OperatorType O)
{
    // Specify differential operator. If unspecified, this has the default value NONE
    Operator = O;
    if (Operator!=NONE)
    {
        c_fg_i = true;
        c_fg_j = true;
        c_fg_k = true;
    }
    if (Operator==GRAD)
    {
        c_ft_in2 = true;
        c_ft_in3 = true;
        r_out_2 = true;
        r_out_3 = true;
    }
//    if (Operator==CURL) {}// Ignore this case for now. Not of interest
//    if (Operator==DIV)  {} // No additional arrays required
}

void Unbounded_Solver_3DV::Specify_Operator(OperatorType O)
{
    // Specify differential operator. If unspecified, this has the default value NONE
    Operator = O;
    if (Operator!=NONE){
        c_fg_i = true;
        c_fg_j = true;
        c_fg_k = true;
    }
    if (Operator==DIV)
    {
//        // No additional arrays required
//        r_out_2 = false;
//        r_out_3 = false;
    }
//    if (Operator==GRAD) // Skip this case for now. This requires nine arrays for the partial gradients of all vars.
    if (Operator==CURL){
        c_dbf_1 = true; // May require a dummy  buffer
        c_dbf_2 = true; // May require a dummy  buffer
        c_dbf_3 = true; // May require a dummy  buffer
        c_dbf_4 = true; // May require a dummy  buffer
//        c_dbf_5 = true; // May require a dummy  buffer
//        c_dbf_6 = true; // May require a dummy  buffer
    }

//    if (Operator==NABLA)        // No additional arrays necessary


}

//------------------------
//--- Greens function spec
//------------------------

typedef Real (*GreenKernel)(const Real &r, const Real &sigma);

//--- 1D

void Unbounded_Solver_1D::Specify_Greens_Function()
{
    // Specify Green's function in 1D

    //---------- Grid constants
    Real sigma = Hx/M_PI;

    // We shall assume for now that this is given by the spectrally compact result of Hejlsen in 1D:
    // Doi: 10.1016/j.aml.2018.09.012

    Real ScaleFac = BFac*Hx;

    OpenMPfor
    for (int i=0; i<NX; i++){
        Real r;
        if (2*i<NX)     r = i*Hx;
        else            r = (NX-i)*Hx;
        Real rho = r/sigma;
        r_FG[i] = ScaleFac*(-sigma/M_PI*(rho*sine_int(rho) + cos(rho)) + 0.5*Lx);       // This has not been checked yet....
    }

    // Convert Greens function to frequency domain
    Prep_Greens_Function_R2C();
}

void Unbounded_Solver_2D::Specify_Greens_Function()
{
    // Specify wavenumber components in x direction

    //---------- Grid constants
    Real H = std::max(Hx,Hy);
    Real sigma;

    //---------- Select green's function

    GreenKernel Gk, G0;
    switch (Greens_Kernel)
    {
        case (HEJ_S0):  {sigma = H/M_PI;    Gk = &Kernel_HEJ2_S0;   G0 = &Kernel_HEJ2_S0;   break;}
        case (HEJ_G2):  {sigma = 2.0*H;     Gk = &Kernel_HEJ2_G2;   G0 = &Kernel_HEJ2_G2_0; break;}
        case (HEJ_G4):  {sigma = 2.0*H;     Gk = &Kernel_HEJ2_G4;   G0 = &Kernel_HEJ2_G4_0; break;}
        case (HEJ_G6):  {sigma = 2.0*H;     Gk = &Kernel_HEJ2_G6;   G0 = &Kernel_HEJ2_G6_0; break;}
        case (HEJ_G8):  {sigma = 2.0*H;     Gk = &Kernel_HEJ2_G8;   G0 = &Kernel_HEJ2_G8_0; break;}
        case (HEJ_G10): {sigma = 2.0*H;     Gk = &Kernel_HEJ2_G10;  G0 = &Kernel_HEJ2_G10_0; break;}
        default:   break;
    }

    Real ScaleFac = BFac*Hx*Hy;

    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++){
            Real x,y;
            if (2*i<NX)     x = i*Hx;
            else            x = (NX-i)*Hx;
            if (2*j<NY)     y = j*Hy;
            else            y = (NY-j)*Hy;
            Real r = sqrt(x*x + y*y);
            // r_FG[GID(i,j,NX,NY)] = ScaleFac*Gk(r,sigma);       // FFTW,CUFFT
            r_FG[GID(j,i,NY,NX)] = ScaleFac*Gk(r,sigma);       // VKFFT
        }
    }

    //-------------------------------------------------

    // Specify value at zero displacement (origin)
    r_FG[0] = ScaleFac*G0(0,sigma);

    // Convert Greens function to frequency domain
    Prep_Greens_Function_R2C();
}

void Unbounded_Solver_3D::Specify_Greens_Function()
{
    // Specify wavenumber components in x direction

    //---------- Grid constants
    Real H = std::max(std::max(Hx,Hy),Hz);
    Real sigma;

    //---------- Select green's function

    GreenKernel Gk, G0;
    switch (Greens_Kernel)
    {
        case (HEJ_S0):  {sigma = H/M_PI;    Gk = &Kernel_HEJ3_S0;   G0 = &Kernel_HEJ3_S0_O; break;}
        case (HEJ_G2):  {sigma = 2.0*H;     Gk = &Kernel_HEJ3_G2;   G0 = &Kernel_HEJ3_G2_0; break;}
        case (HEJ_G4):  {sigma = 2.0*H;     Gk = &Kernel_HEJ3_G4;   G0 = &Kernel_HEJ3_G4_0; break;}
        case (HEJ_G6):  {sigma = 2.0*H;     Gk = &Kernel_HEJ3_G6;   G0 = &Kernel_HEJ3_G6_0; break;}
        case (HEJ_G8):  {sigma = 2.0*H;     Gk = &Kernel_HEJ3_G8;   G0 = &Kernel_HEJ3_G8_0; break;}
        case (HEJ_G10): {sigma = 2.0*H;     Gk = &Kernel_HEJ3_G10;  G0 = &Kernel_HEJ3_G10_0; break;}
        default:   break;
    }

    Real ScaleFac = BFac*Hx*Hy*Hz;

    OpenMPfor
    for (int i=0; i<NX; i++){
        Real x;
        if (2*i<NX)     x = i*Hx;
        else            x = (NX-i)*Hx;
        for (int j=0; j<NY; j++){
            Real y;
            if (2*j<NY)     y = j*Hy;
            else            y = (NY-j)*Hy;
            for (int k=0; k<NZ; k++){
                Real z;
                if (2*k<NZ)     z = k*Hz;
                else            z = (NZ-k)*Hz;
                Real r = sqrt(x*x + y*y + z*z);
                r_FG[GID(dim3(i,j,k),dim3(NX,NY,NZ))] = ScaleFac*Gk(r,sigma);
            }
        }
    }

    //-------------------------------------------------

    // Specify value at zero displacement (origin)
    r_FG[0] = ScaleFac*G0(0,sigma);

    // Convert Greens function to frequency domain
    Prep_Greens_Function_R2C();
}

void Unbounded_Solver_3DV::Specify_Greens_Function()
{
    // This is an exact duplicate of the function for Unbounded_Solver_3D

    //---------- Grid constants
    Real H = std::max(std::max(Hx,Hy),Hz);
    Real sigma;

    //---------- Select green's function

    GreenKernel Gk, G0;
    switch (Greens_Kernel)
    {
        case (HEJ_S0):  {sigma = H/M_PI;    Gk = &Kernel_HEJ3_S0;   G0 = &Kernel_HEJ3_S0_O; break;}
        case (HEJ_G2):  {sigma = 2.0*H;     Gk = &Kernel_HEJ3_G2;   G0 = &Kernel_HEJ3_G2_0; break;}
        case (HEJ_G4):  {sigma = 2.0*H;     Gk = &Kernel_HEJ3_G4;   G0 = &Kernel_HEJ3_G4_0; break;}
        case (HEJ_G6):  {sigma = 2.0*H;     Gk = &Kernel_HEJ3_G6;   G0 = &Kernel_HEJ3_G6_0; break;}
        case (HEJ_G8):  {sigma = 2.0*H;     Gk = &Kernel_HEJ3_G8;   G0 = &Kernel_HEJ3_G8_0; break;}
        case (HEJ_G10): {sigma = 2.0*H;     Gk = &Kernel_HEJ3_G10;  G0 = &Kernel_HEJ3_G10_0; break;}
        default:   break;
    }

    Real ScaleFac = BFac*Hx*Hy*Hz;

    OpenMPfor
    for (int i=0; i<NX; i++){
        Real x;
        if (2*i<NX)     x = i*Hx;
        else            x = (NX-i)*Hx;
        for (int j=0; j<NY; j++){
            Real y;
            if (2*j<NY)     y = j*Hy;
            else            y = (NY-j)*Hy;
            for (int k=0; k<NZ; k++){
                Real z;
                if (2*k<NZ)     z = k*Hz;
                else            z = (NZ-k)*Hz;
                Real r = sqrt(x*x + y*y + z*z);
                r_FG[GID(dim3(i,j,k),dim3(NX,NY,NZ))] = ScaleFac*Gk(r,sigma);
            }
        }
    }

    //-------------------------------------------------

    // Specify value at zero displacement (origin)
    r_FG[0] = ScaleFac*G0(0,sigma);

    // Convert Greens function to frequency domain
    Prep_Greens_Function_R2C();
}

}
