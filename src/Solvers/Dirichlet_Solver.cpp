//-----------------------------------------------------------------------------
//------------------- Poisson Dirichlet Solver Functions ----------------------
//-----------------------------------------------------------------------------

#include "Dirichlet_Solver.h"
#include "Greens_Functions.h"

namespace SailFFish
{

//--- Constructors

Poisson_Dirichlet_1D::Poisson_Dirichlet_1D(Grid_Type G, Bounded_Kernel B)
{
    // Specify arrays to be allocated
    r_in1 = true;
    r_ft_in1 = true;
    r_out_1 = true;
    r_fg = true;

    // Specify transform  and kernel type
    Grid = G;
    Spect_Kernel = B;
    if (Grid==REGULAR)    Transform = DST1;
    if (Grid==STAGGERED)  Transform = DST2;
}

Poisson_Dirichlet_2D::Poisson_Dirichlet_2D(Grid_Type G, Bounded_Kernel B)
{
    // Specify arrays to be allocated
    r_in1 = true;
    r_ft_in1 = true;
    r_out_1 = true;
    r_fg = true;

    // Specify transform  and kernel type
    Grid = G;
    Spect_Kernel = B;
    if (Grid==REGULAR)    Transform = DST1;
    if (Grid==STAGGERED)  Transform = DST2;
}

Poisson_Dirichlet_3D::Poisson_Dirichlet_3D(Grid_Type G, Bounded_Kernel B)
{
    // Specify arrays to be allocated
    r_in1 = true;
    r_ft_in1 = true;
    r_out_1 = true;
    r_fg = true;

    // Specify transform  and kernel type
    Grid = G;
    Spect_Kernel = B;
    if (Grid==REGULAR)    Transform = DST1;
    if (Grid==STAGGERED)  Transform = DST2;
}

Poisson_Dirichlet_3DV::Poisson_Dirichlet_3DV(Grid_Type G, Bounded_Kernel B)
{
    // Specify arrays to be allocated
    r_in1 = true;       r_in2 = true;       r_in3 = true;
    r_out_1 = true;     r_out_2 = true;     r_out_3 = true;
    r_ft_in1 = true;    r_ft_in2 = true;    r_ft_in3 = true;
    r_fg = true;

    // Specify transform  and kernel type
    Grid = G;
    Spect_Kernel = B;
    if (Grid==REGULAR)    Transform = DST1;
    if (Grid==STAGGERED)  Transform = DST2;
}

//--- Specify boundary conditions

void Poisson_Dirichlet_1D::Set_BC(Real AX, Real BX)
{
    // Specifies boundary conditions for the 2D case. This modifies the RHS vector in order to provide the desired values.
    // Nb: The values at the corners of the grid are not used due to the nature of the stencil.

    if (Spect_Kernel==PS)
    {
        cout << "In order to specify (nonzero) boundary conditions for a bounded Poisson solver, the finite difference kernel (SailFFish::FD2) option be specified." << endl;
        cout << "The solution shall be calculated assuming zero boundary conditions." << endl;
        return;
    }

    Real InvxFac;
    if (Grid==REGULAR)    InvxFac = 1.0/(Hx*Hx);
    if (Grid==STAGGERED)  InvxFac = 2.0/(Hx*Hx);

    // Modify f array
    r_Input1[0]     -=  AX*InvxFac;
    r_Input1[NX-1]  -=  BX*InvxFac;
}

void Poisson_Dirichlet_2D::Set_BC(RVector AX, RVector BX, RVector AY, RVector BY)
{
    // Specifies boundary conditions for the 2D case. This modifies the RHS vector in order to provide the desired values.
    // Nb: The values at the corners of the grid are not used due to the nature of the stencil.

    if (Spect_Kernel==PS)
    {
        cout << "In order to specify (nonzero) boundary conditions for a bounded Poisson solver, the finite difference kernel (SailFFish::FD2) option be specified." << endl;
        cout << "The solution shall be calculated assuming zero boundary conditions." << endl;
        return;
    }

    Real InvxFac, InvyFac;
    if (Grid==REGULAR){
        InvxFac = 1.0/(Hx*Hx);
        InvyFac = 1.0/(Hy*Hy);
    }
    if (Grid==STAGGERED){
        InvxFac = 2.0/(Hx*Hx);
        InvyFac = 2.0/(Hy*Hy);
    }

    // X Faces
    OpenMPfor
    for (int j=0; j<NY; j++){
        r_Input1[GID(0,j,NX,NY)]     -=  AX[j]*InvxFac;
        r_Input1[GID(NX-1,j,NX,NY)]  -=  BX[j]*InvxFac;
    }

    // Y Faces
    OpenMPfor
    for (int i=0; i<NX; i++){
        r_Input1[GID(i,0,NX,NY)]     -=  AY[i]*InvyFac;
        r_Input1[GID(i,NY-1,NX,NY)]  -=  BY[i]*InvyFac;
    }
}

void Poisson_Dirichlet_3D::Set_BC(RVector AX, RVector BX, RVector AY, RVector BY, RVector AZ, RVector BZ)
{
    // Specifies boundary conditions for the 2D case. This modifies the RHS vector in order to provide the desired values.
    // Nb: The values at the corners of the grid are not used due to the nature of the stencil.

    if (Spect_Kernel==PS)
    {
        cout << "In order to specify (nonzero) boundary conditions for a bounded Poisson solver, the finite difference kernel (SailFFish::FD2) option be specified." << endl;
        cout << "The solution shall be calculated assuming zero boundary conditions." << endl;
        return;
    }

    //--- Modify f array

    Real InvxFac, InvyFac, InvzFac;
    if (Grid==REGULAR){
        InvxFac = 1.0/(Hx*Hx);
        InvyFac = 1.0/(Hy*Hy);
        InvzFac = 1.0/(Hz*Hz);
    }
    if (Grid==STAGGERED){
        InvxFac = 2.0/(Hx*Hx);
        InvyFac = 2.0/(Hy*Hy);
        InvzFac = 2.0/(Hz*Hz);
    }

    // Faces X = const
    OpenMPfor
    for (int j=0; j<NY; j++){
        for (int k=0; k<NZ; k++){
            r_Input1[GID(0,j,k,NX,NY,NZ)]    -=  AX[GID(j,k,NY,NZ)]*InvxFac;
            r_Input1[GID(NX-1,j,k,NX,NY,NZ)] -=  BX[GID(j,k,NY,NZ)]*InvxFac;
        }
    }

    // Faces Y = const
    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int k=0; k<NZ; k++){
            r_Input1[GID(i,0,k,NX,NY,NZ)]    -=  AY[GID(i,k,NX,NZ)]*InvyFac;
            r_Input1[GID(i,NY-1,k,NX,NY,NZ)] -=  BY[GID(i,k,NX,NZ)]*InvyFac;
        }
    }

    // Faces Z = const
    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++){
            r_Input1[GID(i,j,0,NX,NY,NZ)]    -=  AZ[GID(i,j,NX,NY)]*InvzFac;
            r_Input1[GID(i,j,NZ-1,NX,NY,NZ)] -=  BZ[GID(i,j,NX,NY)]*InvzFac;
        }
    }
}

void Poisson_Dirichlet_3DV::Set_BC( RVector AX1, RVector BX1, RVector AX2, RVector BX2, RVector AX3, RVector BX3,
                                    RVector AY1, RVector BY1, RVector AY2, RVector BY2, RVector AY3, RVector BY3,
                                    RVector AZ1, RVector BZ1, RVector AZ2, RVector BZ2, RVector AZ3, RVector BZ3)
{
    // Specifies boundary conditions for the 3D Vector field case. This modifies the RHS vector in order to provide the desired values.
    // Nb: The values at the corners of the grid are not used due to the nature of the stencil.

    if (Spect_Kernel==PS)
    {
        cout << "In order to specify (nonzero) boundary conditions for a bounded Poisson solver, the finite difference kernel (SailFFish::FD2) option be specified." << endl;
        cout << "The solution shall be calculated assuming zero boundary conditions." << endl;
        return;
    }

    //--- Modify f array

    Real InvxFac, InvyFac, InvzFac;
    if (Grid==REGULAR){
        InvxFac = 1.0/(Hx*Hx);
        InvyFac = 1.0/(Hy*Hy);
        InvzFac = 1.0/(Hz*Hz);
    }
    if (Grid==STAGGERED){
        InvxFac = 2.0/(Hx*Hx);
        InvyFac = 2.0/(Hy*Hy);
        InvzFac = 2.0/(Hz*Hz);
    }

    // Faces X = const
    OpenMPfor
    for (int j=0; j<NY; j++){
        for (int k=0; k<NZ; k++){
            int GL = GID(0,j,k,NX,NY,NZ);
            int GR = GID(NX-1,j,k,NX,NY,NZ);
            int G = GID(j,k,gNY,gNZ);
            r_Input1[GL] -= AX1[G]*InvxFac;
            r_Input1[GR] -= BX1[G]*InvxFac;
            r_Input2[GL] -= AX2[G]*InvxFac;
            r_Input2[GR] -= BX2[G]*InvxFac;
            r_Input3[GL] -= AX3[G]*InvxFac;
            r_Input3[GR] -= BX3[G]*InvxFac;
        }
    }

    // Faces Y = const
    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int k=0; k<NZ; k++){
            int GL = GID(i,0,k,NX,NY,NZ);
            int GR = GID(i,NY-1,k,NX,NY,NZ);
            int G = GID(i,k,gNX,gNZ);
            r_Input1[GL] -= AY1[G]*InvyFac;
            r_Input1[GR] -= BY1[G]*InvyFac;
            r_Input2[GL] -= AY2[G]*InvyFac;
            r_Input2[GR] -= BY2[G]*InvyFac;
            r_Input3[GL] -= AY3[G]*InvyFac;
            r_Input3[GR] -= BY3[G]*InvyFac;
        }
    }

    // Faces Z = const
    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++){
            int GL = GID(i,j,0,NX,NY,NZ);
            int GR = GID(i,j,NZ-1,NX,NY,NZ);
            int G = GID(i,j,gNX,gNY);
            r_Input1[GL] -= AZ1[G]*InvzFac;
            r_Input1[GR] -= BZ1[G]*InvzFac;
            r_Input2[GL] -= AZ2[G]*InvzFac;
            r_Input2[GR] -= BZ2[G]*InvzFac;
            r_Input3[GL] -= AZ3[G]*InvzFac;
            r_Input3[GR] -= BZ3[G]*InvzFac;
        }
    }
}

//--- Greens function spec

void Poisson_Dirichlet_1D::Specify_Greens_Function()
{
    // Specify wavenumber components in x direction

    std::vector<Real> fx;
    if (Grid==REGULAR)
    {
        if (Spect_Kernel==FD2)      for (int i=0; i<NX; i++)  fx.push_back(EV_FD_Reg_Dirichlet(i,NX,Hx));
        if (Spect_Kernel==PS)       for (int i=0; i<NX; i++)  fx.push_back(EV_PS_Reg_Dirichlet(i,Lx));
    }
    if (Grid==STAGGERED)
    {
        if (Spect_Kernel==FD2)      for (int i=0; i<NX; i++)  fx.push_back(EV_FD_Stag_Dirichlet(i,NX,Hx));
        if (Spect_Kernel==PS)       for (int i=0; i<NX; i++)  fx.push_back(EV_PS_Stag_Dirichlet(i,Lx));
    }

    OpenMPfor
    for (int i=0; i<NX; i++)    r_FG[i] = BFac/(fx[i]);
}

void Poisson_Dirichlet_2D::Specify_Greens_Function()
{
    // Specify wavenumber components in x,y, directions
    // Cues taken from :
    // https://stackoverflow.com/questions/35173102/fftw3-for-poisson-with-dirichlet-boundary-condition-for-all-side-of-computationa

    std::vector<Real> fx, fy;
    if (Grid==REGULAR)
    {
        if (Spect_Kernel==FD2){
            for (int i=0; i<NX; i++)  fx.push_back(EV_FD_Reg_Dirichlet(i,NX,Hx));
            for (int j=0; j<NY; j++)  fy.push_back(EV_FD_Reg_Dirichlet(j,NY,Hy));
        }
        if (Spect_Kernel==PS){
            for (int i=0; i<NX; i++)  fx.push_back(EV_PS_Reg_Dirichlet(i,Lx));
            for (int j=0; j<NY; j++)  fy.push_back(EV_PS_Reg_Dirichlet(j,Ly));
        }
    }
    if (Grid==STAGGERED)
    {
        if (Spect_Kernel==FD2){
            for (int i=0; i<NX; i++)  fx.push_back(EV_FD_Stag_Dirichlet(i,NX,Hx));
            for (int j=0; j<NY; j++)  fy.push_back(EV_FD_Stag_Dirichlet(j,NY,Hy));
        }
        if (Spect_Kernel==PS){
            for (int i=0; i<NX; i++)  fx.push_back(EV_PS_Stag_Dirichlet(i,Lx));
            for (int j=0; j<NY; j++)  fy.push_back(EV_PS_Stag_Dirichlet(j,Ly));
        }
    }

    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++) r_FG[i*NY+j] = BFac/(fx[i]+fy[j]);
    }
}

void Poisson_Dirichlet_3D::Specify_Greens_Function()
{
    // Specify wavenumber components in x,y,z directions

    std::vector<Real> fx, fy, fz;
    if (Grid==REGULAR)
    {
        if (Spect_Kernel==FD2){
            for (int i=0; i<NX; i++)  fx.push_back(EV_FD_Reg_Dirichlet(i,NX,Hx));
            for (int j=0; j<NY; j++)  fy.push_back(EV_FD_Reg_Dirichlet(j,NY,Hy));
            for (int k=0; k<NZ; k++)  fz.push_back(EV_FD_Reg_Dirichlet(k,NZ,Hz));
        }
        if (Spect_Kernel==PS){
            for (int i=0; i<NX; i++)  fx.push_back(EV_PS_Reg_Dirichlet(i,Lx));
            for (int j=0; j<NY; j++)  fy.push_back(EV_PS_Reg_Dirichlet(j,Ly));
            for (int k=0; k<NZ; k++)  fz.push_back(EV_PS_Reg_Dirichlet(k,Lz));
        }
    }
    if (Grid==STAGGERED)
    {
        if (Spect_Kernel==FD2){
            for (int i=0; i<NX; i++)  fx.push_back(EV_FD_Stag_Dirichlet(i,NX,Hx));
            for (int j=0; j<NY; j++)  fy.push_back(EV_FD_Stag_Dirichlet(j,NY,Hy));
            for (int k=0; k<NZ; k++)  fz.push_back(EV_FD_Stag_Dirichlet(k,NZ,Hz));
        }
        if (Spect_Kernel==PS){
            for (int i=0; i<NX; i++)  fx.push_back(EV_PS_Stag_Dirichlet(i,Lx));
            for (int j=0; j<NY; j++)  fy.push_back(EV_PS_Stag_Dirichlet(j,Ly));
            for (int k=0; k<NZ; k++)  fz.push_back(EV_PS_Stag_Dirichlet(k,Lz));
        }
    }

    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++){
            for (int k=0; k<NZ; k++) r_FG[GID(i,j,k,NX,NY,NZ)] = BFac/(fx[i]+fy[j]+fz[k]);
        }
    }
}

void Poisson_Dirichlet_3DV::Specify_Greens_Function()
{
    // Specify wavenumber components in x,y, directions

    std::vector<Real> fx, fy, fz;
    if (Grid==REGULAR)
    {
        if (Spect_Kernel==FD2){
            for (int i=0; i<NX; i++)  fx.push_back(EV_FD_Reg_Dirichlet(i,NX,Hx));
            for (int j=0; j<NY; j++)  fy.push_back(EV_FD_Reg_Dirichlet(j,NY,Hy));
            for (int k=0; k<NZ; k++)  fz.push_back(EV_FD_Reg_Dirichlet(k,NZ,Hz));
        }
        if (Spect_Kernel==PS){
            for (int i=0; i<NX; i++)  fx.push_back(EV_PS_Reg_Dirichlet(i,Lx));
            for (int j=0; j<NY; j++)  fy.push_back(EV_PS_Reg_Dirichlet(j,Ly));
            for (int k=0; k<NZ; k++)  fz.push_back(EV_PS_Reg_Dirichlet(k,Lz));
        }
    }
    if (Grid==STAGGERED)
    {
        if (Spect_Kernel==FD2){
            for (int i=0; i<NX; i++)  fx.push_back(EV_FD_Stag_Dirichlet(i,NX,Hx));
            for (int j=0; j<NY; j++)  fy.push_back(EV_FD_Stag_Dirichlet(j,NY,Hy));
            for (int k=0; k<NZ; k++)  fz.push_back(EV_FD_Stag_Dirichlet(k,NZ,Hz));
        }
        if (Spect_Kernel==PS){
            for (int i=0; i<NX; i++)  fx.push_back(EV_PS_Stag_Dirichlet(i,Lx));
            for (int j=0; j<NY; j++)  fy.push_back(EV_PS_Stag_Dirichlet(j,Ly));
            for (int k=0; k<NZ; k++)  fz.push_back(EV_PS_Stag_Dirichlet(k,Lz));
        }
    }

    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++){
            for (int k=0; k<NZ; k++) r_FG[GID(i,j,k,NX,NY,NZ)] = BFac/(fx[i]+fy[j]+fz[k]);
        }
    }
}

}
