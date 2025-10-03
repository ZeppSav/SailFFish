/****************************************************************************
    SailFFish Library
    Copyright (C) 2025 Joseph Saverin j.saverin@tu-berlin.de

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

    -> These are the kernels used during execution of the VPM solver

*****************************************************************************/

#include "../Solvers/Solver_Base.h"

//---- Add eigen library
// #include <../../eigen/Eigen/Eigen>
#include <Eigen/Eigen>               // Dont know... but this isn't working with CMake
#ifdef SinglePrec
    //--------Single Precision----------
    typedef Eigen::MatrixXf             Matrix;
    typedef Eigen::VectorXf             Vector;
    typedef Eigen::Vector3f             Vector3;
#endif
#ifdef DoublePrec
    //--------Double Precision----------
    typedef Eigen::MatrixXd             Matrix;
    typedef Eigen::VectorXd             Vector;
    typedef Eigen::Vector3d             Vector3;
#endif

namespace SailFFish
{
//--- Grid types

typedef     std::vector<RVector>            TensorGrid;
typedef     std::vector<std::vector<int>>   TensorIntGrid;

inline void ConstructTensorGrid(TensorGrid &A, int NN, int Comp)
{
    RVector RD = RVector(NN,0.0);
    for (int i=0; i<Comp; i++) A.push_back(RD);
}

inline void ConstructTensorGrid(TensorGrid &A, const RVector &A1, const RVector &A2, const RVector &A3)
{
    A.push_back(A1);
    A.push_back(A2);
    A.push_back(A3);
}

inline void DestructTensorGrid(TensorGrid &A) {A.clear();}

inline void ConstructTensorIntGrid(TensorIntGrid &A, int NN, int Comp)
{
    std::vector<int> RD = std::vector<int>(NN,0);
    for (int i=0; i<Comp; i++) A.push_back(RD);
}

inline void KER_Clear(TensorGrid &p_Vals, const int &id)
{
    // Reset array values
    p_Vals[0][id] = 0;
    p_Vals[1][id] = 0;
    p_Vals[2][id] = 0;
}

inline void KER_Copy(const TensorGrid &Src, TensorGrid &Dest, const int &id)
{
    // Update the particle property with Eulerian forward (either position or strength)
    Dest[0][id] = Src[0][id];
    Dest[1][id] = Src[1][id];
    Dest[2][id] = Src[2][id];
}

inline void KER_Copy(const TensorGrid &Src1, const TensorGrid &Src2, TensorGrid &Dest1, TensorGrid &Dest2, const int &id)
{
    // Update the particle property with Eulerian forward (either position or strength)
    Dest1[0][id] = Src1[0][id];
    Dest1[1][id] = Src1[1][id];
    Dest1[2][id] = Src1[2][id];
    Dest2[0][id] = Src2[0][id];
    Dest2[1][id] = Src2[1][id];
    Dest2[2][id] = Src2[2][id];
}

enum Mapping            {M2, M3, M4, M4D, M6D, D2, D3};

inline void mapD2(const Real &x, Real &u){
    if (x<=0.5)         u = 1.0-x*x;
    else if (x<1.5)     u = 0.5*(1.0-x)*(2.0-x);
    else                u = 0.0;
}

inline void mapD3(const Real &x, Real &u){
    if (x<1.0)          u = (1.0-x)*(2.0-x)*(3.0-x)/6.0;
    else if (x<2.0)     u = 0.5*(1.0-x*x)*(2.0-x);
    else                u = 0.0;
}

inline void mapM2(const Real &x, Real &u){
    if (x<=1.0)         u = 1.0-x;
    else                u = 0.0;
}

inline void mapM3(const Real &x, Real &u){
    if (x<0.5)          u = 0.75-x*x;
    else if (x<1.5)     u = 0.5*(1.5-x)*(1.5-x);
    else                u = 0.;
}

inline void mapM4(const Real &x, Real &u){
    Real tmx = (2.-x);
    Real omx = (1.-x);
    if (x<1.)          u = tmx*tmx*tmx/6.0 - 4.0*omx*omx*omx/6.0;
    else if (x<2.0)     u = tmx*tmx*tmx/6.0;
    else                u = 0.0;
}

inline void mapM4D(const Real &x, Real &u){
    if (x<1.0)          u = 0.5*(2.0-5.0*x*x+3.0*x*x*x);
    else if (x<2.0)     u = 0.5*(1.0-x)*(2.0-x)*(2.0-x);
    else                u = 0.0;
}

inline void mapM6D(const Real &x, Real &u){
    constexpr Real c1 = -1. / 88.;
    constexpr Real c2 = 1. / 176.;
    constexpr Real c3 = -3. / 176.;
    Real x2 = x*x;
    if (x<1.0)          {u = c1*(x-1.)*(60.*x2*x2-87.*x*x2-87.*x2+88.*x+88.);   }
    else if (x<2.0)     {u = c2*(x-1.)*(x-2.)*(60.*x*x2-261.*x2+257.*x+68.);    }
    else if (x<3.0)     {u = c3*(x-2.)*(4.*x2-17.*x+12.)*(x-3.)*(x-3.);         }
    else                u = 0.0;
}

//--- Diagnostics Kernels

inline RVector DataBlock_Contraction3(const TensorGrid Src, int NB, Real Scale = 1.0)
{
    // Contracts the blocks
    RVector U = RVector(3,0.0);
    for (int i=0; i<NB; i++){
        U[0] += Src[0][i];
        U[1] += Src[1][i];
        U[2] += Src[2][i];
    }
    U[0] *= Scale;
    U[1] *= Scale;
    U[2] *= Scale;
    return U;
}

inline void KER_Diagnostics(const TensorGrid &p_d_Vals,
                            const TensorGrid &p_o_Vals,
                            const TensorGrid &dpdt_Vals,
                            const TensorGrid &NablaU,
                            TensorGrid &Circ,
                            TensorGrid &LinImp,
                            TensorGrid &AngImp,
                            RVector &KinEng1,
                            RVector &KinEng2,
                            RVector &Enstrophy,
                            RVector &Helicity,
                            RVector &OmMagMax,
                            RVector &UMagMax,
                            RVector &SMax,
                            const int &id_src, const int &id_dest)
{
    // This is a single kernel for evaluating the integral properties of the field
    // Maxima are also evaluated

    //--- Specify values shorthand for simplicity/checking
    Real Px = p_d_Vals[0][id_src], Py = p_d_Vals[1][id_src], Pz = p_d_Vals[2][id_src];
    Real Ox = p_o_Vals[0][id_src], Oy = p_o_Vals[1][id_src], Oz = p_o_Vals[2][id_src];
    Real Ux = dpdt_Vals[0][id_src], Uy = dpdt_Vals[1][id_src], Uz = dpdt_Vals[2][id_src];

    //--- Linear diagnostics
    Circ[0][id_dest] += Ox;
    Circ[1][id_dest] += Oy;
    Circ[2][id_dest] += Oz;
    Real cx = Py*Oz - Pz*Oy;
    Real cy = Pz*Ox - Px*Oz;
    Real cz = Px*Oy - Py*Ox;
    LinImp[0][id_dest] += cx*0.5;
    LinImp[1][id_dest] += cy*0.5;
    LinImp[2][id_dest] += cz*0.5;
    AngImp[0][id_dest] += (Py*cz - Pz*cy)/3.0;
    AngImp[1][id_dest] += (Pz*cx - Px*cz)/3.0;
    AngImp[2][id_dest] += (Px*cy - Py*cx)/3.0;

    //--- Quadratic diagnostics
    Real UMag = sqrt(Ux*Ux + Uy*Uy + Uz*Uz);
    Real OmMag = sqrt(Ox*Ox + Oy*Oy + Oz*Oz);
    KinEng1[id_dest] +=  0.5*UMag*UMag;             // Approach 1: Winckelmans
    KinEng2[id_dest] +=  (Ux*cx + Uy*cy + Uz*cz);   // Approach 2: Liska
    Enstrophy[id_dest] += OmMag*OmMag;
    Helicity[id_dest] +=  (Ox*Ux + Oy*Uy + Oz*Uz);

    //--- Maxima
    if (OmMagMax[id_dest] < OmMag)  OmMagMax[id_dest] = OmMag;      // Vorticity
    if (UMagMax[id_dest] < UMag)  UMagMax[id_dest] = UMag;          // Velocity

    // Shear components: Private Correpondence with Gregoire Winckelmans
    // Compute the 1-norm of the deformation tensor, It bounds the 2-norm...
    // Nb: NablaU = [dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz];
    Real dsu = fabs(NablaU[0][id_src]) + 0.5*fabs(NablaU[1][id_src]+NablaU[3][id_src]) + 0.5*fabs(NablaU[2][id_src]+NablaU[6][id_src]);
    Real dsv = 0.5*fabs(NablaU[3][id_src]+NablaU[1][id_src]) + fabs(NablaU[4][id_src]) + 0.5*fabs(NablaU[5][id_src]+NablaU[7][id_src]);
    Real dsw = 0.5*fabs(NablaU[6][id_src]+NablaU[2][id_src]) + 0.5*fabs(NablaU[7][id_src]+NablaU[5][id_src]) + fabs(NablaU[8][id_src]);
    Real dsm = std::max(dsu,std::max(dsv,dsw));
    if (SMax[id_dest] < dsm)  SMax[id_dest] = dsm;          // Stretching

    // Hack for visualisation
    //    dpdt_Vals[0][id_src] = 0.5*UMag*UMag;
    //    dpdt_Vals[1][id_src] = OmMag*OmMag;
}


inline void KER_Diagnostics(const TensorGrid &p_d_Vals,
                            const TensorGrid &p_o_Vals,
                            const TensorGrid &dpdt_Vals,
                            TensorGrid &Circ,
                            TensorGrid &LinImp,
                            TensorGrid &AngImp,
                            RVector &KinEng1,
                            RVector &KinEng2,
                            RVector &Enstrophy,
                            RVector &Helicity,
                            RVector &OmMagMax,
                            RVector &UMagMax,
                            RVector &SMax,
                            const int &id_src, const int &id_dest)
{
    // This is a single kernel for evaluating the integral properties of the field
    // Maxima are also evaluated

    //--- Specify values shorthand for simplicity/checking
    Real Px = p_d_Vals[0][id_src], Py = p_d_Vals[1][id_src], Pz = p_d_Vals[2][id_src];
    Real Ox = p_o_Vals[0][id_src], Oy = p_o_Vals[1][id_src], Oz = p_o_Vals[2][id_src];
    Real Ux = dpdt_Vals[0][id_src], Uy = dpdt_Vals[1][id_src], Uz = dpdt_Vals[2][id_src];

    //--- Linear diagnostics
    Circ[0][id_dest] += Ox;
    Circ[1][id_dest] += Oy;
    Circ[2][id_dest] += Oz;
    Real cx = Py*Oz - Pz*Oy;
    Real cy = Pz*Ox - Px*Oz;
    Real cz = Px*Oy - Py*Ox;
    LinImp[0][id_dest] += cx*0.5;
    LinImp[1][id_dest] += cy*0.5;
    LinImp[2][id_dest] += cz*0.5;
    AngImp[0][id_dest] += (Py*cz - Pz*cy)/3.0;
    AngImp[1][id_dest] += (Pz*cx - Px*cz)/3.0;
    AngImp[2][id_dest] += (Px*cy - Py*cx)/3.0;

    //--- Quadratic diagnostics
    Real UMag = sqrt(Ux*Ux + Uy*Uy + Uz*Uz);
    Real OmMag = sqrt(Ox*Ox + Oy*Oy + Oz*Oz);
    KinEng1[id_dest] +=  0.5*UMag*UMag;             // Approach 1: Winckelmans
    KinEng2[id_dest] +=  (Ux*cx + Uy*cy + Uz*cz);   // Approach 2: Liska
    Enstrophy[id_dest] += OmMag*OmMag;
    Helicity[id_dest] +=  (Ox*Ux + Oy*Uy + Oz*Uz);

    //--- Maxima
    if (OmMagMax[id_dest] < OmMag)  OmMagMax[id_dest] = OmMag;      // Vorticity
    if (UMagMax[id_dest] < UMag)  UMagMax[id_dest] = UMag;          // Velocity

    // Shear components: Private Correpondence with Gregoire Winckelmans
    // Compute the 1-norm of the deformation tensor, It bounds the 2-norm...
    // Nb: NablaU = [dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz];
    //    Real dsu = fabs(NablaU[0][id_src]) + 0.5*fabs(NablaU[1][id_src]+NablaU[3][id_src]) + 0.5*fabs(NablaU[2][id_src]+NablaU[6][id_src]);
    //    Real dsv = 0.5*fabs(NablaU[3][id_src]+NablaU[1][id_src]) + fabs(NablaU[4][id_src]) + 0.5*fabs(NablaU[5][id_src]+NablaU[7][id_src]);
    //    Real dsw = 0.5*fabs(NablaU[6][id_src]+NablaU[2][id_src]) + 0.5*fabs(NablaU[7][id_src]+NablaU[5][id_src]) + fabs(NablaU[8][id_src]);
    //    Real dsm = std::max(dsu,std::max(dsv,dsw));
    //    if (SMax[id_dest] < dsm)  SMax[id_dest] = dsm;          // Stretching

    // Hack for visualisation
    //    dpdt_Vals[0][id_src] = 0.5*UMag*UMag;
    //    dpdt_Vals[1][id_src] = OmMag*OmMag;
}

//--- Remeshing kernel

inline void KER_Remesh(TensorGrid &p_Vals, TensorGrid &o_Vals, const TensorGrid &g_Vals, const int &id)
{
    // This kernel is called during remeshing basically we carry out three operations.
    // i) Set the particle displacement to zero
    p_Vals[0][id] = 0.0;
    p_Vals[1][id] = 0.0;
    p_Vals[2][id] = 0.0;
    // ii) Set the particle vorticity equal to that on the grid
    o_Vals[0][id] = g_Vals[0][id];
    o_Vals[1][id] = g_Vals[1][id];
    o_Vals[2][id] = g_Vals[2][id];
}

//--- Finite difference kernels

static Real const D1C2[2] =  {-0.5, 0.5};
static Real const D1C4[4] =  {1./12., -2./3., 2./3., -1./12.};
static Real const D1C6[6] =  {-1./60., 3./20., -3./4., 3./4., -3./20., 1./60.};
static Real const D1C8[8] =  {1./280., -4./105., 1./5., -4./5., 4./5., -1./5., 4./105., -1./280. };

static Real const D2C2[3] = {1., -2., 1.};
static Real const D2C4[5] =  {-1./12., 4./3., -5./2., 4./3., -1./12.};
static Real const D2C6[7] =  {1./90., -3./20., 3./2., -49./18., 3./2., -3./20., 1./90};
static Real const D2C8[9] =  {-1./560., 8./315., -1./5., 8./5., -205./72., 8./5.,  -1./5., 8./315., -1./560.};

// Isotropic Laplacians
// static const Real L1 = 2./30., L2 = 1./30., L3 = 18./30., L4 = -136./30.;    // Cocle 2008
// const Real L1 = 0., L2 = 1./6., L3 = 1./3., L4 = -4.;                   // Patra 2006 - variant 2 (compact)
// const Real L1 = 1./30., L2 = 3./30., L3 = 14./30., L4 = -128./30.;    // Patra 2006 - variant 5
const Real L1 = 0., L2 = 0., L3 = 1., L4 = -6.;                   // Standard cross (non isotropic!)
static Real const LAP_ISO_3D[27] =  {L1,L2,L1,L2,L3,L2,L1,L2,L1,      L2,L3,L2,L3,L4,L3,L2,L3,L2,     L1,L2,L1,L2,L3,L2,L1,L2,L1};

inline uint GID(const uint &i, const uint &j, const uint &k, const dim3 &D) {return i*D.y*D.z + j*D.z + k;}     // C-style ordering
inline uint GID(const dim3 &I, const dim3 &D) {return I.x*D.y*D.z + I.y*D.z + I.z;}                             // C-style ordering

inline void KER_Stretch_FD2(const TensorGrid &go, const TensorGrid &dpdt, TensorGrid &dodt, TensorGrid &NablaU, TensorGrid &Lap, const dim3 &id, const dim3 &D, const Real &H, const Real &Nu)
{
    // This calculate the rates of change on the grid using finite differences.

    int tid = GID(id.x,id.y,id.z,D);

    //--- Calculate gradient of velocity field
    Real duxdx=0.,duydx=0.,duzdx=0., duxdy=0., duydy=0., duzdy=0., duxdz=0., duydz=0., duzdz=0.;
    int ids;
    ids = GID(id.x-1,id.y  ,id.z  ,D);     duxdx += D1C2[0]*dpdt[0][ids];   duydx += D1C2[0]*dpdt[1][ids];  duzdx += D1C2[0]*dpdt[2][ids];
    ids = GID(id.x+1,id.y  ,id.z  ,D);     duxdx += D1C2[1]*dpdt[0][ids];   duydx += D1C2[1]*dpdt[1][ids];  duzdx += D1C2[1]*dpdt[2][ids];
    ids = GID(id.x  ,id.y-1,id.z  ,D);     duxdy += D1C2[0]*dpdt[0][ids];   duydy += D1C2[0]*dpdt[1][ids];  duzdy += D1C2[0]*dpdt[2][ids];
    ids = GID(id.x  ,id.y+1,id.z  ,D);     duxdy += D1C2[1]*dpdt[0][ids];   duydy += D1C2[1]*dpdt[1][ids];  duzdy += D1C2[1]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-1,D);     duxdz += D1C2[0]*dpdt[0][ids];   duydz += D1C2[0]*dpdt[1][ids];  duzdz += D1C2[0]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+1,D);     duxdz += D1C2[1]*dpdt[0][ids];   duydz += D1C2[1]*dpdt[1][ids];  duzdz += D1C2[1]*dpdt[2][ids];

    // Scale for output
    Real InvH = 1.0/H;
    NablaU[0][tid] = duxdx*InvH;
    NablaU[1][tid] = duxdy*InvH;
    NablaU[2][tid] = duxdz*InvH;
    NablaU[3][tid] = duydx*InvH;
    NablaU[4][tid] = duydy*InvH;
    NablaU[5][tid] = duydz*InvH;
    NablaU[6][tid] = duzdx*InvH;
    NablaU[7][tid] = duzdy*InvH;
    NablaU[8][tid] = duzdz*InvH;

    //--- Calculate Laplacian
    Real lx=0., ly=0., lz=0.;
    ids = GID(id.x-1,id.y  ,id.z  ,D);    lx += D2C2[0]*go[0][ids];  ly += D2C2[0]*go[1][ids];  lz += D2C2[0]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C2[1]*go[0][ids];  ly += D2C2[1]*go[1][ids];  lz += D2C2[1]*go[2][ids];
    ids = GID(id.x+1,id.y  ,id.z  ,D);    lx += D2C2[2]*go[0][ids];  ly += D2C2[2]*go[1][ids];  lz += D2C2[2]*go[2][ids];
    ids = GID(id.x  ,id.y-1,id.z  ,D);    lx += D2C2[0]*go[0][ids];  ly += D2C2[0]*go[1][ids];  lz += D2C2[0]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C2[1]*go[0][ids];  ly += D2C2[1]*go[1][ids];  lz += D2C2[1]*go[2][ids];
    ids = GID(id.x  ,id.y+1,id.z  ,D);    lx += D2C2[2]*go[0][ids];  ly += D2C2[2]*go[1][ids];  lz += D2C2[2]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-1,D);    lx += D2C2[0]*go[0][ids];  ly += D2C2[0]*go[1][ids];  lz += D2C2[0]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C2[1]*go[0][ids];  ly += D2C2[1]*go[1][ids];  lz += D2C2[1]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+1,D);    lx += D2C2[2]*go[0][ids];  ly += D2C2[2]*go[1][ids];  lz += D2C2[2]*go[2][ids];

    // Scale for output
    Real InvH2 = 1.0/H/H;
    Lap[0][tid] = lx*InvH2;
    Lap[1][tid] = ly*InvH2;
    Lap[2][tid] = lz*InvH2;

    // Calculate total stretching
    dodt[0][tid] = NablaU[0][tid]*go[0][tid] + NablaU[1][tid]*go[1][tid] + NablaU[2][tid]*go[2][tid]    +   Nu*Lap[0][tid];
    dodt[1][tid] = NablaU[3][tid]*go[0][tid] + NablaU[4][tid]*go[1][tid] + NablaU[5][tid]*go[2][tid]    +   Nu*Lap[1][tid];
    dodt[2][tid] = NablaU[6][tid]*go[0][tid] + NablaU[7][tid]*go[1][tid] + NablaU[8][tid]*go[2][tid]    +   Nu*Lap[2][tid];
}

inline void KER_Stretch_FD4(const TensorGrid &go, const TensorGrid &dpdt, TensorGrid &dodt, TensorGrid &NablaU, TensorGrid &Lap, const dim3 &id, const dim3 &D, const Real &H, const Real &Nu)
{
    // This calculate the rates of change on the grid using finite differences.

    int tid = GID(id.x,id.y,id.z,D);

    //--- Calculate gradient of velocity field
    Real duxdx=0.,duydx=0.,duzdx=0., duxdy=0., duydy=0., duzdy=0., duxdz=0., duydz=0., duzdz=0.;
    int ids;
    ids = GID(id.x-2,id.y  ,id.z  ,D);     duxdx += D1C4[0]*dpdt[0][ids];   duydx += D1C4[0]*dpdt[1][ids];  duzdx += D1C4[0]*dpdt[2][ids];
    ids = GID(id.x-1,id.y  ,id.z  ,D);     duxdx += D1C4[1]*dpdt[0][ids];   duydx += D1C4[1]*dpdt[1][ids];  duzdx += D1C4[1]*dpdt[2][ids];
    ids = GID(id.x+1,id.y  ,id.z  ,D);     duxdx += D1C4[2]*dpdt[0][ids];   duydx += D1C4[2]*dpdt[1][ids];  duzdx += D1C4[2]*dpdt[2][ids];
    ids = GID(id.x+2,id.y  ,id.z  ,D);     duxdx += D1C4[3]*dpdt[0][ids];   duydx += D1C4[3]*dpdt[1][ids];  duzdx += D1C4[3]*dpdt[2][ids];
    ids = GID(id.x  ,id.y-2,id.z  ,D);     duxdy += D1C4[0]*dpdt[0][ids];   duydy += D1C4[0]*dpdt[1][ids];  duzdy += D1C4[0]*dpdt[2][ids];
    ids = GID(id.x  ,id.y-1,id.z  ,D);     duxdy += D1C4[1]*dpdt[0][ids];   duydy += D1C4[1]*dpdt[1][ids];  duzdy += D1C4[1]*dpdt[2][ids];
    ids = GID(id.x  ,id.y+1,id.z  ,D);     duxdy += D1C4[2]*dpdt[0][ids];   duydy += D1C4[2]*dpdt[1][ids];  duzdy += D1C4[2]*dpdt[2][ids];
    ids = GID(id.x  ,id.y+2,id.z  ,D);     duxdy += D1C4[3]*dpdt[0][ids];   duydy += D1C4[3]*dpdt[1][ids];  duzdy += D1C4[3]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-2,D);     duxdz += D1C4[0]*dpdt[0][ids];   duydz += D1C4[0]*dpdt[1][ids];  duzdz += D1C4[0]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-1,D);     duxdz += D1C4[1]*dpdt[0][ids];   duydz += D1C4[1]*dpdt[1][ids];  duzdz += D1C4[1]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+1,D);     duxdz += D1C4[2]*dpdt[0][ids];   duydz += D1C4[2]*dpdt[1][ids];  duzdz += D1C4[2]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+2,D);     duxdz += D1C4[3]*dpdt[0][ids];   duydz += D1C4[3]*dpdt[1][ids];  duzdz += D1C4[3]*dpdt[2][ids];

    // Scale for output
    Real InvH = 1.0/H;
    NablaU[0][tid] = duxdx*InvH;
    NablaU[1][tid] = duxdy*InvH;
    NablaU[2][tid] = duxdz*InvH;
    NablaU[3][tid] = duydx*InvH;
    NablaU[4][tid] = duydy*InvH;
    NablaU[5][tid] = duydz*InvH;
    NablaU[6][tid] = duzdx*InvH;
    NablaU[7][tid] = duzdy*InvH;
    NablaU[8][tid] = duzdz*InvH;

    //--- Calculate Laplacian
    Real lx=0., ly=0., lz=0.;
    ids = GID(id.x-2,id.y  ,id.z  ,D);    lx += D2C4[0]*go[0][ids];  ly += D2C4[0]*go[1][ids];  lz += D2C4[0]*go[2][ids];
    ids = GID(id.x-1,id.y  ,id.z  ,D);    lx += D2C4[1]*go[0][ids];  ly += D2C4[1]*go[1][ids];  lz += D2C4[1]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C4[2]*go[0][ids];  ly += D2C4[2]*go[1][ids];  lz += D2C4[2]*go[2][ids];
    ids = GID(id.x+1,id.y  ,id.z  ,D);    lx += D2C4[3]*go[0][ids];  ly += D2C4[3]*go[1][ids];  lz += D2C4[3]*go[2][ids];
    ids = GID(id.x+2,id.y  ,id.z  ,D);    lx += D2C4[4]*go[0][ids];  ly += D2C4[4]*go[1][ids];  lz += D2C4[4]*go[2][ids];
    ids = GID(id.x  ,id.y-2,id.z  ,D);    lx += D2C4[0]*go[0][ids];  ly += D2C4[0]*go[1][ids];  lz += D2C4[0]*go[2][ids];
    ids = GID(id.x  ,id.y-1,id.z  ,D);    lx += D2C4[1]*go[0][ids];  ly += D2C4[1]*go[1][ids];  lz += D2C4[1]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C4[2]*go[0][ids];  ly += D2C4[2]*go[1][ids];  lz += D2C4[2]*go[2][ids];
    ids = GID(id.x  ,id.y+1,id.z  ,D);    lx += D2C4[3]*go[0][ids];  ly += D2C4[3]*go[1][ids];  lz += D2C4[3]*go[2][ids];
    ids = GID(id.x  ,id.y+2,id.z  ,D);    lx += D2C4[4]*go[0][ids];  ly += D2C4[4]*go[1][ids];  lz += D2C4[4]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-2,D);    lx += D2C4[0]*go[0][ids];  ly += D2C4[0]*go[1][ids];  lz += D2C4[0]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-1,D);    lx += D2C4[1]*go[0][ids];  ly += D2C4[1]*go[1][ids];  lz += D2C4[1]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C4[2]*go[0][ids];  ly += D2C4[2]*go[1][ids];  lz += D2C4[2]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+1,D);    lx += D2C4[3]*go[0][ids];  ly += D2C4[3]*go[1][ids];  lz += D2C4[3]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+2,D);    lx += D2C4[4]*go[0][ids];  ly += D2C4[4]*go[1][ids];  lz += D2C4[4]*go[2][ids];

    // Scale for output
    Real InvH2 = 1.0/H/H;
    Lap[0][tid] = lx*InvH2;
    Lap[1][tid] = ly*InvH2;
    Lap[2][tid] = lz*InvH2;

    // Calculate total stretching
    dodt[0][tid] = NablaU[0][tid]*go[0][tid] + NablaU[1][tid]*go[1][tid] + NablaU[2][tid]*go[2][tid]    +   Nu*Lap[0][tid];
    dodt[1][tid] = NablaU[3][tid]*go[0][tid] + NablaU[4][tid]*go[1][tid] + NablaU[5][tid]*go[2][tid]    +   Nu*Lap[1][tid];
    dodt[2][tid] = NablaU[6][tid]*go[0][tid] + NablaU[7][tid]*go[1][tid] + NablaU[8][tid]*go[2][tid]    +   Nu*Lap[2][tid];
}

inline void KER_Stretch_FD6(const TensorGrid &go, const TensorGrid &dpdt, TensorGrid &dodt, TensorGrid &NablaU, TensorGrid &Lap, const dim3 &id, const dim3 &D, const Real &H, const Real &Nu)
{
    // This calculate the rates of change on the grid using finite differences.

    int tid = GID(id.x,id.y,id.z,D);

    //--- Calculate gradient of velocity field
    Real duxdx=0.,duydx=0.,duzdx=0., duxdy=0., duydy=0., duzdy=0., duxdz=0., duydz=0., duzdz=0.;
    int ids;
    ids = GID(id.x-3,id.y  ,id.z  ,D);     duxdx += D1C6[0]*dpdt[0][ids];   duydx += D1C6[0]*dpdt[1][ids];  duzdx += D1C6[0]*dpdt[2][ids];
    ids = GID(id.x-2,id.y  ,id.z  ,D);     duxdx += D1C6[1]*dpdt[0][ids];   duydx += D1C6[1]*dpdt[1][ids];  duzdx += D1C6[1]*dpdt[2][ids];
    ids = GID(id.x-1,id.y  ,id.z  ,D);     duxdx += D1C6[2]*dpdt[0][ids];   duydx += D1C6[2]*dpdt[1][ids];  duzdx += D1C6[2]*dpdt[2][ids];
    ids = GID(id.x+1,id.y  ,id.z  ,D);     duxdx += D1C6[3]*dpdt[0][ids];   duydx += D1C6[3]*dpdt[1][ids];  duzdx += D1C6[3]*dpdt[2][ids];
    ids = GID(id.x+2,id.y  ,id.z  ,D);     duxdx += D1C6[4]*dpdt[0][ids];   duydx += D1C6[4]*dpdt[1][ids];  duzdx += D1C6[4]*dpdt[2][ids];
    ids = GID(id.x+3,id.y  ,id.z  ,D);     duxdx += D1C6[5]*dpdt[0][ids];   duydx += D1C6[5]*dpdt[1][ids];  duzdx += D1C6[5]*dpdt[2][ids];
    ids = GID(id.x  ,id.y-3,id.z  ,D);     duxdy += D1C6[0]*dpdt[0][ids];   duydy += D1C6[0]*dpdt[1][ids];  duzdy += D1C6[0]*dpdt[2][ids];
    ids = GID(id.x  ,id.y-2,id.z  ,D);     duxdy += D1C6[1]*dpdt[0][ids];   duydy += D1C6[1]*dpdt[1][ids];  duzdy += D1C6[1]*dpdt[2][ids];
    ids = GID(id.x  ,id.y-1,id.z  ,D);     duxdy += D1C6[2]*dpdt[0][ids];   duydy += D1C6[2]*dpdt[1][ids];  duzdy += D1C6[2]*dpdt[2][ids];
    ids = GID(id.x  ,id.y+1,id.z  ,D);     duxdy += D1C6[3]*dpdt[0][ids];   duydy += D1C6[3]*dpdt[1][ids];  duzdy += D1C6[3]*dpdt[2][ids];
    ids = GID(id.x  ,id.y+2,id.z  ,D);     duxdy += D1C6[4]*dpdt[0][ids];   duydy += D1C6[4]*dpdt[1][ids];  duzdy += D1C6[4]*dpdt[2][ids];
    ids = GID(id.x  ,id.y+3,id.z  ,D);     duxdy += D1C6[5]*dpdt[0][ids];   duydy += D1C6[5]*dpdt[1][ids];  duzdy += D1C6[5]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-3,D);     duxdz += D1C6[0]*dpdt[0][ids];   duydz += D1C6[0]*dpdt[1][ids];  duzdz += D1C6[0]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-2,D);     duxdz += D1C6[1]*dpdt[0][ids];   duydz += D1C6[1]*dpdt[1][ids];  duzdz += D1C6[1]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-1,D);     duxdz += D1C6[2]*dpdt[0][ids];   duydz += D1C6[2]*dpdt[1][ids];  duzdz += D1C6[2]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+1,D);     duxdz += D1C6[3]*dpdt[0][ids];   duydz += D1C6[3]*dpdt[1][ids];  duzdz += D1C6[3]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+2,D);     duxdz += D1C6[4]*dpdt[0][ids];   duydz += D1C6[4]*dpdt[1][ids];  duzdz += D1C6[4]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+3,D);     duxdz += D1C6[5]*dpdt[0][ids];   duydz += D1C6[5]*dpdt[1][ids];  duzdz += D1C6[5]*dpdt[2][ids];

    // Scale for output
    Real InvH = 1.0/H;
    NablaU[0][tid] = duxdx*InvH;
    NablaU[1][tid] = duxdy*InvH;
    NablaU[2][tid] = duxdz*InvH;
    NablaU[3][tid] = duydx*InvH;
    NablaU[4][tid] = duydy*InvH;
    NablaU[5][tid] = duydz*InvH;
    NablaU[6][tid] = duzdx*InvH;
    NablaU[7][tid] = duzdy*InvH;
    NablaU[8][tid] = duzdz*InvH;

    //--- Calculate Laplacian
    Real lx=0., ly=0., lz=0.;
    ids = GID(id.x-3,id.y  ,id.z  ,D);    lx += D2C6[0]*go[0][ids];  ly += D2C6[0]*go[1][ids];  lz += D2C6[0]*go[2][ids];
    ids = GID(id.x-2,id.y  ,id.z  ,D);    lx += D2C6[1]*go[0][ids];  ly += D2C6[1]*go[1][ids];  lz += D2C6[1]*go[2][ids];
    ids = GID(id.x-1,id.y  ,id.z  ,D);    lx += D2C6[2]*go[0][ids];  ly += D2C6[2]*go[1][ids];  lz += D2C6[2]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C6[3]*go[0][ids];  ly += D2C6[3]*go[1][ids];  lz += D2C6[3]*go[2][ids];
    ids = GID(id.x+1,id.y  ,id.z  ,D);    lx += D2C6[4]*go[0][ids];  ly += D2C6[4]*go[1][ids];  lz += D2C6[4]*go[2][ids];
    ids = GID(id.x+2,id.y  ,id.z  ,D);    lx += D2C6[5]*go[0][ids];  ly += D2C6[5]*go[1][ids];  lz += D2C6[5]*go[2][ids];
    ids = GID(id.x+3,id.y  ,id.z  ,D);    lx += D2C6[6]*go[0][ids];  ly += D2C6[6]*go[1][ids];  lz += D2C6[6]*go[2][ids];
    ids = GID(id.x  ,id.y-3,id.z  ,D);    lx += D2C6[0]*go[0][ids];  ly += D2C6[0]*go[1][ids];  lz += D2C6[0]*go[2][ids];
    ids = GID(id.x  ,id.y-2,id.z  ,D);    lx += D2C6[1]*go[0][ids];  ly += D2C6[1]*go[1][ids];  lz += D2C6[1]*go[2][ids];
    ids = GID(id.x  ,id.y-1,id.z  ,D);    lx += D2C6[2]*go[0][ids];  ly += D2C6[2]*go[1][ids];  lz += D2C6[2]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C6[3]*go[0][ids];  ly += D2C6[3]*go[1][ids];  lz += D2C6[3]*go[2][ids];
    ids = GID(id.x  ,id.y+1,id.z  ,D);    lx += D2C6[4]*go[0][ids];  ly += D2C6[4]*go[1][ids];  lz += D2C6[4]*go[2][ids];
    ids = GID(id.x  ,id.y+2,id.z  ,D);    lx += D2C6[5]*go[0][ids];  ly += D2C6[5]*go[1][ids];  lz += D2C6[5]*go[2][ids];
    ids = GID(id.x  ,id.y+3,id.z  ,D);    lx += D2C6[6]*go[0][ids];  ly += D2C6[6]*go[1][ids];  lz += D2C6[6]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-3,D);    lx += D2C6[0]*go[0][ids];  ly += D2C6[0]*go[1][ids];  lz += D2C6[0]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-2,D);    lx += D2C6[1]*go[0][ids];  ly += D2C6[1]*go[1][ids];  lz += D2C6[1]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-1,D);    lx += D2C6[2]*go[0][ids];  ly += D2C6[2]*go[1][ids];  lz += D2C6[2]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C6[3]*go[0][ids];  ly += D2C6[3]*go[1][ids];  lz += D2C6[3]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+1,D);    lx += D2C6[4]*go[0][ids];  ly += D2C6[4]*go[1][ids];  lz += D2C6[4]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+2,D);    lx += D2C6[5]*go[0][ids];  ly += D2C6[5]*go[1][ids];  lz += D2C6[5]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+3,D);    lx += D2C6[6]*go[0][ids];  ly += D2C6[6]*go[1][ids];  lz += D2C6[6]*go[2][ids];

    // Scale for output
    Real InvH2 = 1.0/H/H;
    Lap[0][tid] = lx*InvH2;
    Lap[1][tid] = ly*InvH2;
    Lap[2][tid] = lz*InvH2;

    // Calculate total stretching
    dodt[0][tid] = NablaU[0][tid]*go[0][tid] + NablaU[1][tid]*go[1][tid] + NablaU[2][tid]*go[2][tid]    +   Nu*Lap[0][tid];
    dodt[1][tid] = NablaU[3][tid]*go[0][tid] + NablaU[4][tid]*go[1][tid] + NablaU[5][tid]*go[2][tid]    +   Nu*Lap[1][tid];
    dodt[2][tid] = NablaU[6][tid]*go[0][tid] + NablaU[7][tid]*go[1][tid] + NablaU[8][tid]*go[2][tid]    +   Nu*Lap[2][tid];
}

inline void KER_Stretch_FD8(const TensorGrid &go, const TensorGrid &dpdt, TensorGrid &dodt, TensorGrid &NablaU, TensorGrid &Lap, const dim3 &id, const dim3 &D, const Real &H, const Real &Nu)
{
    // This calculate the rates of change on the grid using finite differences.

    int tid = GID(id.x,id.y,id.z,D);

    //--- Calculate gradient of velocity field
    Real duxdx=0.,duydx=0.,duzdx=0., duxdy=0., duydy=0., duzdy=0., duxdz=0., duydz=0., duzdz=0.;
    int ids;
    ids = GID(id.x-4,id.y  ,id.z  ,D);     duxdx += D1C8[0]*dpdt[0][ids];   duydx += D1C8[0]*dpdt[1][ids];  duzdx += D1C8[0]*dpdt[2][ids];
    ids = GID(id.x-3,id.y  ,id.z  ,D);     duxdx += D1C8[1]*dpdt[0][ids];   duydx += D1C8[1]*dpdt[1][ids];  duzdx += D1C8[1]*dpdt[2][ids];
    ids = GID(id.x-2,id.y  ,id.z  ,D);     duxdx += D1C8[2]*dpdt[0][ids];   duydx += D1C8[2]*dpdt[1][ids];  duzdx += D1C8[2]*dpdt[2][ids];
    ids = GID(id.x-1,id.y  ,id.z  ,D);     duxdx += D1C8[3]*dpdt[0][ids];   duydx += D1C8[3]*dpdt[1][ids];  duzdx += D1C8[3]*dpdt[2][ids];
    ids = GID(id.x+1,id.y  ,id.z  ,D);     duxdx += D1C8[4]*dpdt[0][ids];   duydx += D1C8[4]*dpdt[1][ids];  duzdx += D1C8[4]*dpdt[2][ids];
    ids = GID(id.x+2,id.y  ,id.z  ,D);     duxdx += D1C8[5]*dpdt[0][ids];   duydx += D1C8[5]*dpdt[1][ids];  duzdx += D1C8[5]*dpdt[2][ids];
    ids = GID(id.x+3,id.y  ,id.z  ,D);     duxdx += D1C8[6]*dpdt[0][ids];   duydx += D1C8[6]*dpdt[1][ids];  duzdx += D1C8[6]*dpdt[2][ids];
    ids = GID(id.x+4,id.y  ,id.z  ,D);     duxdx += D1C8[7]*dpdt[0][ids];   duydx += D1C8[7]*dpdt[1][ids];  duzdx += D1C8[7]*dpdt[2][ids];
    ids = GID(id.x  ,id.y-4,id.z  ,D);     duxdy += D1C8[0]*dpdt[0][ids];   duydy += D1C8[0]*dpdt[1][ids];  duzdy += D1C8[0]*dpdt[2][ids];
    ids = GID(id.x  ,id.y-3,id.z  ,D);     duxdy += D1C8[1]*dpdt[0][ids];   duydy += D1C8[1]*dpdt[1][ids];  duzdy += D1C8[1]*dpdt[2][ids];
    ids = GID(id.x  ,id.y-2,id.z  ,D);     duxdy += D1C8[2]*dpdt[0][ids];   duydy += D1C8[2]*dpdt[1][ids];  duzdy += D1C8[2]*dpdt[2][ids];
    ids = GID(id.x  ,id.y-1,id.z  ,D);     duxdy += D1C8[3]*dpdt[0][ids];   duydy += D1C8[3]*dpdt[1][ids];  duzdy += D1C8[3]*dpdt[2][ids];
    ids = GID(id.x  ,id.y+1,id.z  ,D);     duxdy += D1C8[4]*dpdt[0][ids];   duydy += D1C8[4]*dpdt[1][ids];  duzdy += D1C8[4]*dpdt[2][ids];
    ids = GID(id.x  ,id.y+2,id.z  ,D);     duxdy += D1C8[5]*dpdt[0][ids];   duydy += D1C8[5]*dpdt[1][ids];  duzdy += D1C8[5]*dpdt[2][ids];
    ids = GID(id.x  ,id.y+3,id.z  ,D);     duxdy += D1C8[6]*dpdt[0][ids];   duydy += D1C8[6]*dpdt[1][ids];  duzdy += D1C8[6]*dpdt[2][ids];
    ids = GID(id.x  ,id.y+4,id.z  ,D);     duxdy += D1C8[7]*dpdt[0][ids];   duydy += D1C8[7]*dpdt[1][ids];  duzdy += D1C8[7]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-4,D);     duxdz += D1C8[0]*dpdt[0][ids];   duydz += D1C8[0]*dpdt[1][ids];  duzdz += D1C8[0]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-3,D);     duxdz += D1C8[1]*dpdt[0][ids];   duydz += D1C8[1]*dpdt[1][ids];  duzdz += D1C8[1]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-2,D);     duxdz += D1C8[2]*dpdt[0][ids];   duydz += D1C8[2]*dpdt[1][ids];  duzdz += D1C8[2]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-1,D);     duxdz += D1C8[3]*dpdt[0][ids];   duydz += D1C8[3]*dpdt[1][ids];  duzdz += D1C8[3]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+1,D);     duxdz += D1C8[4]*dpdt[0][ids];   duydz += D1C8[4]*dpdt[1][ids];  duzdz += D1C8[4]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+2,D);     duxdz += D1C8[5]*dpdt[0][ids];   duydz += D1C8[5]*dpdt[1][ids];  duzdz += D1C8[5]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+3,D);     duxdz += D1C8[6]*dpdt[0][ids];   duydz += D1C8[6]*dpdt[1][ids];  duzdz += D1C8[6]*dpdt[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+4,D);     duxdz += D1C8[7]*dpdt[0][ids];   duydz += D1C8[7]*dpdt[1][ids];  duzdz += D1C8[7]*dpdt[2][ids];


    // Scale for output
    Real InvH = 1.0/H;
    NablaU[0][tid] = duxdx*InvH;
    NablaU[1][tid] = duxdy*InvH;
    NablaU[2][tid] = duxdz*InvH;
    NablaU[3][tid] = duydx*InvH;
    NablaU[4][tid] = duydy*InvH;
    NablaU[5][tid] = duydz*InvH;
    NablaU[6][tid] = duzdx*InvH;
    NablaU[7][tid] = duzdy*InvH;
    NablaU[8][tid] = duzdz*InvH;

    //--- Calculate Laplacian
    Real lx=0., ly=0., lz=0.;
    ids = GID(id.x-4,id.y  ,id.z  ,D);    lx += D2C8[0]*go[0][ids];  ly += D2C8[0]*go[1][ids];  lz += D2C8[0]*go[2][ids];
    ids = GID(id.x-3,id.y  ,id.z  ,D);    lx += D2C8[1]*go[0][ids];  ly += D2C8[1]*go[1][ids];  lz += D2C8[1]*go[2][ids];
    ids = GID(id.x-2,id.y  ,id.z  ,D);    lx += D2C8[2]*go[0][ids];  ly += D2C8[2]*go[1][ids];  lz += D2C8[2]*go[2][ids];
    ids = GID(id.x-1,id.y  ,id.z  ,D);    lx += D2C8[3]*go[0][ids];  ly += D2C8[3]*go[1][ids];  lz += D2C8[3]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C8[4]*go[0][ids];  ly += D2C8[4]*go[1][ids];  lz += D2C8[4]*go[2][ids];
    ids = GID(id.x+1,id.y  ,id.z  ,D);    lx += D2C8[5]*go[0][ids];  ly += D2C8[5]*go[1][ids];  lz += D2C8[5]*go[2][ids];
    ids = GID(id.x+2,id.y  ,id.z  ,D);    lx += D2C8[6]*go[0][ids];  ly += D2C8[6]*go[1][ids];  lz += D2C8[6]*go[2][ids];
    ids = GID(id.x+3,id.y  ,id.z  ,D);    lx += D2C8[7]*go[0][ids];  ly += D2C8[7]*go[1][ids];  lz += D2C8[7]*go[2][ids];
    ids = GID(id.x+4,id.y  ,id.z  ,D);    lx += D2C8[8]*go[0][ids];  ly += D2C8[8]*go[1][ids];  lz += D2C8[8]*go[2][ids];
    ids = GID(id.x  ,id.y-4,id.z  ,D);    lx += D2C8[0]*go[0][ids];  ly += D2C8[0]*go[1][ids];  lz += D2C8[0]*go[2][ids];
    ids = GID(id.x  ,id.y-3,id.z  ,D);    lx += D2C8[1]*go[0][ids];  ly += D2C8[1]*go[1][ids];  lz += D2C8[1]*go[2][ids];
    ids = GID(id.x  ,id.y-2,id.z  ,D);    lx += D2C8[2]*go[0][ids];  ly += D2C8[2]*go[1][ids];  lz += D2C8[2]*go[2][ids];
    ids = GID(id.x  ,id.y-1,id.z  ,D);    lx += D2C8[3]*go[0][ids];  ly += D2C8[3]*go[1][ids];  lz += D2C8[3]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C8[4]*go[0][ids];  ly += D2C8[4]*go[1][ids];  lz += D2C8[4]*go[2][ids];
    ids = GID(id.x  ,id.y+1,id.z  ,D);    lx += D2C8[5]*go[0][ids];  ly += D2C8[5]*go[1][ids];  lz += D2C8[5]*go[2][ids];
    ids = GID(id.x  ,id.y+2,id.z  ,D);    lx += D2C8[6]*go[0][ids];  ly += D2C8[6]*go[1][ids];  lz += D2C8[6]*go[2][ids];
    ids = GID(id.x  ,id.y+3,id.z  ,D);    lx += D2C8[7]*go[0][ids];  ly += D2C8[7]*go[1][ids];  lz += D2C8[7]*go[2][ids];
    ids = GID(id.x  ,id.y+4,id.z  ,D);    lx += D2C8[8]*go[0][ids];  ly += D2C8[8]*go[1][ids];  lz += D2C8[8]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-4,D);    lx += D2C8[0]*go[0][ids];  ly += D2C8[0]*go[1][ids];  lz += D2C8[0]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-3,D);    lx += D2C8[1]*go[0][ids];  ly += D2C8[1]*go[1][ids];  lz += D2C8[1]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-2,D);    lx += D2C8[2]*go[0][ids];  ly += D2C8[2]*go[1][ids];  lz += D2C8[2]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-1,D);    lx += D2C8[3]*go[0][ids];  ly += D2C8[3]*go[1][ids];  lz += D2C8[3]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C8[4]*go[0][ids];  ly += D2C8[4]*go[1][ids];  lz += D2C8[4]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+1,D);    lx += D2C8[5]*go[0][ids];  ly += D2C8[5]*go[1][ids];  lz += D2C8[5]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+2,D);    lx += D2C8[6]*go[0][ids];  ly += D2C8[6]*go[1][ids];  lz += D2C8[6]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+3,D);    lx += D2C8[7]*go[0][ids];  ly += D2C8[7]*go[1][ids];  lz += D2C8[7]*go[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+4,D);    lx += D2C8[8]*go[0][ids];  ly += D2C8[8]*go[1][ids];  lz += D2C8[8]*go[2][ids];

    // Scale for output
    Real InvH2 = 1.0/H/H;
    Lap[0][tid] = lx*InvH2;
    Lap[1][tid] = ly*InvH2;
    Lap[2][tid] = lz*InvH2;

    // Calculate total stretching
    dodt[0][tid] = NablaU[0][tid]*go[0][tid] + NablaU[1][tid]*go[1][tid] + NablaU[2][tid]*go[2][tid]    +   Nu*Lap[0][tid];
    dodt[1][tid] = NablaU[3][tid]*go[0][tid] + NablaU[4][tid]*go[1][tid] + NablaU[5][tid]*go[2][tid]    +   Nu*Lap[1][tid];
    dodt[2][tid] = NablaU[6][tid]*go[0][tid] + NablaU[7][tid]*go[1][tid] + NablaU[8][tid]*go[2][tid]    +   Nu*Lap[2][tid];
}

//--- Reprojection (finite difference)

inline void KER_Calc_DivOmega2(const TensorGrid &p_Array, RVector &Div, const dim3 &id, const dim3 &D, const Real &H)
{
    //--- Calculate divergence of vorticity field using second-order central finite differences
    Real divOm=0.;
    divOm += D1C2[0]*p_Array[0][GID(id.x-1,id.y  ,id.z  ,D)];
    divOm += D1C2[1]*p_Array[0][GID(id.x+1,id.y  ,id.z  ,D)];
    divOm += D1C2[0]*p_Array[1][GID(id.x  ,id.y-1,id.z  ,D)];
    divOm += D1C2[1]*p_Array[1][GID(id.x  ,id.y+1,id.z  ,D)];
    divOm += D1C2[0]*p_Array[2][GID(id.x  ,id.y  ,id.z-1,D)];
    divOm += D1C2[1]*p_Array[2][GID(id.x  ,id.y  ,id.z+1,D)];
    Div[  GID(id.x  ,id.y  ,id.z  ,D)] =   divOm/H;
}

inline void KER_Calc_DivOmega4(const TensorGrid &p_Array, RVector &Div, const dim3 &id, const dim3 &D, const Real &H)
{
    //--- Calculate divergence of vorticity field using second-order central finite differences
    Real divOm=0.;
    divOm += D1C4[0]*p_Array[0][GID(id.x-2,id.y  ,id.z  ,D)];
    divOm += D1C4[1]*p_Array[0][GID(id.x-1,id.y  ,id.z  ,D)];
    divOm += D1C4[2]*p_Array[0][GID(id.x+1,id.y  ,id.z  ,D)];
    divOm += D1C4[3]*p_Array[0][GID(id.x+2,id.y  ,id.z  ,D)];
    divOm += D1C4[0]*p_Array[1][GID(id.x  ,id.y-2,id.z  ,D)];
    divOm += D1C4[1]*p_Array[1][GID(id.x  ,id.y-1,id.z  ,D)];
    divOm += D1C4[2]*p_Array[1][GID(id.x  ,id.y+1,id.z  ,D)];
    divOm += D1C4[3]*p_Array[1][GID(id.x  ,id.y+2,id.z  ,D)];
    divOm += D1C4[0]*p_Array[2][GID(id.x  ,id.y  ,id.z-2,D)];
    divOm += D1C4[1]*p_Array[2][GID(id.x  ,id.y  ,id.z-1,D)];
    divOm += D1C4[2]*p_Array[2][GID(id.x  ,id.y  ,id.z+1,D)];
    divOm += D1C4[3]*p_Array[2][GID(id.x  ,id.y  ,id.z+2,D)];
    Div[  GID(id.x  ,id.y  ,id.z  ,D)] =   divOm/H;
}

inline void KER_Calc_DivOmega6(const TensorGrid &p_Array, RVector &Div, const dim3 &id, const dim3 &D, const Real &H)
{
    //--- Calculate divergence of vorticity field using second-order central finite differences
    Real divOm=0.;
    divOm += D1C6[0]*p_Array[0][GID(id.x-3,id.y  ,id.z  ,D)];
    divOm += D1C6[1]*p_Array[0][GID(id.x-2,id.y  ,id.z  ,D)];
    divOm += D1C6[2]*p_Array[0][GID(id.x-1,id.y  ,id.z  ,D)];
    divOm += D1C6[3]*p_Array[0][GID(id.x+1,id.y  ,id.z  ,D)];
    divOm += D1C6[4]*p_Array[0][GID(id.x+2,id.y  ,id.z  ,D)];
    divOm += D1C6[5]*p_Array[0][GID(id.x+3,id.y  ,id.z  ,D)];
    divOm += D1C6[0]*p_Array[1][GID(id.x  ,id.y-3,id.z  ,D)];
    divOm += D1C6[1]*p_Array[1][GID(id.x  ,id.y-2,id.z  ,D)];
    divOm += D1C6[2]*p_Array[1][GID(id.x  ,id.y-1,id.z  ,D)];
    divOm += D1C6[3]*p_Array[1][GID(id.x  ,id.y+1,id.z  ,D)];
    divOm += D1C6[4]*p_Array[1][GID(id.x  ,id.y+2,id.z  ,D)];
    divOm += D1C6[5]*p_Array[1][GID(id.x  ,id.y+3,id.z  ,D)];
    divOm += D1C6[0]*p_Array[2][GID(id.x  ,id.y  ,id.z-3,D)];
    divOm += D1C6[1]*p_Array[2][GID(id.x  ,id.y  ,id.z-2,D)];
    divOm += D1C6[2]*p_Array[2][GID(id.x  ,id.y  ,id.z-1,D)];
    divOm += D1C6[3]*p_Array[2][GID(id.x  ,id.y  ,id.z+1,D)];
    divOm += D1C6[4]*p_Array[2][GID(id.x  ,id.y  ,id.z+2,D)];
    divOm += D1C6[5]*p_Array[2][GID(id.x  ,id.y  ,id.z+3,D)];
    Div[  GID(id.x  ,id.y  ,id.z  ,D)] =   divOm/H;
}

inline void KER_Calc_DivOmega8(const TensorGrid &p_Array, RVector &Div, const dim3 &id, const dim3 &D, const Real &H)
{
    //--- Calculate divergence of vorticity field using second-order central finite differences
    Real divOm=0.;
    divOm += D1C8[0]*p_Array[0][GID(id.x-4,id.y  ,id.z  ,D)];
    divOm += D1C8[1]*p_Array[0][GID(id.x-3,id.y  ,id.z  ,D)];
    divOm += D1C8[2]*p_Array[0][GID(id.x-2,id.y  ,id.z  ,D)];
    divOm += D1C8[3]*p_Array[0][GID(id.x-1,id.y  ,id.z  ,D)];
    divOm += D1C8[4]*p_Array[0][GID(id.x+1,id.y  ,id.z  ,D)];
    divOm += D1C8[5]*p_Array[0][GID(id.x+2,id.y  ,id.z  ,D)];
    divOm += D1C8[6]*p_Array[0][GID(id.x+3,id.y  ,id.z  ,D)];
    divOm += D1C8[7]*p_Array[0][GID(id.x+4,id.y  ,id.z  ,D)];
    divOm += D1C8[0]*p_Array[1][GID(id.x  ,id.y-4,id.z  ,D)];
    divOm += D1C8[1]*p_Array[1][GID(id.x  ,id.y-3,id.z  ,D)];
    divOm += D1C8[2]*p_Array[1][GID(id.x  ,id.y-2,id.z  ,D)];
    divOm += D1C8[3]*p_Array[1][GID(id.x  ,id.y-1,id.z  ,D)];
    divOm += D1C8[4]*p_Array[1][GID(id.x  ,id.y+1,id.z  ,D)];
    divOm += D1C8[5]*p_Array[1][GID(id.x  ,id.y+2,id.z  ,D)];
    divOm += D1C8[6]*p_Array[1][GID(id.x  ,id.y+3,id.z  ,D)];
    divOm += D1C8[7]*p_Array[1][GID(id.x  ,id.y+4,id.z  ,D)];
    divOm += D1C8[0]*p_Array[2][GID(id.x  ,id.y  ,id.z-4,D)];
    divOm += D1C8[1]*p_Array[2][GID(id.x  ,id.y  ,id.z-3,D)];
    divOm += D1C8[2]*p_Array[2][GID(id.x  ,id.y  ,id.z-2,D)];
    divOm += D1C8[3]*p_Array[2][GID(id.x  ,id.y  ,id.z-1,D)];
    divOm += D1C8[4]*p_Array[2][GID(id.x  ,id.y  ,id.z+1,D)];
    divOm += D1C8[5]*p_Array[2][GID(id.x  ,id.y  ,id.z+2,D)];
    divOm += D1C8[6]*p_Array[2][GID(id.x  ,id.y  ,id.z+3,D)];
    divOm += D1C8[7]*p_Array[2][GID(id.x  ,id.y  ,id.z+4,D)];
    Div[  GID(id.x  ,id.y  ,id.z  ,D)] =   divOm/H;
}

inline void KER_Reproject2(TensorGrid &p_Array, RVector &F, const int &id, const std::vector<int> &Temp, const Real &H)
{
    //--- Subtracts the projected portion of vorticit field divergence (second-order finite differences)
    Real InvH = 1.0/H;
    p_Array[3][id] -= (F[id+Temp[0]]*D1C2[0] + F[id+Temp[2]]*D1C2[1])*InvH;
    p_Array[4][id] -= (F[id+Temp[3]]*D1C2[0] + F[id+Temp[5]]*D1C2[1])*InvH;
    p_Array[5][id] -= (F[id+Temp[6]]*D1C2[0] + F[id+Temp[8]]*D1C2[1])*InvH;
}

inline void KER_Reproject4(TensorGrid &p_Array, RVector &F, const int &id, const std::vector<int> &Temp, const Real &H)
{
    //--- Subtracts the projected portion of vorticity field divergence (fourth-order finite differences)
    Real InvH = 1.0/H;
    p_Array[3][id] -= (F[id+Temp[0]]*D1C4[0]  + F[id+Temp[1]]*D1C4[1]     + F[id+Temp[3]]*D1C4[2]     + F[id+Temp[4]]*D1C4[3])*InvH;     // dF/dx
    p_Array[4][id] -= (F[id+Temp[5]]*D1C4[0]  + F[id+Temp[6]]*D1C4[1]     + F[id+Temp[8]]*D1C4[2]     + F[id+Temp[9]]*D1C4[3])*InvH;     // dF/dy
    p_Array[5][id] -= (F[id+Temp[10]]*D1C4[0] + F[id+Temp[11]]*D1C4[1]    + F[id+Temp[13]]*D1C4[2]    + F[id+Temp[14]]*D1C4[3])*InvH;    // dF/dz
}

inline void KER_Reproject6(TensorGrid &p_Array, RVector &F, const int &id, const std::vector<int> &Temp, const Real &H)
{
    //--- Subtracts the projected portion of vorticity field divergence (sixth-order finite differences)
    Real InvH = 1.0/H;
    p_Array[3][id] -= (F[id+Temp[0]]*D1C6[0]   + F[id+Temp[1]]*D1C6[1]  + F[id+Temp[2]]*D1C6[2]  + F[id+Temp[4]]*D1C6[3]  + F[id+Temp[5]]*D1C6[4]  + F[id+Temp[6]]*D1C6[5])*InvH;     // dF/dx
    p_Array[4][id] -= (F[id+Temp[7]]*D1C6[0]   + F[id+Temp[8]]*D1C6[1]  + F[id+Temp[9]]*D1C6[2]  + F[id+Temp[11]]*D1C6[3] + F[id+Temp[12]]*D1C6[4] + F[id+Temp[13]]*D1C6[5])*InvH;    // dF/dy
    p_Array[5][id] -= (F[id+Temp[14]]*D1C6[0]  + F[id+Temp[15]]*D1C6[1] + F[id+Temp[16]]*D1C6[2] + F[id+Temp[18]]*D1C6[3] + F[id+Temp[19]]*D1C6[4] + F[id+Temp[20]]*D1C6[5])*InvH;    // dF/dz
}

inline void KER_Reproject8(TensorGrid &p_Array, RVector &F, const int &id, const std::vector<int> &Temp, const Real &H)
{
    //--- Subtracts the projected portion of vorticity field divergence (eighth-order finite differences)
    Real InvH = 1.0/H;
    p_Array[3][id] -= ( F[id+Temp[0]]*D1C8[0]
                       + F[id+Temp[1]]*D1C8[1]
                       + F[id+Temp[2]]*D1C8[2]
                       + F[id+Temp[3]]*D1C8[3]
                       + F[id+Temp[5]]*D1C8[4]
                       + F[id+Temp[6]]*D1C8[5]
                       + F[id+Temp[7]]*D1C8[6]
                       + F[id+Temp[8]]*D1C8[7])*InvH;     // dF/dx
    p_Array[4][id] -= ( F[id+Temp[9]]*D1C8[0]
                       + F[id+Temp[10]]*D1C8[1]
                       + F[id+Temp[11]]*D1C8[2]
                       + F[id+Temp[12]]*D1C8[3]
                       + F[id+Temp[14]]*D1C8[4]
                       + F[id+Temp[15]]*D1C8[5]
                       + F[id+Temp[16]]*D1C8[6]
                       + F[id+Temp[17]]*D1C8[7])*InvH;    // dF/dy
    p_Array[5][id] -= ( F[id+Temp[18]]*D1C8[0]
                       + F[id+Temp[19]]*D1C8[1]
                       + F[id+Temp[20]]*D1C8[2]
                       + F[id+Temp[21]]*D1C8[3]
                       + F[id+Temp[23]]*D1C8[4]
                       + F[id+Temp[24]]*D1C8[5]
                       + F[id+Temp[25]]*D1C8[6]
                       + F[id+Temp[26]]*D1C8[7])*InvH;    // dF/dz
}

//--- Turbulence model kernels

// Hyperviscosity Model

// static const Real turbt0 = 0.0;
// static const Real C_hv = 2.5e-2/turbt0;

inline void KER_HypVisc_FD2(const TensorGrid &Lap, TensorGrid &dgdt_Array, const dim3 &id, const dim3 &D, const Real &H, Real C_hv)
{
    // This kernel implements the hyperviscosity operator described in Cocle et al. doi 10.1016/j.jcp.2007.10.010
    // Input Lap is the Laplacian of the vorticity field (Contains the term 1/H2)

    //--- Calculate Laplacian (2nd order central finite difference)
    Real lx=0., ly=0., lz=0.;
    int ids;
    ids = GID(id.x-1,id.y  ,id.z  ,D);    lx += D2C2[0]*Lap[0][ids];  ly += D2C2[0]*Lap[1][ids];  lz += D2C2[0]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C2[1]*Lap[0][ids];  ly += D2C2[1]*Lap[1][ids];  lz += D2C2[1]*Lap[2][ids];
    ids = GID(id.x+1,id.y  ,id.z  ,D);    lx += D2C2[2]*Lap[0][ids];  ly += D2C2[2]*Lap[1][ids];  lz += D2C2[2]*Lap[2][ids];
    ids = GID(id.x  ,id.y-1,id.z  ,D);    lx += D2C2[0]*Lap[0][ids];  ly += D2C2[0]*Lap[1][ids];  lz += D2C2[0]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C2[1]*Lap[0][ids];  ly += D2C2[1]*Lap[1][ids];  lz += D2C2[1]*Lap[2][ids];
    ids = GID(id.x  ,id.y+1,id.z  ,D);    lx += D2C2[2]*Lap[0][ids];  ly += D2C2[2]*Lap[1][ids];  lz += D2C2[2]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-1,D);    lx += D2C2[0]*Lap[0][ids];  ly += D2C2[0]*Lap[1][ids];  lz += D2C2[0]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C2[1]*Lap[0][ids];  ly += D2C2[1]*Lap[1][ids];  lz += D2C2[1]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+1,D);    lx += D2C2[2]*Lap[0][ids];  ly += D2C2[2]*Lap[1][ids];  lz += D2C2[2]*Lap[2][ids];

    Real H2 = H*H;
    int tid = GID(id.x,id.y,id.z,D);
    dgdt_Array[0][tid] -= C_hv*lx*H2;
    dgdt_Array[1][tid] -= C_hv*ly*H2;
    dgdt_Array[2][tid] -= C_hv*lz*H2;
}

inline void KER_HypVisc_FD4(const TensorGrid &Lap, TensorGrid &dgdt_Array, const dim3 &id, const dim3 &D, const Real &H, Real C_hv)
{
    // This kernel implements the hyperviscosity operator described in Cocle et al. doi 10.1016/j.jcp.2007.10.010
    // Input Lap is the Laplacian of the vorticity field (Contains the term 1/H2)

    //--- Calculate Laplacian (2nd order central finite difference)
    Real lx=0., ly=0., lz=0.;
    int ids;
    ids = GID(id.x-2,id.y  ,id.z  ,D);    lx += D2C4[0]*Lap[0][ids];  ly += D2C4[0]*Lap[1][ids];  lz += D2C4[0]*Lap[2][ids];
    ids = GID(id.x-1,id.y  ,id.z  ,D);    lx += D2C4[1]*Lap[0][ids];  ly += D2C4[1]*Lap[1][ids];  lz += D2C4[1]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C4[2]*Lap[0][ids];  ly += D2C4[2]*Lap[1][ids];  lz += D2C4[2]*Lap[2][ids];
    ids = GID(id.x+1,id.y  ,id.z  ,D);    lx += D2C4[3]*Lap[0][ids];  ly += D2C4[3]*Lap[1][ids];  lz += D2C4[3]*Lap[2][ids];
    ids = GID(id.x+2,id.y  ,id.z  ,D);    lx += D2C4[4]*Lap[0][ids];  ly += D2C4[4]*Lap[1][ids];  lz += D2C4[4]*Lap[2][ids];
    ids = GID(id.x  ,id.y-2,id.z  ,D);    lx += D2C4[0]*Lap[0][ids];  ly += D2C4[0]*Lap[1][ids];  lz += D2C4[0]*Lap[2][ids];
    ids = GID(id.x  ,id.y-1,id.z  ,D);    lx += D2C4[1]*Lap[0][ids];  ly += D2C4[1]*Lap[1][ids];  lz += D2C4[1]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C4[2]*Lap[0][ids];  ly += D2C4[2]*Lap[1][ids];  lz += D2C4[2]*Lap[2][ids];
    ids = GID(id.x  ,id.y+1,id.z  ,D);    lx += D2C4[3]*Lap[0][ids];  ly += D2C4[3]*Lap[1][ids];  lz += D2C4[3]*Lap[2][ids];
    ids = GID(id.x  ,id.y+2,id.z  ,D);    lx += D2C4[4]*Lap[0][ids];  ly += D2C4[4]*Lap[1][ids];  lz += D2C4[4]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-2,D);    lx += D2C4[0]*Lap[0][ids];  ly += D2C4[0]*Lap[1][ids];  lz += D2C4[0]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-1,D);    lx += D2C4[1]*Lap[0][ids];  ly += D2C4[1]*Lap[1][ids];  lz += D2C4[1]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C4[2]*Lap[0][ids];  ly += D2C4[2]*Lap[1][ids];  lz += D2C4[2]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+1,D);    lx += D2C4[3]*Lap[0][ids];  ly += D2C4[3]*Lap[1][ids];  lz += D2C4[3]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+2,D);    lx += D2C4[4]*Lap[0][ids];  ly += D2C4[4]*Lap[1][ids];  lz += D2C4[4]*Lap[2][ids];

    Real H2 = H*H;
    int tid = GID(id.x,id.y,id.z,D);
    dgdt_Array[0][tid] -= C_hv*lx*H2;
    dgdt_Array[1][tid] -= C_hv*ly*H2;
    dgdt_Array[2][tid] -= C_hv*lz*H2;
}

inline void KER_HypVisc_FD6(const TensorGrid &Lap, TensorGrid &dgdt_Array, const dim3 &id, const dim3 &D, const Real &H, Real C_hv)
{
    // This kernel implements the hyperviscosity operator described in Cocle et al. doi 10.1016/j.jcp.2007.10.010
    // Input Lap is the Laplacian of the vorticity field (Contains the term 1/H2)

    //--- Calculate Laplacian (2nd order central finite difference)
    Real lx=0., ly=0., lz=0.;
    int ids;
    ids = GID(id.x-3,id.y  ,id.z  ,D);    lx += D2C6[0]*Lap[0][ids];  ly += D2C6[0]*Lap[1][ids];  lz += D2C6[0]*Lap[2][ids];
    ids = GID(id.x-2,id.y  ,id.z  ,D);    lx += D2C6[1]*Lap[0][ids];  ly += D2C6[1]*Lap[1][ids];  lz += D2C6[1]*Lap[2][ids];
    ids = GID(id.x-1,id.y  ,id.z  ,D);    lx += D2C6[2]*Lap[0][ids];  ly += D2C6[2]*Lap[1][ids];  lz += D2C6[2]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C6[3]*Lap[0][ids];  ly += D2C6[3]*Lap[1][ids];  lz += D2C6[3]*Lap[2][ids];
    ids = GID(id.x+1,id.y  ,id.z  ,D);    lx += D2C6[4]*Lap[0][ids];  ly += D2C6[4]*Lap[1][ids];  lz += D2C6[4]*Lap[2][ids];
    ids = GID(id.x+2,id.y  ,id.z  ,D);    lx += D2C6[5]*Lap[0][ids];  ly += D2C6[5]*Lap[1][ids];  lz += D2C6[5]*Lap[2][ids];
    ids = GID(id.x+3,id.y  ,id.z  ,D);    lx += D2C6[6]*Lap[0][ids];  ly += D2C6[6]*Lap[1][ids];  lz += D2C6[6]*Lap[2][ids];
    ids = GID(id.x  ,id.y-3,id.z  ,D);    lx += D2C6[0]*Lap[0][ids];  ly += D2C6[0]*Lap[1][ids];  lz += D2C6[0]*Lap[2][ids];
    ids = GID(id.x  ,id.y-2,id.z  ,D);    lx += D2C6[1]*Lap[0][ids];  ly += D2C6[1]*Lap[1][ids];  lz += D2C6[1]*Lap[2][ids];
    ids = GID(id.x  ,id.y-1,id.z  ,D);    lx += D2C6[2]*Lap[0][ids];  ly += D2C6[2]*Lap[1][ids];  lz += D2C6[2]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C6[3]*Lap[0][ids];  ly += D2C6[3]*Lap[1][ids];  lz += D2C6[3]*Lap[2][ids];
    ids = GID(id.x  ,id.y+1,id.z  ,D);    lx += D2C6[4]*Lap[0][ids];  ly += D2C6[4]*Lap[1][ids];  lz += D2C6[4]*Lap[2][ids];
    ids = GID(id.x  ,id.y+2,id.z  ,D);    lx += D2C6[5]*Lap[0][ids];  ly += D2C6[5]*Lap[1][ids];  lz += D2C6[5]*Lap[2][ids];
    ids = GID(id.x  ,id.y+3,id.z  ,D);    lx += D2C6[6]*Lap[0][ids];  ly += D2C6[6]*Lap[1][ids];  lz += D2C6[6]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-3,D);    lx += D2C6[0]*Lap[0][ids];  ly += D2C6[0]*Lap[1][ids];  lz += D2C6[0]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-2,D);    lx += D2C6[1]*Lap[0][ids];  ly += D2C6[1]*Lap[1][ids];  lz += D2C6[1]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-1,D);    lx += D2C6[2]*Lap[0][ids];  ly += D2C6[2]*Lap[1][ids];  lz += D2C6[2]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C6[3]*Lap[0][ids];  ly += D2C6[3]*Lap[1][ids];  lz += D2C6[3]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+1,D);    lx += D2C6[4]*Lap[0][ids];  ly += D2C6[4]*Lap[1][ids];  lz += D2C6[4]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+2,D);    lx += D2C6[5]*Lap[0][ids];  ly += D2C6[5]*Lap[1][ids];  lz += D2C6[5]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+3,D);    lx += D2C6[6]*Lap[0][ids];  ly += D2C6[6]*Lap[1][ids];  lz += D2C6[6]*Lap[2][ids];

    Real H2 = H*H;
    int tid = GID(id.x,id.y,id.z,D);
    dgdt_Array[0][tid] -= C_hv*lx*H2;
    dgdt_Array[1][tid] -= C_hv*ly*H2;
    dgdt_Array[2][tid] -= C_hv*lz*H2;
}

inline void KER_HypVisc_FD8(const TensorGrid &Lap, TensorGrid &dgdt_Array, const dim3 &id, const dim3 &D, const Real &H, Real C_hv)
{
    // This kernel implements the hyperviscosity operator described in Cocle et al. doi 10.1016/j.jcp.2007.10.010
    // Input Lap is the Laplacian of the vorticity field (Contains the term 1/H2)

    //--- Calculate Laplacian (2nd order central finite difference)
    Real lx=0., ly=0., lz=0.;
    int ids;
    ids = GID(id.x-4,id.y  ,id.z  ,D);    lx += D2C8[0]*Lap[0][ids];  ly += D2C8[0]*Lap[1][ids];  lz += D2C8[0]*Lap[2][ids];
    ids = GID(id.x-3,id.y  ,id.z  ,D);    lx += D2C8[1]*Lap[0][ids];  ly += D2C8[1]*Lap[1][ids];  lz += D2C8[1]*Lap[2][ids];
    ids = GID(id.x-2,id.y  ,id.z  ,D);    lx += D2C8[2]*Lap[0][ids];  ly += D2C8[2]*Lap[1][ids];  lz += D2C8[2]*Lap[2][ids];
    ids = GID(id.x-1,id.y  ,id.z  ,D);    lx += D2C8[3]*Lap[0][ids];  ly += D2C8[3]*Lap[1][ids];  lz += D2C8[3]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C8[4]*Lap[0][ids];  ly += D2C8[4]*Lap[1][ids];  lz += D2C8[4]*Lap[2][ids];
    ids = GID(id.x+1,id.y  ,id.z  ,D);    lx += D2C8[5]*Lap[0][ids];  ly += D2C8[5]*Lap[1][ids];  lz += D2C8[5]*Lap[2][ids];
    ids = GID(id.x+2,id.y  ,id.z  ,D);    lx += D2C8[6]*Lap[0][ids];  ly += D2C8[6]*Lap[1][ids];  lz += D2C8[6]*Lap[2][ids];
    ids = GID(id.x+3,id.y  ,id.z  ,D);    lx += D2C8[7]*Lap[0][ids];  ly += D2C8[7]*Lap[1][ids];  lz += D2C8[7]*Lap[2][ids];
    ids = GID(id.x+4,id.y  ,id.z  ,D);    lx += D2C8[8]*Lap[0][ids];  ly += D2C8[8]*Lap[1][ids];  lz += D2C8[8]*Lap[2][ids];
    ids = GID(id.x  ,id.y-4,id.z  ,D);    lx += D2C8[0]*Lap[0][ids];  ly += D2C8[0]*Lap[1][ids];  lz += D2C8[0]*Lap[2][ids];
    ids = GID(id.x  ,id.y-3,id.z  ,D);    lx += D2C8[1]*Lap[0][ids];  ly += D2C8[1]*Lap[1][ids];  lz += D2C8[1]*Lap[2][ids];
    ids = GID(id.x  ,id.y-2,id.z  ,D);    lx += D2C8[2]*Lap[0][ids];  ly += D2C8[2]*Lap[1][ids];  lz += D2C8[2]*Lap[2][ids];
    ids = GID(id.x  ,id.y-1,id.z  ,D);    lx += D2C8[3]*Lap[0][ids];  ly += D2C8[3]*Lap[1][ids];  lz += D2C8[3]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C8[4]*Lap[0][ids];  ly += D2C8[4]*Lap[1][ids];  lz += D2C8[4]*Lap[2][ids];
    ids = GID(id.x  ,id.y+1,id.z  ,D);    lx += D2C8[5]*Lap[0][ids];  ly += D2C8[5]*Lap[1][ids];  lz += D2C8[5]*Lap[2][ids];
    ids = GID(id.x  ,id.y+2,id.z  ,D);    lx += D2C8[6]*Lap[0][ids];  ly += D2C8[6]*Lap[1][ids];  lz += D2C8[6]*Lap[2][ids];
    ids = GID(id.x  ,id.y+3,id.z  ,D);    lx += D2C8[7]*Lap[0][ids];  ly += D2C8[7]*Lap[1][ids];  lz += D2C8[7]*Lap[2][ids];
    ids = GID(id.x  ,id.y+4,id.z  ,D);    lx += D2C8[8]*Lap[0][ids];  ly += D2C8[8]*Lap[1][ids];  lz += D2C8[8]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-4,D);    lx += D2C8[0]*Lap[0][ids];  ly += D2C8[0]*Lap[1][ids];  lz += D2C8[0]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-3,D);    lx += D2C8[1]*Lap[0][ids];  ly += D2C8[1]*Lap[1][ids];  lz += D2C8[1]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-2,D);    lx += D2C8[2]*Lap[0][ids];  ly += D2C8[2]*Lap[1][ids];  lz += D2C8[2]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z-1,D);    lx += D2C8[3]*Lap[0][ids];  ly += D2C8[3]*Lap[1][ids];  lz += D2C8[3]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += D2C8[4]*Lap[0][ids];  ly += D2C8[4]*Lap[1][ids];  lz += D2C8[4]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+1,D);    lx += D2C8[5]*Lap[0][ids];  ly += D2C8[5]*Lap[1][ids];  lz += D2C8[5]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+2,D);    lx += D2C8[6]*Lap[0][ids];  ly += D2C8[6]*Lap[1][ids];  lz += D2C8[6]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+3,D);    lx += D2C8[7]*Lap[0][ids];  ly += D2C8[7]*Lap[1][ids];  lz += D2C8[7]*Lap[2][ids];
    ids = GID(id.x  ,id.y  ,id.z+4,D);    lx += D2C8[8]*Lap[0][ids];  ly += D2C8[8]*Lap[1][ids];  lz += D2C8[8]*Lap[2][ids];

    Real H2 = H*H;
    int tid = GID(id.x,id.y,id.z,D);
    dgdt_Array[0][tid] -= C_hv*lx*H2;
    dgdt_Array[1][tid] -= C_hv*ly*H2;
    dgdt_Array[2][tid] -= C_hv*lz*H2;
}

//----------------------------------

// Regularized Variational Multiscale (RVM) Model

// static const Real C_rvmn1 = 1.39*(0.3*0.3*0.3);         // Smagorinski scale (first order filter)
// static const Real C_rvmn2 = 1.27*C_rvmn1;               // Smagorinski scale (second order filter)

enum Axis {Ex,Ey,Ez};

inline void KER_RVM_SS(const TensorGrid &g, const TensorGrid &g_filt, TensorGrid &g_ss, const int &id)
{
    // This kernel extracts the small-scale vorticity from the given field.

    // Small-scale vorticity.
    g_ss[0][id] = g[0][id]-g_filt[0][id];
    g_ss[1][id] = g[1][id]-g_filt[1][id];
    g_ss[2][id] = g[2][id]-g_filt[2][id];
}

inline void KER_RVM_SGS(const TensorGrid &GradU, RVector &SGS, const int &id, const Real &H, const Real &RVMConst)
{
    // The sub-grid scale viscosity is calcualted here as described in Appendix A of the thesis of R. Cocle

    Real s11 = 0.5*(GradU[0][id] + GradU[0][id]);
    Real s12 = 0.5*(GradU[1][id] + GradU[3][id]);
    Real s13 = 0.5*(GradU[2][id] + GradU[6][id]);
    Real s21 = s12;
    Real s22 = 0.5*(GradU[4][id] + GradU[4][id]);
    Real s23 = 0.5*(GradU[5][id] + GradU[7][id]);
    // Real s31 = 0.5*(GradU[0][id] + GradU[0][id]);       // Should be s13!!!
    Real s31 = s13;
    Real s32 = s23;
    Real s33 = 0.5*(GradU[8][id] + GradU[8][id]);
    Real s_ij2 = s11*s11 + s12*s12 + s13*s13 + s21*s21 + s22*s22 + s23*s23 + s31*s31 + s32*s32 + s33*s33;
    SGS[id] = RVMConst*H*H*sqrt(2.0*s_ij2);
}

inline void KER_QCriterion(const TensorGrid &GradU, RVector &Q, const int &id, const Real &H, const Real &RVMConst)
{
    // The sub-grid scale viscosity is calcualted here as described in Appendix A of the thesis of R. Cocle

    Real s11 = 0.5*(GradU[0][id] + GradU[0][id]), q11 = 0.0;
    Real s12 = 0.5*(GradU[1][id] + GradU[3][id]), q12 = 0.5*(GradU[1][id] - GradU[3][id]);
    Real s13 = 0.5*(GradU[2][id] + GradU[6][id]), q13 = 0.5*(GradU[2][id] - GradU[6][id]);
    Real s21 = 0.5*(GradU[3][id] + GradU[1][id]), q21 = 0.5*(GradU[3][id] - GradU[1][id]);
    Real s22 = 0.5*(GradU[4][id] + GradU[4][id]), q22 = 0.0;
    Real s23 = 0.5*(GradU[5][id] + GradU[7][id]), q23 = 0.5*(GradU[5][id] - GradU[7][id]);
    Real s31 = 0.5*(GradU[6][id] + GradU[2][id]), q31 = 0.5*(GradU[6][id] - GradU[2][id]);
    Real s32 = 0.5*(GradU[7][id] + GradU[5][id]), q32 = 0.5*(GradU[7][id] - GradU[5][id]);
    Real s33 = 0.5*(GradU[8][id] + GradU[8][id]), q33 = 0.0;
    Real s2 = s11*s11 + s12*s12 + s13*s13 + s21*s21 + s22*s22 + s23*s23 + s31*s31 + s32*s32 + s33*s33;
    Real q2 = q11*q11 + q12*q12 + q13*q13 + q21*q21 + q22*q22 + q23*q23 + q31*q31 + q32*q32 + q33*q33;
    Q[id] = 0.5*(q2-s2);
}

inline void KER_SG_Disc_Filter(const TensorGrid &g, TensorGrid &g_filt, const dim3 &id, const dim3 &D, const Axis &A)
{
    // This kernel implements the Discrete SGS filter operator described in Jeanmart & Winckelmans doi 10.1063/1.2728935

    int ID1, ID2 = GID(id.x,id.y,id.z,D), ID3;
    switch (A){
    case (Ex):  {ID1 = GID(id.x-1,id.y,id.z,D); ID3 = GID(id.x+1,id.y,id.z,D);   break;}    // X Filter
    case (Ey):  {ID1 = GID(id.x,id.y-1,id.z,D); ID3 = GID(id.x,id.y+1,id.z,D);   break;}    // Y Filter
    case (Ez):  {ID1 = GID(id.x,id.y,id.z-1,D); ID3 = GID(id.x,id.y,id.z+1,D);   break;}    // Z Filter
    default: {break;}
    }

    g_filt[0][ID2] = 0.25*g[0][ID1] + 0.5*g[0][ID2] + 0.25*g[0][ID3];
    g_filt[1][ID2] = 0.25*g[1][ID1] + 0.5*g[1][ID2] + 0.25*g[1][ID3];
    g_filt[2][ID2] = 0.25*g[2][ID1] + 0.5*g[2][ID2] + 0.25*g[2][ID3];
}

inline void KER_RVM_FD2(const TensorGrid &gss, const RVector &sgs, TensorGrid &dgdt, const dim3 &id, const dim3 &D, const Real &H)
{
    // This kernel implements the RVM as described in Appendix A of the thesis of R. Cocle. (Here however the standard 7-point Laplacian FD stencil is applied

    // Average subgrid scales
    Real f = 0.5;
    int tid = GID(id.x,id.y,id.z,D);
    Real sgstid = sgs[tid];
    Real sx[3] = {f*(sgs[GID(id.x-1,id.y  ,id.z  ,D)] + sgstid), sgstid, f*(sgs[GID(id.x+1,id.y  ,id.z  ,D)] + sgstid)};
    Real sy[3] = {f*(sgs[GID(id.x  ,id.y-1,id.z  ,D)] + sgstid), sgstid, f*(sgs[GID(id.x  ,id.y+1,id.z  ,D)] + sgstid)};
    Real sz[3] = {f*(sgs[GID(id.x  ,id.y  ,id.z-1,D)] + sgstid), sgstid, f*(sgs[GID(id.x  ,id.y  ,id.z+1,D)] + sgstid)};

    Real lx=0., ly=0., lz=0.;
    Real gssx = gss[0][tid];
    Real gssy = gss[1][tid];
    Real gssz = gss[2][tid];
    int ids;
    ids = GID(id.x-1,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[0]*D2C2[0];  ly += (gss[1][ids]-gssy)*sx[0]*D2C2[0];  lz += (gss[2][ids]-gssz)*sx[0]*D2C2[0];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[1]*D2C2[1];  ly += (gss[1][ids]-gssy)*sx[1]*D2C2[1];  lz += (gss[2][ids]-gssz)*sx[1]*D2C2[1];
    ids = GID(id.x+1,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[2]*D2C2[2];  ly += (gss[1][ids]-gssy)*sx[2]*D2C2[2];  lz += (gss[2][ids]-gssz)*sx[2]*D2C2[2];
    ids = GID(id.x  ,id.y-1,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[0]*D2C2[0];  ly += (gss[1][ids]-gssy)*sy[0]*D2C2[0];  lz += (gss[2][ids]-gssz)*sy[0]*D2C2[0];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[1]*D2C2[1];  ly += (gss[1][ids]-gssy)*sy[1]*D2C2[1];  lz += (gss[2][ids]-gssz)*sy[1]*D2C2[1];
    ids = GID(id.x  ,id.y+1,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[2]*D2C2[2];  ly += (gss[1][ids]-gssy)*sy[2]*D2C2[2];  lz += (gss[2][ids]-gssz)*sy[2]*D2C2[2];
    ids = GID(id.x  ,id.y  ,id.z-1,D);    lx += (gss[0][ids]-gssx)*sz[0]*D2C2[0];  ly += (gss[1][ids]-gssy)*sz[0]*D2C2[0];  lz += (gss[2][ids]-gssz)*sz[0]*D2C2[0];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sz[1]*D2C2[1];  ly += (gss[1][ids]-gssy)*sz[1]*D2C2[1];  lz += (gss[2][ids]-gssz)*sz[1]*D2C2[1];
    ids = GID(id.x  ,id.y  ,id.z+1,D);    lx += (gss[0][ids]-gssx)*sz[2]*D2C2[2];  ly += (gss[1][ids]-gssy)*sz[2]*D2C2[2];  lz += (gss[2][ids]-gssz)*sz[2]*D2C2[2];

    Real InvH2 = 1.0/H/H;
    dgdt[0][tid] += lx*InvH2;
    dgdt[1][tid] += ly*InvH2;
    dgdt[2][tid] += lz*InvH2;
}

inline void KER_RVM_FD4(const TensorGrid &gss, const RVector &sgs, TensorGrid &dgdt, const dim3 &id, const dim3 &D, const Real &H)
{
    // This kernel implements the RVM as described in Appendix A of the thesis of R. Cocle. (Here however the standard 7-point Laplacian FD stencil is applied

    // Average subgrid scales
    Real f = 0.5;
    int tid = GID(id.x,id.y,id.z,D);
    Real sgstid = sgs[tid];
    Real sx[5] = {  f*(sgs[GID(id.x-2,id.y  ,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x-1,id.y  ,id.z  ,D)] + sgstid),
                  sgstid,
                  f*(sgs[GID(id.x+1,id.y  ,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x+2,id.y  ,id.z  ,D)] + sgstid)};

    Real sy[5] = {  f*(sgs[GID(id.x  ,id.y-2,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y-1,id.z  ,D)] + sgstid),
                  sgstid,
                  f*(sgs[GID(id.x  ,id.y+1,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y+2,id.z  ,D)] + sgstid)};

    Real sz[5] = {  f*(sgs[GID(id.x  ,id.y  ,id.z-2,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y  ,id.z-1,D)] + sgstid),
                  sgstid,
                  f*(sgs[GID(id.x  ,id.y  ,id.z+1,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y  ,id.z+2,D)] + sgstid)};

    Real lx=0., ly=0., lz=0.;
    Real gssx = gss[0][tid];
    Real gssy = gss[1][tid];
    Real gssz = gss[2][tid];
    int ids;
    ids = GID(id.x-2,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[0]*D2C4[0];  ly += (gss[1][ids]-gssy)*sx[0]*D2C4[0];  lz += (gss[2][ids]-gssz)*sx[0]*D2C4[0];
    ids = GID(id.x-1,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[1]*D2C4[1];  ly += (gss[1][ids]-gssy)*sx[1]*D2C4[1];  lz += (gss[2][ids]-gssz)*sx[1]*D2C4[1];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[2]*D2C4[2];  ly += (gss[1][ids]-gssy)*sx[2]*D2C4[2];  lz += (gss[2][ids]-gssz)*sx[2]*D2C4[2];
    ids = GID(id.x+1,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[3]*D2C4[3];  ly += (gss[1][ids]-gssy)*sx[3]*D2C4[3];  lz += (gss[2][ids]-gssz)*sx[3]*D2C4[3];
    ids = GID(id.x+2,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[4]*D2C4[4];  ly += (gss[1][ids]-gssy)*sx[4]*D2C4[4];  lz += (gss[2][ids]-gssz)*sx[4]*D2C4[4];
    ids = GID(id.x  ,id.y-2,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[0]*D2C4[0];  ly += (gss[1][ids]-gssy)*sy[0]*D2C4[0];  lz += (gss[2][ids]-gssz)*sy[0]*D2C4[0];
    ids = GID(id.x  ,id.y-1,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[1]*D2C4[1];  ly += (gss[1][ids]-gssy)*sy[1]*D2C4[1];  lz += (gss[2][ids]-gssz)*sy[1]*D2C4[1];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[2]*D2C4[2];  ly += (gss[1][ids]-gssy)*sy[2]*D2C4[2];  lz += (gss[2][ids]-gssz)*sy[2]*D2C4[2];
    ids = GID(id.x  ,id.y+1,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[3]*D2C4[3];  ly += (gss[1][ids]-gssy)*sy[3]*D2C4[3];  lz += (gss[2][ids]-gssz)*sy[3]*D2C4[3];
    ids = GID(id.x  ,id.y+2,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[4]*D2C4[4];  ly += (gss[1][ids]-gssy)*sy[4]*D2C4[4];  lz += (gss[2][ids]-gssz)*sy[4]*D2C4[4];
    ids = GID(id.x  ,id.y  ,id.z-2,D);    lx += (gss[0][ids]-gssx)*sz[0]*D2C4[0];  ly += (gss[1][ids]-gssy)*sz[0]*D2C4[0];  lz += (gss[2][ids]-gssz)*sz[0]*D2C4[0];
    ids = GID(id.x  ,id.y  ,id.z-1,D);    lx += (gss[0][ids]-gssx)*sz[1]*D2C4[1];  ly += (gss[1][ids]-gssy)*sz[1]*D2C4[1];  lz += (gss[2][ids]-gssz)*sz[1]*D2C4[1];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sz[2]*D2C4[2];  ly += (gss[1][ids]-gssy)*sz[2]*D2C4[2];  lz += (gss[2][ids]-gssz)*sz[2]*D2C4[2];
    ids = GID(id.x  ,id.y  ,id.z+1,D);    lx += (gss[0][ids]-gssx)*sz[3]*D2C4[3];  ly += (gss[1][ids]-gssy)*sz[3]*D2C4[3];  lz += (gss[2][ids]-gssz)*sz[3]*D2C4[3];
    ids = GID(id.x  ,id.y  ,id.z+2,D);    lx += (gss[0][ids]-gssx)*sz[4]*D2C4[4];  ly += (gss[1][ids]-gssy)*sz[4]*D2C4[4];  lz += (gss[2][ids]-gssz)*sz[4]*D2C4[4];

    Real InvH2 = 1.0/H/H;
    dgdt[0][tid] += lx*InvH2;
    dgdt[1][tid] += ly*InvH2;
    dgdt[2][tid] += lz*InvH2;
}

inline void KER_RVM_FD6(const TensorGrid &gss, const RVector &sgs, TensorGrid &dgdt, const dim3 &id, const dim3 &D, const Real &H)
{
    // This kernel implements the RVM as described in Appendix A of the thesis of R. Cocle. (Here however the standard 7-point Laplacian FD stencil is applied

    // Average subgrid scales
    Real f = 0.5;
    int tid = GID(id.x,id.y,id.z,D);
    Real sgstid = sgs[tid];
    Real sx[7] = {  f*(sgs[GID(id.x-3,id.y  ,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x-2,id.y  ,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x-1,id.y  ,id.z  ,D)] + sgstid),
                  sgstid,
                  f*(sgs[GID(id.x+1,id.y  ,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x+2,id.y  ,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x+3,id.y  ,id.z  ,D)] + sgstid)};

    Real sy[7] = {  f*(sgs[GID(id.x  ,id.y-3,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y-2,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y-1,id.z  ,D)] + sgstid),
                  sgstid,
                  f*(sgs[GID(id.x  ,id.y+1,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y+2,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y+3,id.z  ,D)] + sgstid)};

    Real sz[7] = {  f*(sgs[GID(id.x  ,id.y  ,id.z-3,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y  ,id.z-2,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y  ,id.z-1,D)] + sgstid),
                  sgstid,
                  f*(sgs[GID(id.x  ,id.y  ,id.z+1,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y  ,id.z+2,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y  ,id.z+3,D)] + sgstid)};

    Real lx=0., ly=0., lz=0.;
    Real gssx = gss[0][tid];
    Real gssy = gss[1][tid];
    Real gssz = gss[2][tid];
    int ids;
    ids = GID(id.x-3,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[0]*D2C6[0];  ly += (gss[1][ids]-gssy)*sx[0]*D2C6[0];  lz += (gss[2][ids]-gssz)*sx[0]*D2C6[0];
    ids = GID(id.x-2,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[1]*D2C6[1];  ly += (gss[1][ids]-gssy)*sx[1]*D2C6[1];  lz += (gss[2][ids]-gssz)*sx[1]*D2C6[1];
    ids = GID(id.x-1,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[2]*D2C6[2];  ly += (gss[1][ids]-gssy)*sx[2]*D2C6[2];  lz += (gss[2][ids]-gssz)*sx[2]*D2C6[2];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[3]*D2C6[3];  ly += (gss[1][ids]-gssy)*sx[3]*D2C6[3];  lz += (gss[2][ids]-gssz)*sx[3]*D2C6[3];
    ids = GID(id.x+1,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[4]*D2C6[4];  ly += (gss[1][ids]-gssy)*sx[4]*D2C6[4];  lz += (gss[2][ids]-gssz)*sx[4]*D2C6[4];
    ids = GID(id.x+2,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[5]*D2C6[5];  ly += (gss[1][ids]-gssy)*sx[5]*D2C6[5];  lz += (gss[2][ids]-gssz)*sx[5]*D2C6[5];
    ids = GID(id.x+3,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[6]*D2C6[6];  ly += (gss[1][ids]-gssy)*sx[6]*D2C6[6];  lz += (gss[2][ids]-gssz)*sx[6]*D2C6[6];
    ids = GID(id.x  ,id.y-3,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[0]*D2C6[0];  ly += (gss[1][ids]-gssy)*sy[0]*D2C6[0];  lz += (gss[2][ids]-gssz)*sy[0]*D2C6[0];
    ids = GID(id.x  ,id.y-2,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[1]*D2C6[1];  ly += (gss[1][ids]-gssy)*sy[1]*D2C6[1];  lz += (gss[2][ids]-gssz)*sy[1]*D2C6[1];
    ids = GID(id.x  ,id.y-1,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[2]*D2C6[2];  ly += (gss[1][ids]-gssy)*sy[2]*D2C6[2];  lz += (gss[2][ids]-gssz)*sy[2]*D2C6[2];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[3]*D2C6[3];  ly += (gss[1][ids]-gssy)*sy[3]*D2C6[3];  lz += (gss[2][ids]-gssz)*sy[3]*D2C6[3];
    ids = GID(id.x  ,id.y+1,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[4]*D2C6[4];  ly += (gss[1][ids]-gssy)*sy[4]*D2C6[4];  lz += (gss[2][ids]-gssz)*sy[4]*D2C6[4];
    ids = GID(id.x  ,id.y+2,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[5]*D2C6[5];  ly += (gss[1][ids]-gssy)*sy[5]*D2C6[5];  lz += (gss[2][ids]-gssz)*sy[5]*D2C6[5];
    ids = GID(id.x  ,id.y+3,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[6]*D2C6[6];  ly += (gss[1][ids]-gssy)*sy[6]*D2C6[6];  lz += (gss[2][ids]-gssz)*sy[6]*D2C6[6];
    ids = GID(id.x  ,id.y  ,id.z-3,D);    lx += (gss[0][ids]-gssx)*sz[0]*D2C6[0];  ly += (gss[1][ids]-gssy)*sz[0]*D2C6[0];  lz += (gss[2][ids]-gssz)*sz[0]*D2C6[0];
    ids = GID(id.x  ,id.y  ,id.z-2,D);    lx += (gss[0][ids]-gssx)*sz[1]*D2C6[1];  ly += (gss[1][ids]-gssy)*sz[1]*D2C6[1];  lz += (gss[2][ids]-gssz)*sz[1]*D2C6[1];
    ids = GID(id.x  ,id.y  ,id.z-1,D);    lx += (gss[0][ids]-gssx)*sz[2]*D2C6[2];  ly += (gss[1][ids]-gssy)*sz[2]*D2C6[2];  lz += (gss[2][ids]-gssz)*sz[2]*D2C6[2];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sz[3]*D2C6[3];  ly += (gss[1][ids]-gssy)*sz[3]*D2C6[3];  lz += (gss[2][ids]-gssz)*sz[3]*D2C6[3];
    ids = GID(id.x  ,id.y  ,id.z+1,D);    lx += (gss[0][ids]-gssx)*sz[4]*D2C6[4];  ly += (gss[1][ids]-gssy)*sz[4]*D2C6[4];  lz += (gss[2][ids]-gssz)*sz[4]*D2C6[4];
    ids = GID(id.x  ,id.y  ,id.z+2,D);    lx += (gss[0][ids]-gssx)*sz[5]*D2C6[5];  ly += (gss[1][ids]-gssy)*sz[5]*D2C6[5];  lz += (gss[2][ids]-gssz)*sz[5]*D2C6[5];
    ids = GID(id.x  ,id.y  ,id.z+3,D);    lx += (gss[0][ids]-gssx)*sz[6]*D2C6[6];  ly += (gss[1][ids]-gssy)*sz[6]*D2C6[6];  lz += (gss[2][ids]-gssz)*sz[6]*D2C6[6];

    Real InvH2 = 1.0/H/H;
    dgdt[0][tid] += lx*InvH2;
    dgdt[1][tid] += ly*InvH2;
    dgdt[2][tid] += lz*InvH2;
}

inline void KER_RVM_FD8(const TensorGrid &gss, const RVector &sgs, TensorGrid &dgdt, const dim3 &id, const dim3 &D, const Real &H)
{
    // This kernel implements the RVM as described in Appendix A of the thesis of R. Cocle. (Here however the standard 7-point Laplacian FD stencil is applied

    // Average subgrid scales
    Real f = 0.5;
    int tid = GID(id.x,id.y,id.z,D);
    Real sgstid = sgs[tid];
    Real sx[9] = {  f*(sgs[GID(id.x-4,id.y  ,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x-3,id.y  ,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x-2,id.y  ,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x-1,id.y  ,id.z  ,D)] + sgstid),
                  sgstid,
                  f*(sgs[GID(id.x+1,id.y  ,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x+2,id.y  ,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x+3,id.y  ,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x+4,id.y  ,id.z  ,D)] + sgstid)};

    Real sy[9] = {  f*(sgs[GID(id.x  ,id.y-4,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y-3,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y-2,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y-1,id.z  ,D)] + sgstid),
                  sgstid,
                  f*(sgs[GID(id.x  ,id.y+1,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y+2,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y+3,id.z  ,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y+4,id.z  ,D)] + sgstid)};

    Real sz[9] = {  f*(sgs[GID(id.x  ,id.y  ,id.z-4,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y  ,id.z-3,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y  ,id.z-2,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y  ,id.z-1,D)] + sgstid),
                  sgstid,
                  f*(sgs[GID(id.x  ,id.y  ,id.z+1,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y  ,id.z+2,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y  ,id.z+3,D)] + sgstid),
                  f*(sgs[GID(id.x  ,id.y  ,id.z+4,D)] + sgstid)};

    Real lx=0., ly=0., lz=0.;
    Real gssx = gss[0][tid];
    Real gssy = gss[1][tid];
    Real gssz = gss[2][tid];
    int ids;
    ids = GID(id.x-4,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[0]*D2C8[0];  ly += (gss[1][ids]-gssy)*sx[0]*D2C8[0];  lz += (gss[2][ids]-gssz)*sx[0]*D2C8[0];
    ids = GID(id.x-3,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[1]*D2C8[1];  ly += (gss[1][ids]-gssy)*sx[1]*D2C8[1];  lz += (gss[2][ids]-gssz)*sx[1]*D2C8[1];
    ids = GID(id.x-2,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[2]*D2C8[2];  ly += (gss[1][ids]-gssy)*sx[2]*D2C8[2];  lz += (gss[2][ids]-gssz)*sx[2]*D2C8[2];
    ids = GID(id.x-1,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[3]*D2C8[3];  ly += (gss[1][ids]-gssy)*sx[3]*D2C8[3];  lz += (gss[2][ids]-gssz)*sx[3]*D2C8[3];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[4]*D2C8[4];  ly += (gss[1][ids]-gssy)*sx[4]*D2C8[4];  lz += (gss[2][ids]-gssz)*sx[4]*D2C8[4];
    ids = GID(id.x+1,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[5]*D2C8[5];  ly += (gss[1][ids]-gssy)*sx[5]*D2C8[5];  lz += (gss[2][ids]-gssz)*sx[5]*D2C8[5];
    ids = GID(id.x+2,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[6]*D2C8[6];  ly += (gss[1][ids]-gssy)*sx[6]*D2C8[6];  lz += (gss[2][ids]-gssz)*sx[6]*D2C8[6];
    ids = GID(id.x+3,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[7]*D2C8[7];  ly += (gss[1][ids]-gssy)*sx[7]*D2C8[7];  lz += (gss[2][ids]-gssz)*sx[7]*D2C8[7];
    ids = GID(id.x+4,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sx[8]*D2C8[8];  ly += (gss[1][ids]-gssy)*sx[8]*D2C8[8];  lz += (gss[2][ids]-gssz)*sx[8]*D2C8[8];
    ids = GID(id.x  ,id.y-4,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[0]*D2C8[0];  ly += (gss[1][ids]-gssy)*sy[0]*D2C8[0];  lz += (gss[2][ids]-gssz)*sy[0]*D2C8[0];
    ids = GID(id.x  ,id.y-3,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[1]*D2C8[1];  ly += (gss[1][ids]-gssy)*sy[1]*D2C8[1];  lz += (gss[2][ids]-gssz)*sy[1]*D2C8[1];
    ids = GID(id.x  ,id.y-2,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[2]*D2C8[2];  ly += (gss[1][ids]-gssy)*sy[2]*D2C8[2];  lz += (gss[2][ids]-gssz)*sy[2]*D2C8[2];
    ids = GID(id.x  ,id.y-1,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[3]*D2C8[3];  ly += (gss[1][ids]-gssy)*sy[3]*D2C8[3];  lz += (gss[2][ids]-gssz)*sy[3]*D2C8[3];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[4]*D2C8[4];  ly += (gss[1][ids]-gssy)*sy[4]*D2C8[4];  lz += (gss[2][ids]-gssz)*sy[4]*D2C8[4];
    ids = GID(id.x  ,id.y+1,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[5]*D2C8[5];  ly += (gss[1][ids]-gssy)*sy[5]*D2C8[5];  lz += (gss[2][ids]-gssz)*sy[5]*D2C8[5];
    ids = GID(id.x  ,id.y+2,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[6]*D2C8[6];  ly += (gss[1][ids]-gssy)*sy[6]*D2C8[6];  lz += (gss[2][ids]-gssz)*sy[6]*D2C8[6];
    ids = GID(id.x  ,id.y+3,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[7]*D2C8[7];  ly += (gss[1][ids]-gssy)*sy[7]*D2C8[7];  lz += (gss[2][ids]-gssz)*sy[7]*D2C8[7];
    ids = GID(id.x  ,id.y+4,id.z  ,D);    lx += (gss[0][ids]-gssx)*sy[8]*D2C8[8];  ly += (gss[1][ids]-gssy)*sy[8]*D2C8[8];  lz += (gss[2][ids]-gssz)*sy[8]*D2C8[8];
    ids = GID(id.x  ,id.y  ,id.z-4,D);    lx += (gss[0][ids]-gssx)*sz[0]*D2C8[0];  ly += (gss[1][ids]-gssy)*sz[0]*D2C8[0];  lz += (gss[2][ids]-gssz)*sz[0]*D2C8[0];
    ids = GID(id.x  ,id.y  ,id.z-3,D);    lx += (gss[0][ids]-gssx)*sz[1]*D2C8[1];  ly += (gss[1][ids]-gssy)*sz[1]*D2C8[1];  lz += (gss[2][ids]-gssz)*sz[1]*D2C8[1];
    ids = GID(id.x  ,id.y  ,id.z-2,D);    lx += (gss[0][ids]-gssx)*sz[2]*D2C8[2];  ly += (gss[1][ids]-gssy)*sz[2]*D2C8[2];  lz += (gss[2][ids]-gssz)*sz[2]*D2C8[2];
    ids = GID(id.x  ,id.y  ,id.z-1,D);    lx += (gss[0][ids]-gssx)*sz[3]*D2C8[3];  ly += (gss[1][ids]-gssy)*sz[3]*D2C8[3];  lz += (gss[2][ids]-gssz)*sz[3]*D2C8[3];
    ids = GID(id.x  ,id.y  ,id.z  ,D);    lx += (gss[0][ids]-gssx)*sz[4]*D2C8[4];  ly += (gss[1][ids]-gssy)*sz[4]*D2C8[4];  lz += (gss[2][ids]-gssz)*sz[4]*D2C8[4];
    ids = GID(id.x  ,id.y  ,id.z+1,D);    lx += (gss[0][ids]-gssx)*sz[5]*D2C8[5];  ly += (gss[1][ids]-gssy)*sz[5]*D2C8[5];  lz += (gss[2][ids]-gssz)*sz[5]*D2C8[5];
    ids = GID(id.x  ,id.y  ,id.z+2,D);    lx += (gss[0][ids]-gssx)*sz[6]*D2C8[6];  ly += (gss[1][ids]-gssy)*sz[6]*D2C8[6];  lz += (gss[2][ids]-gssz)*sz[6]*D2C8[6];
    ids = GID(id.x  ,id.y  ,id.z+3,D);    lx += (gss[0][ids]-gssx)*sz[7]*D2C8[7];  ly += (gss[1][ids]-gssy)*sz[7]*D2C8[7];  lz += (gss[2][ids]-gssz)*sz[7]*D2C8[7];
    ids = GID(id.x  ,id.y  ,id.z+4,D);    lx += (gss[0][ids]-gssx)*sz[8]*D2C8[8];  ly += (gss[1][ids]-gssy)*sz[8]*D2C8[8];  lz += (gss[2][ids]-gssz)*sz[8]*D2C8[8];

    Real InvH2 = 1.0/H/H;
    dgdt[0][tid] += lx*InvH2;
    dgdt[1][tid] += ly*InvH2;
    dgdt[2][tid] += lz*InvH2;
}

//--- Integration schemes

inline void KER_Update(TensorGrid &p_d_Vals,
                       TensorGrid &p_o_Vals,
                       const TensorGrid &dpdt_d_Vals,
                       const TensorGrid &dpdt_o_Vals,
                       const int &id, const Real &dT)
{
    // Update the particle property with Eulerian forward (either position or strength)
    p_d_Vals[0][id] += dpdt_d_Vals[0][id]*dT;
    p_d_Vals[1][id] += dpdt_d_Vals[1][id]*dT;
    p_d_Vals[2][id] += dpdt_d_Vals[2][id]*dT;
    p_o_Vals[0][id] += dpdt_o_Vals[0][id]*dT;
    p_o_Vals[1][id] += dpdt_o_Vals[1][id]*dT;
    p_o_Vals[2][id] += dpdt_o_Vals[2][id]*dT;
}


inline void KER_Update_RK(const TensorGrid &p_d_Vals,
                          const TensorGrid &p_o_Vals,
                          const TensorGrid &dpdt_d_Vals,
                          const TensorGrid &dpdt_o_Vals,
                          TensorGrid &k_d_Array,
                          TensorGrid &k_o_Array,
                          const int &id, const Real &dT)
{
    // Update the particle property with Eulerian forward (either position or strength)
    k_d_Array[0][id] = p_d_Vals[0][id] + dpdt_d_Vals[0][id]*dT;
    k_d_Array[1][id] = p_d_Vals[1][id] + dpdt_d_Vals[1][id]*dT;
    k_d_Array[2][id] = p_d_Vals[2][id] + dpdt_d_Vals[2][id]*dT;
    k_o_Array[0][id] = p_o_Vals[0][id] + dpdt_o_Vals[0][id]*dT;
    k_o_Array[1][id] = p_o_Vals[1][id] + dpdt_o_Vals[1][id]*dT;
    k_o_Array[2][id] = p_o_Vals[2][id] + dpdt_o_Vals[2][id]*dT;
}

inline void KER_Update_RKLS(TensorGrid &p_d_Vals,
                            TensorGrid &p_o_Vals,
                            const TensorGrid &dpdt_d_Vals,
                            const TensorGrid &dpdt_o_Vals,
                            TensorGrid &int_d_Array,
                            TensorGrid &int_o_Array,
                            const Real &A, const Real &B,
                            const int &id, const Real &h)
{
    // Update the particle property for low-storage RK explicite scheme

    // Intermediate vars
    Real s21 = A*int_d_Array[0][id] + h*dpdt_d_Vals[0][id];
    Real s22 = A*int_d_Array[1][id] + h*dpdt_d_Vals[1][id];
    Real s23 = A*int_d_Array[2][id] + h*dpdt_d_Vals[2][id];
    Real s24 = A*int_o_Array[0][id] + h*dpdt_o_Vals[0][id];
    Real s25 = A*int_o_Array[1][id] + h*dpdt_o_Vals[1][id];
    Real s26 = A*int_o_Array[2][id] + h*dpdt_o_Vals[2][id];

    // Update intermediate vector
    int_d_Array[0][id] = s21;
    int_d_Array[1][id] = s22;
    int_d_Array[2][id] = s23;
    int_o_Array[0][id] = s24;
    int_o_Array[1][id] = s25;
    int_o_Array[2][id] = s26;

    // Update state vector
    p_d_Vals[0][id] += B*s21;
    p_d_Vals[1][id] += B*s22;
    p_d_Vals[2][id] += B*s23;
    p_o_Vals[0][id] += B*s24;
    p_o_Vals[1][id] += B*s25;
    p_o_Vals[2][id] += B*s26;
}

inline void KER_Update_RK2(TensorGrid &p_d_Vals,
                           TensorGrid &p_o_Vals,
                           const TensorGrid &k1d,
                           const TensorGrid &k1o,
                           const TensorGrid &k2d,
                           const TensorGrid &k2o,
                           const int &id,
                           const Real &dT)
{
    // Update the particle property with Eulerian forward (either position or strength)
    Real f = 0.5*dT;
    p_d_Vals[0][id] += f*(k1d[0][id]+k2d[0][id]);
    p_d_Vals[1][id] += f*(k1d[1][id]+k2d[1][id]);
    p_d_Vals[2][id] += f*(k1d[2][id]+k2d[2][id]);
    p_o_Vals[0][id] += f*(k1o[0][id]+k2o[0][id]);
    p_o_Vals[1][id] += f*(k1o[1][id]+k2o[1][id]);
    p_o_Vals[2][id] += f*(k1o[2][id]+k2o[2][id]);
}

inline void KER_Update_RK3(TensorGrid &p_d_Vals,
                           TensorGrid &p_o_Vals,
                           const TensorGrid &k1d,
                           const TensorGrid &k1o,
                           const TensorGrid &k2d,
                           const TensorGrid &k2o,
                           const TensorGrid &k3d,
                           const TensorGrid &k3o,
                           const int &id,
                           const Real &dT)
{
    // Update the particle property with Eulerian forward (either position or strength)
    Real f = 1.0/6.0*dT;
    p_d_Vals[0][id] += f*(k1d[0][id] + 4.0*k2d[0][id] + k3d[0][id]);
    p_d_Vals[1][id] += f*(k1d[1][id] + 4.0*k2d[1][id] + k3d[1][id]);
    p_d_Vals[2][id] += f*(k1d[2][id] + 4.0*k2d[2][id] + k3d[2][id]);
    p_o_Vals[0][id] += f*(k1o[0][id] + 4.0*k2o[0][id] + k3o[0][id]);
    p_o_Vals[1][id] += f*(k1o[1][id] + 4.0*k2o[1][id] + k3o[1][id]);
    p_o_Vals[2][id] += f*(k1o[2][id] + 4.0*k2o[2][id] + k3o[2][id]);
}

inline void KER_Update_RK4(TensorGrid &p_d_Vals,
                           TensorGrid &p_o_Vals,
                           const TensorGrid &k1d,
                           const TensorGrid &k1o,
                           const TensorGrid &k2d,
                           const TensorGrid &k2o,
                           const TensorGrid &k3d,
                           const TensorGrid &k3o,
                           const TensorGrid &k4d,
                           const TensorGrid &k4o,
                           const int &id,
                           const Real &dT)
{
    // Update the particle property with Eulerian forward (either position or strength)
    Real f = 1.0/6.0*dT;
    p_d_Vals[0][id] += f*(k1d[0][id] + 2.0*k2d[0][id] + 2.0*k3d[0][id] + k4d[0][id]);
    p_d_Vals[1][id] += f*(k1d[1][id] + 2.0*k2d[1][id] + 2.0*k3d[1][id] + k4d[1][id]);
    p_d_Vals[2][id] += f*(k1d[2][id] + 2.0*k2d[2][id] + 2.0*k3d[2][id] + k4d[2][id]);
    p_o_Vals[0][id] += f*(k1o[0][id] + 2.0*k2o[0][id] + 2.0*k3o[0][id] + k4o[0][id]);
    p_o_Vals[1][id] += f*(k1o[1][id] + 2.0*k2o[1][id] + 2.0*k3o[1][id] + k4o[1][id]);
    p_o_Vals[2][id] += f*(k1o[2][id] + 2.0*k2o[2][id] + 2.0*k3o[2][id] + k4o[2][id]);
}

inline void KER_Update_AB2LF(TensorGrid &p_d_Vals,
                             TensorGrid &p_o_Vals,
                             const TensorGrid &dpdt_d_Vals,
                             const TensorGrid &dpdt_o_Vals,
                             const TensorGrid &p_d_Vals_prev,
                             const TensorGrid &dpdt_o_Vals_prev,
                             const int &id, const Real &dT)
{
    // Update the particle property with leapfrog order 2 for the position and AB2 for circulation

    p_d_Vals[0][id] = p_d_Vals_prev[0][id] + 2.0*dpdt_d_Vals[0][id]*dT;
    p_d_Vals[1][id] = p_d_Vals_prev[1][id] + 2.0*dpdt_d_Vals[1][id]*dT;
    p_d_Vals[2][id] = p_d_Vals_prev[2][id] + 2.0*dpdt_d_Vals[2][id]*dT;
    p_o_Vals[0][id] += (1.5*dpdt_o_Vals[0][id]-0.5*dpdt_o_Vals_prev[0][id])*dT;
    p_o_Vals[1][id] += (1.5*dpdt_o_Vals[1][id]-0.5*dpdt_o_Vals_prev[1][id])*dT;
    p_o_Vals[2][id] += (1.5*dpdt_o_Vals[2][id]-0.5*dpdt_o_Vals_prev[2][id])*dT;
}


//--- Field mapping

inline void KER_M2_Map_Coeffs(const TensorGrid &Disp, TensorGrid &Map, const int &id, const Real &H)
{
    // This kernel specifies the mapping factors for an M_4' type mapping (stencil width 4)
    Real InvH = 1.0/H;
    Real xf = Disp[0][id]*InvH;
    Real yf = Disp[1][id]*InvH;
    Real zf = Disp[2][id]*InvH;

    mapM2(1.0+xf,   Map[0][id]);
    mapM2(fabs(xf), Map[1][id]);
    mapM2(1.0-xf,   Map[2][id]);

    mapM2(1.0+yf,   Map[3][id]);
    mapM2(fabs(yf), Map[4][id]);
    mapM2(1.0-yf,   Map[5][id]);

    mapM2(1.0+zf,   Map[6][id]);
    mapM2(fabs(zf), Map[7][id]);
    mapM2(1.0-zf,   Map[8][id]);
}

inline void KER_M4_Map_Coeffs(const TensorGrid &Disp, TensorGrid &Map, const int &id, const Real &H)
{
    // This kernel specifies the mapping factors for an M4 type mapping (stencil width 4)
    Real InvH = 1.0/H;
    Real xf = Disp[0][id]*InvH;
    Real yf = Disp[1][id]*InvH;
    Real zf = Disp[2][id]*InvH;

    mapM4(2.0+xf,  Map[0][id]);
    mapM4(1.0+xf,  Map[1][id]);
    mapM4(fabs(xf),Map[2][id]);
    mapM4(1.0-xf,  Map[3][id]);
    mapM4(2.0-xf,  Map[4][id]);

    mapM4(2.0+yf,  Map[5][id]);
    mapM4(1.0+yf,  Map[6][id]);
    mapM4(fabs(yf),Map[7][id]);
    mapM4(1.0-yf,  Map[8][id]);
    mapM4(2.0-yf,  Map[9][id]);

    mapM4(2.0+zf,  Map[10][id]);
    mapM4(1.0+zf,  Map[11][id]);
    mapM4(fabs(zf),Map[12][id]);
    mapM4(1.0-zf,  Map[13][id]);
    mapM4(2.0-zf,  Map[14][id]);
}

inline void KER_M4D_Map_Coeffs(const TensorGrid &Disp, TensorGrid &Map, const int &id, const Real &H)
{
    // This kernel specifies the mapping factors for an M_4' type mapping (stencil width 4)
    Real InvH = 1.0/H;
    Real xf = Disp[0][id]*InvH;
    Real yf = Disp[1][id]*InvH;
    Real zf = Disp[2][id]*InvH;

    mapM4D(2.0+xf,  Map[0][id]);
    mapM4D(1.0+xf,  Map[1][id]);
    mapM4D(fabs(xf),Map[2][id]);
    mapM4D(1.0-xf,  Map[3][id]);
    mapM4D(2.0-xf,  Map[4][id]);

    mapM4D(2.0+yf,  Map[5][id]);
    mapM4D(1.0+yf,  Map[6][id]);
    mapM4D(fabs(yf),Map[7][id]);
    mapM4D(1.0-yf,  Map[8][id]);
    mapM4D(2.0-yf,  Map[9][id]);

    mapM4D(2.0+zf,  Map[10][id]);
    mapM4D(1.0+zf,  Map[11][id]);
    mapM4D(fabs(zf),Map[12][id]);
    mapM4D(1.0-zf,  Map[13][id]);
    mapM4D(2.0-zf,  Map[14][id]);
}

inline void KER_M6D_Map_Coeffs(const TensorGrid &Disp, TensorGrid &Map, const int &id, const Real &H)
{
    // This kernel specifies the mapping factors for an M_4' type mapping (stencil width 4)
    Real InvH = 1.0/H;
    Real xf = Disp[0][id]*InvH;
    Real yf = Disp[1][id]*InvH;
    Real zf = Disp[2][id]*InvH;

    mapM6D(3.0+xf,  Map[0][id]);
    mapM6D(2.0+xf,  Map[1][id]);
    mapM6D(1.0+xf,  Map[2][id]);
    mapM6D(fabs(xf),Map[3][id]);
    mapM6D(1.0-xf,  Map[4][id]);
    mapM6D(2.0-xf,  Map[5][id]);
    mapM6D(3.0-xf,  Map[6][id]);

    mapM6D(3.0+yf,  Map[7][id]);
    mapM6D(2.0+yf,  Map[8][id]);
    mapM6D(1.0+yf,  Map[9][id]);
    mapM6D(fabs(yf),Map[10][id]);
    mapM6D(1.0-yf,  Map[11][id]);
    mapM6D(2.0-yf,  Map[12][id]);
    mapM6D(3.0-yf,  Map[13][id]);

    mapM6D(3.0+zf,  Map[14][id]);
    mapM6D(2.0+zf,  Map[15][id]);
    mapM6D(1.0+zf,  Map[16][id]);
    mapM6D(fabs(zf),Map[17][id]);
    mapM6D(1.0-zf,  Map[18][id]);
    mapM6D(2.0-zf,  Map[19][id]);
    mapM6D(3.0-zf,  Map[20][id]);
}

inline void KER_Map_Arrays3(const TensorGrid &Field_In, const TensorGrid &Fac, const TensorIntGrid &Tmpl, TensorGrid &Field_Mapped, const int &id, const int &NM)
{
    // This function takes the input field (Field_In) and uses the Mapping Factors (Fac) and Interaction Template (Tmlp)
    // calculated previously to generatethe new Mapped Field (Field_Mapped)

    // Note: Tmpl[i][0] is the relativ index of the source node
    // Note: Tmpl[i][1] is the x index of the mapping factor (0-2)
    // Note: Tmpl[i][2] is the y index of the mapping factor (3-5)
    // Note: Tmpl[i][3] is the z index of the mapping factor (6-8)

    Real v3 = 0.0, v4 = 0.0, v5 = 0.0;
    for (int i=0; i<NM; i++){
        int sid = id + Tmpl[i][0];
        Real F = Fac[ Tmpl[i][1] ][sid] * Fac[ Tmpl[i][2] ][sid] * Fac[ Tmpl[i][3] ][sid];
        if (F==0) continue;
        // v3 += Field_In[3][sid] * F;
        // v4 += Field_In[4][sid] * F;
        // v5 += Field_In[5][sid] * F;
        v3 = fmadd(Field_In[0][sid],F,v3);
        v4 = fmadd(Field_In[1][sid],F,v4);
        v5 = fmadd(Field_In[2][sid],F,v5);
    }
    Field_Mapped[0][id] += v3;
    Field_Mapped[1][id] += v4;
    Field_Mapped[2][id] += v5;

}

inline void KER_Map(const TensorGrid &Field_In, const TensorGrid &Fac, TensorGrid &Field_Mapped, const dim3 &id, const dim3 &D, const Mapping &tMap)
{
    // This function takes the input field (Field_In) and uses the Mapping Factors (Fac) and Interaction Template (Tmlp)
    // calculated previously to generatethe new Mapped Field (Field_Mapped)

    // Set shifting terms
    int idsh = 0, nc = 0, ncx = 0, ncy = 0, ncz = 0;
    switch (tMap)
    {
    case (M2):  {idsh = -1;     nc = 3;    ncx = 2;    ncy = 5;     ncz = 8;    break;}
    case (M4):  {idsh = -2;     nc = 5;    ncx = 4;    ncy = 9;     ncz = 14;   break;}
    case (M4D): {idsh = -2;     nc = 5;    ncx = 4;    ncy = 9;     ncz = 14;   break;}
    case (M6D): {idsh = -3;     nc = 7;    ncx = 6;    ncy = 13;    ncz = 20;   break;}
    case (M3):  {                                                               break;}
    case (D2):  {                                                               break;}
    case (D3):  {                                                               break;}
    default:    {std::cout << "KER_Map: Mapping unknown" << std::endl;          break;}
    }

    // Map coefficients
    Real v1 = 0.0, v2 = 0.0, v3 = 0.0;
    for (int i=0; i<nc; i++){
        for (int j=0; j<nc; j++){
            for (int k=0; k<nc; k++){
                int sid = GID(id.x+idsh+i, id.y+idsh+j, id.z+idsh+k, D);
                Real F = Fac[ncx-i][sid] * Fac[ncy-j][sid] * Fac[ncz-k][sid];
                if (F==0) continue;
                v1 = fmadd(Field_In[0][sid],F,v1);
                v2 = fmadd(Field_In[1][sid],F,v2);
                v3 = fmadd(Field_In[2][sid],F,v3);
            }
        }
    }

    // Append output
    int tid = GID(id,D);
    Field_Mapped[0][tid] += v1;
    Field_Mapped[1][tid] += v2;
    Field_Mapped[2][tid] += v3;
}

//--- Field interpolation

inline void KER_Interpolation(  const TensorGrid &Sp,        // Source grid p
                              const TensorGrid &So,        // Source grid o
                              const TensorGrid &P,        // Particle displacmenets
                              const TensorGrid &G,        // Particle global positions
                              const Real *C,              // Grid global corners
                              // const int *Shift,           // Shifting array for local interpolation nodes
                              TensorGrid &Dp,              // Output (interpolated) grid
                              TensorGrid &Do,              // Output (interpolated) grid
                              const dim3 &tid,
                              const dim3 &D,
                              const Mapping &tMap)        // mapping function
{
    // This is kernel implementation of the M4D mapping
    int id = GID(tid,D);
    Real rpx = (P[0][id]+G[0][id])-C[0];
    Real rpy = (P[1][id]+G[1][id])-C[1];
    Real rpz = (P[2][id]+G[2][id])-C[2];
    Real H_Grid = C[3];
    int ti = int(rpx/H_Grid);
    int tj = int(rpy/H_Grid);
    int tk = int(rpz/H_Grid);

    // The "Map_Flag" catch called before this function ensures that out-of-bound nodes aren't called

    // Set interpolation factors.
    Real fx = (rpx-H_Grid*ti)/H_Grid;
    Real fy = (rpy-H_Grid*tj)/H_Grid;
    Real fz = (rpz-H_Grid*tk)/H_Grid;

    // Set interpolation coefficients
    int nc=0, ns=0;
    Vector Mx, My, Mz;
    switch (tMap)
    {
    case (M2):      {
        nc = 2; ns = 0;
        Mx = Vector(2), My = Vector(2), Mz = Vector(2);
        mapM2(fx, Mx(0)); mapM2(1.0-fx, Mx(1));
        mapM2(fy, My(0)); mapM2(1.0-fy, My(1));
        mapM2(fz, Mz(0)); mapM2(1.0-fz, Mz(1));
        break;
    }
    case (M4):      {
        nc = 4; ns = -1;
        Mx = Vector(4), My = Vector(4), Mz = Vector(4);
        mapM4(1.0+fx,Mx(0));     mapM4(fx,Mx(1));  mapM4(1.0-fx,Mx(2));  mapM4(2.0-fx,Mx(3));
        mapM4(1.0+fy,My(0));     mapM4(fy,My(1));  mapM4(1.0-fy,My(2));  mapM4(2.0-fy,My(3));
        mapM4(1.0+fz,Mz(0));     mapM4(fz,Mz(1));  mapM4(1.0-fz,Mz(2));  mapM4(2.0-fz,Mz(3));
        break;
    }
    case (M4D):     {
        nc = 4; ns = -1;
        Mx = Vector(4), My = Vector(4), Mz = Vector(4);
        mapM4D(1.0+fx,Mx(0));    mapM4D(fx,Mx(1)); mapM4D(1.0-fx,Mx(2)); mapM4D(2.0-fx,Mx(3));
        mapM4D(1.0+fy,My(0));    mapM4D(fy,My(1)); mapM4D(1.0-fy,My(2)); mapM4D(2.0-fy,My(3));
        mapM4D(1.0+fz,Mz(0));    mapM4D(fz,Mz(1)); mapM4D(1.0-fz,Mz(2)); mapM4D(2.0-fz,Mz(3));
        break;
    }
    case (M6D):     {
        nc = 6; ns = -2;
        Mx = Vector(6), My = Vector(6), Mz = Vector(6);
        mapM6D(2.0+fx,Mx(0)); mapM6D(1.0+fx,Mx(1)); mapM6D(fx,Mx(2)); mapM6D(1.0-fx,Mx(3)); mapM6D(2.0-fx,Mx(4)); mapM6D(3.0-fx,Mx(5));
        mapM6D(2.0+fy,My(0)); mapM6D(1.0+fy,My(1)); mapM6D(fy,My(2)); mapM6D(1.0-fy,My(3)); mapM6D(2.0-fy,My(4)); mapM6D(3.0-fy,My(5));
        mapM6D(2.0+fz,Mz(0)); mapM6D(1.0+fz,Mz(1)); mapM6D(fz,Mz(2)); mapM6D(1.0-fz,Mz(3)); mapM6D(2.0-fz,Mz(4)); mapM6D(3.0-fz,Mz(5));
        break;
    }
    default:        {std::cout << "VPM_3D_Solver::Map_Grid_Sources: Mapping unknown" << std::endl;    break;}
    }

    // Now map back to the receiver node
    Real v0 = 0., v1 = 0., v2 = 0., d0 = 0., d1 = 0., d2 = 0.;
    for (int i=0; i<nc; i++){
        for (int j=0; j<nc; j++){
            for (int k=0; k<nc; k++){
                int ids = GID(ti+ns+i, tj+ns+j, tk+ns+k, D);
                Real Fac =  Mx(i)*My(j)*Mz(k);
                v0 += Fac*Sp[0][ids];
                v1 += Fac*Sp[1][ids];
                v2 += Fac*Sp[2][ids];
                d0 += Fac*So[0][ids];
                d1 += Fac*So[1][ids];
                d2 += Fac*So[2][ids];
            }
        }
    }

    // Store output
    Dp[0][id] = v0;
    Dp[1][id] = v1;
    Dp[2][id] = v2;
    Do[0][id] = d0;
    Do[1][id] = d1;
    Do[2][id] = d2;
}

//--- Vorticity field initialisation

//--- Vortex ring.

inline void KER_Vortex_Ring(const Real &x, const Real &y, const Real &z, Real &Ox, Real &Oy, Real &Oz, Real &R, Real &a, Real &Rey, Real &Nu)
{
    // This function uses the given x,y,z position and the (prespecified) array of phase angles to generate a vortex ring

    Real Phi = atan2(z,y);
    Real rx = 0.0, ry = R*cos(Phi), rz = R*sin(Phi);                  // Position of ring core
    //    Real rx = 0.2*R*cos(2.0*Phi), ry = R*cos(Phi), rz = R*sin(Phi);     // Displaced ring
    Real tx = 0.0, ty = sin(Phi), tz = -cos(Phi);       // Tangent vector of ring core
    Real dx = x-rx, dy = y-ry, dz = z-rz;               // Relative position vector to ring core
    Real r2 = dx*dx + dy*dy + dz*dz;

    Real a2 = a*a;
    Real Gamma0 = Rey*Nu;
    Real f = Gamma0/M_PI/a2*exp(-r2/a2);                // Magnitude

    // Specify vorticity
    Ox = -f*tx;
    Oy = -f*ty;
    Oz = -f*tz;

    //    Real re2 = x*x + y*y + z*z;
    //    Real omf =  Gamma0/M_PI/a2*exp(-re2/(4.0*a2));
    //    Ox = 1.0*omf;
    //    Oy = 0.5*omf;
    //    Oz = 0.0;
}

inline void KER_Collision(const Real &x, const Real &y, const Real &z, Real &Ox, Real &Oy, Real &Oz, Real &R, Real &a, Real &Rey, Real &Nu)
{
    // This function uses the given x,y,z position and a given geometry to create colliding vortices
    // which are inclined to each other at an initial angle of Alpha and a separation of D.

    Real Alpha = 15.0*D2R;
    Real D = 2.7;

    // Must convert position into relative position within new coordinate system of modified ring. Real Y1[3] =

    Real a2 = a*a;
    Real Gamma0 = Rey*Nu;
    //    Real t0 = R*R/Gamma0;

    // Ring 1
    // Ring CS
    Real C1[3] = {0.0, Real(0.5*D), 0.0};
    Real X1[3] = {1.0,0.0,0.0};
    Real Y1[3] = {0.0,Real(cos(Alpha)),Real(-sin(Alpha))};
    Real Z1[3] = {0.0,Real(sin(Alpha)),Real(cos(Alpha))};

    // Ring position and tangent vectors
    Real p1[3] = {x-C1[0], y-C1[1], z-C1[2]};
    Real x1 = p1[0]*X1[0] + p1[1]*X1[1] +  p1[2]*X1[2];
    Real y1 = p1[0]*Y1[0] + p1[1]*Y1[1] +  p1[2]*Y1[2];
    Real z1 = p1[0]*Z1[0] + p1[1]*Z1[1] +  p1[2]*Z1[2];

    Real Phi1 = atan2(y1,x1);
    Real rx1 = R*cos(Phi1), ry1 = R*sin(Phi1), rz1 = 0.0;       // Position of ring core (in CS 1)
    Real dx1 = x1-rx1, dy1 = y1-ry1, dz1 = z1-rz1;              // Relative position vector to ring core
    Real r21 = dx1*dx1 + dy1*dy1 + dz1*dz1;
    Real f1 = Gamma0/M_PI/a2*exp(-r21/a2);                      // Magnitude
    Real tx = -sin(Phi1);
    Real ty = cos(Phi1);
    Real tz = 0.0;

    Real txv[3] = {tx*X1[0], tx*X1[1], tx*X1[2]};
    Real tyv[3] = {ty*Y1[0], ty*Y1[1], ty*Y1[2]};
    Real tzv[3] = {tz*Z1[0], tz*Z1[1], tz*Z1[2]};

    Ox += -f1*(txv[0] + tyv[0] + tzv[0]);
    Oy += -f1*(txv[1] + tyv[1] + tzv[1]);
    Oz += -f1*(txv[2] + tyv[2] + tzv[2]);

    // Ring 2

    // Ring CS
    Real C2[3] = {0.0, Real(-0.5*D), 0.0};
    Real X2[3] = {1.0,0.0,0.0};
    Real Y2[3] = {0.0,Real(cos(-Alpha)),Real(-sin(-Alpha))};
    Real Z2[3] = {0.0,Real(sin(-Alpha)),Real(cos(-Alpha))};

    // Ring position and tangent vectors
    Real p2[3] = {x-C2[0], y-C2[1], z-C2[2]};
    Real x2 = p2[0]*X2[0] + p2[1]*X2[1] +  p2[2]*X2[2];
    Real y2 = p2[0]*Y2[0] + p2[1]*Y2[1] +  p2[2]*Y2[2];
    Real z2 = p2[0]*Z2[0] + p2[1]*Z2[1] +  p2[2]*Z2[2];

    Real Phi2 = atan2(y2,x2);
    Real rx2 = R*cos(Phi2), ry2 = R*sin(Phi2), rz2 = 0.0;       // Position of ring core (in CS 1)
    Real dx2 = x2-rx2, dy2 = y2-ry2, dz2 = z2-rz2;              // Relative position vector to ring core
    Real r22 = dx2*dx2 + dy2*dy2 + dz2*dz2;
    Real f2 = Gamma0/M_PI/a2*exp(-r22/a2);                      // Magnitude
    tx = -sin(Phi2);
    ty = cos(Phi2);
    tz = 0.0;

    Real txv2[3] = {tx*X2[0], tx*X2[1], tx*X2[2]};
    Real tyv2[3] = {ty*Y2[0], ty*Y2[1], ty*Y2[2]};
    Real tzv2[3] = {tz*Z2[0], tz*Z2[1], tz*Z2[2]};

    Ox += -f2*(txv2[0] + tyv2[0] + tzv2[0]);
    Oy += -f2*(txv2[1] + tyv2[1] + tzv2[1]);
    Oz += -f2*(txv2[2] + tyv2[2] + tzv2[2]);

}

//--- This generates the perturbed vortex ring for the breakdown / turbulence model tests as in Cocle

inline void KER_Perturbed_Vortex_Ring(const Real &x, const Real &y, const Real &z, Real &Ox, Real &Oy, Real &Oz, RVector &PhaseShift, Real &R, Real &a, Real &Rey, Real &Nu, Real &eps)
{
    // This function uses the given x,y,z position and the (prespecified) array of phase angles to generate a perturbed vortex ring

    Real a2 = a*a;
    Real Gamma0 = Rey*Nu;
    //    Real t0 = R*R/Gamma0;

    //        // Cocle test 1
    //        Real a = 0.4131;
    //        Real Gamma0 = 5500*Vars.Kin_Visc;
    //        Real t0 = R*R/Gamma0;s
    //        Vars.Turb_Model = SMG;
    //        Vars.C_smg = 2.5e-2;       // Constant
    //        Vars.T_smg = t0;           // Global time scale? Hopefully correct
    //        Real H2 = Grid_Vars.H_Grid*Grid_Vars.H_Grid;
    //        Vars.SMG_Fac = Vars.C_smg*(H2*H2)/Vars.T_smg;
    //        Real eps = 0.0002;                //--- Specify Displacement parameter perturbations
    //        Vars.dT = 0.05*t0;               //--- Specify Timestep

    // Real a = 0.2;
    // Real Gamma0 = 25000*Vars.Kin_Visc;
    // Real t0 = R*R/Gamma0;

    // Vars.Turb_Model = RVM;
    // Vars.T_smg = t0;           // Global time scale
    // Vars.T_Char = Vars.T_smg;
    // Vars.rvm_n = 2;            // Trial first order!
    // if (Vars.rvm_n==1) Vars.C_n = pow(0.3,3)*1.39;
    // if (Vars.rvm_n==2) Vars.C_n = 1.27*pow(0.3,3)*1.39;
    // Real eps = 0.0001;                  //--- Specify displacement parameter perturbations
    // Vars.dT = 0.005*t0;                 //--- Specify Timestep
    // //    Vars.dT *= 0.5;                  // HACK JOE!
    // Vars.N_Max = 200*t0/Vars.dT;      //--- Total sim run time
    // Vars.N_Max = 12000;      //--- Total sim run time
    // Vars.N_Max = 1;      //--- Total sim run time


    Real Phi = atan2(z,y);
    Real rx = 0.0, ry = R*cos(Phi), rz = R*sin(Phi);                        // Position of unshifted ring core
    Real tx = 0.0, ty = sin(Phi), tz = -cos(Phi);                           // Tangent vector of ring core

    Real G = 0, dG = 0;
    // Real eps = 0.0002;
    int NModesCocle = 24;
    for (int m=0; m<NModesCocle; m++){
        G += sin((m+1)*Phi + PhaseShift[m]);
        //        dG += (m+1)*cos((m+1)*Phi + PhaseShift[m]); // HACK! AVOID CHANGING TANGENT....
    }
    Real srx = 0.0, sry = (1.0+eps*G)*ry, srz = (1.0+eps*G)*rz;             // Perturbed position of shifted ring core
    Real stx = tx + eps*dG*rx, sty = ty + eps*dG*ry, stz = tz + eps*dG*rz;  // Perturbed ring tangent

    Real dx = x-srx, dy = y-sry, dz = z-srz;            // Relative position vector to ring core
    Real r2 = dx*dx + dy*dy + dz*dz;
    Real f = Gamma0/M_PI/a2*exp(-r2/a2);

    // Specify vorticity
    Ox = -f*tx;
    Oy = -f*ty;
    Oz = -f*tz;
}

}
