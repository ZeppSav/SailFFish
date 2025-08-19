/****************************************************************************
    SailFFish Library
    Copyright (C) 2022 Joseph Saverin j.saverin@tu-berlin.de

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

    -> Test cases for each of the solver types.

*****************************************************************************/

#ifndef TEST_FUNCTIONS_H
#define TEST_FUNCTIONS_H

#include "../src/SailFFish_Math_Types.h"
#include <chrono>

//-----------------------------------
//--- Helper defs
//-----------------------------------

static Real Ms2s = 1.0e-3;
unsigned int stopwatch()
{
    static auto start_time  = std::chrono::steady_clock::now();
    auto        end_time    = std::chrono::steady_clock::now();
    auto        delta       = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    start_time = end_time;
    return delta.count();
}

//-----------------------------------
//--- Error functions
//-----------------------------------

Real Error_LInf(RVector &V, RVector &Sol, Real Fac)
{
    if (V.size() != Sol.size()){
        std::cout << "Error array inputs have non-matching sizes." << std::endl;
        return 0.0;
    }
    Real e = 0.0;
    for (int i=0; i<V.size(); i++)  e += (V[i]-Sol[i])*(V[i]-Sol[i])*Fac;
    return e;
}

Real E_Inf(RVector &V, RVector &Sol)
{
    if (V.size() != Sol.size()){
        std::cout << "Error array inputs have non-matching sizes." << std::endl;
        return 0.0;
    }
    Real e = 0.0;
    for (int i=0; i<V.size(); i++){
        Real val = fabs(V[i]-Sol[i]);
        if (val>e)  e = val;
    }
    return e;
}

//-----------------------------------
//--- Grid Vars
//-----------------------------------

Real UnitX[2] = {-1.0, 1.0};
Real UnitY[2] = {-1.0, 1.0};
Real UnitZ[2] = {-1.0, 1.0};

//-----------------------------------
//--- Test Cases bounded solver
//-----------------------------------

//--- Neumann (zero-gradient) boundary conditions

//inline Real NTest_Omega(Real x, Real Lx)                                    {return 2.0;}
//inline Real NTest_Phi(Real x, Real Lx)                                      {Real Cx = x/Lx;    return 2.0*Cx*Cx;}
//inline Real NTest_Omega(Real x, Real Lx, Real y, Real Ly)                   {return -M_PI/Lx*cos(M_PI/Lx*x)-M_PI/Ly*cos(M_PI/Ly*y);}
//inline Real NTest_Phi(Real x, Real Lx, Real y, Real Ly)                     {return cos(M_PI/Lx*x) + cos(M_PI/Ly*y);}
//inline Real NTest_Omega(Real x, Real Lx, Real y, Real Ly, Real z, Real Lz)  {return -M_PI/Lx*cos(M_PI/Lx*x)-M_PI/Ly*cos(M_PI/Ly*y)-M_PI/Lz*cos(M_PI/Lz*z);}
//inline Real NTest_Phi(Real x, Real Lx, Real y, Real Ly, Real z, Real Lz)    {return cos(M_PI/Lx*x) + cos(M_PI/Ly*y) + cos(M_PI/Lz*z);}

//-----------------------------------
//--- Test Cases dirichlet BCs
//-----------------------------------



inline Real STest_Omega(Real x, Real Lx)
{
    Real Cx = 4.*M_PI/Lx;
    return -Cx*Cx*sin(x*Cx);
}
inline Real STest_Phi(Real x, Real Lx)                                      {return sin(4.*M_PI*x/Lx);}
inline Real STest_Omega(Real x, Real Lx, Real y, Real Ly)
{
    Real Cx = 2.*M_PI/Lx, Cy = 2.*M_PI/Ly;
    return - (Cx*Cx + Cy*Cy)*sin(x*Cx)*sin(y*Cy);
}
inline Real STest_Phi(Real x, Real Lx, Real y, Real Ly)                     {return sin(2.*M_PI*x/Lx)*sin(2.*M_PI*y/Ly);}
inline Real STest_Omega(Real x, Real Lx, Real y, Real Ly, Real z, Real Lz)
{
    Real Cx = 4.*M_PI/Lx, Cy = 4.*M_PI/Ly, Cz = 4.*M_PI/Lz;
    return - (Cx*Cx + Cy*Cy + Cz*Cz)*sin(x*Cx)*sin(y*Cy)*sin(z*Cz);
}
inline Real STest_Phi(Real x, Real Lx, Real y, Real Ly, Real z, Real Lz)    {return sin(4.*M_PI*x/Lx)*sin(4.*M_PI*y/Ly)*sin(4.*M_PI*z/Lz);}

//--- Inhomogenous boundary conditions

inline Real IH_BC_Omega(    Real x, Real Lx, Real a, Real b)    {return 2.0*(a+b)/Lx/Lx;}
inline Real IH_BC_Phi(      Real x, Real Lx, Real a, Real b)    {Real nx = x/Lx; return a*(1.0-nx)*(1.0-nx) + b*nx*nx;}
inline Real IH_BC_GradPhi(  Real x, Real Lx, Real a, Real b)    {Real nx = x/Lx; return (2.0*a*(nx-1.0) + 2.0*b*nx)/Lx;}

inline Real IH_BC_Omega(    Real x, Real Lx, Real a, Real b,
                            Real y, Real Ly, Real c, Real d)    {return IH_BC_Omega(x,Lx,a,b)+IH_BC_Omega(y,Ly,c,d);}
inline Real IH_BC_GradPhi(  Real x, Real Lx, Real a, Real b,
                            Real y, Real Ly, Real c, Real d)    {return IH_BC_GradPhi(x,Lx,a,b)+IH_BC_GradPhi(y,Ly,c,d);}
inline Real IH_BC_Phi(      Real x, Real Lx, Real a, Real b,
                            Real y, Real Ly, Real c, Real d)    {return IH_BC_Phi(x,Lx,a,b) + IH_BC_Phi(y,Ly,c,d);}

inline Real IH_BC_Omega(    Real x, Real Lx, Real a, Real b,
                            Real y, Real Ly, Real c, Real d,
                            Real z, Real Lz, Real e, Real f)    {return IH_BC_Omega(x,Lx,a,b) + IH_BC_Omega(y,Ly,c,d) + IH_BC_Omega(z,Lz,e,f);}
inline Real IH_BC_Phi(      Real x, Real Lx, Real a, Real b,
                            Real y, Real Ly, Real c, Real d,
                            Real z, Real Lz, Real e, Real f)    {return IH_BC_Phi(x,Lx,a,b) + IH_BC_Phi(y,Ly,c,d) + IH_BC_Phi(z,Lz,e,f);}

//----------------------

//-----------------------------------
//--- Test Cases periodic BCs
//-----------------------------------

inline Real CTest_Omega(Real x, Real Lx)
{
    Real Cx = 4.*M_PI/Lx;
    return -Cx*Cx*cos(x*Cx);
}
inline Real CTest_Phi(Real x, Real Lx)                                      {return cos(4.*M_PI*x/Lx);}
inline Real CTest_Omega(Real x, Real Lx, Real y, Real Ly)
{
    Real Cx = 4.*M_PI/Lx, Cy = 4.*M_PI/Ly;
    return - (Cx*Cx + Cy*Cy)*cos(x*Cx)*cos(y*Cy);
}
inline Real CTest_Phi(Real x, Real Lx, Real y, Real Ly)                     {return cos(4.*M_PI*x/Lx)*cos(4.*M_PI*y/Ly);}
inline Real CTest_Omega(Real x, Real Lx, Real y, Real Ly, Real z, Real Lz)
{
    Real Cx = 4.*M_PI/Lx, Cy = 4.*M_PI/Ly, Cz = 4.*M_PI/Lz;
    return - (Cx*Cx + Cy*Cy + Cz*Cz)*cos(Cx*x)*cos(Cy*y)*cos(Cz*z);
}
inline Real CTest_Phi(Real x, Real Lx, Real y, Real Ly, Real z, Real Lz)    {return cos(4.*M_PI*x/Lx)*cos(4.*M_PI*y/Ly)*cos(4.*M_PI*z/Lz);}

//-----------------------------------
//--- Test Cases unbounded solvers
//-----------------------------------

// These cases are all essentially the bump function.

static Real Cbf = 10.0;

//--- 2D cases

inline Real UTest_Omega(Real x)
{
    if (fabs(x)<1.0)
    {
        Real d = 1.0/(1-x*x);
        return -(4.0*Cbf*Cbf*x*x*d*d*d*d - 2.0*Cbf*d*d - 8.0*Cbf*x*x*d*d*d)*exp(-Cbf*d);
    }
    else    return 0.0;
}

inline Real UTest_Phi(Real x)
{
    Real d = 1.0/(1-x*x);
    if (fabs(x)<1.0)    return exp(-Cbf*d);
    else                return 0.0;
}

inline Real UTest_Phi(Real x, Real y)      {return UTest_Phi(sqrt(x*x + y*y));}

inline Real UTest_Omega_Hejlesen(Real x, Real y, Real r0)
{
    // Function taken directly from Hejlesen solver input
    Real r = sqrt(x*x + y*y);
    Real Omega = 0;
    if( r < r0 )
    {
        Omega = 4.0 * Cbf * pow(r0,2)
                * exp(- Cbf * pow(r0,2)/(pow(r0,2) - pow(x,2) - pow(y,2)))
                * ( pow(r0,4) - pow(x,4) - pow(y,4) - 2.0*pow(x,2)*pow(y,2)- Cbf*pow(x,2)*pow(r0,2)- Cbf*pow(y,2)*pow(r0,2) )
                * pow(pow(r0,2) - pow(x,2) - pow(y,2),-4);
        //				Bx[ij] = pow(1.0 - r*r, m);
    }
    return Omega;
}

inline Real UTest_XGrad_Hejlesen(Real x, Real y, Real r0)
{
    // This is the analytical solution to the velocity field of the bump funtion in 2D.
    Real r = sqrt(x*x + y*y);
    Real sol = 0.0;
    if (r<=r0)
    {
        sol = - 2.0 * Cbf * pow(r0,2) * x
              * exp( -Cbf * pow(r0,2)/(pow(r0,2) - pow(x,2) - pow(y,2)))
              * pow(pow(r0,2) - pow(x,2) - pow(y,2), -2);
    }

    return sol;
}

inline Real UTest_YGrad_Hejlesen(Real x, Real y, Real r0)
{
    // This is the analytical solution to the velocity field of the bump funtion in 2D.
    Real r = sqrt(x*x + y*y);
    Real sol = 0.0;
    if (r<=r0)
    {
        sol = - 2.0 * Cbf * pow(r0,2) * y
              * exp( -Cbf * pow(r0,2)/(pow(r0,2) - pow(x,2) - pow(y,2)))
              * pow(pow(r0,2) - pow(x,2) - pow(y,2), -2);
    }

    return sol;
    //    return solY;
}

//--- 3D cases

inline void UTest_Omega_Hejlesen(Real x, Real y, Real z, Real r0, Real &ox, Real &oy, Real &oz)
{
    // 3D vortex ring bump function

    Real rho   = sqrt(z*z + y*y);
    Real phi   = sqrt( pow(rho - r0,2) + pow(x,2) );
    Real theta = atan2(z,y);
    // Real Bx = 0.0, By = 0.0, Bz = 0.0;

    if( phi < r0)
    {
        Real Bmag = -exp(- Cbf*pow(r0,2)/(2.0*r0*rho - pow(rho,2) - pow(x,2))) *
                    ( 4.0*pow(Cbf,2)*pow(r0,4)*pow(x,2)*pow(rho,2)
                     - 16.0*pow(r0,4)*pow(rho,4)
                     + 32.0*pow(r0,3)*pow(rho,5)
                     - 24.0*pow(r0,2)*pow(rho,6)
                     + 8.0*r0*pow(rho,7)
                     - 4.0*pow(rho,6)*pow(x,2)
                     - 6.0*pow(rho,4)*pow(x,4)
                     - 4.0*pow(rho,2)*pow(x,6)
                     - 8.0*Cbf*pow(r0,5)*pow(rho,3)
                     + 8.0*Cbf*pow(r0,4)*pow(rho,4)
                     - 6.0*Cbf*pow(r0,3)*pow(rho,5)
                     + 4.0*pow(Cbf,2)*pow(r0,6)*pow(rho,2)
                     - 8.0*pow(Cbf,2)*pow(r0,5)*pow(rho,3)
                     + 4.0*pow(Cbf,2)*pow(r0,4)*pow(rho,4)
                     + 2.0*Cbf*pow(r0,2)*pow(rho,6)
                     + 32.0*pow(r0,3)*pow(rho,3)*pow(x,2)
                     - 48.0*pow(r0,2)*pow(rho,4)*pow(x,2)
                     - 24.0*pow(r0,2)*pow(rho,2)*pow(x,4)
                     + 24.0*r0*pow(rho,5)*pow(x,2)
                     + 24.0*r0*pow(rho,3)*pow(x,4)
                     + 8.0*r0*rho*pow(x,6)
                     + 2.0*Cbf*pow(r0,3)*rho*pow(x,4)
                     + 2.0*Cbf*pow(r0,2)*pow(rho,2)*pow(x,4)
                     - 4.0*Cbf*pow(r0,3)*pow(rho,3)*pow(x,2)
                     + 4.0*Cbf*pow(r0,2)*pow(rho,4)*pow(x,2)
                     - pow(rho,8) - pow(x,8))
                    * pow(2.0*r0*rho - pow(rho,2) - pow(x,2),-4) * pow(rho,-2);

        // Bx =   0.0;
        // By = - sin(theta)*Bmag;
        // Bz =   cos(theta)*Bmag;
        ox =   0.0;
        oy = - sin(theta)*Bmag;
        oz =   cos(theta)*Bmag;
    }
}

inline void UTest_Phi_Hejlesen(Real x, Real y, Real z, Real r0, Real &px, Real &py, Real &pz)
{
    // 3D vortex ring bump function. Potential components

    // Approach 2
    Real ryz = sqrt(z*z + y*y);
    Real d = (ryz-r0)*(ryz-r0) + x*x;
    Real a = sqrt(d)/r0;
    Real u = -Cbf/(1.0-a*a);                // Scale this to avoid extremely small values
    Real theta = atan2(z,y);
    if( a < 1.0 )
    {
        px = 0.0;
        py = sin(theta)*exp(u);
        pz = -cos(theta)*exp(u);
    }
    else
    {
        px = 0.0;
        py = 0.0;
        pz = 0.0;
    }
}

inline void UTest_Velocity_Hejlesen(Real x, Real y, Real z, Real r0, Real &ux, Real &uy, Real &uz)
{
    // 3D vortex ring bump function. Velocity components

    Real rho   = sqrt(z*z + y*y);
    Real phi   = sqrt( pow(rho - r0,2) + pow(x,2) );
    Real theta = atan2(z,y);
    // Real solX = 0.0, solY = 0.0, solZ = 0.0;

    if( phi < r0 )
    {
        Real Amag = 2.0 * Cbf * pow(r0,2) * x
                    * exp(-Cbf*pow(r0,2)/(2.0*r0*rho - pow(rho,2) - pow(x,2)))
                    * pow(2.0*r0*rho - pow(rho,2) - pow(x,2),-2);

        // solX = exp( -Cbf * pow(r0,2)/(2.0 * r0 * rho - pow(rho,2) - pow(x,2)))
        //           * ( 4.0*pow(r0,2) *pow(rho,2)
        //             - 4.0*r0*pow(rho,3)
        //             - 4.0*r0*rho*pow(x,2)
        //             + pow(rho,4) + 2.0*pow(rho,2)*pow(x,2)
        //             + pow(x,4) + 2.0*Cbf*pow(r0,3)*rho
        //             - 2.0*Cbf*pow(r0,2)*pow(rho,2) )
        //           * pow(2.0*r0*rho - pow(rho,2) - pow(x,2),-2)*pow(rho,-1);
        // solY = cos(theta)*Amag;
        // solZ = sin(theta)*Amag;

        ux = exp( -Cbf * pow(r0,2)/(2.0 * r0 * rho - pow(rho,2) - pow(x,2)))
             * ( 4.0*pow(r0,2) *pow(rho,2)
                - 4.0*r0*pow(rho,3)
                - 4.0*r0*rho*pow(x,2)
                + pow(rho,4) + 2.0*pow(rho,2)*pow(x,2)
                + pow(x,4) + 2.0*Cbf*pow(r0,3)*rho
                - 2.0*Cbf*pow(r0,2)*pow(rho,2) )
             * pow(2.0*r0*rho - pow(rho,2) - pow(x,2),-2)*pow(rho,-1);
        uy = cos(theta)*Amag;
        uz = sin(theta)*Amag;
    }
}

inline void UTest_Gradient_Hejlesen(    Real x, Real y, Real z, Real r0,
                                    Real &dfidx, Real &dfidy, Real &dfidz,
                                    Real &dfjdx, Real &dfjdy, Real &dfjdz,
                                    Real &dfkdx, Real &dfkdy, Real &dfkdz)
{
    // This function calculates the gradients of the bump function. Useful for testing finite differences.
    Real ryz = sqrt(z*z + y*y);
    Real d = (ryz-r0)*(ryz-r0) + x*x;
    Real a = sqrt(d)/r0;
    if( a < 1.0 )
    {
        Real u = -Cbf/(1.0-a*a);
        Real f = exp(u);
        Real dfda = -Cbf*2.0*a/pow(1-a*a,2.0)*exp(u);       // (df/du)*(du/da)
        Real theta = atan2(z,y);
        Real r2 = ryz*ryz;

        // Gradient terms
        Real dadx = x*a/d;
        Real dady = y*(ryz-r0)/ryz*a/d;
        Real dadz = z*(ryz-r0)/ryz*a/d;
        Real dsindy = cos(theta)*-z/r2;
        Real dsindz = cos(theta)*y/r2;
        Real dcosdy = -sin(theta)*-z/r2;
        Real dcosdz = -sin(theta)*y/r2;
        dfidx = 0;
        dfidy = 0;
        dfidz = 0;
        dfjdx = (dfda)*(dadx)*sin(theta);
        dfjdy = (dfda)*(dady)*sin(theta) + f*dsindy;
        dfjdz = (dfda)*(dadz)*sin(theta) + f*dsindz;
        dfkdx = (dfda)*(dadx)*-cos(theta);
        dfkdy = (dfda)*(dady)*-cos(theta) + -f*dcosdy;
        dfkdz = (dfda)*(dadz)*-cos(theta) + -f*dcosdz;
    }
    else
    {
        dfidx = 0;
        dfidy = 0;
        dfidz = 0;
        dfjdx = 0;
        dfjdy = 0;
        dfjdz = 0;
        dfkdx = 0;
        dfkdy = 0;
        dfkdz = 0;
    }
}

inline void UTest_Nabla_Hejlesen(Real x, Real y, Real z, Real r0, Real &nablaPsi)
{
    // This function calculates the gradients of the bump function. Useful for testing finite differences.
    Real ryz = sqrt(z*z + y*y);
    Real d = (ryz-r0)*(ryz-r0) + x*x;
    Real a = sqrt(d)/r0;
    if( a < 1.0 )
    {
        Real u = -Cbf/(1.0-a*a);
        Real f = exp(u);
        Real dfda = -Cbf*2.0*a/pow(1-a*a,2.0)*exp(u);       // (df/du)*(du/da)
        Real theta = atan2(z,y);
        Real r2 = ryz*ryz;

        // Gradient terms
        Real dadx = x*a/d;
        Real dady = y*(ryz-r0)/ryz*a/d;
        Real dadz = z*(ryz-r0)/ryz*a/d;
        Real dsindy = cos(theta)*-z/r2;
        Real dsindz = cos(theta)*y/r2;
        Real dcosdy = -sin(theta)*-z/r2;
        Real dcosdz = -sin(theta)*y/r2;

        Real dfidx = 0;
        Real dfjdy = (dfda)*(dady)*sin(theta) + f*dsindy;
        // Real dfjdy = 0;
        Real dfkdz = (dfda)*(dadz)*-cos(theta) + -f*dcosdz;
        // Real dfkdz = 0;
        nablaPsi = dfidx + dfjdy + dfkdz;
    }
    else
    {
        nablaPsi = 0;
    }
}

#endif // TEST_FUNCTIONS_H
