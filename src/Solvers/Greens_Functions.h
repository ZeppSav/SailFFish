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

    -> Greens functions for the fast Poisson solver.

*****************************************************************************/

#ifndef GREENS_FUNCTIONS_H
#define GREENS_FUNCTIONS_H

#include "../SailFFish_Math_Types.h"   // Math vars
#include "Special_functions.cpp"    // Sine integral
#include <cmath>                    // Error function

namespace SailFFish
{

//--------------------------------------------
//--------- Frequency-space kernels ----------
//--------------------------------------------

//--- Pseudo-spectral kernels

inline Real EV_PS_Reg_Periodic(const int &i, const int &N, const Real &L)
{
    Real f;
    if (2*i<N)      f = M_2PI*i/L;
    else            f = M_2PI*(N-i)/L;
    return -f*f;
}
inline Real EV_PS_Reg_Dirichlet(const int &i, const Real &L)    {Real f = M_PI*(i+1)/L; return -f*f;}     // Cell-boundary
inline Real EV_PS_Reg_Neumann(const int &i, const Real &L)      {Real f = M_PI*i/L; return -f*f;}          // Cell-boundary
inline Real EV_PS_Stag_Dirichlet(const int &i, const Real &L)   {Real f = M_PI*(i+1)/L; return -f*f;}      // Cell-centre
inline Real EV_PS_Stag_Neumann(const int &i, const Real &L)     {Real f = M_PI*i/L; return -f*f;}          // Cell-centre

//--- Finite Difference kernels (2nd order)

// Regular grid points
inline Real EV_FD_Reg_Periodic(const int &i, const int &N, const Real &H)     {Real f = 2.0*sin(M_PI*i/N); return -f*f/H/H;}
inline Real EV_FD_Reg_Dirichlet(const int &i, const int &N, const Real &H)    {Real f = 2.0*sin((M_PI*(i+1))/(2.0*(N+1))); return -f*f/H/H;}
inline Real EV_FD_Reg_Neumann(const int &i, const int &N, const Real &H)      {Real f = 2.0*sin(M_PI*i/(2.0*(N-1))); return -f*f/H/H;}
inline Real EV_FD_Stag_Dirichlet(const int &i, const int &N, const Real &H)   {Real f = 2.0*sin((M_PI*(i+1))/(2.0*N)); return -f*f/H/H;}
inline Real EV_FD_Stag_Neumann(const int &i, const int &N, const Real &H)     {Real f = 2.0*sin(M_PI*i/(2.0*N)); return -f*f/H/H;}


//--------------------------------------------
//--------- Real-space kernels ---------------
//--------------------------------------------

//--- 2D

inline Real P2_6(const Real &rho2) {return (3.0/4-rho2/8);}
inline Real P2_8(const Real &rho2) {return 11.0/12-7.0/24*rho2+1.0/48*rho2*rho2;}
inline Real P2_10(const Real &rho2){return 25.0/24-23.0/48*rho2+13.0/192*rho2*rho2-1.0/384*rho2*rho2*rho2;}

inline Real Kernel_HEJ2_S0(const Real &r, const Real &sigma)     {                                      return -M_INV2PI * ( bessel_int_J0( r/sigma ) + log(2.0*sigma) - M_GAMMA );}
inline Real Kernel_HEJ2_G2(const Real &r, const Real &sigma)     {Real rho = r/sigma, rho2 = rho*rho;   return -M_INV2PI*(log(r) + 0.5*expint_ei(0.5*rho2));}
inline Real Kernel_HEJ2_G2_0(const Real &r, const Real &sigma)   {                                      return  M_INV2PI*(0.5*M_GAMMA - log(M_SQRT2*sigma));}
inline Real Kernel_HEJ2_G4(const Real &r, const Real &sigma)     {Real rho = r/sigma, rho2 = rho*rho;   return -M_INV2PI*(log(r) + 0.5*expint_ei(0.5*rho2) - 0.5*exp(-0.5*rho2));}
inline Real Kernel_HEJ2_G4_0(const Real &r, const Real &sigma)   {                                      return  M_INV2PI*(0.5*M_GAMMA - log(M_SQRT2*sigma) + 0.5);}
inline Real Kernel_HEJ2_G6(const Real &r, const Real &sigma)     {Real rho = r/sigma, rho2 = rho*rho;   return -M_INV2PI*(log(r) + 0.5*expint_ei(0.5*rho2) - P2_6(rho2)*exp(-0.5*rho2));}
inline Real Kernel_HEJ2_G6_0(const Real &r, const Real &sigma)   {                                      return  M_INV2PI*(0.5*M_GAMMA - log(M_SQRT2*sigma) + 0.75);}
inline Real Kernel_HEJ2_G8(const Real &r, const Real &sigma)     {Real rho = r/sigma, rho2 = rho*rho;   return -M_INV2PI*(log(r) + 0.5*expint_ei(0.5*rho2) - P2_8(rho2)*exp(-0.5*rho2));}
inline Real Kernel_HEJ2_G8_0(const Real &r, const Real &sigma)   {                                      return  M_INV2PI*(0.5*M_GAMMA - log(M_SQRT2*sigma) + 11.0/12.0);}
inline Real Kernel_HEJ2_G10(const Real &r, const Real &sigma)    {Real rho = r/sigma, rho2 = rho*rho;   return -M_INV2PI*(log(r) + 0.5*expint_ei(0.5*rho2) - P2_10(rho2)*exp(-0.5*rho2));}
inline Real Kernel_HEJ2_G10_0(const Real &r, const Real &sigma)  {                                      return  M_INV2PI*(0.5*M_GAMMA - log(M_SQRT2*sigma) + 25.0/24.0);}

//--- 3D

inline Real Q3_4(const Real &p){return M_INVSQRT2PI*p;}
inline Real Q3_6(const Real &p){return M_INVSQRT2PI*(7./4.*p-1./4.*p*p*p);}
inline Real Q3_8(const Real &p){Real p2=p*p; return M_INVSQRT2PI*(19./8.*p-2./3.*p2*p+1./24.*p2*p2*p);}
inline Real Q3_10(const Real &p){Real p2=p*p, p4=p2*p2; return M_INVSQRT2PI*(187./64.*p-233./192.*p2*p+29./192.*p4*p-1./192.*p4*p2*p);}

inline Real Kernel_HEJ3_S0_O(const Real &r, const Real &s)  {return -M_INV2PISQ/s;                                  }
inline Real Kernel_HEJ3_G2_0(const Real &r, const Real &s)  {return -1./4.*M_SQRT2/(s*sqrt(M_PI*M_PI*M_PI));      }
inline Real Kernel_HEJ3_G4_0(const Real &r, const Real &s)  {return -3./8.*M_SQRT2/(s*sqrt(M_PI*M_PI*M_PI));      }
inline Real Kernel_HEJ3_G6_0(const Real &r, const Real &s)  {return -15./32.*M_SQRT2/(s*sqrt(M_PI*M_PI*M_PI));    }
inline Real Kernel_HEJ3_G8_0(const Real &r, const Real &s)  {return -35./64.*M_SQRT2/(s*sqrt(M_PI*M_PI*M_PI));    }
inline Real Kernel_HEJ3_G10_0(const Real &r, const Real &s) {return -315./512.*M_SQRT2/(s*sqrt(M_PI*M_PI*M_PI));  }

inline Real Kernel_HEJ3_S0(const Real &r, const Real &s)    {Real p = r/s;  return -M_INV2PISQ*sine_int(p)/r;     }
inline Real Kernel_HEJ3_G2(const Real &r, const Real &s)    {Real p = r/s;  return -M_INV4PI/r*(                        erf(p*M_INVSQRT2));}
inline Real Kernel_HEJ3_G4(const Real &r, const Real &s)    {Real p = r/s;  return -M_INV4PI/r*(Q3_4(p)*exp(-0.5*p*p) + erf(p*M_INVSQRT2));}
inline Real Kernel_HEJ3_G6(const Real &r, const Real &s)    {Real p = r/s;  return -M_INV4PI/r*(Q3_6(p)*exp(-0.5*p*p) + erf(p*M_INVSQRT2));}
inline Real Kernel_HEJ3_G8(const Real &r, const Real &s)    {Real p = r/s;  return -M_INV4PI/r*(Q3_8(p)*exp(-0.5*p*p) + erf(p*M_INVSQRT2));}
inline Real Kernel_HEJ3_G10(const Real &r, const Real &s)   {Real p = r/s;  return -M_INV4PI/r*(Q3_10(p)*exp(-0.5*p*p) + erf(p*M_INVSQRT2));}

inline Real KQ3_2(const Real &p)  {                                                     return M_INVSQRT2PI*(-2.*p);}
inline Real KQ3_4(const Real &p)  {Real p2 = p*p, p3=p*p2;                              return M_INVSQRT2PI*(-2.*p + p3);}
inline Real KQ3_6(const Real &p)  {Real p2 = p*p, p3=p*p2, p5=p3*p2;                    return M_INVSQRT2PI*(-2.*p + (9.*p3 - p5)/4.);}
inline Real KQ3_8(const Real &p)  {Real p2 = p*p, p3=p*p2, p5=p3*p2, p7=p5*p2;          return M_INVSQRT2PI*(-2.*p + (89.*p3 - 20.*p5 + p7)/24.);}
inline Real KQ3_10(const Real &p) {Real p2 = p*p, p3=p*p2, p5=p3*p2, p7=p5*p2, p9=p7*p2;return M_INVSQRT2PI*(-2.*p + (1027.*p3 - 349.*p5 + 35.*p7 - p9)/192.);}

inline Real Kernel_HEJ3_K2(const Real &r, const Real &s)    {Real p =r/s;   return -M_INV4PI/(r*r*r)*(KQ3_2(p)*exp(-0.5*p*p) + erf(p*M_INVSQRT2));}
inline Real Kernel_HEJ3_K4(const Real &r, const Real &s)    {Real p =r/s;   return -M_INV4PI/(r*r*r)*(KQ3_4(p)*exp(-0.5*p*p) + erf(p*M_INVSQRT2));}
inline Real Kernel_HEJ3_K6(const Real &r, const Real &s)    {Real p =r/s;   return -M_INV4PI/(r*r*r)*(KQ3_6(p)*exp(-0.5*p*p) + erf(p*M_INVSQRT2));}
inline Real Kernel_HEJ3_K8(const Real &r, const Real &s)    {Real p =r/s;   return -M_INV4PI/(r*r*r)*(KQ3_8(p)*exp(-0.5*p*p) + erf(p*M_INVSQRT2));}
inline Real Kernel_HEJ3_K10(const Real &r, const Real &s)   {Real p =r/s;   return -M_INV4PI/(r*r*r)*(KQ3_10(p)*exp(-0.5*p*p) + erf(p*M_INVSQRT2));}


}

#endif // GREENS_FUNCTIONS_H
