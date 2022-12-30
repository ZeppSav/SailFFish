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

inline Real Q3_4(const Real &rho){return M_INVSQRT2PI*rho;}
inline Real Q3_6(const Real &rho){return M_INVSQRT2PI*(7.0/4.0*rho-1.0/4.0*rho*rho*rho);}
inline Real Q3_8(const Real &rho){Real rho2=rho*rho; return M_INVSQRT2PI*(19.0/8.0*rho-2.0/3.0*rho2*rho+1.0/24.0*rho2*rho2*rho);}
inline Real Q3_10(const Real &rho){Real rho2=rho*rho, rho4=rho2*rho2; return M_INVSQRT2PI*(187.0/64.0*rho-233.0/192.0*rho2*rho+29.0/192.0*rho4*rho-1.0/192.0*rho4*rho2*rho);}

inline Real Kernel_HEJ3_S0(const Real &r, const Real &sigma)    {Real rho = r/sigma;    return -M_INV2PISQ*sine_int(rho)/r;     }
inline Real Kernel_HEJ3_S0_O(const Real &r, const Real &sigma)  {                       return -M_INV2PISQ/sigma;               }
inline Real Kernel_HEJ3_G2(const Real &r, const Real &sigma)    {Real rho = r/sigma;    return -M_INV4PI/r*(erf(rho*M_INVSQRT2));}
inline Real Kernel_HEJ3_G2_0(const Real &r, const Real &sigma)  {                       return -1.0/4.0*M_SQRT2/(sigma*sqrt(M_PI*M_PI*M_PI));}
inline Real Kernel_HEJ3_G4(const Real &r, const Real &sigma)    {Real rho = r/sigma;    return -M_INV4PI/r*(Q3_4(rho)*exp(-0.5*rho*rho) + erf(rho*M_INVSQRT2));}
inline Real Kernel_HEJ3_G4_0(const Real &r, const Real &sigma)  {                       return -3.0/8.0*M_SQRT2/(sigma*sqrt(M_PI*M_PI*M_PI));}
inline Real Kernel_HEJ3_G6(const Real &r, const Real &sigma)    {Real rho = r/sigma;    return -M_INV4PI/r*(Q3_6(rho)*exp(-0.5*rho*rho) + erf(rho*M_INVSQRT2));}
inline Real Kernel_HEJ3_G6_0(const Real &r, const Real &sigma)  {                       return -15.0/32.0*M_SQRT2/(sigma*sqrt(M_PI*M_PI*M_PI));}
inline Real Kernel_HEJ3_G8(const Real &r, const Real &sigma)    {Real rho=r/sigma;      return -M_INV4PI/r*(Q3_8(rho)*exp(-0.5*rho*rho) + erf(rho*M_INVSQRT2));}
inline Real Kernel_HEJ3_G8_0(const Real &r, const Real &sigma)  {                       return -35.0/64.0*M_SQRT2/(sigma*sqrt(M_PI*M_PI*M_PI));}
inline Real Kernel_HEJ3_G10(const Real &r, const Real &sigma)   {Real rho=r/sigma;      return -M_INV4PI/r*(Q3_10(rho)*exp(-0.5*rho*rho) + erf(rho*M_INVSQRT2));}
inline Real Kernel_HEJ3_G10_0(const Real &r, const Real &sigma) {                       return -315.0/512.0*M_SQRT2/(sigma*sqrt(M_PI*M_PI*M_PI));}


}

#endif // GREENS_FUNCTIONS_H
