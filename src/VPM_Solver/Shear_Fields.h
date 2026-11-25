/****************************************************************************
    SailFFish Library
    Copyright (C) 2026 Joseph Saverin j.saverin@tu-berlin.de

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

    -> The definitions for the different shear field options are defined here

*****************************************************************************/


#ifndef SHEAR_FIELDS_H
#define SHEAR_FIELDS_H

#include "../SailFFish_Math_Types.h"
#include <functional>

namespace SailFFish
{

enum Shear_Profile  {SHEAR_UNIFORM, SHEAR_LINEAR, SHEAR_POWER, SHEAR_LOG, SHEAR_EKMAN, SHEAR_TABLE};
enum Veer_Profile   {VEER_NONE, VEER_LINEAR};

struct Shear_Point
{
    Real U = 0., V = 0.;            // Velocity
    Real dU = 0., dV = 0.;          // First derivative  d/dz
    Real d2U = 0., d2V = 0.;        // Second derivative d2/dz2
};

struct Shear_Arrays
{
    RVector U, V, dU, dV, d2U, d2V;
    void Resize(int N)  {U.assign(N,0.); V.assign(N,0.); dU.assign(N,0.); dV.assign(N,0.); d2U.assign(N,0.); d2V.assign(N,0.);}
    bool Omega_Zero() const {
        for (size_t i=0; i<dU.size(); i++) if (dU[i]!=0. || dV[i]!=0.) return false;
        return true;
    }

    //--- Print profile on grid z-nodes z_k = Z_Min + k*H. Omega_bg = (-V', U', 0)
    void Print(Real Z_Min, Real H, int Stride = 1, std::ostream &os = std::cout) const {
        os << std::setw(6) << "k" << std::setw(14) << "z" << std::setw(14) << "U" << std::setw(14) << "V"
           << std::setw(14) << "dU/dz" << std::setw(14) << "dV/dz" << std::setw(14) << "d2U/dz2" << std::setw(14) << "d2V/dz2"
           << std::setw(14) << "Omega_x" << std::setw(14) << "Omega_y" << std::endl;
        os << std::scientific << std::setprecision(5);
        for (size_t k=0; k<U.size(); k+=std::max(Stride,1)){
            os << std::setw(6) << k << std::setw(14) << Z_Min + Real(k)*H << std::setw(14) << U[k] << std::setw(14) << V[k]
               << std::setw(14) << dU[k] << std::setw(14) << dV[k] << std::setw(14) << d2U[k] << std::setw(14) << d2V[k]
               << std::setw(14) << -dV[k] << std::setw(14) << dU[k] << std::endl;
        }
        os << std::defaultfloat;
    }
};

struct Natural_Spline
{
    // Natural cubic spline, used for tabulated profiles
    RVector X, Y, M;                // Nodes, values, second derivatives

    void Set(const RVector &x, const RVector &y) {
        X = x; Y = y;
        int N = int(X.size());
        M.assign(N,0.);
        if (N<3) return;
        RVector a(N,0.), b(N,1.), c(N,0.), r(N,0.);
        for (int i=1; i<N-1; i++){
            Real h0 = X[i]-X[i-1], h1 = X[i+1]-X[i];
            a[i] = h0/6.; b[i] = (h0+h1)/3.; c[i] = h1/6.;
            r[i] = (Y[i+1]-Y[i])/h1 - (Y[i]-Y[i-1])/h0;
        }
        for (int i=1; i<N; i++){                                // Thomas algorithm
            Real m = a[i]/b[i-1];
            b[i] -= m*c[i-1];
            r[i] -= m*r[i-1];
        }
        M[N-1] = r[N-1]/b[N-1];
        for (int i=N-2; i>=0; i--) M[i] = (r[i]-c[i]*M[i+1])/b[i];
    }

    void Eval(Real x, Real &y, Real &dy, Real &d2y) const {
        int N = int(X.size());
        if (N==0)   {y = 0.; dy = 0.; d2y = 0.; return;}
        if (N==1)   {y = Y[0]; dy = 0.; d2y = 0.; return;}
        if (x<=X[0])    {y = Y[0];   dy = 0.; d2y = 0.; return;}    // Constant extrapolation
        if (x>=X[N-1])  {y = Y[N-1]; dy = 0.; d2y = 0.; return;}
        int k = int(std::upper_bound(X.begin(),X.end(),x) - X.begin()) - 1;
        Real h = X[k+1]-X[k], A = (X[k+1]-x)/h, B = (x-X[k])/h;
        y   = A*Y[k] + B*Y[k+1] + ((A*A*A-A)*M[k] + (B*B*B-B)*M[k+1])*h*h/6.;
        dy  = (Y[k+1]-Y[k])/h - (3.*A*A-1.)*h/6.*M[k] + (3.*B*B-1.)*h/6.*M[k+1];
        d2y = A*M[k] + B*M[k+1];
    }
};

struct Shear_Field
{
private:

    Natural_Spline SplU, SplV;

    //--- Monin-Obukhov (Businger-Dyer) stability functions, zeta = h/L
    static Real Psi_M(Real zeta) {
        if (zeta>=0.) return -5.*zeta;
        Real x = pow(1.-16.*zeta,0.25);
        return 2.*log(0.5*(1.+x)) + log(0.5*(1.+x*x)) - 2.*atan(x) + 0.5*M_PI;
    }
    static Real Phi_M(Real zeta) {
        if (zeta>=0.) return 1.+5.*zeta;
        return pow(1.-16.*zeta,-0.25);
    }
    static Real dPhi_M(Real zeta) {
        if (zeta>=0.) return 5.;
        return 4.*pow(1.-16.*zeta,-1.25);
    }

    //--- Speed S(h) and derivatives in the flow-aligned frame
    void Speed(Real h, Real &S, Real &dS, Real &d2S) const {
        S = 0.; dS = 0.; d2S = 0.;
        switch (Profile)
        {
        case SHEAR_UNIFORM: {S = U_Ref;                             break;}
        case SHEAR_LINEAR:  {S = U_Ref + Gradient*(h-Z_Ref); dS = Gradient;   break;}
        case SHEAR_POWER:   {
            S = U_Ref*pow(h/Z_Ref,Alpha);
            dS = Alpha*S/h;
            d2S = Alpha*(Alpha-1.)*S/(h*h);
            break;
        }
        case SHEAR_LOG:     {
            Real zeta = h*Inv_L, c = U_Star/Kappa;
            S = c*(log(h/Z0) - Psi_M(zeta));
            dS = c*Phi_M(zeta)/h;
            d2S = c*(dPhi_M(zeta)*Inv_L/h - Phi_M(zeta)/(h*h));
            break;
        }
        default: break;
        }
    }

    //--- Direction phi(h) and derivatives
    void Direction(Real h, Real &phi, Real &dphi) const {
        phi = Dir; dphi = 0.;
        if (Veer!=VEER_LINEAR || Veer_Z2<=Veer_Z1) return;
        Real rate = Veer_Angle/(Veer_Z2-Veer_Z1);
        if (h<=Veer_Z1)         phi += 0.;
        else if (h>=Veer_Z2)    phi += Veer_Angle;
        else                    {phi += rate*(h-Veer_Z1); dphi = rate;}
    }

    //--- Rotate flow-frame components (Uf,Vf) by angle phi into grid frame
    static void Rotate(Real phi, Shear_Point &P) {
        Real c = cos(phi), s = sin(phi);
        Real u, v;
        u = c*P.U   - s*P.V;    v = s*P.U   + c*P.V;    P.U = u;   P.V = v;
        u = c*P.dU  - s*P.dV;   v = s*P.dU  + c*P.dV;   P.dU = u;  P.dV = v;
        u = c*P.d2U - s*P.d2V;  v = s*P.d2U + c*P.d2V;  P.d2U = u; P.d2V = v;
    }

    Shear_Point Eval_Above(Real h) const {
        Shear_Point P;

        if (Profile==SHEAR_TABLE){                          // Components given directly in flow frame
            SplU.Eval(h,P.U,P.dU,P.d2U);
            SplV.Eval(h,P.V,P.dV,P.d2V);
            Rotate(Dir,P);
            return P;
        }

        if (Profile==SHEAR_EKMAN){                          // Flow frame: geostrophic wind along x
            Real g = Gamma_Ek, e = exp(-g*h), c = cos(g*h), s = sin(g*h);
            P.U   = U_Ref*(1. - e*c);
            P.V   = U_Ref*e*s;
            P.dU  = U_Ref*g*e*(c + s);
            P.dV  = U_Ref*g*e*(c - s);
            P.d2U = -2.*U_Ref*g*g*e*s;
            P.d2V = -2.*U_Ref*g*g*e*c;
            Rotate(Dir,P);
            return P;
        }

        //--- Analytic speed profile with optional veer: U = S cos(phi), V = S sin(phi)
        Real S, dS, d2S, phi, dphi;
        Speed(h,S,dS,d2S);
        Direction(h,phi,dphi);
        Real c = cos(phi), s = sin(phi);
        P.U   = S*c;
        P.V   = S*s;
        P.dU  = dS*c - S*dphi*s;
        P.dV  = dS*s + S*dphi*c;
        P.d2U = d2S*c - 2.*dS*dphi*s - S*dphi*dphi*c;       // d2phi = 0 within linear veer
        P.d2V = d2S*s + 2.*dS*dphi*c - S*dphi*dphi*s;
        return P;
    }

public:

    //--- Profile type
    Shear_Profile Profile = SHEAR_UNIFORM;
    Veer_Profile Veer = VEER_NONE;

    //--- Geometry
    Real Z_Ground = 0.;             // Grid z-coordinate of the ground plane (height h = z - Z_Ground)
    Real Z_Cut = 1.;                // Below this height: linear ramp to zero at h=0 (log/power singularity). Zero for h<0.

    //--- Constant freestream (Ux,Uy,Uz), superposed on the profile. Special case: SHEAR_UNIFORM with U_Ref = 0
    Real U_Const[3] = {0., 0., 0.};

    //--- Reference values
    Real U_Ref = 0.;                // Reference speed at Z_Ref
    Real Z_Ref = 100.;              // Reference height
    Real Dir = 0.;                  // Base flow direction [rad], measured from x towards y

    //--- Profile-specific
    Real Gradient = 0.;             // SHEAR_LINEAR: dU/dz
    Real Alpha = 0.14;              // SHEAR_POWER: exponent
    Real Z0 = 2.e-4;                // SHEAR_LOG: roughness length
    Real U_Star = 0.;               // SHEAR_LOG: friction velocity (if 0, computed from U_Ref at Z_Ref)
    Real Inv_L = 0.;                // SHEAR_LOG: 1/Obukhov length (0 = neutral, >0 stable, <0 unstable)
    Real Kappa = 0.41;              // SHEAR_LOG: von Karman constant
    Real Gamma_Ek = 0.;             // SHEAR_EKMAN: sqrt(f/(2K)); U_Ref is the geostrophic speed
    RVector Tab_Z, Tab_U, Tab_V;    // SHEAR_TABLE: heights (ascending, above ground), components in flow frame

    //--- Veer (analytic profiles only)
    Real Veer_Z1 = 0., Veer_Z2 = 0.;    // VEER_LINEAR: veer region (heights above ground)
    Real Veer_Angle = 0.;               // VEER_LINEAR: total direction change across [Veer_Z1,Veer_Z2] [rad]

    //--- Time modulation: all quantities scaled by f(t) (e.g. gust shapes). Empty = 1.
    std::function<Real(Real)> Time_Factor;

    //--- Profile evaluated on the grid z-nodes z_k = Grid_Z_Min + k*Grid_H
    Shear_Arrays Grid;
    Real Grid_Z_Min = 0., Grid_H = 0.;

    //--- Initialisation (call after setting parameters)
    void Initialise() {
        if (Profile==SHEAR_LOG && U_Star==0.){
            Real zr = Z_Ref, zeta = zr*Inv_L;
            U_Star = U_Ref*Kappa/(log(zr/Z0) - Psi_M(zeta));
        }
        if (Profile==SHEAR_TABLE){
            SplU.Set(Tab_Z,Tab_U);
            SplV.Set(Tab_Z,Tab_V);
        }
    }

    //--- Vertical velocity (constant only)
    Real W() const              {return U_Const[2];}

    //--- Evaluation at a single grid z-coordinate
    Shear_Point Eval(Real z, Real t = 0.) const {
        Shear_Point P;
        Real h = z - Z_Ground;
        Real hc = Z_Cut;
        if (Profile==SHEAR_UNIFORM || Profile==SHEAR_LINEAR || Profile==SHEAR_EKMAN || Profile==SHEAR_TABLE) hc = 0.;

        if (h<0. && hc>0.)  {}                                      // Below ground (log/power): no profile contribution
        else if (hc>0. && h<hc){                                    // Linear ramp below cut-off (log/power only)
            Shear_Point Pc = Eval_Above(hc);
            Real f = h/hc;
            P.U = Pc.U*f;       P.V = Pc.V*f;
            P.dU = Pc.U/hc;     P.dV = Pc.V/hc;
        }
        else P = Eval_Above(h);

        if (Time_Factor){
            Real f = Time_Factor(t);
            P.U *= f; P.V *= f; P.dU *= f; P.dV *= f; P.d2U *= f; P.d2V *= f;
        }

        P.U += U_Const[0];                                          // Constant freestream (not time-modulated)
        P.V += U_Const[1];
        return P;
    }

    //--- Evaluation on the grid: z_k = Z_Min + k*H, k = 0..NZ-1
    void Eval_Grid(Real Z_Min, Real H, int NZ, Real t, Shear_Arrays &A) const {
        A.Resize(NZ);
        for (int k=0; k<NZ; k++){
            Shear_Point P = Eval(Z_Min + k*H, t);
            A.U[k] = P.U;   A.V[k] = P.V;
            A.dU[k] = P.dU; A.dV[k] = P.dV;
            A.d2U[k] = P.d2U; A.d2V[k] = P.d2V;
        }
    }

    //--- Evaluation on the grid, stored internally in Grid
    void Eval_Grid(Real Z_Min, Real H, int NZ, Real t = 0.) {
        Grid_Z_Min = Z_Min; Grid_H = H;
        Eval_Grid(Z_Min, H, NZ, t, Grid);
    }
    void Print_Grid(int Stride = 1, std::ostream &os = std::cout) const    {Grid.Print(Grid_Z_Min, Grid_H, Stride, os);}

    bool Time_Dependent() const {return bool(Time_Factor);}
    bool Active() const         {return !(Profile==SHEAR_UNIFORM && U_Ref==0.) || U_Const[0]!=0. || U_Const[1]!=0. || U_Const[2]!=0.;}   // Is any background velocity present?
    bool Sheared() const        {return Active() && !Grid.Omega_Zero();}             // Is background vorticity present on the grid?

};

}

#endif // SHEAR_FIELDS_H
