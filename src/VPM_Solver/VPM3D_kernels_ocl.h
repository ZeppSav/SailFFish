//--------------------------------------------------------------------------------
//-------------------------- VPM 3D: ocl Kernels --------------------------------
//--------------------------------------------------------------------------------

// Note: Prior to initialiization, the "Real" type must be defined along with float/double dependent functions

// #include "../SailFFish_Math_Types.h"
#include <string>

namespace SailFFish
{

const std::string VPM3D_ocl_kernels_float = R"CLC(
typedef float Real;
__device__   float  fastma(const float &a, const float &b, const float &c)    {return fmaf(a,b,c);}
__device__   float  fab(const float &a)   {return fabs(a);}
__device__   float  mymax(float a, float b)   {return fmaxf(a,b);}
)CLC";

const std::string VPM3D_ocl_kernels_double = R"CLC(
typedef double Real;
__device__ double fastma(const double &a, const double &b, const double &c) {return fma(a,b,c);}
__device__ double fab(const double &a)  {return abs(a);}
__device__ double mymax(double a, double b)  {return max(a,b);}
)CLC";

const std::string VPM3D_ocl_kernels_update = R"CLC(

__kernel void update(__global Real* d,
                     __global Real* o,
                     __global const Real* d_d,
                     __global const Real* d_o,
                     const Real dt)
{
    // Thread index in global space
    int i = get_global_id(0);

    // Load values
    Real p1  = d[i];
    Real p2  = d[i + NT];
    Real p3  = d[i + 2*NT];
    Real o1  = o[i];
    Real o2  = o[i + NT];
    Real o3  = o[i + 2*NT];
    Real dp1 = d_d[i];
    Real dp2 = d_d[i + NT];
    Real dp3 = d_d[i + 2*NT];
    Real do1 = d_o[i];
    Real do2 = d_o[i + NT];
    Real do3 = d_o[i + 2*NT];

    // Write updated values
    d[i     ] = p1 + dt * dp1;
    d[i+1*NT] = p2 + dt * dp2;
    d[i+2*NT] = p3 + dt * dp3;
    o[i     ] = o1 + dt * do1;
    o[i+1*NT] = o2 + dt * do2;
    o[i+2*NT] = o3 + dt * do3;
}

)CLC";

// const std::string VPM3D_ocl_kernels_update = R"CLC(

// )CLC";

}
