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

const std::string ocl_GID_functions = R"CLC(
inline int gid (int i,int j, int k, int nx, int ny, int nz)     {return k*nx*ny + j*nx + i;}    // Grid id in F-style ordering
inline int gidc(int i,int j, int k, int nx, int ny, int nz)     {return i*ny*nz + j*nz + k;}    // Grid id in C-style ordering
inline int gidb(int i,int j, int k){                                                            // Global block node id
    const int ib = i/BX, jb = j/BY, kb = k/BZ;
    const int gl = gidc(i-ib*BX, j-jb*BY, k-kb*BZ, BX, BY, BZ);
    const int bls = gidc(ib, jb, kb, NBX, NBY, NBZ);
    return gl + BT*bls;
}

)CLC";

const std::string VPM3D_ocl_kernels_monolith_to_block = R"CLC(
__kernel void  Monolith_to_Block(   __global Real* src,
                                    __global Real* dst)
{
    const int gx = get_local_id(0) + get_group_id(0)*BX;
    const int gy = get_local_id(1) + get_group_id(1)*BY;
    const int gz = get_local_id(2) + get_group_id(2)*BZ;
    const int mid = gid(gx,gy,gz,NX,NY,NZ);           // Global id (Monolithic)
    const int bid = gidb(gx,gy,gz);                   // Global id (Block)

    dst[bid     ] += src[mid     ];
    dst[bid+NT  ] += src[mid+NT  ];
    dst[bid+2*NT] += src[mid+2*NT];
}
)CLC";

const std::string VPM3D_ocl_kernels_block_to_monolith = R"CLC(
__kernel void  Block_to_Monolith(   __global Real* src,
                                    __global Real* dst)
{
    const int gx = get_local_id(0) + get_group_id(0)*BX;
    const int gy = get_local_id(1) + get_group_id(1)*BY;
    const int gz = get_local_id(2) + get_group_id(2)*BZ;
    const int mid = gid(gx,gy,gz,NX,NY,NZ);           // Global id (Monolithic)
    const int bid = gidb(gx,gy,gz);                   // Global id (Block)

    dst[mid     ] += src[bid     ];
    dst[mid+NT  ] += src[bid+NT  ];
    dst[mid+2*NT] += src[bid+2*NT];
}
)CLC";

const std::string VPM3D_ocl_kernels_map_to_unbounded = R"CLC(
__kernel void  Map_toUnbounded( __global Real* src,
                                __global Real* dst1,
                                __global Real* dst2,
                                __global Real* dst3)
{
    const int gx = get_local_id(0) + get_group_id(0)*BX;
    const int gy = get_local_id(1) + get_group_id(1)*BY;
    const int gz = get_local_id(2) + get_group_id(2)*BZ;
    const int mid = gid(gx,gy,gz,2*NX,2*NY,2*NZ);       // Global id (Monolithic)
    const int bid = gidb(gx,gy,gz);                     // Global id (Block)

    dst1[mid] = src[bid     ];
    dst2[mid] = src[bid+NT  ];
    dst3[mid] = src[bid+2*NT];
}
)CLC";

const std::string VPM3D_ocl_kernels_map_from_unbounded = R"CLC(
__kernel void  Map_fromUnbounded(   __global Real* src1,
                                    __global Real* src2,
                                    __global Real* src3,
                                    __global Real* dst)
{
    const int gx = get_local_id(0) + get_group_id(0)*BX;
    const int gy = get_local_id(1) + get_group_id(1)*BY;
    const int gz = get_local_id(2) + get_group_id(2)*BZ;
    const int mid = gid(gx,gy,gz,2*NX,2*NY,2*NZ);       // Global id (Monolithic-unbounded)
    const int bid = gidb(gx,gy,gz);                     // Global id (Block)

    dst[bid     ] = src1[mid];
    dst[bid+NT  ] = src2[mid];
    dst[bid+2*NT] = src3[mid];
}
)CLC";

// const int gx = threadIdx.x + blockIdx.x*BX;
// const int gy = threadIdx.y + blockIdx.y*BY;
// const int gz = threadIdx.z + blockIdx.z*BZ;
// const int mid = gid(gx,gy,gz,2*NX,2*NY,2*NZ);
// const int bid = gidb(gx,gy,gz);

// dst[bid     ]  = src1[mid];
// dst[bid+1*NT]  = src2[mid];
// dst[bid+2*NT]  = src3[mid];

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
