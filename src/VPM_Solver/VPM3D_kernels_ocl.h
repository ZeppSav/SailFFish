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
// inline float fastma(const float &a, const float &b, const float &c)    {return fmaf(a,b,c);}
inline float fab(float a)   {return fabs(a);}
// inline float mymax(float a, float b)   {return fmaxf(a,b);}
)CLC";

const std::string VPM3D_ocl_kernels_double = R"CLC(
typedef double Real;
// inline double fastma(const double &a, const double &b, const double &c) {return fma(a,b,c);}
inline double fab(float a)  {return abs(a);}
// inline double mymax(double a, double b)  {return max(a,b);}
)CLC";

const std::string ocl_GID_functions = R"CLC(
inline int gidf(int i,int j, int k, int nx, int ny, int nz)     {return k*nx*ny + j*nx + i;}    // Grid id in F-style ordering
inline int gid (int i,int j, int k, int nx, int ny, int nz)     {return i*ny*nz + j*nz + k;}    // Grid id in C-style ordering
inline int gidb(int i,int j, int k){                                                            // Global block node id
    const int ib = i/BX, jb = j/BY, kb = k/BZ;
    const int gl = gid(i-ib*BX, j-jb*BY, k-kb*BZ, BX, BY, BZ);
    const int bls = gid(ib, jb, kb, NBX, NBY, NBZ);
    return gl + BT*bls;
}

)CLC";

// const std::string VPM3D_ocl_kernels_monolith_to_block = R"CLC(       // OBSOLETE
// __kernel void  Monolith_to_Block(   __global Real* src,
//                                     __global Real* dst)
// {
//     const int gx = get_local_id(0) + get_group_id(0)*BX;
//     const int gy = get_local_id(1) + get_group_id(1)*BY;
//     const int gz = get_local_id(2) + get_group_id(2)*BZ;
//     const int mid = gid(gx,gy,gz,NX,NY,NZ);           // Global id (Monolithic)
//     const int bid = gidb(gx,gy,gz);                   // Global id (Block)

//     dst[bid     ] += src[mid     ];
//     dst[bid+NT  ] += src[mid+NT  ];
//     dst[bid+2*NT] += src[mid+2*NT];
// }
// )CLC";

const std::string VPM3D_ocl_kernels_block_to_monolith = R"CLC(
__kernel void  Block_to_Monolith(   __global Real* src,
                                    __global Real* dst)
{
    const int gx = get_local_id(0) + get_group_id(0)*BX;
    const int gy = get_local_id(1) + get_group_id(1)*BY;
    const int gz = get_local_id(2) + get_group_id(2)*BZ;
    const int mid = gidf(gx,gy,gz,NX,NY,NZ);        // Global id (Monolithic)
    const int bid = gidb(gx,gy,gz);                 // Global id (Block)

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
    const int mid = gidf(gx,gy,gz,2*NX,2*NY,2*NZ);      // Global id (Monolithic)
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
    const int mid = gidf(gx,gy,gz,2*NX,2*NY,2*NZ);      // Global id (Monolithic-unbounded)
    const int bid = gidb(gx,gy,gz);                     // Global id (Block)

    dst[bid     ] = src1[mid];
    dst[bid+NT  ] = src2[mid];
    dst[bid+2*NT] = src3[mid];
}
)CLC";

//------------- Mapping kernels

const std::string VPM3D_ocl_kernels_mapping_functions = R"CLC(
inline void mapM2(Real x, Real *u){
   if (x< (Real)1.0) *u = (Real)1.0-x;
   else              *u = (Real)0.0;
}

inline void mapM4(Real x, Real *u){
    if (x<(Real)1.0)        *u = ((Real)2.0-x)*((Real)2.0-x)*((Real)2.0-x)/(Real)6.0 - (Real)4.0*((Real)1.0-x)*((Real)1.0-x)*((Real)1.0-x)/(Real)6.0;
    else if (x<(Real)2.0)   *u = ((Real)2.0-x)*((Real)2.0-x)*((Real)2.0-x)/(Real)6.0;
    else                    *u = (Real)0.0;
}

inline void mapM4D(Real x, Real *u){
   if (x<(Real)1.0)         *u = (Real)0.5*((Real)2.0-(Real)5.0*x*x+(Real)3.0*x*x*x);
   else if (x<(Real)2.0)    *u = (Real)0.5*((Real)1.0-x)*((Real)2.0-x)*((Real)2.0-x);
   else                     *u = (Real)0.0;
}

inline void mapM6D(Real x, Real *u){
    const Real x2 = x*x;
    if (x<(Real)1.0)        *u = (Real)(-1.0/88.0) *(x-(Real)1.0)*((Real)60.0*x2*x2-(Real)87.0*x*x2-(Real)87.0*x2+(Real)88.0*x+(Real)88.0);
    else if (x<(Real)2.0)   *u = (Real)( 1.0/176.0)*(x-(Real)1.0)*(x-(Real)2.0)*((Real)60.0*x*x2-(Real)261.0*x2+(Real)257.0*x+(Real)68.0);
    else if (x<(Real)3.0)   *u = (Real)(-3.0/176.0)*(x-(Real)2.0)*((Real)4.0*x2-(Real)17.0*x+(Real)12.0*(x-(Real)3.0)*(x-(Real)3.0));
    else                    *u = (Real)0.0;
}
)CLC";

const std::string VPM3D_ocl_kernels_map = R"CLC(

__kernel void MapKernel(__global const Real* f,
                        __global const Real* d,
                        __global const int* hs,
                        __global Real* interpf) {

__local Real sx[NHT];
__local Real sy[NHT];
__local Real sz[NHT];
__local Real dx[NHT];
__local Real dy[NHT];
__local Real dz[NHT];

// Prepare relevant indices
const int gx0 = get_group_id(0)*BX;
const int gy0 = get_group_id(1)*BY;
const int gz0 = get_group_id(2)*BZ;
const int tx = get_local_id(0);
const int ty = get_local_id(1);
const int tz = get_local_id(2);
const int gx = tx + gx0;
const int gy = ty + gy0;
const int gz = tz + gz0;
const int txh = get_local_id(0) + Halo;                                          // Local x id within padded grid
const int tyh = get_local_id(1) + Halo;                                          // Local y id within padded grid
const int tzh = get_local_id(2) + Halo;                                          // Local z id within padded grid

// Monolith structure
// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);
const int lid = gid(tx,ty,tz,BX,BY,BZ);                                     // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);                            // Local id within padded block

// Fill centre volume source & displacement coeff arrays
sx[pid] = f[bid     ];
sy[pid] = f[bid+NT  ];
sz[pid] = f[bid+2*NT];
dx[pid] = d[bid     ];
dy[pid] = d[bid+NT  ];
dz[pid] = d[bid+2*NT];

barrier(CLK_LOCAL_MEM_FENCE);

// Fill Halo (with coalesced index read)
for (int i=0; i<NHIT; i++){
   const int hid = BT*i + lid;
   const int hsx = hs[hid           ];          // global x-shift relative to position
   const int hsy = hs[hid+BT*NHIT   ];          // global y-shift relative to position
   const int hsz = hs[hid+2*BT*NHIT ];          // global z-shift relative to position
   if (hsx<NFDX){                               // Catch: is id within padded indices?
        const int ghx = gx0-Halo+hsx;           // Global x-value of retrieved node
        const int ghy = gy0-Halo+hsy;           // Global y-value of retrieved node
        const int ghz = gz0-Halo+hsz;           // Global z-value of retrieved node
        const int lhid = gid(hsx,hsy,hsz,NFDX,NFDY,NFDZ);

        // if (Data==Monolith) Kernel.append(const int bhid = gid(ghx,ghy,ghz,NX,NY,NZ););
        // if (Data==Block)    Kernel.append(const int bhid = gidb(ghx,ghy,ghz););
        const int bhid = gidb(ghx,ghy,ghz);

        const bool exx = (ghx<0 || ghx>=NX);      // Is x coordinate outside of the domain?
        const bool exy = (ghy<0 || ghy>=NY);      // Is y coordinate outside of the domain?
        const bool exz = (ghz<0 || ghz>=NZ);      // Is z coordinate outside of the domain?
        if (exx || exy || exz){                    // Catch: is id outside of domain?
            sx[lhid] = (Real)0.0;
            sy[lhid] = (Real)0.0;
            sz[lhid] = (Real)0.0;
            dx[lhid] = (Real)0.0;
            dy[lhid] = (Real)0.0;
            dz[lhid] = (Real)0.0;
        }
        else {
            sx[lhid] = f[bhid     ];
            sy[lhid] = f[bhid+1*NT];
            sz[lhid] = f[bhid+2*NT];
            dx[lhid] = d[bhid     ];
            dy[lhid] = d[bhid+1*NT];
            dz[lhid] = d[bhid+2*NT];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

// Now carry out interpolation using shared memory
// The displacement of the particles is used to calculate which nodes shall be used for mapping.
// The interpolation coefficients for this are then calculated and used to map the source field from the shared memory arrays.
Real mx = (Real)0.0, my = (Real)0.0, mz = (Real)0.0;     // Interpolated values
const bool mxx = (gx>(Halo-1) && gx<(NX-1-(Halo-1)));
const bool mxy = (gy>(Halo-1) && gy<(NY-1-(Halo-1)));
const bool mxz = (gz>(Halo-1) && gz<(NZ-1-(Halo-1)));
if (mxx && mxy && mxz){

    // Set interpolation limits
    int BL, BU;							// Upper and lower bounds
    #if (Map==2)
        {BL = -1; BU = 1;}		// M2 mapping
    #elif (Map==4)
        {BL = -2; BU = 2;}		// M4 mapping
    #elif (Map==42)
        {BL = -2; BU = 2;}		// M4D mapping
    #elif (Map==6)
        {BL = -3; BU = 3;}		// M6D mapping
    #else
        {BL = 0; BU = 0;}		// Mapping not recognised
    #endif

    // Carry out interpolation
    Real fx,fy,fz;
    for (int i=BL; i<=BU; i++){
        for (int j=BL; j<=BU; j++){
            for (int k=BL; k<=BU; k++){
                const int ids = gid(txh+i,tyh+j,tzh+k,NFDX,NFDY,NFDZ);
                #if (Map==2)
                    mapM2(fab((Real)i+dx[ids]/hx), &fx);
                    mapM2(fab((Real)j+dy[ids]/hy), &fy);
                    mapM2(fab((Real)k+dz[ids]/hz), &fz);
                #elif (Map==4)
                    mapM4(fab((Real)i+dx[ids]/hx), &fx);
                    mapM4(fab((Real)j+dy[ids]/hy), &fy);
                    mapM4(fab((Real)k+dz[ids]/hz), &fz);
                #elif (Map==42)
                    mapM4D(fab((Real)i+dx[ids]/hx), &fx);
                    mapM4D(fab((Real)j+dy[ids]/hy), &fy);
                    mapM4D(fab((Real)k+dz[ids]/hz), &fz);
                #elif (Map==6)
                    mapM6D(fab((Real)i+dx[ids]/hx), &fx);
                    mapM6D(fab((Real)j+dy[ids]/hy), &fy);
                    mapM6D(fab((Real)k+dz[ids]/hz), &fz);
                #else
                    // Do nothing- undefined
                #endif

                const Real Fac = fx*fy*fz;
                mx += Fac*sx[ids];
                my += Fac*sy[ids];
                mz += Fac*sz[ids];
            }
        }
    }
}
barrier(CLK_LOCAL_MEM_FENCE);

// Write output
interpf[bid     ] = mx;
interpf[bid+1*NT] = my;
interpf[bid+2*NT] = mz;
}
)CLC";

//-- Interpolation kernels

const std::string VPM3D_ocl_kernels_interp = R"CLC(
__kernel void InterpKernel( __global const Real* f1,
                            __global const Real* f2,
                            __global const Real* d,
                            __global const int* hs,
                            __global Real* interpf1,
                            __global Real* interpf2) {

__local Real sx[NHT];
__local Real sy[NHT];
__local Real sz[NHT];
__local Real ux[NHT];
__local Real uy[NHT];
__local Real uz[NHT];

// Prepare relevant indices
const int gx0 = get_group_id(0)*BX;
const int gy0 = get_group_id(1)*BY;
const int gz0 = get_group_id(2)*BZ;
const int tx = get_local_id(0);
const int ty = get_local_id(1);
const int tz = get_local_id(2);
const int gx = tx + gx0;
const int gy = ty + gy0;
const int gz = tz + gz0;
const int txh = get_local_id(0) + Halo;                                          // Local x id within padded grid
const int tyh = get_local_id(1) + Halo;                                          // Local y id within padded grid
const int tzh = get_local_id(2) + Halo;                                          // Local z id within padded grid

// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);

const int lid = gid(tx,ty,tz,BX,BY,BZ);                 // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);        // Local id within padded block

// Fill centre volume source arrays & mapping coeff arrays
sx[pid] = f1[bid        ];
sy[pid] = f1[bid+NT     ];
sz[pid] = f1[bid+2*NT   ];
ux[pid] = f2[bid        ];
uy[pid] = f2[bid+NT     ];
uz[pid] = f2[bid+2*NT   ];

// Load particle displacements
const Real dx = d[bid];
const Real dy = d[bid+NT];
const Real dz = d[bid+2*NT];

barrier(CLK_LOCAL_MEM_FENCE);

// Fill Halo (with coalesced index read)
for (int i=0; i<NHIT; i++){
    const int hid = BT*i + lid;
    const int hsx = hs[hid          ];          // global x-shift relative to position
    const int hsy = hs[hid+BT*NHIT  ];          // global y-shift relative to position
    const int hsz = hs[hid+2*BT*NHIT];          // global z-shift relative to position
    if (hsx<NFDX){                              // Catch: is id within padded indices?
        const int ghx = gx0-Halo+hsx;           // Global x-value of retrieved node
        const int ghy = gy0-Halo+hsy;           // Global y-value of retrieved node
        const int ghz = gz0-Halo+hsz;           // Global z-value of retrieved node
        const int lhid = gid(hsx,hsy,hsz,NFDX,NFDY,NFDZ);

        // if (Data==Monolith) Kernel.append(const int bhid = gid(ghx,ghy,ghz,NX,NY,NZ););
        // if (Data==Block)    Kernel.append(const int bhid = gidb(ghx,ghy,ghz););
        const int bhid = gidb(ghx,ghy,ghz);

        const bool exx = (ghx<0 || ghx>=NX);      // Is x coordinate outside of the domain?
        const bool exy = (ghy<0 || ghy>=NY);      // Is y coordinate outside of the domain?
        const bool exz = (ghz<0 || ghz>=NZ);      // Is z coordinate outside of the domain?
        if (exx || exy || exz){                    // Catch: is id outside of domain?
            sx[lhid] = (Real)0.0;
            sy[lhid] = (Real)0.0;
            sz[lhid] = (Real)0.0;
            ux[lhid] = (Real)0.0;
            uy[lhid] = (Real)0.0;
            uz[lhid] = (Real)0.0;
        }
        else {
            sx[lhid] = f1[bhid     ];
            sy[lhid] = f1[bhid+1*NT];
            sz[lhid] = f1[bhid+2*NT];
            ux[lhid] = f2[bhid     ];
            uy[lhid] = f2[bhid+1*NT];
            uz[lhid] = f2[bhid+2*NT];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

// Now carry out interpolation using shared memory
// The displacement of the particles is used to calculate which nodes shall be used for mapping.
// The interpolation coefficients for this are then calculated and used to map the source field from the shared memory arrays.
Real m1x = (Real)0.0, m1y = (Real)0.0, m1z = (Real)0.0;     // Interpolated values
Real m2x = (Real)0.0, m2y = (Real)0.0, m2z = (Real)0.0;     // Interpolated values
// const bool mxx = (gx>0 && gx<(NX-1));
// const bool mxy = (gy>0 && gy<(NY-1));
// const bool mxz = (gz>0 && gz<(NZ-1));
const bool mxx = (gx>(Halo-1) && gx<(NX-1-(Halo-1)));
const bool mxy = (gy>(Halo-1) && gy<(NY-1-(Halo-1)));
const bool mxz = (gz>(Halo-1) && gz<(NZ-1-(Halo-1)));
if (mxx && mxy && mxz){
    int iix, iiy, iiz;                                // Interpolation id
    const Real dxh = dx/hx, dyh = dy/hy, dzh = dz/hz;    // Normalized distances

    // Calculate interpolation weights
    const int H2 = Halo*2;
    Real cx[H2], cy[H2], cz[H2];			// Interpolation weights
    int NS;								// Shift for node id

    #if (Map==2)    // M2 interpolation

        NS = 0;
        if (dxh>=(Real)0.0){iix = txh;      mapM2(dxh,&cx[0]);              mapM2((Real)1.0-dxh,&cx[1]);    }
        else {              iix = txh-1;    mapM2((Real)1.0+dxh,&cx[0]);    mapM2(-dxh,&cx[1]);             }

        if (dyh>=(Real)0.0){iiy = tyh;      mapM2(dyh,&cy[0]);              mapM2((Real)1.0-dyh,&cy[1]);    }
        else {             	iiy = tyh-1;    mapM2((Real)1.0+dyh,&cy[0]);    mapM2(-dyh,&cy[1]);             }

        if (dzh>=(Real)0.0){iiz = tzh;      mapM2(dzh,&cz[0]);              mapM2((Real)1.0-dzh,&cz[1]); 	}
        else {             	iiz = tzh-1;    mapM2((Real)1.0+dzh,&cz[0]);    mapM2(-dzh,&cz[1]);             }

    #elif (Map==4)  // M4 interpolation

        NS = -1;
        if (dxh>=(Real)0.0){iix = txh;      mapM4((Real)1.0+dxh,&cx[0]);  mapM4(dxh,&cx[1]);    		mapM4((Real)1.0-dxh,&cx[2]);    mapM4((Real)2.0-dxh,&cx[3]);}
        else {             	iix = txh-1;    mapM4((Real)2.0+dxh,&cx[0]);  mapM4((Real)1.0+dxh,&cx[1]);  mapM4(-dxh,&cx[2]);             mapM4((Real)1.0-dxh,&cx[3]);}

        if (dyh>=(Real)0.0){iiy = tyh;    	mapM4((Real)1.0+dyh,&cy[0]);  mapM4(dyh,&cy[1]);            mapM4((Real)1.0-dyh,&cy[2]);	mapM4((Real)2.0-dyh,&cy[3]);}
        else {             	iiy = tyh-1;    mapM4((Real)2.0+dyh,&cy[0]);  mapM4((Real)1.0+dyh,&cy[1]);	mapM4(-dyh,&cy[2]);             mapM4((Real)1.0-dyh,&cy[3]);}

        if (dzh>=(Real)0.0){iiz = tzh;		mapM4((Real)1.0+dzh,&cz[0]);  mapM4(dzh,&cz[1]);            mapM4((Real)1.0-dzh,&cz[2]);	mapM4((Real)2.0-dzh,&cz[3]);}
        else {             	iiz = tzh-1;	mapM4((Real)2.0+dzh,&cz[0]);  mapM4((Real)1.0+dzh,&cz[1]);	mapM4(-dzh,&cz[2]);             mapM4((Real)1.0-dzh,&cz[3]);}

    #elif (Map==42) // M4' interpolation

        NS = -1;
        if (dxh>=(Real)0.0){iix = txh;      mapM4D((Real)1.0+dxh,&cx[0]);  mapM4D(dxh,&cx[1]);    			mapM4D((Real)1.0-dxh,&cx[2]);   mapM4D((Real)2.0-dxh,&cx[3]);}
        else {             	iix = txh-1;    mapM4D((Real)2.0+dxh,&cx[0]);  mapM4D((Real)1.0+dxh,&cx[1]);   	mapM4D(-dxh,&cx[2]);            mapM4D((Real)1.0-dxh,&cx[3]);}

        if (dyh>=(Real)0.0){iiy = tyh;      mapM4D((Real)1.0+dyh,&cy[0]);  mapM4D(dyh,&cy[1]);   			mapM4D((Real)1.0-dyh,&cy[2]);	mapM4D((Real)2.0-dyh,&cy[3]);}
        else {             	iiy = tyh-1;    mapM4D((Real)2.0+dyh,&cy[0]);  mapM4D((Real)1.0+dyh,&cy[1]);   	mapM4D(-dyh,&cy[2]);            mapM4D((Real)1.0-dyh,&cy[3]);}

        if (dzh>=(Real)0.0){iiz = tzh;      mapM4D((Real)1.0+dzh,&cz[0]);  mapM4D(dzh,&cz[1]);   			mapM4D((Real)1.0-dzh,&cz[2]);	mapM4D((Real)2.0-dzh,&cz[3]);}
        else {             	iiz = tzh-1;    mapM4D((Real)2.0+dzh,&cz[0]);  mapM4D((Real)1.0+dzh,&cz[1]);   	mapM4D(-dzh,&cz[2]);            mapM4D((Real)1.0-dzh,&cz[3]);}

    #elif (Map==6) // M6' interpolation

        NS = -2;
        if (dxh>=(Real)0.0){iix = txh;      mapM6D((Real)2.0+dxh,&cx[0]);  mapM6D((Real)1.0+dxh,&cx[1]);  mapM6D(      dxh,&cx[2]);         mapM6D((Real)1.0-dxh,&cx[3]); 	mapM6D((Real)2.0-dxh,&cx[4]);  	mapM6D((Real)3.0-dxh,&cx[5]);}
        else {             	iix = txh-1;    mapM6D((Real)3.0+dxh,&cx[0]);  mapM6D((Real)2.0+dxh,&cx[1]);  mapM6D((Real)1.0+dxh,&cx[2]);  	mapM6D(     -dxh,&cx[3]); 		mapM6D((Real)1.0-dxh,&cx[4]);  	mapM6D((Real)2.0-dxh,&cx[5]);}

        if (dyh>=(Real)0.0){iiy = tyh;      mapM6D((Real)2.0+dyh,&cy[0]);  mapM6D((Real)1.0+dyh,&cy[1]);  mapM6D(      dyh,&cy[2]);         mapM6D((Real)1.0-dyh,&cy[3]); 	mapM6D((Real)2.0-dyh,&cy[4]);  	mapM6D((Real)3.0-dyh,&cy[5]);}
        else {             	iiy = tyh-1;    mapM6D((Real)3.0+dyh,&cy[0]);  mapM6D((Real)2.0+dyh,&cy[1]);  mapM6D((Real)1.0+dyh,&cy[2]);  	mapM6D(     -dyh,&cy[3]); 		mapM6D((Real)1.0-dyh,&cy[4]);  	mapM6D((Real)2.0-dyh,&cy[5]);}

        if (dzh>=(Real)0.0){iiz = tzh;      mapM6D((Real)2.0+dzh,&cz[0]);  mapM6D((Real)1.0+dzh,&cz[1]);  mapM6D(      dzh,&cz[2]);         mapM6D((Real)1.0-dzh,&cz[3]); 	mapM6D((Real)2.0-dzh,&cz[4]);  	mapM6D((Real)3.0-dzh,&cz[5]);}
        else {             	iiz = tzh-1;    mapM6D((Real)3.0+dzh,&cz[0]);  mapM6D((Real)2.0+dzh,&cz[1]);  mapM6D((Real)1.0+dzh,&cz[2]);  	mapM6D(     -dzh,&cz[3]); 		mapM6D((Real)1.0-dzh,&cz[4]);  	mapM6D((Real)2.0-dzh,&cz[5]);}

    #endif

    // Carry out interpolation
    for (int i=0; i<H2; i++){
        for (int j=0; j<H2; j++){
            for (int k=0; k<H2; k++){
                const int idsx =  iix + NS + i;
                const int idsy =  iiy + NS + j;
                const int idsz =  iiz + NS + k;
                const int ids =  gid(idsx,idsy,idsz,NFDX,NFDY,NFDZ);
                const Real fac =  cx[i]*cy[j]*cz[k];
                m1x += fac*sx[ids];
                m1y += fac*sy[ids];
                m1z += fac*sz[ids];
                m2x += fac*ux[ids];
                m2y += fac*uy[ids];
                m2z += fac*uz[ids];
            }
        }
    }
}
barrier(CLK_LOCAL_MEM_FENCE);

// Write output
interpf1[bid     ] = m1x;
interpf1[bid+1*NT] = m1y;
interpf1[bid+2*NT] = m1z;
interpf2[bid     ] = m2x;
interpf2[bid+1*NT] = m2y;
interpf2[bid+2*NT] = m2z;
}
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
