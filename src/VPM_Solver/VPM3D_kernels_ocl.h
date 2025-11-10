//--------------------------------------------------------------------------------
//-------------------------- VPM 3D: ocl Kernels --------------------------------
//--------------------------------------------------------------------------------

// Note: Prior to initialiization, the "Real" type must be defined along with float/double dependent functions

// #include "../SailFFish_Math_Types.h"
#include <string>

namespace SailFFish
{

const std::string VPM3D_ocl_kernels_float = R"CLC(
typedef float   Real;
typedef float2  Complex;
// inline float fastma(const float &a, const float &b, const float &c)    {return fmaf(a,b,c);}
inline float fab(float a)   {return fabs(a);}
)CLC";

const std::string VPM3D_ocl_kernels_double = R"CLC(
typedef double  Real;
typedef double2 Complex;
// inline double fastma(const double &a, const double &b, const double &c) {return fma(a,b,c);}
inline double fab(float a)  {return abs(a);}
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

//--- Convolution kernels

const std::string VPM3D_ocl_kernels_reprojection = R"CLC(

__kernel void  vpm_reprojection(__global const Complex *OX,
                                __global const Complex *OY,
                                __global const Complex *OZ,
                                __global const Complex *GF,
                                __global const Complex *iX,
                                __global const Complex *iY,
                                __global const Complex *iZ,
                                __global Complex *UX,
                                __global Complex *UY,
                                __global Complex *UZ) {

// Specify grid id
unsigned int i = get_group_id(0)*get_local_size(0) + get_local_id(0);

// Load values for this node into memory
const Complex ox = OX[i];
const Complex oy = OY[i];
const Complex oz = OZ[i];
const Complex gf = GF[i];
const Complex ix = iX[i];
const Complex iy = iY[i];
const Complex iz = iZ[i];

barrier(CLK_LOCAL_MEM_FENCE);

// Calculate divergence in spectral space
const Complex duxdx = {ox.x*ix.x - ox.y*ix.y , ox.x*ix.y + ox.y*ix.x};
const Complex duydy = {oy.x*iy.x - oy.y*iy.y , oy.x*iy.y + oy.y*iy.x};
const Complex duzdz = {oz.x*iz.x - oz.y*iz.y , oz.x*iz.y + oz.y*iz.x};
const Complex divom = {duxdx.x + duydy.x + duzdz.x, duxdx.y + duydy.y + duzdz.y};

// Solve for Nabla^2 (F) = divOm
const Complex F = {divom.x*gf.x - divom.y*gf.y , divom.x*gf.y + divom.y*gf.x};

// Calculate gradient of F
const Complex dFdx = {F.x*ix.x - F.y*ix.y , F.x*ix.y + F.y*ix.x};
const Complex dFdy = {F.x*iy.x - F.y*iy.y , F.x*iy.y + F.y*iy.x};
const Complex dFdz = {F.x*iz.x - F.y*iz.y , F.x*iz.y + F.y*iz.x};

// Reprojected output calculated by scaling input field and subtracting gradients of F term
const Real BFac = (Real)1.0/(Real)(NT*8);    // NT const is actually 'NNT'-> Grid dimension for the unextended domain
const Complex rOx = {ox.x*BFac - dFdx.x , ox.y*BFac - dFdx.y};
const Complex rOy = {oy.x*BFac - dFdy.x , oy.y*BFac - dFdy.y};
const Complex rOz = {oz.x*BFac - dFdz.x , oz.y*BFac - dFdz.y};

barrier(CLK_LOCAL_MEM_FENCE);

// Write outputs
UX[i] = rOx;
UY[i] = rOy;
UZ[i] = rOz;
}
)CLC";

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

//--- Mapping kernels

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

//--- Interpolation kernels

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

const std::string VPM3D_ocl_kernels_InterpBlock = R"CLC(

__kernel void Interp_Block( __global const Real* fd,         // Source grid
                            __global const Real* dsx,        // Positions of particles to be mapped (in local block coordinate system)
                            __global const Real* dsy,
                            __global const Real* dsz,
                            __global const int *blX,         // Block indices
                            __global const int *blY,
                            __global const int *blZ,
                            __global const int* hs,          // Halo indices
                            __global Real* ux,               // Interpolated values from grid
                            __global Real* uy,
                            __global Real* uz)
{
// Declare shared memory vars (velocity field in shared memory)
__local Real sx[NHT];
__local Real sy[NHT];
__local Real sz[NHT];

// We are still executing the blocks with blockdim [BX, BY, BZ], however are are using the Griddims in a 1d sense.
// So we need to modify the x-dim based on the number of active blocks.

// Prepare relevant indices
const int BlockID = get_group_id(0);
const int gx0 = blX[BlockID]*BX;
const int gy0 = blY[BlockID]*BY;
const int gz0 = blZ[BlockID]*BZ;
const int tx = get_local_id(0);
const int ty = get_local_id(1);
const int tz = get_local_id(2);
const int gx = tx + gx0;
const int gy = ty + gy0;
const int gz = tz + gz0;
const int txh = tx + Halo;                              // Local x id within padded grid
const int tyh = ty + Halo;                              // Local y id within padded grid
const int tzh = tz + Halo;                              // Local z id within padded grid

// Monolith structure
// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);
const int lid = gid(tx,ty,tz,BX,BY,BZ);                 // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);        // Local id within padded block


//-------------------------------------------------------
// Step 1) Copy global memory to shared & local memory
//-------------------------------------------------------

// Specify source particle vorticity, position
const int llid = lid + BlockID*BT;
const Real pX = dsx[llid];
const Real pY = dsy[llid];
const Real pZ = dsz[llid];

// Specify centre volume arrays
sx[pid] = fd[bid       ];
sy[pid] = fd[bid + 1*NT];
sz[pid] = fd[bid + 2*NT];

barrier(CLK_LOCAL_MEM_FENCE);

// Fill Halo (with coalesced index read)
for (int i=0; i<NHIT; i++){
   const int hid = BT*i + lid;
   const int hsx = hs[hid];                    		// global x-shift relative to position
   const int hsy = hs[hid+BT*NHIT];               	// global y-shift relative to position
   const int hsz = hs[hid+2*BT*NHIT];             	// global z-shift relative to position
   if (hsx<NFDX){                                 	// Catch: is id within padded indices?
        const int ghx = gx0-Halo+hsx;             	// Global x-value of retrieved node
        const int ghy = gy0-Halo+hsy;             	// Global y-value of retrieved node
        const int ghz = gz0-Halo+hsz;             	// Global z-value of retrieved node
        const int lhid = gid(hsx,hsy,hsz,NFDX,NFDY,NFDZ);

        // if (Data==Monolith) Kernel.append(const int bhid = gid(ghx,ghy,ghz,NX,NY,NZ););
        // if (Data==Block)    Kernel.append(const int bhid = gidb(ghx,ghy,ghz););
        const int bhid = gidb(ghx,ghy,ghz);

        const bool exx = (ghx<0 || ghx>=NX);      	// Is x coordinate outside of the domain?
        const bool exy = (ghy<0 || ghy>=NY);      	// Is y coordinate outside of the domain?
        const bool exz = (ghz<0 || ghz>=NZ);      	// Is z coordinate outside of the domain?
        if (exx || exy || exz){                    	// Catch: is id outside of domain?
            sx[lhid] = (Real)0.0;
            sy[lhid] = (Real)0.0;
            sz[lhid] = (Real)0.0;
        }
        else {
            sx[lhid] = fd[bhid       ];
            sy[lhid] = fd[bhid + 1*NT];
            sz[lhid] = fd[bhid + 2*NT];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// Step 2) Interpolate values from grid
//------------------------------------------------------------------------------

Real m1x = (Real)0.0, m1y = (Real)0.0, m1z = (Real)0.0;     // Interpolated values

// Calculate relative indices and displacement factors
Real pmx, pmy, pmz;
#if (Map==2)
    pmx = (Real)0.5*hx + pX;
    pmy = (Real)0.5*hy + pY;
    pmz = (Real)0.5*hz + pZ;
#elif (Map==4)
    pmx = (Real)1.5*hx + pX;
    pmy = (Real)1.5*hy + pY;
    pmz = (Real)1.5*hz + pZ;
#elif (Map==42)
    pmx = (Real)1.5*hx + pX;
    pmy = (Real)1.5*hy + pY;
    pmz = (Real)1.5*hz + pZ;
#elif (Map==62)
    pmx = (Real)2.5*hx + pX;
    pmy = (Real)2.5*hy + pY;
    pmz = (Real)2.5*hz + pZ;
#endif

const int iix = (int)pmx/hx;
const int iiy = (int)pmy/hy;
const int iiz = (int)pmz/hz;
const Real dxh = (pmx-hx*iix)/hx;
const Real dyh = (pmy-hy*iiy)/hy;
const Real dzh = (pmz-hz*iiz)/hz;

// Calculate interpolation weights
const int H2 = Halo*2;
Real cx[H2], cy[H2], cz[H2];			// Interpolation weights
int NS;								// Shift for node id

#if (Map==2)
    NS = 0;
    mapM2(dxh,&cx[0]);               mapM2((Real)1.0-dxh,&cx[1]);
    mapM2(dyh,&cy[0]);               mapM2((Real)1.0-dyh,&cy[1]);
    mapM2(dzh,&cz[0]);               mapM2((Real)1.0-dzh,&cz[1]);
#elif (Map==4)
    NS = -1;
    mapM4((Real)1.0+dxh,&cx[0]);     mapM4(dxh,&cx[1]);               mapM4((Real)1.0-dxh,&cx[2]);  	mapM4((Real)2.0-dxh,&cx[3]);
    mapM4((Real)1.0+dyh,&cy[0]);     mapM4(dyh,&cy[1]);               mapM4((Real)1.0-dyh,&cy[2]);  	mapM4((Real)2.0-dyh,&cy[3]);
    mapM4((Real)1.0+dzh,&cz[0]);     mapM4(dzh,&cz[1]);               mapM4((Real)1.0-dzh,&cz[2]);  	mapM4((Real)2.0-dzh,&cz[3]);
#elif (Map==42)
    NS = -1;
    mapM4D((Real)1.0+dxh,&cx[0]);    mapM4D(dxh,&cx[1]);              mapM4D((Real)1.0-dxh,&cx[2]);  	mapM4D((Real)2.0-dxh,&cx[3]);
    mapM4D((Real)1.0+dyh,&cy[0]);    mapM4D(dyh,&cy[1]);              mapM4D((Real)1.0-dyh,&cy[2]);  	mapM4D((Real)2.0-dyh,&cy[3]);
    mapM4D((Real)1.0+dzh,&cz[0]);    mapM4D(dzh,&cz[1]);              mapM4D((Real)1.0-dzh,&cz[2]);  	mapM4D((Real)2.0-dzh,&cz[3]);
#elif (Map==6)
    NS = -2;
    mapM6D((Real)2.0+dxh,&cx[0]);    mapM6D((Real)1.0+dxh,&cx[1]);    mapM6D(dxh,&cx[2]);              mapM6D((Real)1.0-dxh,&cx[3]); 	mapM6D((Real)2.0-dxh,&cx[4]);  	mapM6D((Real)3.0-dxh,&cx[5]);
    mapM6D((Real)2.0+dyh,&cy[0]);    mapM6D((Real)1.0+dyh,&cy[1]);    mapM6D(dyh,&cy[2]);              mapM6D((Real)1.0-dyh,&cy[3]); 	mapM6D((Real)2.0-dyh,&cy[4]);  	mapM6D((Real)3.0-dyh,&cy[5]);
    mapM6D((Real)2.0+dzh,&cz[0]);    mapM6D((Real)1.0+dzh,&cz[1]);    mapM6D(dzh,&cz[2]);              mapM6D((Real)1.0-dzh,&cz[3]); 	mapM6D((Real)2.0-dzh,&cz[4]);  	mapM6D((Real)3.0-dzh,&cz[5]);
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
            m1x +=  fac*sx[ids];
            m1y +=  fac*sy[ids];
            m1z +=  fac*sz[ids];
       }
    }
}

barrier(CLK_LOCAL_MEM_FENCE);

//------------------------------------------------------------------------------
// Step 3) Transfer interpolated values back to array
//------------------------------------------------------------------------------

ux[llid] = m1x;
uy[llid] = m1y;
uz[llid] = m1z;
}

)CLC";

//--- Timestepping kernels

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
    Real p1  = d[i          ];
    Real p2  = d[i + NT     ];
    Real p3  = d[i + 2*NT   ];
    Real o1  = o[i          ];
    Real o2  = o[i + NT     ];
    Real o3  = o[i + 2*NT   ];
    Real dp1 = d_d[i        ];
    Real dp2 = d_d[i + NT   ];
    Real dp3 = d_d[i + 2*NT ];
    Real do1 = d_o[i        ];
    Real do2 = d_o[i + NT   ];
    Real do3 = d_o[i + 2*NT ];

    // Write updated values
    d[i     ] = p1 + dt * dp1;
    d[i+1*NT] = p2 + dt * dp2;
    d[i+2*NT] = p3 + dt * dp3;
    o[i     ] = o1 + dt * do1;
    o[i+1*NT] = o2 + dt * do2;
    o[i+2*NT] = o3 + dt * do3;
}

)CLC";

const std::string VPM3D_ocl_kernels_updateRK = R"CLC(
__kernel void updateRK( __global const Real* d,
                        __global const Real* o,
                        __global const Real* d_d,
                        __global const Real* d_o,
                        __global Real* kd,
                        __global Real* ko,
                        const Real dt)
{

    // Set grid id
    unsigned int i = get_group_id(0)*get_local_size(0) + get_local_id(0);

    const Real p1 = d[i     ];
    const Real p2 = d[i+NT  ];
    const Real p3 = d[i+2*NT];
    const Real o1 = o[i     ];
    const Real o2 = o[i+NT  ];
    const Real o3 = o[i+2*NT];

    const Real dp1 = d_d[i      ];
    const Real dp2 = d_d[i+NT   ];
    const Real dp3 = d_d[i+2*NT ];
    const Real do1 = d_o[i      ];
    const Real do2 = d_o[i+NT   ];
    const Real do3 = d_o[i+2*NT ];

    barrier(CLK_LOCAL_MEM_FENCE);       // Probably not necessary

    // Set outputs
    kd[i     ] = p1 + dt*dp1;
    kd[i+1*NT] = p2 + dt*dp2;
    kd[i+2*NT] = p3 + dt*dp3;
    ko[i     ] = o1 + dt*do1;
    ko[i+1*NT] = o2 + dt*do2;
    ko[i+2*NT] = o3 + dt*do3;
}

)CLC";

const std::string VPM3D_ocl_kernels_updateRK2 = R"CLC(
__kernel void updateRK2(__global Real* d,
                        __global Real* o,
                        __global const Real* k1d,
                        __global const Real* k1o,
                        __global const Real* k2d,
                        __global const Real* k2o,
                        const Real dt) {

    // Set grid id
    unsigned int i = get_group_id(0)*get_local_size(0) + get_local_id(0);

    const Real p1 = d[i     ];
    const Real p2 = d[i+NT  ];
    const Real p3 = d[i+2*NT];
    const Real o1 = o[i     ];
    const Real o2 = o[i+NT  ];
    const Real o3 = o[i+2*NT];

    const Real dp1k1 = k1d[i    ];      const Real dp1k2 = k2d[i    ];
    const Real dp2k1 = k1d[i+NT ];		const Real dp2k2 = k2d[i+NT ];
    const Real dp3k1 = k1d[i+2*NT];		const Real dp3k2 = k2d[i+2*NT];
    const Real do1k1 = k1o[i    ];      const Real do1k2 = k2o[i    ];
    const Real do2k1 = k1o[i+NT ];		const Real do2k2 = k2o[i+NT ];
    const Real do3k1 = k1o[i+2*NT];		const Real do3k2 = k2o[i+2*NT];

    barrier(CLK_LOCAL_MEM_FENCE);       // Probably not necessary

    // Set outputs
    const Real f = (Real)0.5*dt;
    d[i     ] = p1 + f*(dp1k1+dp1k2);
    d[i+1*NT] = p2 + f*(dp2k1+dp2k2);
    d[i+2*NT] = p3 + f*(dp3k1+dp3k2);
    o[i     ] = o1 + f*(do1k1+do1k2);
    o[i+1*NT] = o2 + f*(do2k1+do2k2);
    o[i+2*NT] = o3 + f*(do3k1+do3k2);
}

)CLC";

const std::string VPM3D_ocl_kernels_updateRK3 = R"CLC(
__kernel void updateRK3(__global Real* d,
                        __global Real* o,
                        __global const Real* k1d,
                        __global const Real* k1o,
                        __global const Real* k2d,
                        __global const Real* k2o,
                        __global const Real* k3d,
                        __global const Real* k3o,
                        const Real dt) {

// Set grid id
unsigned int i = get_group_id(0)*get_local_size(0) + get_local_id(0);

const Real p1 = d[i     ];
const Real p2 = d[i+NT  ];
const Real p3 = d[i+2*NT];
const Real o1 = o[i     ];
const Real o2 = o[i+NT  ];
const Real o3 = o[i+2*NT];

const Real dp1k1 = k1d[i    ];      const Real dp1k2 = k2d[i    ];      const Real dp1k3 = k3d[i    ];
const Real dp2k1 = k1d[i+NT ];		const Real dp2k2 = k2d[i+NT ];		const Real dp2k3 = k3d[i+NT ];
const Real dp3k1 = k1d[i+2*NT];		const Real dp3k2 = k2d[i+2*NT];		const Real dp3k3 = k3d[i+2*NT];
const Real do1k1 = k1o[i    ];      const Real do1k2 = k2o[i    ];      const Real do1k3 = k3o[i    ];
const Real do2k1 = k1o[i+NT ];		const Real do2k2 = k2o[i+NT ];		const Real do2k3 = k3o[i+NT ];
const Real do3k1 = k1o[i+2*NT];		const Real do3k2 = k2o[i+2*NT];		const Real do3k3 = k3o[i+2*NT];


barrier(CLK_LOCAL_MEM_FENCE);       // Probably not necessary

// Set outputs
const Real f = dt/(Real)6.0;
d[i     ] = p1 + f*(dp1k1 + (Real)4.0*dp1k2 + dp1k3);
d[i+1*NT] = p2 + f*(dp2k1 + (Real)4.0*dp2k2 + dp2k3);
d[i+2*NT] = p3 + f*(dp3k1 + (Real)4.0*dp3k2 + dp3k3);
o[i     ] = o1 + f*(do1k1 + (Real)4.0*do1k2 + do1k3);
o[i+1*NT] = o2 + f*(do2k1 + (Real)4.0*do2k2 + do2k3);
o[i+2*NT] = o3 + f*(do3k1 + (Real)4.0*do3k2 + do3k3);
}

)CLC";

const std::string VPM3D_ocl_kernels_updateRK4 = R"CLC(
__kernel void updateRK4(__global Real* d,
                        __global Real* o,
                        __global const Real* k1d,
                        __global const Real* k1o,
                        __global const Real* k2d,
                        __global const Real* k2o,
                        __global const Real* k3d,
                        __global const Real* k3o,
                        __global  const Real* k4d,
                        __global const Real* k4o,
                        const Real dt) {

// Set grid id
unsigned int i = get_group_id(0)*get_local_size(0) + get_local_id(0);

const Real p1 = d[i     ];
const Real p2 = d[i+NT  ];
const Real p3 = d[i+2*NT];
const Real o1 = o[i     ];
const Real o2 = o[i+NT  ];
const Real o3 = o[i+2*NT];

const Real dp1k1 = k1d[i    ];      const Real dp1k2 = k2d[i    ];      const Real dp1k3 = k3d[i    ];      const Real dp1k4 = k4d[i    ];
const Real dp2k1 = k1d[i+NT ];		const Real dp2k2 = k2d[i+NT ];		const Real dp2k3 = k3d[i+NT ];		const Real dp2k4 = k4d[i+NT ];
const Real dp3k1 = k1d[i+2*NT];		const Real dp3k2 = k2d[i+2*NT];		const Real dp3k3 = k3d[i+2*NT];		const Real dp3k4 = k4d[i+2*NT];
const Real do1k1 = k1o[i    ];      const Real do1k2 = k2o[i    ];      const Real do1k3 = k3o[i    ];      const Real do1k4 = k4o[i    ];
const Real do2k1 = k1o[i+NT ];		const Real do2k2 = k2o[i+NT ];		const Real do2k3 = k3o[i+NT ];		const Real do2k4 = k4o[i+NT ];
const Real do3k1 = k1o[i+2*NT];		const Real do3k2 = k2o[i+2*NT];		const Real do3k3 = k3o[i+2*NT];		const Real do3k4 = k4o[i+2*NT];

barrier(CLK_LOCAL_MEM_FENCE);

// Set outputs
const Real f = dt/(Real)6.0;
d[i     ] = p1 + f*(dp1k1 + (Real)2.0*dp1k2 + (Real)2.0*dp1k3 + dp1k4);
d[i+1*NT] = p2 + f*(dp2k1 + (Real)2.0*dp2k2 + (Real)2.0*dp2k3 + dp2k4);
d[i+2*NT] = p3 + f*(dp3k1 + (Real)2.0*dp3k2 + (Real)2.0*dp3k3 + dp3k4);
o[i     ] = o1 + f*(do1k1 + (Real)2.0*do1k2 + (Real)2.0*do1k3 + do1k4);
o[i+1*NT] = o2 + f*(do2k1 + (Real)2.0*do2k2 + (Real)2.0*do2k3 + do2k4);
o[i+2*NT] = o3 + f*(do3k1 + (Real)2.0*do3k2 + (Real)2.0*do3k3 + do3k4);

}

)CLC";

const std::string VPM3D_ocl_kernels_updateRKLS = R"CLC(
__kernel void updateRK_LS(  __global  Real* d,
                            __global Real* o,
                            __global const Real* d_d,
                            __global const Real* d_o,
                            __global Real* s_d,
                            __global Real* s_o,
                            const Real A,
                            const Real B,
                            const Real h)
{

// Set grid id
unsigned int i = get_group_id(0)*get_local_size(0) + get_local_id(0);

// Intermediate vars
Real s21 = A*s_d[i      ]   + h*d_d[i       ];
Real s22 = A*s_d[i+NT   ]   + h*d_d[i+NT    ];
Real s23 = A*s_d[i+2*NT ] 	+ h*d_d[i+2*NT  ];
Real s24 = A*s_o[i      ]   + h*d_o[i       ];
Real s25 = A*s_o[i+NT   ]   + h*d_o[i+NT    ];
Real s26 = A*s_o[i+2*NT ]	+ h*d_o[i+2*NT  ];

barrier(CLK_LOCAL_MEM_FENCE);       // Probably unecessary

// Update intermediate vector
s_d[i     ] = s21;
s_d[i+1*NT] = s22;
s_d[i+2*NT] = s23;
s_o[i     ] = s24;
s_o[i+1*NT] = s25;
s_o[i+2*NT] = s26;

// Update state vector
d[i     ] += B*s21;
d[i+1*NT] += B*s22;
d[i+2*NT] += B*s23;
o[i     ] += B*s24;
o[i+1*NT] += B*s25;
o[i+2*NT] += B*s26;
}

)CLC";

//--- Turbulence Kernels

const std::string VPM3D_ocl_kernels_subgrid_discfilter = R"CLC(

__kernel void SubGrid_DiscFilter(   __global const Real* Om,
                                    __global const int* hs,
                                    __global const Real *O,
                                    __global Real *filtO,
                                    const int option) {

// Declare shared memory arrays
__local Real ox[NHT];
__local Real oy[NHT];
__local Real oz[NHT];

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

const int lid = gid(tx,ty,tz,BX,BY,BZ);           // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);                             // Local id within padded block

// Fill centre volume
ox[pid] = O[bid		];
oy[pid] = O[bid+1*NT];
oz[pid] = O[bid+2*NT];

// Extract vorticity here
const Real Omx = Om[bid     ];
const Real Omy = Om[bid+1*NT];
const Real Omz = Om[bid+2*NT];

barrier(CLK_LOCAL_MEM_FENCE);

// Fill Halo (with coalesced index read)
for (int i=0; i<NHIT; i++){
    const int hid = BT*i + lid;
    const int hsx = hs[hid];                           // global x-shift relative to position
    const int hsy = hs[hid+BT*NHIT];                 // global y-shift relative to position
    const int hsz = hs[hid+2*BT*NHIT];               // global z-shift relative to position
    if (hsx<NFDX){                                 // Catch: is id within padded indices?
        const int ghx = gx0-Halo+hsx;             // Global x-value of retrieved node
        const int ghy = gy0-Halo+hsy;             // Global y-value of retrieved node
        const int ghz = gz0-Halo+hsz;             // Global z-value of retrieved node
        const int lhid = gid(hsx,hsy,hsz,NFDX,NFDY,NFDZ);

        // if (Data==Monolith) Kernel.append(const int bhid = gid(ghx,ghy,ghz,NX,NY,NZ););
        // if (Data==Block)    Kernel.append(const int bhid = gidb(ghx,ghy,ghz););
        const int bhid = gidb(ghx,ghy,ghz);

        const bool exx = (ghx<0 || ghx>=NX);      // Is x coordinate outside of the domain?
        const bool exy = (ghy<0 || ghy>=NY);      // Is x coordinate outside of the domain?
        const bool exz = (ghz<0 || ghz>=NZ);      // Is x coordinate outside of the domain?
        if (exx || exy || exz){                    // Catch: is id within domain?
            ox[lhid] = (Real)0.0;
            oy[lhid] = (Real)0.0;
            oz[lhid] = (Real)0.0;
        }
        else {
            ox[lhid] = O[bhid     ];
            oy[lhid] = O[bhid+1*NT];
            oz[lhid] = O[bhid+2*NT];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

// Calculate discrete filter

Real fx = (Real)0.0, fy = (Real)0.0, fz = (Real)0.0;                                                                                                         // Laplacian of Omega
const bool mxx = (gx>(Halo-1) && gx<(NX-Halo));
const bool mxy = (gy>(Halo-1) && gy<(NY-Halo));
const bool mxz = (gz>(Halo-1) && gz<(NZ-Halo));

if (mxx && mxy && mxz){

    #if (Map==0)            // Carry out discrete filtering operation in x direction
        fx = (Real)0.25*ox[gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + (Real)0.5*ox[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + (Real)0.25*ox[gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)];
        fy = (Real)0.25*oy[gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + (Real)0.5*oy[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + (Real)0.25*oy[gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)];
        fz = (Real)0.25*oz[gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + (Real)0.5*oz[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + (Real)0.25*oz[gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)];
    #elif (Map==1)          // Carry out discrete filtering operation in y direction
        fx = (Real)0.25*ox[gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ)] + (Real)0.5*ox[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + (Real)0.25*ox[gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ)];
        fy = (Real)0.25*oy[gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ)] + (Real)0.5*oy[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + (Real)0.25*oy[gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ)];
        fz = (Real)0.25*oz[gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ)] + (Real)0.5*oz[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + (Real)0.25*oz[gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ)];
    #elif (Map==2)      // Carry out discrete filtering operation in z direction
        fx = (Real)0.25*ox[gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ)] + (Real)0.5*ox[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + (Real)0.25*ox[gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ)];
        fy = (Real)0.25*oy[gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ)] + (Real)0.5*oy[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + (Real)0.25*oy[gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ)];
        fz = (Real)0.25*oz[gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ)] + (Real)0.5*oz[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + (Real)0.25*oz[gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ)];
    #elif (Map==3)      // Subtract filtered scale
        fx = Omx - ox[pid];
        fy = Omy - oy[pid];
        fz = Omz - oz[pid];
    #endif

}

barrier(CLK_LOCAL_MEM_FENCE);

// Assign output vars
filtO[bid]      = fx;
filtO[bid+NT]   = fy;
filtO[bid+2*NT] = fz;

}

)CLC";

const std::string VPM3D_ocl_kernels_RVM = R"CLC(

// Central FD Constants
__constant Real D1C2[2] =  {-0.5, 0.5};
__constant Real D1C4[4] =  {1.0/12.0,-2.0/3.0, 2.0/3.0, -1.0/12.0};
__constant Real D1C6[6] =  {-1.0/60.0, 3.0/20.0, -3.0/4.0, 3.0/4.0, -3.0/20.0, 1.0/60.0};
__constant Real D1C8[8] =  {1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0, 4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0 };

__constant Real D2C2[3] = {1.0, -2.0, 1.0};
__constant Real D2C4[5] = {-1.0/12.0, 4.0/3.0, -5.0/2.0, 4.0/3.0, -1.0/12.0};
__constant Real D2C6[7] = {1.0/90.0, -3.0/20.0, 3.0/2.0, -49.0/18.0, 3.0/2.0, -3.0/20.0, 1.0/90.0};
__constant Real D2C8[9] = {-1.0/560.0, 8.0/315.0, -1.0/5.0, 8.0/5.0, -205.0/72.0, 8.0/5.0, -1.0/5.0, 8.0/315.0, -1.0/560.0};

// Isotropic Laplacians
// __constant Real L1 = 2.0/30.0, L2 = 1.0/30.0, L3 = 18.0/30.0, L4 = -136.0/30.0;    // Cocle 2008
// __constant Real L1 =0.0, L2 = 1.0/6.0, L3 = 1.0/3.0, L4 = -4.0;                   // Patra 2006 - variant 2 (compact)
// __constant Real L1 = 1.0/30.0, L2 = 3.0/30.0, L3 = 14.0/30.0, L4 = -128.0/30.0;    // Patra 2006 - variant 5
// __constant Real LAP_ISO_3D[27] = {   L1,L2,L1,L2,L3,L2,L1,L2,L1,
//                                      L2,L3,L2,L3,L4,L3,L2,L3,L2,
//                                      L1,L2,L1,L2,L3,L2,L1,L2,L1};
__constant Real LAP_ISO_3D[27] = {   1.0/30.0,3.0/30.0,1.0/30.0,3.0/30.0,14.0/30.0,3.0/30.0,1.0/30.0,3.0/30.0,1.0/30.0,
                                     3.0/30.0,14.0/30.0,3.0/30.0,14.0/30.0,-128.0/30.0,14.0/30.0,3.0/30.0,14.0/30.0,3.0/30.0,
                                     1.0/30.0,3.0/30.0,1.0/30.0,3.0/30.0,14.0/30.0,3.0/30.0,1.0/30.0,3.0/30.0,1.0/30.0};

__kernel void RVM_turbulentstress(  __global const Real* fO,
                                    __global const Real* sgs,
                                    __global const int* hs,
                                    __global Real *dOmdt,
                                    const Real smag) {

// Declare shared memory arrays
__local Real gsx[NHT];
__local Real gsy[NHT];
__local Real gsz[NHT];
__local Real sg[NHT];

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

const int lid = gid(tx,ty,tz,BX,BY,BZ);             // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);    // Local id within padded block

// Fill centre volume
gsx[pid] = fO[bid       ];
gsy[pid] = fO[bid+NT    ];
gsz[pid] = fO[bid+2*NT  ];
sg[pid] = smag*sgs[bid];

barrier(CLK_LOCAL_MEM_FENCE);

// Fill Halo (with coalesced index read)
for (int i=0; i<NHIT; i++){
   const int hid = BT*i + lid;
   const int hsx = hs[hid];                           // global x-shift relative to position
   const int hsy = hs[hid+BT*NHIT];                 // global y-shift relative to position
   const int hsz = hs[hid+2*BT*NHIT];               // global z-shift relative to position
   if (hsx<NFDX){                                 // Catch: is id within padded indices?
        const int ghx = gx0-Halo+hsx;             // Global x-value of retrieved node
        const int ghy = gy0-Halo+hsy;             // Global y-value of retrieved node
        const int ghz = gz0-Halo+hsz;             // Global z-value of retrieved node
        const int lhid = gid(hsx,hsy,hsz,NFDX,NFDY,NFDZ);

        // if (Data==Monolith) Kernel.append(const int bhid = gid(ghx,ghy,ghz,NX,NY,NZ););
        // if (Data==Block)    Kernel.append(const int bhid = gidb(ghx,ghy,ghz););
        const int bhid = gidb(ghx,ghy,ghz);

        const bool exx = (ghx<0 || ghx>=NX);      // Is x coordinate outside of the domain?
        const bool exy = (ghy<0 || ghy>=NY);      // Is x coordinate outside of the domain?
        const bool exz = (ghz<0 || ghz>=NZ);      // Is x coordinate outside of the domain?
        if (exx || exy || exz){                    // Catch: is id within domain?
            gsx[lhid] = (Real)0.0;
            gsy[lhid] = (Real)0.0;
            gsz[lhid] = (Real)0.0;
            sg[lhid] = (Real)0.0;
        }
        else {
            gsx[lhid] = fO[bhid     ];
            gsy[lhid] = fO[bhid+1*NT];
            gsz[lhid] = fO[bhid+2*NT];
            sg[lhid] = smag*sgs[bhid];
        }
   }
   barrier(CLK_LOCAL_MEM_FENCE);
}

// Calculate turbulent shear stress

Real lx = (Real)0.0, ly = (Real)0.0, lz = (Real)0.0;           	// Laplacian of Omega
const bool mxx = (gx>(Halo-1) && gx<(NX-Halo));
const bool mxy = (gy>(Halo-1) && gy<(NY-Halo));
const bool mxz = (gz>(Halo-1) && gz<(NZ-Halo));

if (mxx && mxy && mxz){

    int ids;  // Dummy value

    // Specify centre values
    Real gxm = gsx[pid];
    Real gym = gsy[pid];
    Real gzm = gsz[pid];
    Real sgm = sg[pid];
    Real lfm;

    const Real Hlf = (Real)0.5;

    #if (Map==2)
        ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[0]*(gsx[ids]-gxm)*lfm;  ly += D2C2[0]*(gsy[ids]-gym)*lfm;  lz += D2C2[0]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[1]*(gsx[ids]-gxm)*lfm;  ly += D2C2[1]*(gsy[ids]-gym)*lfm;  lz += D2C2[1]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[2]*(gsx[ids]-gxm)*lfm;  ly += D2C2[2]*(gsy[ids]-gym)*lfm;  lz += D2C2[2]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[0]*(gsx[ids]-gxm)*lfm;  ly += D2C2[0]*(gsy[ids]-gym)*lfm;  lz += D2C2[0]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[1]*(gsx[ids]-gxm)*lfm;  ly += D2C2[1]*(gsy[ids]-gym)*lfm;  lz += D2C2[1]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[2]*(gsx[ids]-gxm)*lfm;  ly += D2C2[2]*(gsy[ids]-gym)*lfm;  lz += D2C2[2]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[0]*(gsx[ids]-gxm)*lfm;  ly += D2C2[0]*(gsy[ids]-gym)*lfm;  lz += D2C2[0]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[1]*(gsx[ids]-gxm)*lfm;  ly += D2C2[1]*(gsy[ids]-gym)*lfm;  lz += D2C2[1]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[2]*(gsx[ids]-gxm)*lfm;  ly += D2C2[2]*(gsy[ids]-gym)*lfm;  lz += D2C2[2]*(gsz[ids]-gzm)*lfm;
   #elif (Map==4)
        ids = gid(txh-2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C4[0]*(gsx[ids]-gxm)*lfm;  ly += D2C4[0]*(gsy[ids]-gym)*lfm;  lz += D2C4[0]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C4[1]*(gsx[ids]-gxm)*lfm;  ly += D2C4[1]*(gsy[ids]-gym)*lfm;  lz += D2C4[1]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C4[2]*(gsx[ids]-gxm)*lfm;  ly += D2C4[2]*(gsy[ids]-gym)*lfm;  lz += D2C4[2]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C4[3]*(gsx[ids]-gxm)*lfm;  ly += D2C4[3]*(gsy[ids]-gym)*lfm;  lz += D2C4[3]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh+2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C4[4]*(gsx[ids]-gxm)*lfm;  ly += D2C4[4]*(gsy[ids]-gym)*lfm;  lz += D2C4[4]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh-2,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C4[0]*(gsx[ids]-gxm)*lfm;  ly += D2C4[0]*(gsy[ids]-gym)*lfm;  lz += D2C4[0]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C4[1]*(gsx[ids]-gxm)*lfm;  ly += D2C4[1]*(gsy[ids]-gym)*lfm;  lz += D2C4[1]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C4[2]*(gsx[ids]-gxm)*lfm;  ly += D2C4[2]*(gsy[ids]-gym)*lfm;  lz += D2C4[2]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C4[3]*(gsx[ids]-gxm)*lfm;  ly += D2C4[3]*(gsy[ids]-gym)*lfm;  lz += D2C4[3]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh+2,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C4[4]*(gsx[ids]-gxm)*lfm;  ly += D2C4[4]*(gsy[ids]-gym)*lfm;  lz += D2C4[4]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh-2,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C4[0]*(gsx[ids]-gxm)*lfm;  ly += D2C4[0]*(gsy[ids]-gym)*lfm;  lz += D2C4[0]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C4[1]*(gsx[ids]-gxm)*lfm;  ly += D2C4[1]*(gsy[ids]-gym)*lfm;  lz += D2C4[1]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C4[2]*(gsx[ids]-gxm)*lfm;  ly += D2C4[2]*(gsy[ids]-gym)*lfm;  lz += D2C4[2]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C4[3]*(gsx[ids]-gxm)*lfm;  ly += D2C4[3]*(gsy[ids]-gym)*lfm;  lz += D2C4[3]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh+2,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C4[4]*(gsx[ids]-gxm)*lfm;  ly += D2C4[4]*(gsy[ids]-gym)*lfm;  lz += D2C4[4]*(gsz[ids]-gzm)*lfm;
   #elif (Map==6)
        ids = gid(txh-3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[0]*(gsx[ids]-gxm)*lfm;  ly += D2C6[0]*(gsy[ids]-gym)*lfm;  lz += D2C6[0]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh-2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[1]*(gsx[ids]-gxm)*lfm;  ly += D2C6[1]*(gsy[ids]-gym)*lfm;  lz += D2C6[1]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[2]*(gsx[ids]-gxm)*lfm;  ly += D2C6[2]*(gsy[ids]-gym)*lfm;  lz += D2C6[2]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[3]*(gsx[ids]-gxm)*lfm;  ly += D2C6[3]*(gsy[ids]-gym)*lfm;  lz += D2C6[3]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[4]*(gsx[ids]-gxm)*lfm;  ly += D2C6[4]*(gsy[ids]-gym)*lfm;  lz += D2C6[4]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh+2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[5]*(gsx[ids]-gxm)*lfm;  ly += D2C6[5]*(gsy[ids]-gym)*lfm;  lz += D2C6[5]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh+3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[6]*(gsx[ids]-gxm)*lfm;  ly += D2C6[6]*(gsy[ids]-gym)*lfm;  lz += D2C6[6]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh-3,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[0]*(gsx[ids]-gxm)*lfm;  ly += D2C6[0]*(gsy[ids]-gym)*lfm;  lz += D2C6[0]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh-2,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[1]*(gsx[ids]-gxm)*lfm;  ly += D2C6[1]*(gsy[ids]-gym)*lfm;  lz += D2C6[1]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[2]*(gsx[ids]-gxm)*lfm;  ly += D2C6[2]*(gsy[ids]-gym)*lfm;  lz += D2C6[2]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[3]*(gsx[ids]-gxm)*lfm;  ly += D2C6[3]*(gsy[ids]-gym)*lfm;  lz += D2C6[3]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[4]*(gsx[ids]-gxm)*lfm;  ly += D2C6[4]*(gsy[ids]-gym)*lfm;  lz += D2C6[4]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh+2,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[5]*(gsx[ids]-gxm)*lfm;  ly += D2C6[5]*(gsy[ids]-gym)*lfm;  lz += D2C6[5]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh+3,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[6]*(gsx[ids]-gxm)*lfm;  ly += D2C6[6]*(gsy[ids]-gym)*lfm;  lz += D2C6[6]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh-3,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[0]*(gsx[ids]-gxm)*lfm;  ly += D2C6[0]*(gsy[ids]-gym)*lfm;  lz += D2C6[0]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh-2,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[1]*(gsx[ids]-gxm)*lfm;  ly += D2C6[1]*(gsy[ids]-gym)*lfm;  lz += D2C6[1]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[2]*(gsx[ids]-gxm)*lfm;  ly += D2C6[2]*(gsy[ids]-gym)*lfm;  lz += D2C6[2]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[3]*(gsx[ids]-gxm)*lfm;  ly += D2C6[3]*(gsy[ids]-gym)*lfm;  lz += D2C6[3]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[4]*(gsx[ids]-gxm)*lfm;  ly += D2C6[4]*(gsy[ids]-gym)*lfm;  lz += D2C6[4]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh+2,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[5]*(gsx[ids]-gxm)*lfm;  ly += D2C6[5]*(gsy[ids]-gym)*lfm;  lz += D2C6[5]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh+3,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C6[6]*(gsx[ids]-gxm)*lfm;  ly += D2C6[6]*(gsy[ids]-gym)*lfm;  lz += D2C6[6]*(gsz[ids]-gzm)*lfm;
    #elif (Map==8)
        ids = gid(txh-4,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[0]*(gsx[ids]-gxm)*lfm;  ly += D2C8[0]*(gsy[ids]-gym)*lfm;  lz += D2C8[0]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh-3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[1]*(gsx[ids]-gxm)*lfm;  ly += D2C8[1]*(gsy[ids]-gym)*lfm;  lz += D2C8[1]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh-2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[2]*(gsx[ids]-gxm)*lfm;  ly += D2C8[2]*(gsy[ids]-gym)*lfm;  lz += D2C8[2]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[3]*(gsx[ids]-gxm)*lfm;  ly += D2C8[3]*(gsy[ids]-gym)*lfm;  lz += D2C8[3]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[4]*(gsx[ids]-gxm)*lfm;  ly += D2C8[4]*(gsy[ids]-gym)*lfm;  lz += D2C8[4]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[5]*(gsx[ids]-gxm)*lfm;  ly += D2C8[5]*(gsy[ids]-gym)*lfm;  lz += D2C8[5]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh+2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[6]*(gsx[ids]-gxm)*lfm;  ly += D2C8[6]*(gsy[ids]-gym)*lfm;  lz += D2C8[6]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh+3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[7]*(gsx[ids]-gxm)*lfm;  ly += D2C8[7]*(gsy[ids]-gym)*lfm;  lz += D2C8[7]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh+4,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[8]*(gsx[ids]-gxm)*lfm;  ly += D2C8[8]*(gsy[ids]-gym)*lfm;  lz += D2C8[8]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh-4,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[0]*(gsx[ids]-gxm)*lfm;  ly += D2C8[0]*(gsy[ids]-gym)*lfm;  lz += D2C8[0]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh-3,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[1]*(gsx[ids]-gxm)*lfm;  ly += D2C8[1]*(gsy[ids]-gym)*lfm;  lz += D2C8[1]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh-2,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[2]*(gsx[ids]-gxm)*lfm;  ly += D2C8[2]*(gsy[ids]-gym)*lfm;  lz += D2C8[2]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[3]*(gsx[ids]-gxm)*lfm;  ly += D2C8[3]*(gsy[ids]-gym)*lfm;  lz += D2C8[3]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[4]*(gsx[ids]-gxm)*lfm;  ly += D2C8[4]*(gsy[ids]-gym)*lfm;  lz += D2C8[4]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[5]*(gsx[ids]-gxm)*lfm;  ly += D2C8[5]*(gsy[ids]-gym)*lfm;  lz += D2C8[5]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh+2,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[6]*(gsx[ids]-gxm)*lfm;  ly += D2C8[6]*(gsy[ids]-gym)*lfm;  lz += D2C8[6]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh+3,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[7]*(gsx[ids]-gxm)*lfm;  ly += D2C8[7]*(gsy[ids]-gym)*lfm;  lz += D2C8[7]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh+4,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[8]*(gsx[ids]-gxm)*lfm;  ly += D2C8[8]*(gsy[ids]-gym)*lfm;  lz += D2C8[8]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh-4,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[0]*(gsx[ids]-gxm)*lfm;  ly += D2C8[0]*(gsy[ids]-gym)*lfm;  lz += D2C8[0]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh-3,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[1]*(gsx[ids]-gxm)*lfm;  ly += D2C8[1]*(gsy[ids]-gym)*lfm;  lz += D2C8[1]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh-2,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[2]*(gsx[ids]-gxm)*lfm;  ly += D2C8[2]*(gsy[ids]-gym)*lfm;  lz += D2C8[2]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[2]*(gsx[ids]-gxm)*lfm;  ly += D2C8[3]*(gsy[ids]-gym)*lfm;  lz += D2C8[3]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[4]*(gsx[ids]-gxm)*lfm;  ly += D2C8[4]*(gsy[ids]-gym)*lfm;  lz += D2C8[4]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[5]*(gsx[ids]-gxm)*lfm;  ly += D2C8[5]*(gsy[ids]-gym)*lfm;  lz += D2C8[5]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh+2,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[6]*(gsx[ids]-gxm)*lfm;  ly += D2C8[6]*(gsy[ids]-gym)*lfm;  lz += D2C8[6]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh+3,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[7]*(gsx[ids]-gxm)*lfm;  ly += D2C8[7]*(gsy[ids]-gym)*lfm;  lz += D2C8[7]*(gsz[ids]-gzm)*lfm;
        ids = gid(txh  ,tyh  ,tzh+4,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C8[8]*(gsx[ids]-gxm)*lfm;  ly += D2C8[8]*(gsy[ids]-gym)*lfm;  lz += D2C8[8]*(gsz[ids]-gzm)*lfm;
    #endif

    // Scale for grid size
    lx *= (Real)1.0/hx/hx;
    ly *= (Real)1.0/hy/hy;
    lz *= (Real)1.0/hz/hz;
}

barrier(CLK_LOCAL_MEM_FENCE);

// Assign output vars
dOmdt[bid]      += lx;
dOmdt[bid+NT]   += ly;
dOmdt[bid+2*NT] += lz;

}


)CLC";

//--- Diagnostics Kernel

const std::string VPM3D_ocl_kernels_Diagnostics = R"CLC(

// Unroll telescoping sum
inline void warpReduceVector(   __local volatile Real* sx,
                                __local volatile Real* sy,
                                __local volatile Real* sz,
                                uint tid) {
    if (BT >= 64)   {sx[tid] += sx[tid + 32];     sy[tid] += sy[tid + 32];  sz[tid] += sz[tid + 32]; }
    if (BT >= 32)   {sx[tid] += sx[tid + 16];     sy[tid] += sy[tid + 16];  sz[tid] += sz[tid + 16]; }
    if (BT >= 16)   {sx[tid] += sx[tid +  8];     sy[tid] += sy[tid +  8];  sz[tid] += sz[tid +  8]; }
    if (BT >= 8)    {sx[tid] += sx[tid +  4];     sy[tid] += sy[tid +  4];  sz[tid] += sz[tid +  4]; }
    if (BT >= 4)    {sx[tid] += sx[tid +  2];     sy[tid] += sy[tid +  2];  sz[tid] += sz[tid +  2]; }
    if (BT >= 2)    {sx[tid] += sx[tid +  1];     sy[tid] += sy[tid +  1];  sz[tid] += sz[tid +  1]; }
}

inline void warpReduceScalar(__local volatile Real* src, uint tid) {
    if (BT >= 64)   src[tid] += src[tid + 32];
    if (BT >= 32)   src[tid] += src[tid + 16];
    if (BT >= 16)   src[tid] += src[tid + 8];
    if (BT >= 8)    src[tid] += src[tid + 4];
    if (BT >= 4)    src[tid] += src[tid + 2];
    if (BT >= 2)    src[tid] += src[tid + 1];
}

inline void warpReduceMag(__local volatile Real* src, uint tid) {
    if (BT >= 64)   {src[tid] = fmax(src[tid],src[tid+32]);  }
    if (BT >= 32)   {src[tid] = fmax(src[tid],src[tid+16]);  }
    if (BT >= 16)   {src[tid] = fmax(src[tid],src[tid+ 8]);  }
    if (BT >= 8)    {src[tid] = fmax(src[tid],src[tid+ 4]);  }
    if (BT >= 4)    {src[tid] = fmax(src[tid],src[tid+ 2]);  }
    if (BT >= 2)    {src[tid] = fmax(src[tid],src[tid+ 1]);  }
}

// __constant__ int NDiags = 15;          // Number of diagnostic outputs

__kernel void DiagnosticsKernel(__global const Real *omega,
                                __global const Real *vel,
                                __global Real *d) {

    // Declare shared memory arrays for diagnostic vales
    __local Real C[3][BT];      	// Circulation
    __local Real L[3][BT];      	// Linear Impulse
    __local Real A[3][BT];      	// Angular Impulse
    __local Real K1[BT];        	// Kinetic energy 1
    __local Real K2[BT];        	// Kinetic energy 2
    __local Real E[BT];         	// Enstropy
    __local Real H[BT];         	// Helicity
    __local Real normO[BT];      // Magnitude of vorticity
    __local Real normU[BT];      // Magnitude of velocity

    // Specify grid ids
    unsigned int tid = get_local_id(0);
    unsigned int i = get_group_id(0)*get_local_size(0) + get_local_id(0);

    // Calculate global indices based on index
    const int K = (int)i/(BX*BY*BZ);
    const int KX = (int)K/(NBY*NBZ);
    const int KY = (int)(K-KX*NBY*NBZ)/NBZ;
    const int KZ = K-KX*NBY*NBZ-KY*NBZ;
    const int ib = i-K*BX*BY*BZ;
    const int bx = (int)ib/(BY*BZ);
    const int by = (int)(ib-bx*BY*BZ)/BZ;
    const int bz = ib-bx*BY*BZ-by*BZ;
    const Real Px = XN1 + (KX*BX+bx)*hx;
    const Real Py = YN1 + (KY*BY+by)*hy;
    const Real Pz = ZN1 + (KZ*BZ+bz)*hz;

    const Real Ox = omega[i     ];
    const Real Oy = omega[i+1*NT];
    const Real Oz = omega[i+2*NT];
    const Real Ux = vel[i     ];
    const Real Uy = vel[i+1*NT];
    const Real Uz = vel[i+2*NT];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Linear diagnostics
    C[0][tid] = Ox;
    C[1][tid] = Oy;
    C[2][tid] = Oz;
    const Real cx = Py*Oz - Pz*Oy;
    const Real cy = Pz*Ox - Px*Oz;
    const Real cz = Px*Oy - Py*Ox;
    L[0][tid] = cx*(Real)0.5;
    L[1][tid] = cy*(Real)0.5;
    L[2][tid] = cz*(Real)0.5;
    A[0][tid] = (Py*cz - Pz*cy)/(Real)3.0;
    A[1][tid] = (Pz*cx - Px*cz)/(Real)3.0;
    A[2][tid] = (Px*cy - Py*cx)/(Real)3.0;

    // Quadratic diagnostics
    normO[tid] = sqrt(Ox*Ox + Oy*Oy + Oz*Oz);
    normU[tid] = sqrt(Ux*Ux + Uy*Uy + Uz*Uz);
    K1[tid] = (Real)0.5*normU[tid]*normU[tid];      // Approach 1: Winckelmans
    K2[tid] =  (Ux*cx + Uy*cy + Uz*cz);             // Approach 2: Liska
    E[tid] = normO[tid]*normO[tid];
    H[tid] = Ox*Ux + Oy*Uy + Oz*Uz;

    barrier(CLK_LOCAL_MEM_FENCE);

    // // // Shear components: Private Correspondence with Gregoire Winckelmans
    // // // Compute the 1-norm of the deformation tensor, It bounds the 2-norm...
    // // // Nb: NablaU = [dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz];
    // // Real dsu = fabs(NablaU[0][id_src]) + 0.5*fabs(NablaU[1][id_src]+NablaU[3][id_src]) + 0.5*fabs(NablaU[2][id_src]+NablaU[6][id_src]);
    // // Real dsv = 0.5*fabs(NablaU[3][id_src]+NablaU[1][id_src]) + fabs(NablaU[4][id_src]) + 0.5*fabs(NablaU[5][id_src]+NablaU[7][id_src]);
    // // Real dsw = 0.5*fabs(NablaU[6][id_src]+NablaU[2][id_src]) + 0.5*fabs(NablaU[7][id_src]+NablaU[5][id_src]) + fabs(NablaU[8][id_src]);
    // // Real dsm = std::max(dsu,std::max(dsv,dsw));
    // // if (SMax[id_dest] < dsm)  SMax[id_dest] = dsm;          // Stretching

    // Initial step to accumulate the values  above  the lowest power of two... and then after this, continue with NVidia approach
    unsigned int p = 2;
    while (p<BT)  p*=2;
    if (p-BT!=0){                               // If the block has size power of 2, go straight to reduction
        p /= 2;                                 // Otherwise, increment terms
        if (tid>=p){
            C[0][tid-p] += C[0][tid] ;
            C[1][tid-p] += C[1][tid] ;
            C[2][tid-p] += C[2][tid] ;
            L[0][tid-p] += L[0][tid] ;
            L[1][tid-p] += L[1][tid] ;
            L[2][tid-p] += L[2][tid] ;
            A[0][tid-p] += A[0][tid] ;
            A[1][tid-p] += A[1][tid] ;
            A[2][tid-p] += A[2][tid] ;
            K1  [tid-p] += K1  [tid] ;
            K2  [tid-p] += K2  [tid] ;
            E   [tid-p] += E   [tid] ;
            H   [tid-p] += H   [tid] ;
            normO[tid-p] = fmax(normO[tid-p],normO[tid]);
            normU[tid-p] = fmax(normU[tid-p],normU[tid]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Carry out first reduction stage. I have written this as a for loop simply to improve readability
    for (int k=512; k>=64; k/=2){
        if (BT >= 2*k){
            if (tid<k){
                C[0][tid] += C[0][tid+k] ;
                C[1][tid] += C[1][tid+k] ;
                C[2][tid] += C[2][tid+k] ;
                L[0][tid] += L[0][tid+k] ;
                L[1][tid] += L[1][tid+k] ;
                L[2][tid] += L[2][tid+k] ;
                A[0][tid] += A[0][tid+k] ;
                A[1][tid] += A[1][tid+k] ;
                A[2][tid] += A[2][tid+k] ;
                K1  [tid] += K1  [tid+k] ;
                K2  [tid] += K2  [tid+k] ;
                E   [tid] += E   [tid+k] ;
                H   [tid] += H   [tid+k] ;
                normO[tid] = fmax(normO[tid+k],normO[tid]);
                normU[tid] = fmax(normU[tid+k],normU[tid]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Optimised unrolled final warp
    if (tid < 32){
        warpReduceVector(C[0], C[1], C[2], tid);
        warpReduceVector(L[0], L[1], L[2], tid);
        warpReduceVector(A[0], A[1], A[2], tid);
        warpReduceScalar(K1, tid);
        warpReduceScalar(K2, tid);
        warpReduceScalar(E, tid);
        warpReduceScalar(H, tid);
        warpReduceMag(normO, tid);
        warpReduceMag(normU, tid);
    }

    // Write result for this block to global mem
    // Note: This is output in such a way that the values can be transferred directly to the next reduction pass

    //  if (i==0) printf(\ gridDim x %i circulation %f %f %f \\n\ , get_num_groups(0), C[0][0], C[0][1], C[0][2]);\n

    if (tid == 0){
        const int ngroups = get_num_groups(0);
        d[ 0*ngroups + get_group_id(0)] = C[0][0];
        d[ 1*ngroups + get_group_id(0)] = C[1][0];
        d[ 2*ngroups + get_group_id(0)] = C[2][0];
        d[ 3*ngroups + get_group_id(0)] = L[0][0];
        d[ 4*ngroups + get_group_id(0)] = L[1][0];
        d[ 5*ngroups + get_group_id(0)] = L[2][0];
        d[ 6*ngroups + get_group_id(0)] = A[0][0];
        d[ 7*ngroups + get_group_id(0)] = A[1][0];
        d[ 8*ngroups + get_group_id(0)] = A[2][0];
        d[ 9*ngroups + get_group_id(0)] = K1[0];
        d[10*ngroups + get_group_id(0)] = K2[0];
        d[11*ngroups + get_group_id(0)] = E[0];
        d[12*ngroups + get_group_id(0)] = H[0];
        d[13*ngroups + get_group_id(0)] = normO[0];
        d[14*ngroups + get_group_id(0)] = normU[0];
    }
}
)CLC";

//--- Finite difference kernels

// Note: In the kernel below the variable "Map" is highjacked as this variable is used in other kernels
// It represents the order of the finite differences

const std::string VPM3D_ocl_kernels_ShearStress = R"CLC(

// Central FD Constants
__constant Real D1C2[2] =  {-0.5, 0.5};
__constant Real D1C4[4] =  {1.0/12.0,-2.0/3.0, 2.0/3.0, -1.0/12.0};
__constant Real D1C6[6] =  {-1.0/60.0, 3.0/20.0, -3.0/4.0, 3.0/4.0, -3.0/20.0, 1.0/60.0};
__constant Real D1C8[8] =  {1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0, 4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0 };

__constant Real D2C2[3] = {1.0, -2.0, 1.0};
__constant Real D2C4[5] = {-1.0/12.0, 4.0/3.0, -5.0/2.0, 4.0/3.0, -1.0/12.0};
__constant Real D2C6[7] = {1.0/90.0, -3.0/20.0, 3.0/2.0, -49.0/18.0, 3.0/2.0, -3.0/20.0, 1.0/90.0};
__constant Real D2C8[9] = {-1.0/560.0, 8.0/315.0, -1.0/5.0, 8.0/5.0, -205.0/72.0, 8.0/5.0, -1.0/5.0, 8.0/315.0, -1.0/560.0};

// Isotropic Laplacians
// __constant Real L1 = 2.0/30.0, L2 = 1.0/30.0, L3 = 18.0/30.0, L4 = -136.0/30.0;    // Cocle 2008
// __constant Real L1 =0.0, L2 = 1.0/6.0, L3 = 1.0/3.0, L4 = -4.0;                   // Patra 2006 - variant 2 (compact)
// __constant Real L1 = 1.0/30.0, L2 = 3.0/30.0, L3 = 14.0/30.0, L4 = -128.0/30.0;    // Patra 2006 - variant 5
// __constant Real LAP_ISO_3D[27] = {   L1,L2,L1,L2,L3,L2,L1,L2,L1,
//                                      L2,L3,L2,L3,L4,L3,L2,L3,L2,
//                                      L1,L2,L1,L2,L3,L2,L1,L2,L1};
__constant Real LAP_ISO_3D[27] = {   1.0/30.0,3.0/30.0,1.0/30.0,3.0/30.0,14.0/30.0,3.0/30.0,1.0/30.0,3.0/30.0,1.0/30.0,
                                     3.0/30.0,14.0/30.0,3.0/30.0,14.0/30.0,-128.0/30.0,14.0/30.0,3.0/30.0,14.0/30.0,3.0/30.0,
                                     1.0/30.0,3.0/30.0,1.0/30.0,3.0/30.0,14.0/30.0,3.0/30.0,1.0/30.0,3.0/30.0,1.0/30.0};

__kernel void Shear_Stress( __global const Real* w,
                            __global const Real* u,
                            __global const int* hs,
                            __global Real* grad,
                            __global Real* smag,
                            __global Real* qcrit) {

// Declare shared memory arrays
__local Real wx[NHT];
__local Real wy[NHT];
__local Real wz[NHT];
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
const int txh = tx + Halo;                                          // Local x id within padded grid
const int tyh = ty + Halo;                                          // Local y id within padded grid
const int tzh = tz + Halo;                                          // Local z id within padded grid

// Monolith structure
// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);

const int lid = gid(tx,ty,tz,BX,BY,BZ);           // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);                             // Local id within padded block

// Fill centre volume
wx[pid] = w[bid     ];
wy[pid] = w[bid+NT  ];
wz[pid] = w[bid+2*NT];
ux[pid] = u[bid     ];
uy[pid] = u[bid+NT  ];
uz[pid] = u[bid+2*NT];

barrier(CLK_LOCAL_MEM_FENCE);

// Fill Halo (with coalesced index read)
for (int i=0; i<NHIT; i++){
   const int hid = BT*i + lid;
   const int hsx = hs[hid];                           // global x-shift relative to position
   const int hsy = hs[hid+BT*NHIT];                 // global y-shift relative to position
   const int hsz = hs[hid+2*BT*NHIT];               // global z-shift relative to position
   if (hsx<NFDX){                                 // Catch: is id within padded indices?
        const int ghx = gx0-Halo+hsx;             // Global x-value of retrieved node
        const int ghy = gy0-Halo+hsy;             // Global y-value of retrieved node
        const int ghz = gz0-Halo+hsz;             // Global z-value of retrieved node
        const int lhid = gid(hsx,hsy,hsz,NFDX,NFDY,NFDZ);

        // if (Data==Monolith) Kernel.append(const int bhid = gid(ghx,ghy,ghz,NX,NY,NZ););
        // if (Data==Block)    Kernel.append(const int bhid = gidb(ghx,ghy,ghz););
        const int bhid = gidb(ghx,ghy,ghz);

        const bool exx = (ghx<0 || ghx>=NX);      // Is x coordinate outside of the domain?
        const bool exy = (ghy<0 || ghy>=NY);      // Is x coordinate outside of the domain?
        const bool exz = (ghz<0 || ghz>=NZ);      // Is x coordinate outside of the domain?
        if (exx || exy || exz){                    // Catch: is id within domain?
            wx[lhid] = (Real)0.0;
            wy[lhid] = (Real)0.0;
            wz[lhid] = (Real)0.0;
            ux[lhid] = (Real)0.0;
            uy[lhid] = (Real)0.0;
            uz[lhid] = (Real)0.0;
        }
        else {
            wx[lhid] = w[bhid     ];
            wy[lhid] = w[bhid+1*NT];
            wz[lhid] = w[bhid+2*NT];
            ux[lhid] = u[bhid     ];
            uy[lhid] = u[bhid+1*NT];
            uz[lhid] = u[bhid+2*NT];
        }
   }
   barrier(CLK_LOCAL_MEM_FENCE);
}

// Calculate finite differences

Real duxdx = (Real)0.0;
Real duxdy = (Real)0.0;
Real duxdz = (Real)0.0;
Real duydx = (Real)0.0;
Real duydy = (Real)0.0;
Real duydz = (Real)0.0;
Real duzdx = (Real)0.0;
Real duzdy = (Real)0.0;
Real duzdz = (Real)0.0;    // Nabla U
Real sgs = (Real)0.0;
Real qc = (Real)0.0;																													// Smagorinksi term
Real lx = (Real)0.0, ly = (Real)0.0, lz = (Real)0.0;                                                                                                     	// Laplacian of Omega
const bool mxx = (gx>(Halo-1) && gx<(NX-Halo));
const bool mxy = (gy>(Halo-1) && gy<(NY-Halo));
const bool mxz = (gz>(Halo-1) && gz<(NZ-Halo));

if (mxx && mxy && mxz){

    int ids;	// Dummy integer

    //-- Calculate velocity gradients
    #if (Map==2)
        ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C2[0]*ux[ids];   duydx += D1C2[0]*uy[ids];  duzdx += D1C2[0]*uz[ids];
        ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C2[1]*ux[ids];   duydx += D1C2[1]*uy[ids];  duzdx += D1C2[1]*uz[ids];
        ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C2[0]*ux[ids];   duydy += D1C2[0]*uy[ids];  duzdy += D1C2[0]*uz[ids];
        ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C2[1]*ux[ids];   duydy += D1C2[1]*uy[ids];  duzdy += D1C2[1]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    duxdz += D1C2[0]*ux[ids];   duydz += D1C2[0]*uy[ids];  duzdz += D1C2[0]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    duxdz += D1C2[1]*ux[ids];   duydz += D1C2[1]*uy[ids];  duzdz += D1C2[1]*uz[ids];
    #elif (Map==4)
        ids = gid(txh-2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C4[0]*ux[ids];   duydx += D1C4[0]*uy[ids];  duzdx += D1C4[0]*uz[ids];
        ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C4[1]*ux[ids];   duydx += D1C4[1]*uy[ids];  duzdx += D1C4[1]*uz[ids];
        ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C4[2]*ux[ids];   duydx += D1C4[2]*uy[ids];  duzdx += D1C4[2]*uz[ids];
        ids = gid(txh+2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C4[3]*ux[ids];   duydx += D1C4[3]*uy[ids];  duzdx += D1C4[3]*uz[ids];
        ids = gid(txh  ,tyh-2,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C4[0]*ux[ids];   duydy += D1C4[0]*uy[ids];  duzdy += D1C4[0]*uz[ids];
        ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C4[1]*ux[ids];   duydy += D1C4[1]*uy[ids];  duzdy += D1C4[1]*uz[ids];
        ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C4[2]*ux[ids];   duydy += D1C4[2]*uy[ids];  duzdy += D1C4[2]*uz[ids];
        ids = gid(txh  ,tyh+2,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C4[3]*ux[ids];   duydy += D1C4[3]*uy[ids];  duzdy += D1C4[3]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh-2,NFDX,NFDY,NFDZ);    duxdz += D1C4[0]*ux[ids];   duydz += D1C4[0]*uy[ids];  duzdz += D1C4[0]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    duxdz += D1C4[1]*ux[ids];   duydz += D1C4[1]*uy[ids];  duzdz += D1C4[1]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    duxdz += D1C4[2]*ux[ids];   duydz += D1C4[2]*uy[ids];  duzdz += D1C4[2]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh+2,NFDX,NFDY,NFDZ);    duxdz += D1C4[3]*ux[ids];   duydz += D1C4[3]*uy[ids];  duzdz += D1C4[3]*uz[ids];
    #elif (Map==6)
        ids = gid(txh-3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C6[0]*ux[ids];   duydx += D1C6[0]*uy[ids];  duzdx += D1C6[0]*uz[ids];
        ids = gid(txh-2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C6[1]*ux[ids];   duydx += D1C6[1]*uy[ids];  duzdx += D1C6[1]*uz[ids];
        ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C6[2]*ux[ids];   duydx += D1C6[2]*uy[ids];  duzdx += D1C6[2]*uz[ids];
        ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C6[3]*ux[ids];   duydx += D1C6[3]*uy[ids];  duzdx += D1C6[3]*uz[ids];
        ids = gid(txh+2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C6[4]*ux[ids];   duydx += D1C6[4]*uy[ids];  duzdx += D1C6[4]*uz[ids];
        ids = gid(txh+3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C6[5]*ux[ids];   duydx += D1C6[5]*uy[ids];  duzdx += D1C6[5]*uz[ids];
        ids = gid(txh  ,tyh-3,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C6[0]*ux[ids];   duydy += D1C6[0]*uy[ids];  duzdy += D1C6[0]*uz[ids];
        ids = gid(txh  ,tyh-2,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C6[1]*ux[ids];   duydy += D1C6[1]*uy[ids];  duzdy += D1C6[1]*uz[ids];
        ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C6[2]*ux[ids];   duydy += D1C6[2]*uy[ids];  duzdy += D1C6[2]*uz[ids];
        ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C6[3]*ux[ids];   duydy += D1C6[3]*uy[ids];  duzdy += D1C6[3]*uz[ids];
        ids = gid(txh  ,tyh+2,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C6[4]*ux[ids];   duydy += D1C6[4]*uy[ids];  duzdy += D1C6[4]*uz[ids];
        ids = gid(txh  ,tyh+3,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C6[5]*ux[ids];   duydy += D1C6[5]*uy[ids];  duzdy += D1C6[5]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh-3,NFDX,NFDY,NFDZ);    duxdz += D1C6[0]*ux[ids];   duydz += D1C6[0]*uy[ids];  duzdz += D1C6[0]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh-2,NFDX,NFDY,NFDZ);    duxdz += D1C6[1]*ux[ids];   duydz += D1C6[1]*uy[ids];  duzdz += D1C6[1]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    duxdz += D1C6[2]*ux[ids];   duydz += D1C6[2]*uy[ids];  duzdz += D1C6[2]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    duxdz += D1C6[3]*ux[ids];   duydz += D1C6[3]*uy[ids];  duzdz += D1C6[3]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh+2,NFDX,NFDY,NFDZ);    duxdz += D1C6[4]*ux[ids];   duydz += D1C6[4]*uy[ids];  duzdz += D1C6[4]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh+3,NFDX,NFDY,NFDZ);    duxdz += D1C6[5]*ux[ids];   duydz += D1C6[5]*uy[ids];  duzdz += D1C6[5]*uz[ids];
    #elif (Map==8)
        ids = gid(txh-4,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C8[0]*ux[ids];   duydx += D1C8[0]*uy[ids];  duzdx += D1C8[0]*uz[ids];
        ids = gid(txh-3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C8[1]*ux[ids];   duydx += D1C8[1]*uy[ids];  duzdx += D1C8[1]*uz[ids];
        ids = gid(txh-2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C8[2]*ux[ids];   duydx += D1C8[2]*uy[ids];  duzdx += D1C8[2]*uz[ids];
        ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C8[3]*ux[ids];   duydx += D1C8[3]*uy[ids];  duzdx += D1C8[3]*uz[ids];
        ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C8[4]*ux[ids];   duydx += D1C8[4]*uy[ids];  duzdx += D1C8[4]*uz[ids];
        ids = gid(txh+2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C8[5]*ux[ids];   duydx += D1C8[5]*uy[ids];  duzdx += D1C8[5]*uz[ids];
        ids = gid(txh+3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C8[6]*ux[ids];   duydx += D1C8[6]*uy[ids];  duzdx += D1C8[6]*uz[ids];
        ids = gid(txh+4,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C8[7]*ux[ids];   duydx += D1C8[7]*uy[ids];  duzdx += D1C8[7]*uz[ids];
        ids = gid(txh  ,tyh-4,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[0]*ux[ids];   duydy += D1C8[0]*uy[ids];  duzdy += D1C8[0]*uz[ids];
        ids = gid(txh  ,tyh-3,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[1]*ux[ids];   duydy += D1C8[1]*uy[ids];  duzdy += D1C8[1]*uz[ids];
        ids = gid(txh  ,tyh-2,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[2]*ux[ids];   duydy += D1C8[2]*uy[ids];  duzdy += D1C8[2]*uz[ids];
        ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[3]*ux[ids];   duydy += D1C8[3]*uy[ids];  duzdy += D1C8[3]*uz[ids];
        ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[4]*ux[ids];   duydy += D1C8[4]*uy[ids];  duzdy += D1C8[4]*uz[ids];
        ids = gid(txh  ,tyh+2,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[5]*ux[ids];   duydy += D1C8[5]*uy[ids];  duzdy += D1C8[5]*uz[ids];
        ids = gid(txh  ,tyh+3,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[6]*ux[ids];   duydy += D1C8[6]*uy[ids];  duzdy += D1C8[6]*uz[ids];
        ids = gid(txh  ,tyh+4,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[7]*ux[ids];   duydy += D1C8[7]*uy[ids];  duzdy += D1C8[7]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh-4,NFDX,NFDY,NFDZ);    duxdz += D1C8[0]*ux[ids];   duydz += D1C8[0]*uy[ids];  duzdz += D1C8[0]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh-3,NFDX,NFDY,NFDZ);    duxdz += D1C8[1]*ux[ids];   duydz += D1C8[1]*uy[ids];  duzdz += D1C8[1]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh-2,NFDX,NFDY,NFDZ);    duxdz += D1C8[2]*ux[ids];   duydz += D1C8[2]*uy[ids];  duzdz += D1C8[2]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    duxdz += D1C8[3]*ux[ids];   duydz += D1C8[3]*uy[ids];  duzdz += D1C8[3]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    duxdz += D1C8[4]*ux[ids];   duydz += D1C8[4]*uy[ids];  duzdz += D1C8[4]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh+2,NFDX,NFDY,NFDZ);    duxdz += D1C8[5]*ux[ids];   duydz += D1C8[5]*uy[ids];  duzdz += D1C8[5]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh+3,NFDX,NFDY,NFDZ);    duxdz += D1C8[6]*ux[ids];   duydz += D1C8[6]*uy[ids];  duzdz += D1C8[6]*uz[ids];
        ids = gid(txh  ,tyh  ,tzh+4,NFDX,NFDY,NFDZ);    duxdz += D1C8[7]*ux[ids];   duydz += D1C8[7]*uy[ids];  duzdz += D1C8[7]*uz[ids];
    #endif

    // Scale gradients for grid size
    const Real invhx = (Real)1.0/hx, invhy = (Real)1.0/hy, invhz = (Real)1.0/hz;
    duxdx *= invhx;
    duydx *= invhx;
    duzdx *= invhx;
    duxdy *= invhy;
    duydy *= invhy;
    duzdy *= invhy;
    duxdz *= invhz;
    duydz *= invhz;
    duzdz *= invhz;

    // Calculate subgrid scale term for this grid node
    const Real s11 = duxdx                    , q11 = (Real)0.0;
    const Real s12 = (Real)0.5*(duxdy + duydx), q12 = (Real)0.5*(duxdy - duydx)  ;
    const Real s13 = (Real)0.5*(duxdz + duzdx), q13 = (Real)0.5*(duxdz - duzdx)  ;
    const Real s21 = (Real)0.5*(duydx + duxdy), q21 = (Real)0.5*(duydx - duxdy)  ;
    const Real s22 = duydy                    , q22 = (Real)0.0;
    const Real s23 = (Real)0.5*(duydz + duzdy), q23 = (Real)0.5*(duydz - duzdy)  ;
    const Real s31 = (Real)0.5*(duzdx + duxdz), q31 = (Real)0.5*(duzdx - duxdz)  ;
    const Real s32 = (Real)0.5*(duzdy + duydz), q32 = (Real)0.5*(duzdy - duydz)  ;
    const Real s33 = duzdz                    , q33 = (Real)0.0;
    const Real s_ij2 = s11*s11 + s12*s12 + s13*s13 + s21*s21 + s22*s22 + s23*s23 + s31*s31 + s32*s32 + s33*s33;
    const Real q_ij2 = q11*q11 + q12*q12 + q13*q13 + q21*q21 + q22*q22 + q23*q23 + q31*q31 + q32*q32 + q33*q33;
    // This assumes uniform grid spacing!
    sgs = hx*hx*sqrt((Real)2.0*s_ij2);
    qc = (Real)0.5*(q_ij2-s_ij2);

    // Calculate Laplacian
    #if (Map==2)
        ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[0]*wx[ids];  ly += D2C2[0]*wy[ids];  lz += D2C2[0]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[1]*wx[ids];  ly += D2C2[1]*wy[ids];  lz += D2C2[1]*wz[ids];
        ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[2]*wx[ids];  ly += D2C2[2]*wy[ids];  lz += D2C2[2]*wz[ids];
        ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[0]*wx[ids];  ly += D2C2[0]*wy[ids];  lz += D2C2[0]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[1]*wx[ids];  ly += D2C2[1]*wy[ids];  lz += D2C2[1]*wz[ids];
        ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[2]*wx[ids];  ly += D2C2[2]*wy[ids];  lz += D2C2[2]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    lx += D2C2[0]*wx[ids];  ly += D2C2[0]*wy[ids];  lz += D2C2[0]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[1]*wx[ids];  ly += D2C2[1]*wy[ids];  lz += D2C2[1]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    lx += D2C2[2]*wx[ids];  ly += D2C2[2]*wy[ids];  lz += D2C2[2]*wz[ids];
    #elif (Map==4)
        ids = gid(txh-2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C4[0]*wx[ids];  ly += D2C4[0]*wy[ids];  lz += D2C4[0]*wz[ids];
        ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C4[1]*wx[ids];  ly += D2C4[1]*wy[ids];  lz += D2C4[1]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C4[2]*wx[ids];  ly += D2C4[2]*wy[ids];  lz += D2C4[2]*wz[ids];
        ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C4[3]*wx[ids];  ly += D2C4[3]*wy[ids];  lz += D2C4[3]*wz[ids];
        ids = gid(txh+2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C4[4]*wx[ids];  ly += D2C4[4]*wy[ids];  lz += D2C4[4]*wz[ids];
        ids = gid(txh  ,tyh-2,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C4[0]*wx[ids];  ly += D2C4[0]*wy[ids];  lz += D2C4[0]*wz[ids];
        ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C4[1]*wx[ids];  ly += D2C4[1]*wy[ids];  lz += D2C4[1]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C4[2]*wx[ids];  ly += D2C4[2]*wy[ids];  lz += D2C4[2]*wz[ids];
        ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C4[3]*wx[ids];  ly += D2C4[3]*wy[ids];  lz += D2C4[3]*wz[ids];
        ids = gid(txh  ,tyh+2,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C4[4]*wx[ids];  ly += D2C4[4]*wy[ids];  lz += D2C4[4]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh-2,NFDX,NFDY,NFDZ);    lx += D2C4[0]*wx[ids];  ly += D2C4[0]*wy[ids];  lz += D2C4[0]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    lx += D2C4[1]*wx[ids];  ly += D2C4[1]*wy[ids];  lz += D2C4[1]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C4[2]*wx[ids];  ly += D2C4[2]*wy[ids];  lz += D2C4[2]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    lx += D2C4[3]*wx[ids];  ly += D2C4[3]*wy[ids];  lz += D2C4[3]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh+2,NFDX,NFDY,NFDZ);    lx += D2C4[4]*wx[ids];  ly += D2C4[4]*wy[ids];  lz += D2C4[4]*wz[ids];
     #elif (Map==6)
        ids = gid(txh-3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C6[0]*wx[ids];  ly += D2C6[0]*wy[ids];  lz += D2C6[0]*wz[ids];
        ids = gid(txh-2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C6[1]*wx[ids];  ly += D2C6[1]*wy[ids];  lz += D2C6[1]*wz[ids];
        ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C6[2]*wx[ids];  ly += D2C6[2]*wy[ids];  lz += D2C6[2]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C6[3]*wx[ids];  ly += D2C6[3]*wy[ids];  lz += D2C6[3]*wz[ids];
        ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C6[4]*wx[ids];  ly += D2C6[4]*wy[ids];  lz += D2C6[4]*wz[ids];
        ids = gid(txh+2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C6[5]*wx[ids];  ly += D2C6[5]*wy[ids];  lz += D2C6[5]*wz[ids];
        ids = gid(txh+3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C6[6]*wx[ids];  ly += D2C6[6]*wy[ids];  lz += D2C6[6]*wz[ids];
        ids = gid(txh  ,tyh-3,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C6[0]*wx[ids];  ly += D2C6[0]*wy[ids];  lz += D2C6[0]*wz[ids];
        ids = gid(txh  ,tyh-2,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C6[1]*wx[ids];  ly += D2C6[1]*wy[ids];  lz += D2C6[1]*wz[ids];
        ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C6[2]*wx[ids];  ly += D2C6[2]*wy[ids];  lz += D2C6[2]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C6[3]*wx[ids];  ly += D2C6[3]*wy[ids];  lz += D2C6[3]*wz[ids];
        ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C6[4]*wx[ids];  ly += D2C6[4]*wy[ids];  lz += D2C6[4]*wz[ids];
        ids = gid(txh  ,tyh+2,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C6[5]*wx[ids];  ly += D2C6[5]*wy[ids];  lz += D2C6[5]*wz[ids];
        ids = gid(txh  ,tyh+3,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C6[6]*wx[ids];  ly += D2C6[6]*wy[ids];  lz += D2C6[6]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh-3,NFDX,NFDY,NFDZ);    lx += D2C6[0]*wx[ids];  ly += D2C6[0]*wy[ids];  lz += D2C6[0]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh-2,NFDX,NFDY,NFDZ);    lx += D2C6[1]*wx[ids];  ly += D2C6[1]*wy[ids];  lz += D2C6[1]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    lx += D2C6[2]*wx[ids];  ly += D2C6[2]*wy[ids];  lz += D2C6[2]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C6[3]*wx[ids];  ly += D2C6[3]*wy[ids];  lz += D2C6[3]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    lx += D2C6[4]*wx[ids];  ly += D2C6[4]*wy[ids];  lz += D2C6[4]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh+2,NFDX,NFDY,NFDZ);    lx += D2C6[5]*wx[ids];  ly += D2C6[5]*wy[ids];  lz += D2C6[5]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh+3,NFDX,NFDY,NFDZ);    lx += D2C6[6]*wx[ids];  ly += D2C6[6]*wy[ids];  lz += D2C6[6]*wz[ids];
     #elif (Map==8)
        ids = gid(txh-4,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[0]*wx[ids];  ly += D2C8[0]*wy[ids];  lz += D2C8[0]*wz[ids];
        ids = gid(txh-3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[1]*wx[ids];  ly += D2C8[1]*wy[ids];  lz += D2C8[1]*wz[ids];
        ids = gid(txh-2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[2]*wx[ids];  ly += D2C8[2]*wy[ids];  lz += D2C8[2]*wz[ids];
        ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[3]*wx[ids];  ly += D2C8[3]*wy[ids];  lz += D2C8[3]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[4]*wx[ids];  ly += D2C8[4]*wy[ids];  lz += D2C8[4]*wz[ids];
        ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[5]*wx[ids];  ly += D2C8[5]*wy[ids];  lz += D2C8[5]*wz[ids];
        ids = gid(txh+2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[6]*wx[ids];  ly += D2C8[6]*wy[ids];  lz += D2C8[6]*wz[ids];
        ids = gid(txh+3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[7]*wx[ids];  ly += D2C8[7]*wy[ids];  lz += D2C8[7]*wz[ids];
        ids = gid(txh+4,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[8]*wx[ids];  ly += D2C8[8]*wy[ids];  lz += D2C8[8]*wz[ids];
        ids = gid(txh  ,tyh-4,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[0]*wx[ids];  ly += D2C8[0]*wy[ids];  lz += D2C8[0]*wz[ids];
        ids = gid(txh  ,tyh-3,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[1]*wx[ids];  ly += D2C8[1]*wy[ids];  lz += D2C8[1]*wz[ids];
        ids = gid(txh  ,tyh-2,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[2]*wx[ids];  ly += D2C8[2]*wy[ids];  lz += D2C8[2]*wz[ids];
        ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[3]*wx[ids];  ly += D2C8[3]*wy[ids];  lz += D2C8[3]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[4]*wx[ids];  ly += D2C8[4]*wy[ids];  lz += D2C8[4]*wz[ids];
        ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[5]*wx[ids];  ly += D2C8[5]*wy[ids];  lz += D2C8[5]*wz[ids];
        ids = gid(txh  ,tyh+2,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[6]*wx[ids];  ly += D2C8[6]*wy[ids];  lz += D2C8[6]*wz[ids];
        ids = gid(txh  ,tyh+3,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[7]*wx[ids];  ly += D2C8[7]*wy[ids];  lz += D2C8[7]*wz[ids];
        ids = gid(txh  ,tyh+4,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[8]*wx[ids];  ly += D2C8[8]*wy[ids];  lz += D2C8[8]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh-4,NFDX,NFDY,NFDZ);    lx += D2C8[0]*wx[ids];  ly += D2C8[0]*wy[ids];  lz += D2C8[0]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh-3,NFDX,NFDY,NFDZ);    lx += D2C8[1]*wx[ids];  ly += D2C8[1]*wy[ids];  lz += D2C8[1]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh-2,NFDX,NFDY,NFDZ);    lx += D2C8[2]*wx[ids];  ly += D2C8[2]*wy[ids];  lz += D2C8[2]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    lx += D2C8[2]*wx[ids];  ly += D2C8[3]*wy[ids];  lz += D2C8[3]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C8[4]*wx[ids];  ly += D2C8[4]*wy[ids];  lz += D2C8[4]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    lx += D2C8[5]*wx[ids];  ly += D2C8[5]*wy[ids];  lz += D2C8[5]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh+2,NFDX,NFDY,NFDZ);    lx += D2C8[6]*wx[ids];  ly += D2C8[6]*wy[ids];  lz += D2C8[6]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh+3,NFDX,NFDY,NFDZ);    lx += D2C8[7]*wx[ids];  ly += D2C8[7]*wy[ids];  lz += D2C8[7]*wz[ids];
        ids = gid(txh  ,tyh  ,tzh+4,NFDX,NFDY,NFDZ);    lx += D2C8[8]*wx[ids];  ly += D2C8[8]*wy[ids];  lz += D2C8[8]*wz[ids];
     #endif

    // Scale laplacian for grid size
    lx *= (Real)1.0/hx/hx;
    ly *= (Real)1.0/hy/hy;
    lz *= (Real)1.0/hz/hz;
}

// Calculate outputs
const Real sx = duxdx*wx[pid] + duxdy*wy[pid] + duxdz*wz[pid] + KinVisc*lx;
const Real sy = duydx*wx[pid] + duydy*wy[pid] + duydz*wz[pid] + KinVisc*ly;
const Real sz = duzdx*wx[pid] + duzdy*wy[pid] + duzdz*wz[pid] + KinVisc*lz;

barrier(CLK_LOCAL_MEM_FENCE);

// Assign output vars
grad[bid]      = sx;
grad[bid+NT]   = sy;
grad[bid+2*NT] = sz;
smag[bid]	   = sgs;
qcrit[bid]   = qc;

}

)CLC";

//--- Freestream kernels

const std::string VPM3D_ocl_kernels_freestream = R"CLC(
__kernel void AddFreestream(__global Real *u,
                            const Real Ux,
                            const Real Uy,
                            const Real Uz) {

    unsigned int i = get_group_id(0)*get_local_size(0) + get_local_id(0);

    u[i		] += Ux;
    u[i+1*NT] += Uy;
    u[i+2*NT] += Uz;
}
)CLC";

//--- External sources

// const std::string VPM3D_cuda_kernels_XXX = R"CLC(

// )CLC";

const std::string VPM3D_cuda_kernels_MapExt = R"CLC(

__kernel void Map_Ext_Bounded(  __global const Real* sx,
                                __global const Real* sy,
                                __global const Real* sz,    // Source grid
                                __global const int *blX,
                                __global const int *blY,
                                __global const int *blZ,    // Block indices
                                __global Real* u)           // Destination grid
{
    // Prepare relevant indices
    const int BlockID = get_group_id(0);
    const int gx0 = blX[BlockID]*BX;
    const int gy0 = blY[BlockID]*BY;
    const int gz0 = blZ[BlockID]*BZ;
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int tz = get_local_id(2);
    const int gx = tx + gx0;
    const int gy = ty + gy0;
    const int gz = tz + gz0;

    const int lid = gid(tx,ty,tz,BX,BY,BZ);     // Local id within block
    const int sid = lid + BlockID*BT;           // Positions within array of source
    const int did = gidb(gx,gy,gz);             // Destination grid index

    u[did       ] += sx[sid];
    u[did + 1*NT] += sy[sid];
    u[did + 2*NT] += sz[sid];
}
)CLC";

const std::string VPM3D_ocl_kernels_ExtSourceInterp = R"CLC(

__kernel void Interp_Block_Ext( __global const Real* src,            // Source grid values (permanently mapped particles)
                                __global const int *blX,             // Block indices
                                __global const int *blY,
                                __global const int *blZ,
                                __global const Real* disp,           // Particle displacement
                                __global const int* hs,              // Halo indices
                                __global Real* dest)                 // Destination grid (vorticity)
{
// Declare shared memory vars (mapped vorticity field)
__local Real sx[NHT];
__local Real sy[NHT];
__local Real sz[NHT];

// We are still executing the blocks with blockdim [BX, BY, BZ], however are are using the Griddims in a 1d sense.
// So we need to modify the x-dim based on the number of active blocks.

// Prepare relevant indices
const int BlockID = get_group_id(0);
const int gx0 = blX[BlockID]*BX;
const int gy0 = blY[BlockID]*BY;
const int gz0 = blZ[BlockID]*BZ;
const int tx = get_local_id(0);
const int ty = get_local_id(1);
const int tz = get_local_id(2);
const int gx = tx + gx0;
const int gy = ty + gy0;
const int gz = tz + gz0;
const int txh = tx + Halo;                          // Local x id within padded grid
const int tyh = ty + Halo;                          // Local y id within padded grid
const int tzh = tz + Halo;                          // Local z id within padded grid												// How many source points shalll be loaded for this block?

// Monolith structure
// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);
const int lid = gid(tx,ty,tz,BX,BY,BZ);             // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);    // Local id within padded block


// Step 1) Copy global memory to shared & local memory

// Specify node displacement
const Real pX = disp[bid       ];
const Real pY = disp[bid + 1*NT];
const Real pZ = disp[bid + 2*NT];

// Specify centre volume arrays
sx[pid] = src[bid       ];
sy[pid] = src[bid + 1*NT];
sz[pid] = src[bid + 2*NT];

barrier(CLK_LOCAL_MEM_FENCE);

// Fill Halo (with coalesced index read)
for (int i=0; i<NHIT; i++){
    const int hid = BT*i + lid;
    const int hsx = hs[hid];                    		// global x-shift relative to position
    const int hsy = hs[hid+BT*NHIT];               	// global y-shift relative to position
    const int hsz = hs[hid+2*BT*NHIT];             	// global z-shift relative to position
    if (hsx<NFDX){                                 	// Catch: is id within padded indices?
        const int ghx = gx0-Halo+hsx;             	// Global x-value of retrieved node
        const int ghy = gy0-Halo+hsy;             	// Global y-value of retrieved node
        const int ghz = gz0-Halo+hsz;             	// Global z-value of retrieved node
        const int lhid = gid(hsx,hsy,hsz,NFDX,NFDY,NFDZ);

        // if (Data==Monolith) Kernel.append(const int bhid = gid(ghx,ghy,ghz,NX,NY,NZ););
        // if (Data==Block)    Kernel.append(const int bhid = gidb(ghx,ghy,ghz););
        const int bhid = gidb(ghx,ghy,ghz);

        const bool exx = (ghx<0 || ghx>=NX);      	// Is x coordinate outside of the domain?
        const bool exy = (ghy<0 || ghy>=NY);      	// Is y coordinate outside of the domain?
        const bool exz = (ghz<0 || ghz>=NZ);      	// Is z coordinate outside of the domain?
        if (exx || exy || exz){                    	// Catch: is id outside of domain?
            sx[lhid] = (Real)0.0;
            sy[lhid] = (Real)0.0;
            sz[lhid] = (Real)0.0;
        }
        else {
            sx[lhid] = src[bhid       ];
            sy[lhid] = src[bhid + 1*NT];
            sz[lhid] = src[bhid + 2*NT];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

//------------------------------------------------------------------------------
// Step 2) Interpolate values from grid
//------------------------------------------------------------------------------

Real m1x = (Real)0.0, m1y = (Real)0.0, m1z = (Real)0.0;     // Interpolated values

int iix, iiy, iiz;                                // Interpolation id
const Real dxh = pX/hx, dyh = pY/hy, dzh = pZ/hz;    // Normalized distances

// Calculate interpolation weights
const int H2 = Halo*2;
Real cx[H2], cy[H2], cz[H2];			// Interpolation weights
int NS;									// Shift for node id

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
                const Real fac = cx[i]*cy[j]*cz[k];
                m1x += fac*sx[ids];
                m1y += fac*sy[ids];
                m1z += fac*sz[ids];
            }
        }
    }

barrier(CLK_LOCAL_MEM_FENCE);

//------------------------------------------------------------------------------
// Step 3) Transfer interpolated values back to array
//------------------------------------------------------------------------------

dest[bid       ] += m1x;
dest[bid + 1*NT] += m1y;
dest[bid + 2*NT] += m1z;

}

)CLC";

}
