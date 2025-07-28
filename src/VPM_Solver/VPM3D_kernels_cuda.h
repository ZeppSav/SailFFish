//--------------------------------------------------------------------------------
//-------------------------- VPM 3D: cuda Kernels --------------------------------
//--------------------------------------------------------------------------------

// Note: Prior to initialiization, the "Real" type must be defined along with float/double dependent functions

#include "../SailFFish_Math_Types.h"

namespace SailFFish
{

const std::string VPM3D_cuda_kernels_float = R"CLC(
typedef float Real;
__device__   float  fastma(const float &a, const float &b, const float &c)    {return fmaf(a,b,c);}
__device__   float  fab(const float &a)   {return fabs(a);}
__device__   float  mymax(float a, float b)   {return fmaxf(a,b);}
)CLC";

const std::string VPM3D_cuda_kernels_double = R"CLC(
typedef double Real;
__device__ double fastma(const double &a, const double &b, const double &c) {return fma(a,b,c);}
__device__ double fab(const double &a)  {return abs(a);}
__device__ double mymax(double a, double b)  {return max(a,b);}
)CLC";

const std::string VPM3D_cuda_kernels_source = R"CLC(

//---------------
// Grid constants
//---------------

__constant__ int NX, NY, NZ, NT;      
__constant__ int BX, BY, BZ, BT;      
__constant__ int NBX, NBY, NBZ;       
__constant__ int NFDX, NFDY, NFDZ;    
__constant__ int NHIT;       
__constant__ Real hx, hy, hz;                                                  

// gid takes the global x,y,z indices and calculates array positions
__device__ int gid(const int i,const int j, const int k, const int DX, const int DY, const int DZ){ 
return i*DY*DZ + j*DZ + k; 
} 
 
// gidb returns the block-ordered id of a node in global position im,jm,km
__device__ int gidb(const int im, const int jm, const int km){ 
const int ib = im/BX, jb = jm/BY, kb = km/BZ;
const int gl = gid(im-ib*BX, jm-jb*BY, km-kb*BZ, BX, BY, BZ);
const int bls = gid(ib,jb,kb,NBX,NBY,NBZ);
return gl + BT*bls; 
}

// gidb returns the block-ordered id of a node in global position im,jm,km. This version is used for the auxiliary grid
__device__ int gidb(const int im, const int jm, const int km, const int tNBX, const int tNBY, const int tNBZ){ 
const int ib = im/BX, jb = jm/BY, kb = km/BZ;
const int gl = gid(im-ib*BX, jm-jb*BY, km-kb*BZ, BX, BY, BZ);
const int bls = gid(ib,jb,kb,tNBX,tNBY,tNBZ);
return gl +  BT*bls; 
}

//---------------------
// Convolution kernels 
//---------------------

template<typename T>
__device__ void multiply(const T &C1, const T &C2, T &Cout){  
    Cout.x =  C1.x*C2.x - C1.y*C2.y;   
	Cout.y =  C1.x*C2.y + C1.y*C2.x;     
}  

template<typename T>
__device__ void subtract(const T &C1, const T &C2, T &Cout){  
    Cout.x =  C1.x - C2.x;   
	Cout.y =  C1.y - C2.y;    
}  

template<typename T>
__device__ void add3(const T &C1, const T &C2, const T &C3, T &Cout){  
    Cout.x =  C1.x + C2.x + C3.x;   
	Cout.y =  C1.y + C2.y + C3.y;    
} 

template<typename T> 
__global__ void vpm_convolution(const T *OX, const T *OY, const T *OZ, const T *GF, const T *iX, const T *iY, const T *iZ, T *UX, T *UY, T *UZ) { 

// Specify grid id
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 

// Load values for this node into memory
const T ox = OX[i]; 
const T oy = OY[i]; 
const T oz = OZ[i]; 
const T gf = GF[i];
const T ix = iX[i]; 
const T iy = iY[i]; 
const T iz = iZ[i];

__syncthreads();

// Carry out convolution in frequency space.
T gfx, gfy, gfz;
multiply(ox,gf,gfx);
multiply(oy,gf,gfy);
multiply(oz,gf,gfz);

// Extract curl of vector in frequency space
T ux1, ux2, uy1, uy2, uz1, uz2, tUX, tUY, tUZ;
multiply(gfz,iy,ux1);
multiply(gfy,iz,ux2);
multiply(gfx,iz,uy1);
multiply(gfz,ix,uy2);
multiply(gfy,ix,uz1);
multiply(gfx,iy,uz2);
subtract(ux2,ux1,tUX);
subtract(uy2,uy1,tUY);
subtract(uz2,uz1,tUZ);

// __syncthreads();

// Write outputs 
UX[i] = tUX;
UY[i] = tUY;
UZ[i] = tUZ;	
} 


template<typename T> 
__global__ void vpm_reprojection(const T *OX, const T *OY, const T *OZ, const T *GF, const T *iX, const T *iY, const T *iZ, Real BFac, T *UX, T *UY, T *UZ) { 

// Specify grid id
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 

// Load values for this node into memory
const T ox = OX[i]; 
const T oy = OY[i]; 
const T oz = OZ[i]; 
const T gf = GF[i];
const T ix = iX[i]; 
const T iy = iY[i]; 
const T iz = iZ[i];

__syncthreads();

// Declare spectral variables 
T duxdx, duydy, duzdz, divom, F, dFdx, dFdy, dFdz;

// Calculate divergence in spectral space
multiply(ox,ix,duxdx);
multiply(oy,iy,duydy);
multiply(oz,iz,duzdz);
add3(duxdx,duydy,duzdz,divom);

// Solve for Nabla^2 (F) = divOm
multiply(divom,gf,F);

// Calculate gradient of F
multiply(F,ix,dFdx);
multiply(F,iy,dFdy);
multiply(F,iz,dFdz);

// Reprojected output calculated by scaling input field and subtracting gradients of F term
T rOx, rOy, rOz;
rOx.x = ox.x*BFac - dFdx.x;
rOx.y = ox.y*BFac - dFdx.y;
rOy.x = oy.x*BFac - dFdy.x;
rOy.y = oy.y*BFac - dFdy.y;
rOz.x = oz.x*BFac - dFdz.x;
rOz.y = oz.y*BFac - dFdz.y;

// __syncthreads();

// Write outputs 
UX[i] = rOx;
UY[i] = rOy;
UZ[i] = rOz;	
} 

//---------------------------------------
// Block & monolithic data setup kernels 
//---------------------------------------

__global__ void Monolith_to_Block(const Real* src, Real* dst) {

    const int gx = threadIdx.x + blockIdx.x*BX; 
    const int gy = threadIdx.y + blockIdx.y*BY; 
    const int gz = threadIdx.z + blockIdx.z*BZ; 
    const int mid = gid(gx,gy,gz,NX,NY,NZ);           // Global id (Monolithic)
    const int bid = gidb(gx,gy,gz);                   // Global id (Block)

    dst[bid]       += src[mid];
    dst[bid+NT]    += src[mid+NT];
    dst[bid+2*NT]  += src[mid+2*NT];
}

__global__ void Block_to_Monolith(const Real* src, Real* dst) {

	const int gx = threadIdx.x + blockIdx.x*BX; 
	const int gy = threadIdx.y + blockIdx.y*BY; 
	const int gz = threadIdx.z + blockIdx.z*BZ; 
	const int mid = gid(gx,gy,gz,NX,NY,NZ);
	const int bid = gidb(gx,gy,gz);

	dst[mid]       = src[bid];
	dst[mid+NT]    = src[bid+NT];
	dst[mid+2*NT]  = src[bid+2*NT];
}

__global__ void Map_toUnbounded(const Real* src, Real* dst1, Real* dst2, Real* dst3) {

	const int gx = threadIdx.x + blockIdx.x*BX; 
	const int gy = threadIdx.y + blockIdx.y*BY; 
	const int gz = threadIdx.z + blockIdx.z*BZ; 
	const int mid = gid(gx,gy,gz,2*NX,2*NY,2*NZ);
	const int bid = gidb(gx,gy,gz);

	dst1[mid]  = src[bid];
	dst2[mid]  = src[bid+NT];
	dst3[mid]  = src[bid+2*NT];
}

__global__ void Map_fromUnbounded(const Real* src1, const Real* src2, const Real* src3, Real* dst) {

	const int gx = threadIdx.x + blockIdx.x*BX; 
	const int gy = threadIdx.y + blockIdx.y*BY; 
	const int gz = threadIdx.z + blockIdx.z*BZ; 
	const int mid = gid(gx,gy,gz,2*NX,2*NY,2*NZ);
	const int bid = gidb(gx,gy,gz);

	dst[bid     ]  = src1[mid];
	dst[bid+1*NT]  = src2[mid];
	dst[bid+2*NT]  = src3[mid];
}

//---------------
// Mappings kernels
//---------------

__device__ void mapM2(const Real &x, Real &u){                                  
   if (x< Real(1.)) u = Real(1.)-x;                                
   else            	u = Real(0.);                                  
}     

__device__ void mapM4(const Real &x, Real &u){                                     
	if (x<Real(1.))        	u = (Real(2.)-x)*(Real(2.)-x)*(Real(2.)-x)/Real(6.) - Real(4.)*(Real(1.)-x)*(Real(1.)-x)*(Real(1.)-x)/Real(6.);
	else if (x<Real(2.))   	u = (Real(2.)-x)*(Real(2.)-x)*(Real(2.)-x)/Real(6.);
	else                	u = Real(0.);
}

__device__ void mapM4D(const Real &x, Real &u){                                     
   if (x<Real(1.))          u = Real(0.5)*(Real(2.)-Real(5.)*x*x+Real(3.)*x*x*x); 
   else if (x<Real(2.))     u = Real(0.5)*(Real(1.)-x)*(Real(2.)-x)*(Real(2.)-x); 
   else                  	u = Real(0.);                                
}    

__constant__ double cm6d1 = -1./88.;
__constant__ double cm6d2 = 1./176.;
__constant__ double cm6d3 = -3./176.;
__device__ void mapM6D(const Real &x, Real &u){ 
	const Real x2 = x*x;
	if (x<Real(1.0))       	u = Real(cm6d1)*(x-Real(1.))*(Real(60.)*x2*x2-Real(87.)*x*x2-Real(87.)*x2+Real(88.)*x+Real(88.));
	else if (x<Real(2.0))  	u = Real(cm6d2)*(x-Real(1.))*(x-Real(2.))*(Real(60.)*x*x2-Real(261.)*x2+Real(257.)*x+Real(68.));
	else if (x<Real(3.0))  	u = Real(cm6d3)*(x-Real(2.))*(Real(4.)*x2-Real(17.)*x+Real(12.))*(x-Real(3.))*(x-Real(3.));
	else                	u = 0.0;                             
}  

template<int Halo, int Map, int NHT>                    // Pass padded grid size as template argument
__global__ void MapKernel(const Real* f, const Real* d, const int* hs, Real* interpf) {

__shared__ Real sx[NHT];
__shared__ Real sy[NHT];
__shared__ Real sz[NHT];
__shared__ Real dx[NHT];
__shared__ Real dy[NHT];
__shared__ Real dz[NHT];

// Prepare relevant indices
const int gx0 = blockIdx.x*BX; 
const int gy0 = blockIdx.y*BY; 
const int gz0 = blockIdx.z*BZ; 
const int gx = threadIdx.x + gx0; 
const int gy = threadIdx.y + gy0; 
const int gz = threadIdx.z + gz0; 
const int txh = threadIdx.x + Halo;                                          // Local x id within padded grid
const int tyh = threadIdx.y + Halo;                                          // Local y id within padded grid
const int tzh = threadIdx.z + Halo;                                          // Local z id within padded grid

// Monolith structure
// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);
const int lid = gid(threadIdx.x,threadIdx.y,threadIdx.z,BX,BY,BZ);           // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);                             // Local id within padded block

// Fill centre volume source & displacement coeff arrays
sx[pid] = f[bid];
sy[pid] = f[bid+NT];
sz[pid] = f[bid+2*NT];
dx[pid] = d[bid];
dy[pid] = d[bid+NT];
dz[pid] = d[bid+2*NT];

__syncthreads();

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
		const bool exy = (ghy<0 || ghy>=NY);      // Is y coordinate outside of the domain?
		const bool exz = (ghz<0 || ghz>=NZ);      // Is z coordinate outside of the domain?
		if (exx || exy || exz){                    // Catch: is id outside of domain?
			sx[lhid] = Real(0.);                   
			sy[lhid] = Real(0.);                   
			sz[lhid] = Real(0.);                   
			dx[lhid] = Real(0.);                   
			dy[lhid] = Real(0.);                   
			dz[lhid] = Real(0.);                   
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
   __syncthreads();
}

// Now carry out interpolation using shared memory
// The displacement of the particles is used to calculate which nodes shall be used for mapping.
// The interpolation coefficients for this are then calculated and used to map the source field from the shared memory arrays.
Real mx = Real(0.), my = Real(0.), mz = Real(0.);     // Interpolated values
const bool mxx = (gx>(Halo-1) && gx<(NX-1-(Halo-1)));
const bool mxy = (gy>(Halo-1) && gy<(NY-1-(Halo-1)));
const bool mxz = (gz>(Halo-1) && gz<(NZ-1-(Halo-1)));
if (mxx && mxy && mxz){

	// Set interpolation limits
	int BL, BU;							// Upper and lower bounds  
	if (Map==2)	{BL = -1; BU = 1;}		// M2 mapping	
	if (Map==4)	{BL = -2; BU = 2;}		// M4 mapping
	if (Map==42){BL = -2; BU = 2;}		// M4D mapping
	if (Map==6) {BL = -3; BU = 3;}		// M6D mapping
	
	// Carry out interpolation
	Real fx,fy,fz;
	for (int i=BL; i<=BU; i++){                                           
		for (int j=BL; j<=BU; j++){                                        
			for (int k=BL; k<=BU; k++){                                    
				const int ids = gid(txh+i,tyh+j,tzh+k,NFDX,NFDY,NFDZ);   
				if (Map==2){
					mapM2(fab(Real(i)+dx[ids]/hx), fx);                       
					mapM2(fab(Real(j)+dy[ids]/hy), fy);                       
					mapM2(fab(Real(k)+dz[ids]/hz), fz);
				}
				if (Map==4){
					mapM4(fab(Real(i)+dx[ids]/hx), fx);                       
					mapM4(fab(Real(j)+dy[ids]/hy), fy);                       
					mapM4(fab(Real(k)+dz[ids]/hz), fz);
				}
				if (Map==42){
					mapM4D(fab(Real(i)+dx[ids]/hx), fx);                       
					mapM4D(fab(Real(j)+dy[ids]/hy), fy);                       
					mapM4D(fab(Real(k)+dz[ids]/hz), fz);    
				}   
				if (Map==6){
					mapM6D(fab(Real(i)+dx[ids]/hx), fx);                       
					mapM6D(fab(Real(j)+dy[ids]/hy), fy);                       
					mapM6D(fab(Real(k)+dz[ids]/hz), fz);  
				} 
				
				const Real Fac = fx*fy*fz;              
				mx = fastma(Fac,sx[ids],mx);                             
				my = fastma(Fac,sy[ids],my);                             
				mz = fastma(Fac,sz[ids],mz);                             
			}
		}
	}
}
__syncthreads();

// Write output
interpf[bid     ] = mx;
interpf[bid+1*NT] = my;
interpf[bid+2*NT] = mz;
}

//---------------------
// Interpolation kernel 
//---------------------

template<int Halo, int Mapping, int NHT>                    // Pass padded grid size as template argument
__global__ void InterpKernel(const Real* f1, const Real* f2, const Real* d, const int* hs, Real* interpf1, Real* interpf2) {

__shared__ Real sx[NHT];
__shared__ Real sy[NHT];
__shared__ Real sz[NHT];
__shared__ Real ux[NHT];
__shared__ Real uy[NHT];
__shared__ Real uz[NHT];

// Prepare relevant indices
const int gx0 = blockIdx.x*BX; 
const int gy0 = blockIdx.y*BY; 
const int gz0 = blockIdx.z*BZ; 
const int gx = threadIdx.x + gx0; 
const int gy = threadIdx.y + gy0; 
const int gz = threadIdx.z + gz0; 
const int txh = threadIdx.x + Halo;                                          // Local x id within padded grid
const int tyh = threadIdx.y + Halo;                                          // Local y id within padded grid
const int tzh = threadIdx.z + Halo;                                          // Local z id within padded grid

// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);

const int lid = gid(threadIdx.x,threadIdx.y,threadIdx.z,BX,BY,BZ);           // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);                             // Local id within padded block

// Fill centre volume source arrays & mapping coeff arrays
sx[pid] = f1[bid];
sy[pid] = f1[bid+NT];
sz[pid] = f1[bid+2*NT];
ux[pid] = f2[bid];
uy[pid] = f2[bid+NT];
uz[pid] = f2[bid+2*NT];

// Load particle displacements
const Real dx = d[bid];
const Real dy = d[bid+NT];
const Real dz = d[bid+2*NT];
__syncthreads();


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
		const bool exy = (ghy<0 || ghy>=NY);      // Is y coordinate outside of the domain?
		const bool exz = (ghz<0 || ghz>=NZ);      // Is z coordinate outside of the domain?
		if (exx || exy || exz){                    // Catch: is id outside of domain?
			sx[lhid] = Real(0.);                   
			sy[lhid] = Real(0.);                   
			sz[lhid] = Real(0.); 
			ux[lhid] = Real(0.);                   
			uy[lhid] = Real(0.);                   
			uz[lhid] = Real(0.);
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
	__syncthreads();
}

// Now carry out interpolation using shared memory
// The displacement of the particles is used to calculate which nodes shall be used for mapping.
// The interpolation coefficients for this are then calculated and used to map the source field from the shared memory arrays.
Real m1x = Real(0.), m1y = Real(0.), m1z = Real(0.);     // Interpolated values
Real m2x = Real(0.), m2y = Real(0.), m2z = Real(0.);     // Interpolated values
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
	
	// M2 interpolation
	if (Mapping==2){
		NS = 0;	
		if (dxh>=Real(0.)){ iix = txh;      mapM2(dxh,cx[0]);         	mapM2(Real(1.)-dxh,cx[1]);  }
		else {             	iix = txh-1;    mapM2(Real(1.)+dxh,cx[0]);  mapM2(-dxh,cx[1]);        	}

		if (dyh>=Real(0.)){ iiy = tyh;      mapM2(dyh,cy[0]);         	mapM2(Real(1.)-dyh,cy[1]);  }
		else {             	iiy = tyh-1;    mapM2(Real(1.)+dyh,cy[0]);  mapM2(-dyh,cy[1]);        	}

		if (dzh>=Real(0.)){ iiz = tzh;      mapM2(dzh,cz[0]);         	mapM2(Real(1.)-dzh,cz[1]); 	}
		else {             	iiz = tzh-1;    mapM2(Real(1.)+dzh,cz[0]);  mapM2(-dzh,cz[1]);        	}
	}
	
	// M4 interpolation
	if (Mapping==4){
		NS = -1;	
		if (dxh>=Real(0.)){ iix = txh;      mapM4(Real(1.)+dxh,cx[0]);  mapM4(dxh,cx[1]);    		mapM4(Real(1.)-dxh,cx[2]);  mapM4(Real(2.)-dxh,cx[3]);}
		else {             	iix = txh-1;    mapM4(Real(2.)+dxh,cx[0]);  mapM4(Real(1.)+dxh,cx[1]);  mapM4(-dxh,cx[2]);  		mapM4(Real(1.)-dxh,cx[3]);}

		if (dyh>=Real(0.)){ iiy = tyh;    	mapM4(Real(1.)+dyh,cy[0]);  mapM4(dyh,cy[1]);   		mapM4(Real(1.)-dyh,cy[2]);	mapM4(Real(2.)-dyh,cy[3]);}
		else {             	iiy = tyh-1;    mapM4(Real(2.)+dyh,cy[0]);  mapM4(Real(1.)+dyh,cy[1]);	mapM4(-dyh,cy[2]);  		mapM4(Real(1.)-dyh,cy[3]);}

		if (dzh>=Real(0.)){ iiz = tzh;		mapM4(Real(1.)+dzh,cz[0]);  mapM4(dzh,cz[1]);   		mapM4(Real(1.)-dzh,cz[2]);	mapM4(Real(2.)-dzh,cz[3]);}
		else {             	iiz = tzh-1;	mapM4(Real(2.)+dzh,cz[0]);  mapM4(Real(1.)+dzh,cz[1]);	mapM4(-dzh,cz[2]);  		mapM4(Real(1.)-dzh,cz[3]);}     
	}
	
	// M4' interpolation
	if (Mapping==42){
		NS = -1;	
		if (dxh>=Real(0.)){ iix = txh;      mapM4D(Real(1.)+dxh,cx[0]);  mapM4D(dxh,cx[1]);    			mapM4D(Real(1.)-dxh,cx[2]);	mapM4D(Real(2.)-dxh,cx[3]);}
		else {             	iix = txh-1;    mapM4D(Real(2.)+dxh,cx[0]);  mapM4D(Real(1.)+dxh,cx[1]);   	mapM4D(-dxh,cx[2]);  		mapM4D(Real(1.)-dxh,cx[3]);}

		if (dyh>=Real(0.)){ iiy = tyh;      mapM4D(Real(1.)+dyh,cy[0]);  mapM4D(dyh,cy[1]);   			mapM4D(Real(1.)-dyh,cy[2]);	mapM4D(Real(2.)-dyh,cy[3]);}
		else {             	iiy = tyh-1;    mapM4D(Real(2.)+dyh,cy[0]);  mapM4D(Real(1.)+dyh,cy[1]);   	mapM4D(-dyh,cy[2]);  		mapM4D(Real(1.)-dyh,cy[3]);}

		if (dzh>=Real(0.)){	iiz = tzh;      mapM4D(Real(1.)+dzh,cz[0]);  mapM4D(dzh,cz[1]);   			mapM4D(Real(1.)-dzh,cz[2]);	mapM4D(Real(2.)-dzh,cz[3]);}
		else {             	iiz = tzh-1;    mapM4D(Real(2.)+dzh,cz[0]);  mapM4D(Real(1.)+dzh,cz[1]);   	mapM4D(-dzh,cz[2]); 		mapM4D(Real(1.)-dzh,cz[3]);}
	}
	
	// M6' interpolation
	
	if (Mapping==6){
		NS = -2;	
		if (dxh>=Real(0.)){	iix = txh;      mapM6D(Real(2.)+dxh,cx[0]);  mapM6D(Real(1.)+dxh,cx[1]);  mapM6D(      dxh,cx[2]);  	mapM6D(Real(1.)-dxh,cx[3]); 	mapM6D(Real(2.)-dxh,cx[4]);  	mapM6D(Real(3.)-dxh,cx[5]);}
		else {             	iix = txh-1;    mapM6D(Real(3.)+dxh,cx[0]);  mapM6D(Real(2.)+dxh,cx[1]);  mapM6D(Real(1.)+dxh,cx[2]);  	mapM6D(     -dxh,cx[3]); 		mapM6D(Real(1.)-dxh,cx[4]);  	mapM6D(Real(2.)-dxh,cx[5]);}
	
		if (dyh>=Real(0.)){	iiy = tyh;      mapM6D(Real(2.)+dyh,cy[0]);  mapM6D(Real(1.)+dyh,cy[1]);  mapM6D(      dyh,cy[2]);  	mapM6D(Real(1.)-dyh,cy[3]); 	mapM6D(Real(2.)-dyh,cy[4]);  	mapM6D(Real(3.)-dyh,cy[5]);}
		else {             	iiy = tyh-1;    mapM6D(Real(3.)+dyh,cy[0]);  mapM6D(Real(2.)+dyh,cy[1]);  mapM6D(Real(1.)+dyh,cy[2]);  	mapM6D(     -dyh,cy[3]); 		mapM6D(Real(1.)-dyh,cy[4]);  	mapM6D(Real(2.)-dyh,cy[5]);}
	
		if (dzh>=Real(0.)){	iiz = tzh;      mapM6D(Real(2.)+dzh,cz[0]);  mapM6D(Real(1.)+dzh,cz[1]);  mapM6D(      dzh,cz[2]);  	mapM6D(Real(1.)-dzh,cz[3]); 	mapM6D(Real(2.)-dzh,cz[4]);  	mapM6D(Real(3.)-dzh,cz[5]);}
		else {             	iiz = tzh-1;    mapM6D(Real(3.)+dzh,cz[0]);  mapM6D(Real(2.)+dzh,cz[1]);  mapM6D(Real(1.)+dzh,cz[2]);  	mapM6D(     -dzh,cz[3]); 		mapM6D(Real(1.)-dzh,cz[4]);  	mapM6D(Real(2.)-dzh,cz[5]);}
	}
	
	// Carry out interpolation
	for (int i=0; i<H2; i++){                                       
	   for (int j=0; j<H2; j++){                                    
		   for (int k=0; k<H2; k++){                                
			   const int idsx =  iix + NS + i;                           
			   const int idsy =  iiy + NS + j;                           
			   const int idsz =  iiz + NS + k;                           
			   const int ids =  gid(idsx,idsy,idsz,NFDX,NFDY,NFDZ); 
			   const Real fac =  cx[i]*cy[j]*cz[k];                   
			   m1x =  fastma(fac,sx[ids],m1x);                       
			   m1y =  fastma(fac,sy[ids],m1y);                       
			   m1z =  fastma(fac,sz[ids],m1z); 
			   m2x =  fastma(fac,ux[ids],m2x);                       
			   m2y =  fastma(fac,uy[ids],m2y);                       
			   m2z =  fastma(fac,uz[ids],m2z);                       
		   }                                                       
	   }                                                           
	}     
	
}
__syncthreads();

// Write output
interpf1[bid     ] = m1x;
interpf1[bid+1*NT] = m1y;
interpf1[bid+2*NT] = m1z;
interpf2[bid     ] = m2x;
interpf2[bid+1*NT] = m2y;
interpf2[bid+2*NT] = m2z;

}

//-----------------------------
// Time-marching scheme kernels 
//-----------------------------

__global__ void update(Real* d, Real* o, const Real* d_d, const Real* d_o, const Real dt) {

// Set grid id
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 

const Real p1 = d[i];
const Real p2 = d[i+NT];
const Real p3 = d[i+2*NT];
const Real o1 = o[i];
const Real o2 = o[i+NT];
const Real o3 = o[i+2*NT];
const Real dp1 = d_d[i];
const Real dp2 = d_d[i+NT];
const Real dp3 = d_d[i+2*NT];
const Real do1 = d_o[i];
const Real do2 = d_o[i+NT];
const Real do3 = d_o[i+2*NT];

// if (i==0) printer(dt);

__syncthreads(); 

// Set outputs
d[i     ] = p1 + dt*dp1;
d[i+1*NT] = p2 + dt*dp2;
d[i+2*NT] = p3 + dt*dp3;
o[i     ] = o1 + dt*do1;
o[i+1*NT] = o2 + dt*do2;
o[i+2*NT] = o3 + dt*do3;

}

__global__ void updateRK(const Real* d, const Real* o, const Real* d_d, const Real* d_o, Real* kd, Real* ko, const Real dt) {

// Set grid id
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 

const Real p1 = d[i];
const Real p2 = d[i+NT];
const Real p3 = d[i+2*NT];
const Real o1 = o[i];
const Real o2 = o[i+NT];
const Real o3 = o[i+2*NT];

const Real dp1 = d_d[i];
const Real dp2 = d_d[i+NT];
const Real dp3 = d_d[i+2*NT];
const Real do1 = d_o[i];
const Real do2 = d_o[i+NT];
const Real do3 = d_o[i+2*NT];

__syncthreads(); 

// Set outputs
kd[i     ] = p1 + dt*dp1;
kd[i+1*NT] = p2 + dt*dp2;
kd[i+2*NT] = p3 + dt*dp3;
ko[i     ] = o1 + dt*do1;
ko[i+1*NT] = o2 + dt*do2;
ko[i+2*NT] = o3 + dt*do3;
}

__global__ void updateRK_LS(Real* d, Real* o, const Real* d_d, const Real* d_o, Real* s_d, Real* s_o, const Real A, const Real B, const Real h) {

// Set grid id
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 

// Intermediate vars
Real s21 = A*s_d[i]			+ h*d_d[i];			
Real s22 = A*s_d[i+NT]		+ h*d_d[i+NT];	 
Real s23 = A*s_d[i+2*NT] 	+ h*d_d[i+2*NT]; 
Real s24 = A*s_o[i]			+ h*d_o[i];		 
Real s25 = A*s_o[i+NT] 		+ h*d_o[i+NT]; 	
Real s26 = A*s_o[i+2*NT]	+ h*d_o[i+2*NT];

__syncthreads();

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

__global__ void updateRK2(Real* d, Real* o, const Real* k1d, const Real* k1o, const Real* k2d, const Real* k2o, const Real dt) {

// Set grid id
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 

const Real p1 = d[i];
const Real p2 = d[i+NT];
const Real p3 = d[i+2*NT];
const Real o1 = o[i];
const Real o2 = o[i+NT];
const Real o3 = o[i+2*NT];

const Real dp1k1 = k1d[i];			const Real dp1k2 = k2d[i];
const Real dp2k1 = k1d[i+NT];		const Real dp2k2 = k2d[i+NT];
const Real dp3k1 = k1d[i+2*NT];		const Real dp3k2 = k2d[i+2*NT];
const Real do1k1 = k1o[i];			const Real do1k2 = k2o[i];
const Real do2k1 = k1o[i+NT];		const Real do2k2 = k2o[i+NT];
const Real do3k1 = k1o[i+2*NT];		const Real do3k2 = k2o[i+2*NT];

__syncthreads(); 

// Set outputs
const Real f = Real(0.5)*dt;
d[i     ] = p1 + f*(dp1k1+dp1k2);
d[i+1*NT] = p2 + f*(dp2k1+dp2k2);
d[i+2*NT] = p3 + f*(dp3k1+dp3k2);
o[i     ] = o1 + f*(do1k1+do1k2);
o[i+1*NT] = o2 + f*(do2k1+do2k2);
o[i+2*NT] = o3 + f*(do3k1+do3k2);
}

__global__ void updateRK3(Real* d, Real* o, const Real* k1d, const Real* k1o, const Real* k2d, const Real* k2o, const Real* k3d, const Real* k3o, const Real dt) {

// Set grid id
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 

const Real p1 = d[i];
const Real p2 = d[i+NT];
const Real p3 = d[i+2*NT];
const Real o1 = o[i];
const Real o2 = o[i+NT];
const Real o3 = o[i+2*NT];

const Real dp1k1 = k1d[i];			const Real dp1k2 = k2d[i];			const Real dp1k3 = k3d[i];
const Real dp2k1 = k1d[i+NT];		const Real dp2k2 = k2d[i+NT];		const Real dp2k3 = k3d[i+NT];
const Real dp3k1 = k1d[i+2*NT];		const Real dp3k2 = k2d[i+2*NT];		const Real dp3k3 = k3d[i+2*NT];
const Real do1k1 = k1o[i];			const Real do1k2 = k2o[i];			const Real do1k3 = k3o[i];
const Real do2k1 = k1o[i+NT];		const Real do2k2 = k2o[i+NT];		const Real do2k3 = k3o[i+NT];
const Real do3k1 = k1o[i+2*NT];		const Real do3k2 = k2o[i+2*NT];		const Real do3k3 = k3o[i+2*NT];


__syncthreads(); 

// Set outputs
const Real f = dt/Real(6.);
d[i     ] = p1 + f*(dp1k1 + Real(4.)*dp1k2 + dp1k3);
d[i+1*NT] = p2 + f*(dp2k1 + Real(4.)*dp2k2 + dp2k3);
d[i+2*NT] = p3 + f*(dp3k1 + Real(4.)*dp3k2 + dp3k3);
o[i     ] = o1 + f*(do1k1 + Real(4.)*do1k2 + do1k3);
o[i+1*NT] = o2 + f*(do2k1 + Real(4.)*do2k2 + do2k3);
o[i+2*NT] = o3 + f*(do3k1 + Real(4.)*do3k2 + do3k3);
}

__global__ void updateRK4(Real* d, Real* o, const Real* k1d, const Real* k1o, const Real* k2d, const Real* k2o, const Real* k3d, const Real* k3o, const Real* k4d, const Real* k4o, const Real dt) {

// Set grid id
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 

const Real p1 = d[i];
const Real p2 = d[i+NT];
const Real p3 = d[i+2*NT];
const Real o1 = o[i];
const Real o2 = o[i+NT];
const Real o3 = o[i+2*NT];

const Real dp1k1 = k1d[i];			const Real dp1k2 = k2d[i];			const Real dp1k3 = k3d[i];			const Real dp1k4 = k4d[i];
const Real dp2k1 = k1d[i+NT];		const Real dp2k2 = k2d[i+NT];		const Real dp2k3 = k3d[i+NT];		const Real dp2k4 = k4d[i+NT];
const Real dp3k1 = k1d[i+2*NT];		const Real dp3k2 = k2d[i+2*NT];		const Real dp3k3 = k3d[i+2*NT];		const Real dp3k4 = k4d[i+2*NT];
const Real do1k1 = k1o[i];			const Real do1k2 = k2o[i];			const Real do1k3 = k3o[i];			const Real do1k4 = k4o[i];
const Real do2k1 = k1o[i+NT];		const Real do2k2 = k2o[i+NT];		const Real do2k3 = k3o[i+NT];		const Real do2k4 = k4o[i+NT];
const Real do3k1 = k1o[i+2*NT];		const Real do3k2 = k2o[i+2*NT];		const Real do3k3 = k3o[i+2*NT];		const Real do3k4 = k4o[i+2*NT];

__syncthreads(); 

// Set outputs
const Real f = dt/Real(6.);
d[i     ] = p1 + f*(dp1k1 + Real(2.)*dp1k2 + Real(2.)*dp1k3 + dp1k4);
d[i+1*NT] = p2 + f*(dp2k1 + Real(2.)*dp2k2 + Real(2.)*dp2k3 + dp2k4);
d[i+2*NT] = p3 + f*(dp3k1 + Real(2.)*dp3k2 + Real(2.)*dp3k3 + dp3k4);
o[i     ] = o1 + f*(do1k1 + Real(2.)*do1k2 + Real(2.)*do1k3 + do1k4);
o[i+1*NT] = o2 + f*(do2k1 + Real(2.)*do2k2 + Real(2.)*do2k3 + do2k4);
o[i+2*NT] = o3 + f*(do3k1 + Real(2.)*do3k2 + Real(2.)*do3k3 + do3k4);

}

//-------------------------------
// Vortex stretching & FD kernels
//-------------------------------

__constant__ Real D1C2[2] =  {-0.5f, 0.5f};
__constant__ Real D1C4[4] =  {1.f/12.f, -2.f/3.f, 2.f/3.f, -1.f/12.f};
__constant__ Real D1C6[6] =  {-1.f/60.f, 3.f/20.f, -3.f/4.f, 3.f/4.f, -3.f/20.f, 1.f/60.f};
__constant__ Real D1C8[8] =  {1.f/280.f, -4.f/105.f, 1.f/5.f, -4.f/5.f, 4.f/5.f, -1.f/5.f, 4.f/105.f, -1.f/280.f };

__constant__ Real D2C2[3] = {1.f, -2.f, 1.f};
__constant__ Real D2C4[5] =  {-1.f/12.f, 4.f/3.f, -5.f/2.f, 4.f/3.f, -1.f/12.f};
__constant__ Real D2C6[7] =  {1.f/90.f, -3.f/20.f, 3.f/2.f, -49.f/18.f, 3.f/2.f, -3.f/20.f, 1.f/90.f};
__constant__ Real D2C8[9] =  {-1.f/560.f, 8.f/315.f, -1.f/5.f, 8.f/5.f, -205.f/72.f, 8.f/5.f, -1.f/5.f, 8.f/315.f, -1.f/560.f};

// Isotropic Laplacians
// __constant__ Real L1 = 2.f/30.f, L2 = 1.f/30.f, L3 = 18.f/30.f, L4 = -136.f/30.f;    // Cocle 2008
// __constant__ Real L1 = 0.f, L2 = 1.f/6.f, L3 = 1.f/3.f, L4 = -4.f;                   // Patra 2006 - variant 2 (compact)
// __constant__ Real L1 = 1.f/30.f, L2 = 3.f/30.f, L3 = 14.f/30.f, L4 = -128.f/30.0;    // Patra 2006 - variant 5
// __constant__ Real LAP_ISO_3D[27] = {  2.f/30.f,1.f/30.f,2.f/30.f,1.f/30.f,18.f/30.f,1.f/30.f,2.f/30.f,1.f/30.f,2.f/30.f,                 // Cocle
//                                        1.f/30.f,18.f/30.f,1.f/30.f,18.f/30.f,-136.f/30.f,18.f/30.f,1.f/30.f,18.f/30.f,1.f/30.f,    
//                                        2.f/30.f,1.f/30.f,2.f/30.f,1.f/30.f,18.f/30.f,1.f/30.f,2.f/30.f,1.f/30.f,2.f/30.f};         
__constant__ Real LAP_ISO_3D[27] = {  1.f/30.f,3.f/30.f,1.f/30.f,3.f/30.f,14.f/30.f,3.f/30.f,1.f/30.f,3.f/30.f,1.f/30.f,              // Patra 2006 - variant 5
                                       3.f/30.f,14.f/30.f,3.f/30.f,14.f/30.f,14.f/30.f,14.f/30.f,3.f/30.f,14.f/30.f,3.f/30.f,   
                                       1.f/30.f,3.f/30.f,1.f/30.f,3.f/30.f,14.f/30.f,3.f/30.f,1.f/30.f,3.f/30.f,1.f/30.f};      
// __constant__ double LAP_ISO_3D[27] = {  L1,L2,L1,L2,L3,L2,L1,L2,L1,         
//                                        L2,L3,L2,L3,L4,L3,L2,L3,L2,         
//                                        L1,L2,L1,L2,L3,L2,L1,L2,L1};        

__constant__ Real  KinVisc;

template<int Halo, int Order, int NFDGrid>                    
__global__ void Shear_Stress(const Real* w, const Real* u, const int* hs, Real* grad, Real* smag) {

// Declare shared memory arrays
__shared__ Real wx[NFDGrid];
__shared__ Real wy[NFDGrid];
__shared__ Real wz[NFDGrid];
__shared__ Real ux[NFDGrid];
__shared__ Real uy[NFDGrid];
__shared__ Real uz[NFDGrid];

// Prepare relevant indices
const int gx0 = blockIdx.x*BX; 
const int gy0 = blockIdx.y*BY; 
const int gz0 = blockIdx.z*BZ; 
const int gx = threadIdx.x + gx0; 
const int gy = threadIdx.y + gy0; 
const int gz = threadIdx.z + gz0; 
const int txh = threadIdx.x + Halo;                                          // Local x id within padded grid
const int tyh = threadIdx.y + Halo;                                          // Local y id within padded grid
const int tzh = threadIdx.z + Halo;                                          // Local z id within padded grid
	
// Monolith structure
// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);

const int lid = gid(threadIdx.x,threadIdx.y,threadIdx.z,BX,BY,BZ);           // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);                             // Local id within padded block

// Fill centre volume
wx[pid] = w[bid];
wy[pid] = w[bid+NT];
wz[pid] = w[bid+2*NT];
ux[pid] = u[bid];
uy[pid] = u[bid+NT];
uz[pid] = u[bid+2*NT];
__syncthreads();

//--- Fill Halo (with coalesced index read)
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
			wx[lhid] = Real(0.);                     
			wy[lhid] = Real(0.);                     
			wz[lhid] = Real(0.);                     
			ux[lhid] = Real(0.);                     
			uy[lhid] = Real(0.);                     
			uz[lhid] = Real(0.);                     
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
   __syncthreads();
}

//------- Calculate finite differences

Real duxdx = Real(0.), duxdy = Real(0.), duxdz = Real(0.), duydx = Real(0.), duydy = Real(0.), duydz = Real(0.), duzdx = Real(0.), duzdy = Real(0.), duzdz = Real(0.);    // Nabla U
Real sgs = Real(0.);																																// Smagorinksi term
Real lx = Real(0.), ly = Real(0.), lz = Real(0.);                                                                                                     	// Laplacian of Omega
const bool mxx = (gx>(Halo-1) && gx<(NX-Halo));
const bool mxy = (gy>(Halo-1) && gy<(NY-Halo));
const bool mxz = (gz>(Halo-1) && gz<(NZ-Halo));

if (mxx && mxy && mxz){
	
	int ids;	// Dummy integer
	
	//-- Calculate velocity gradients
	if (Order==2){
		ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C2[0]*ux[ids];   duydx += D1C2[0]*uy[ids];  duzdx += D1C2[0]*uz[ids];  
		ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    duxdx += D1C2[1]*ux[ids];   duydx += D1C2[1]*uy[ids];  duzdx += D1C2[1]*uz[ids];  
		ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C2[0]*ux[ids];   duydy += D1C2[0]*uy[ids];  duzdy += D1C2[0]*uz[ids];  
		ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C2[1]*ux[ids];   duydy += D1C2[1]*uy[ids];  duzdy += D1C2[1]*uz[ids];  
		ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    duxdz += D1C2[0]*ux[ids];   duydz += D1C2[0]*uy[ids];  duzdz += D1C2[0]*uz[ids];  
		ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    duxdz += D1C2[1]*ux[ids];   duydz += D1C2[1]*uy[ids];  duzdz += D1C2[1]*uz[ids];  
	}
	if (Order==4){
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
	}
	if (Order==6){
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
	}
	if (Order==8){
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
	}

	// Scale gradients for grid size
	const Real invhx = Real(1.)/hx, invhy = Real(1.)/hy, invhz = Real(1.)/hz;
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
	const Real s11 = duxdx;
    const Real s12 = Real(0.5)*(duxdy + duydx);
    const Real s13 = Real(0.5)*(duxdz + duzdx);
    const Real s21 = s12;
    const Real s22 = duydy;
    const Real s23 = Real(0.5)*(duydz + duzdy);
    const Real s31 = s13;
    const Real s32 = s23;
    const Real s33 = duzdz;
    const Real s_ij2 = s11*s11 + s12*s12 + s13*s13 + s21*s21 + s22*s22 + s23*s23 + s31*s31 + s32*s32 + s33*s33;
	// This assumes uniform grid spacing!
    sgs = hx*hx*sqrt(Real(2.0)*s_ij2);
	

	//--- Calculate Laplacian
	if (Order==2){ 
		ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[0]*wx[ids];  ly += D2C2[0]*wy[ids];  lz += D2C2[0]*wz[ids];  
		ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[1]*wx[ids];  ly += D2C2[1]*wy[ids];  lz += D2C2[1]*wz[ids];  
		ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[2]*wx[ids];  ly += D2C2[2]*wy[ids];  lz += D2C2[2]*wz[ids];  
		ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[0]*wx[ids];  ly += D2C2[0]*wy[ids];  lz += D2C2[0]*wz[ids];  
		ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[1]*wx[ids];  ly += D2C2[1]*wy[ids];  lz += D2C2[1]*wz[ids];  
		ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[2]*wx[ids];  ly += D2C2[2]*wy[ids];  lz += D2C2[2]*wz[ids];  
		ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    lx += D2C2[0]*wx[ids];  ly += D2C2[0]*wy[ids];  lz += D2C2[0]*wz[ids];  
		ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[1]*wx[ids];  ly += D2C2[1]*wy[ids];  lz += D2C2[1]*wz[ids];  
		ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    lx += D2C2[2]*wx[ids];  ly += D2C2[2]*wy[ids];  lz += D2C2[2]*wz[ids];  
	}
	if (Order==4){ 
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
	}
	if (Order==6){
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
	}
	if (Order==8){
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
	}

	// Scale laplacian for grid size
	lx *= Real(1.)/hx/hx; 
	ly *= Real(1.)/hy/hy; 
	lz *= Real(1.)/hz/hz;

}
// Calculate outputs
const Real sx = duxdx*wx[pid] + duxdy*wy[pid] + duxdz*wz[pid] + KinVisc*lx;
const Real sy = duydx*wx[pid] + duydy*wy[pid] + duydz*wz[pid] + KinVisc*ly;
const Real sz = duzdx*wx[pid] + duzdy*wy[pid] + duzdz*wz[pid] + KinVisc*lz;

__syncthreads();

// Assign output vars
grad[bid]      = sx;
grad[bid+NT]   = sy;
grad[bid+2*NT] = sz;
smag[bid]	   = sgs;

}

//-------------------
// Diagnostics kernel
//-------------------

// Unroll telescoping sum
template<int blockSize> 
__device__ void warpReduceVector(volatile Real* sx, volatile Real* sy, volatile Real* sz, unsigned int tid) { 
	if (blockSize >= 64)   {sx[tid] += sx[tid + 32];     sy[tid] += sy[tid + 32];  sz[tid] += sz[tid + 32]; } 
	if (blockSize >= 32)   {sx[tid] += sx[tid + 16];     sy[tid] += sy[tid + 16];  sz[tid] += sz[tid + 16]; } 
	if (blockSize >= 16)   {sx[tid] += sx[tid +  8];     sy[tid] += sy[tid +  8];  sz[tid] += sz[tid +  8]; } 
	if (blockSize >= 8)    {sx[tid] += sx[tid +  4];     sy[tid] += sy[tid +  4];  sz[tid] += sz[tid +  4]; } 
	if (blockSize >= 4)    {sx[tid] += sx[tid +  2];     sy[tid] += sy[tid +  2];  sz[tid] += sz[tid +  2]; } 
	if (blockSize >= 2)    {sx[tid] += sx[tid +  1];     sy[tid] += sy[tid +  1];  sz[tid] += sz[tid +  1]; } 
} 

template<int blockSize> 
__device__ void warpReduceScalar(volatile Real* src, unsigned int tid) { 
	if (blockSize >= 64)   src[tid] += src[tid + 32]; 
	if (blockSize >= 32)   src[tid] += src[tid + 16]; 
	if (blockSize >= 16)   src[tid] += src[tid + 8]; 
	if (blockSize >= 8)    src[tid] += src[tid + 4]; 
	if (blockSize >= 4)    src[tid] += src[tid + 2]; 
	if (blockSize >= 2)    src[tid] += src[tid + 1]; 
} 

template<int blockSize>    // Issues rising here Joe
__device__ void warpReduceMag(volatile Real* src, unsigned int tid) { 
	if (blockSize >= 64)   {src[tid] = mymax(src[tid],src[tid+32]);  } 
	if (blockSize >= 32)   {src[tid] = mymax(src[tid],src[tid+16]);  } 
	if (blockSize >= 16)   {src[tid] = mymax(src[tid],src[tid+ 8]);  } 
	if (blockSize >= 8)    {src[tid] = mymax(src[tid],src[tid+ 4]);  } 
	if (blockSize >= 4)    {src[tid] = mymax(src[tid],src[tid+ 2]);  } 
	if (blockSize >= 2)    {src[tid] = mymax(src[tid],src[tid+ 1]);  } 
} 

__constant__ int NDiags = 15;          // Number of diagnostic outputs

template<int blockSize> 
__global__ void DiagnosticsKernel(const Real x1, const Real y1, const Real z1, const Real *omega, const Real *vel, Real *d) { 

// Declare shared memory arrays for diagnostic vales
__shared__ Real C[3][blockSize];      	// Circulation
__shared__ Real L[3][blockSize];      	// Linear Impulse
__shared__ Real A[3][blockSize];      	// Angular Impulse
__shared__ Real K1[blockSize];        	// Kinetic energy 1
__shared__ Real K2[blockSize];        	// Kinetic energy 2
__shared__ Real E[blockSize];         	// Enstropy
__shared__ Real H[blockSize];         	// Helicity
__shared__ Real normO[blockSize];      // Magnitude of vorticity
__shared__ Real normU[blockSize];      // Magnitude of velocity

// Specify grid ids
unsigned int tid = threadIdx.x; 
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

// Calculate global indices based on index
const int K = int(i/(BX*BY*BZ));
const int KX = int(K/(NBY*NBZ));
const int KY = int((K-KX*NBY*NBZ)/NBZ);
const int KZ = K-KX*NBY*NBZ-KY*NBZ;
const int ib = i-K*BX*BY*BZ;
const int bx = int(ib/(BY*BZ));
const int by = int((ib-bx*BY*BZ)/BZ);
const int bz = ib-bx*BY*BZ-by*BZ;
const Real Px = x1 + (KX*BX+bx)*hx; 
const Real Py = y1 + (KY*BY+by)*hy; 
const Real Pz = z1 + (KZ*BZ+bz)*hz; 		

const Real Ox = omega[i     ]; 
const Real Oy = omega[i+1*NT]; 
const Real Oz = omega[i+2*NT]; 
const Real Ux = vel[i     ]; 
const Real Uy = vel[i+1*NT]; 
const Real Uz = vel[i+2*NT]; 

__syncthreads(); 

// Linear diagnostics
C[0][tid] = Ox; 
C[1][tid] = Oy; 
C[2][tid] = Oz; 
const Real cx = Py*Oz - Pz*Oy; 
const Real cy = Pz*Ox - Px*Oz; 
const Real cz = Px*Oy - Py*Ox; 
L[0][tid] = cx*Real(0.5); 
L[1][tid] = cy*Real(0.5); 
L[2][tid] = cz*Real(0.5); 
A[0][tid] = (Py*cz - Pz*cy)/Real(3.); 
A[1][tid] = (Pz*cx - Px*cz)/Real(3.); 
A[2][tid] = (Px*cy - Py*cx)/Real(3.); 

// Quadratic diagnostics
normO[tid] = sqrt(Ox*Ox + Oy*Oy + Oz*Oz); 
normU[tid] = sqrt(Ux*Ux + Uy*Uy + Uz*Uz); 
K1[tid] = Real(0.5)*normU[tid]*normU[tid];            // Approach 1: Winckelmans
K2[tid] =  (Ux*cx + Uy*cy + Uz*cz);                // Approach 2: Liska
E[tid] = normO[tid]*normO[tid]; 
H[tid] = Ox*Ux + Oy*Uy + Oz*Uz; 

__syncthreads(); 

// // // Shear components: Private Correpondence with Gregoire Winckelmans
// // // Compute the 1-norm of the deformation tensor, It bounds the 2-norm...
// // // Nb: NablaU = [dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz];
// // Real dsu = fabs(NablaU[0][id_src]) + 0.5*fabs(NablaU[1][id_src]+NablaU[3][id_src]) + 0.5*fabs(NablaU[2][id_src]+NablaU[6][id_src]);
// // Real dsv = 0.5*fabs(NablaU[3][id_src]+NablaU[1][id_src]) + fabs(NablaU[4][id_src]) + 0.5*fabs(NablaU[5][id_src]+NablaU[7][id_src]);
// // Real dsw = 0.5*fabs(NablaU[6][id_src]+NablaU[2][id_src]) + 0.5*fabs(NablaU[7][id_src]+NablaU[5][id_src]) + fabs(NablaU[8][id_src]);
// // Real dsm = std::max(dsu,std::max(dsv,dsw));
// // if (SMax[id_dest] < dsm)  SMax[id_dest] = dsm;          // Stretching

// Initial step to accumulate the values  above  the lowest power of two... and then after this, continue with NVidia approach
unsigned int p = 2; 
while (p<blockSize)  p*=2;                      
if (p-blockSize!=0){                               // If the block has size power of 2, go straight to reduction
	p /= 2;                                     // Otherwise, increment terms
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
		normO[tid-p] = mymax(normO[tid-p],normO[tid]);       
	normU[tid-p] = mymax(normU[tid-p],normU[tid]);       
	}                                                        
}                                                               
__syncthreads(); 

// Carry out first reduction stage. I have written this as a for loop simply to improve readability
for (int k=512; k>=64; k/=2){                                   
	if (blockSize >= 2*k){                                       
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
		normO[tid] = mymax(normO[tid+k],normO[tid]);         
		normU[tid] = mymax(normU[tid+k],normU[tid]);         
	}                                                        
__syncthreads();                                         
	}                                                            
}                                                               

// Optimised unrolled final warp
if (tid < 32){ 
	warpReduceVector<blockSize>(C[0], C[1], C[2], tid);        
	warpReduceVector<blockSize>(L[0], L[1], L[2], tid);        
	warpReduceVector<blockSize>(A[0], A[1], A[2], tid);        
	warpReduceScalar<blockSize>(K1, tid);                      
	warpReduceScalar<blockSize>(K2, tid);                      
	warpReduceScalar<blockSize>(E, tid);                       
	warpReduceScalar<blockSize>(H, tid);                       
	warpReduceMag<blockSize>(normO, tid);                      
	warpReduceMag<blockSize>(normU, tid);                      
}  

// Write result for this block to global mem
// Note: This is output in such a way that the values can be transferred directly to the next reduction pass

//  if (i==0) printf(\ gridDim x %i circulation %f %f %f \\n\ , gridDim.x, C[0][0], C[0][1], C[0][2]);\n 

if (tid == 0){  
	d[ 0*gridDim.x + blockIdx.x] = C[0][0]; 
	d[ 1*gridDim.x + blockIdx.x] = C[1][0]; 
	d[ 2*gridDim.x + blockIdx.x] = C[2][0]; 
	d[ 3*gridDim.x + blockIdx.x] = L[0][0]; 
	d[ 4*gridDim.x + blockIdx.x] = L[1][0]; 
	d[ 5*gridDim.x + blockIdx.x] = L[2][0]; 
	d[ 6*gridDim.x + blockIdx.x] = A[0][0]; 
	d[ 7*gridDim.x + blockIdx.x] = A[1][0]; 
	d[ 8*gridDim.x + blockIdx.x] = A[2][0]; 
	d[ 9*gridDim.x + blockIdx.x] = K1[0]; 
	d[10*gridDim.x + blockIdx.x] = K2[0]; 
	d[11*gridDim.x + blockIdx.x] = E[0]; 
	d[12*gridDim.x + blockIdx.x] = H[0]; 
	d[13*gridDim.x + blockIdx.x] = normO[0]; 
	d[14*gridDim.x + blockIdx.x] = normU[0]; 
} 

} 

//-------------------------------------------
// Vorticity field magnitude clipping kernels
//-------------------------------------------

// Unroll telescoping sum

template<int blockSize> 
__device__ void warpReduceCount(volatile int* src, unsigned int tid) { 
	if (blockSize >= 64)   src[tid] += src[tid + 32]; 
	if (blockSize >= 32)   src[tid] += src[tid + 16]; 
	if (blockSize >= 16)   src[tid] += src[tid + 8]; 
	if (blockSize >= 8)    src[tid] += src[tid + 4]; 
	if (blockSize >= 4)    src[tid] += src[tid + 2]; 
	if (blockSize >= 2)    src[tid] += src[tid + 1]; 
} 

template<int blockSize> 
__global__ void MagnitudeFiltering_Step1(const Real *omega, Real *d) { 

// Declare shared memory arrays for diagnostic values
__shared__ Real normO[blockSize];      // Magnitude of vorticity

// Specify grid ids
unsigned int tid = threadIdx.x; 
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 

// Load values for this node into memory
const Real Ox = omega[i     ]; 
const Real Oy = omega[i+1*NT]; 
const Real Oz = omega[i+2*NT]; 
normO[tid] = sqrt(Ox*Ox + Oy*Oy + Oz*Oz); 
__syncthreads(); 

// Initial step to accumulate the values  above  the lowest power of two... and then after this, continue with NVidia approach
unsigned int p = 2; 
while (p<blockSize)  p*=2;                      
if (p-blockSize!=0){                               		
	p /= 2;                                     		
	if (tid>=p)	normO[tid-p] = mymax(normO[tid-p],normO[tid]);                                                         
}                                                               
__syncthreads(); 

// Carry out first reduction stage. I have written this as a for loop simply to improve readability
for (int k=512; k>=64; k/=2){                                   
	if (blockSize >= 2*k){                                       
		if (tid<k)	normO[tid] = mymax(normO[tid+k],normO[tid]);                                                    
		__syncthreads();                                         
	}                                                            
}                                                               


// Optimised unrolled final warp
if (tid < 32)	warpReduceMag<blockSize>(normO, tid);

if (tid == 0)	d[blockIdx.x] = normO[0]; 

} 

template<int blockSize> 
__global__ void MagnitudeFiltering_Step2(Real *omega, Real *removed_omega, int *removed_count, Real ommin) { 

// Declare shared memory arrays for diagnostic values
__shared__ Real removedOx[blockSize];      // vorticity removed x
__shared__ Real removedOy[blockSize];      // vorticity removed y
__shared__ Real removedOz[blockSize];      // vorticity removed z
__shared__ int count[blockSize];   		// Count of vortex particles with non-zero strength

// Specify grid ids
unsigned int tid = threadIdx.x; 
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 

// Load values for this node into memory
const Real Ox = omega[i     ]; 
const Real Oy = omega[i+1*NT]; 
const Real Oz = omega[i+2*NT]; 
const Real normO = sqrt(Ox*Ox + Oy*Oy + Oz*Oz); 
__syncthreads(); 

// The first step is to set the value of omega to be removed and the new value of omega

Real newOx, newOy, newOz; 
if (normO<ommin){
	removedOx[tid] = Ox;
	removedOy[tid] = Oy;
	removedOz[tid] = Oz;
	newOx = Real(0.);
	newOy = Real(0.);
	newOz = Real(0.);
	count[tid] = 0;
}  
else{
	removedOx[tid] = Real(0.);
	removedOy[tid] = Real(0.);
	removedOz[tid] = Real(0.);
	newOx = Ox;
	newOy = Oy;
	newOz = Oz;
	count[tid] = 1;
}

// Update omega array
omega[i     ] = newOx; 
omega[i+1*NT] = newOy; 
omega[i+2*NT] = newOz;
__syncthreads();

// Accumulate values of removed vorticity and number of nonzero vortex particles

// Initial step to accumulate the values  above  the lowest power of two
unsigned int p = 2; 
while (p<blockSize)  p*=2;                      
if (p-blockSize!=0){                               		
	p /= 2;                                     		
	if (tid>=p){
		removedOx[tid-p] += removedOx[tid];
		removedOy[tid-p] += removedOy[tid];
		removedOz[tid-p] += removedOz[tid];
		count[tid-p] += count[tid];
	}
}                                                               
__syncthreads(); 

// Carry out first reduction stage. I have written this as a for loop simply to improve readability
for (int k=512; k>=64; k/=2){                                   
	if (blockSize >= 2*k){                                       
		if (tid<k){
			removedOx[tid] += removedOx[tid+k];
			removedOy[tid] += removedOy[tid+k];
			removedOz[tid] += removedOz[tid+k];
			count[tid] += count[tid+k];
		}		
		__syncthreads();                                         
	}                                                            
}                                                               


// Optimised unrolled final warp
if (tid < 32){	
	warpReduceVector<blockSize>(removedOx, removedOy, removedOz, tid); 
	warpReduceCount<blockSize>(count,tid);
}

// Set outputs
if (tid == 0){	
	removed_omega[ 0*gridDim.x + blockIdx.x] = removedOx[0];
	removed_omega[ 1*gridDim.x + blockIdx.x] = removedOy[0];
	removed_omega[ 2*gridDim.x + blockIdx.x] = removedOz[0];
	removed_count[ blockIdx.x] = count[0];
} 

}

__global__ void MagnitudeFiltering_Step3(Real *omega, Real omincx, Real omincy, Real omincz) { 

// Specify grid ids
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 

// Load values for this node into memory
const Real Ox = omega[i     ]; 
const Real Oy = omega[i+1*NT]; 
const Real Oz = omega[i+2*NT]; 
const Real normO = sqrt(Ox*Ox + Oy*Oy + Oz*Oz); 
// Set new (incremented) value of vorticity
Real newOx = Real(0.);
Real newOy = Real(0.);
Real newOz = Real(0.); 
if (normO!=Real(0.)){
	newOx = Ox + omincx;
	newOy = Oy + omincy;
	newOz = Oz + omincz;
}  
__syncthreads(); 

// Update omega array
omega[i     ] = newOx; 
omega[i+1*NT] = newOy; 
omega[i+2*NT] = newOz;
}

//-------------------
// Freestream kernel
//-------------------

__global__ void AddFreestream(Real *u, const Real Ux, const Real Uy, const Real Uz) { 
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; 

u[i		] += Ux; 
u[i+1*NT] += Uy; 
u[i+2*NT] += Uz; 
}

//-------------------------------------
// Auxiliary grid communication kernels
//-------------------------------------

// This is an "interp" function as we interpolate the value of underlying grid (auxiliary grid) at the current (displaced) 
// node positions.    

template<int Halo, int BXShift, int BYShift, int BZShift, int Mapping, int NHT>                    // Pass padded grid size as template argument
__global__ void Interpolation_Aux(const Real* f1, const Real* d, const int* hs, Real* interpf1) {

__shared__ Real sx[NHT];
__shared__ Real sy[NHT];
__shared__ Real sz[NHT];

// Prepare relevant indices
const int gx0 = (blockIdx.x + BXShift)*BX; 
const int gy0 = (blockIdx.y + BYShift)*BY; 
const int gz0 = (blockIdx.z + BZShift)*BZ; 
const int gx = threadIdx.x + gx0; 
const int gy = threadIdx.y + gy0; 
const int gz = threadIdx.z + gz0; 
const int txh = threadIdx.x + Halo;                                          // Local x id within padded grid
const int tyh = threadIdx.y + Halo;                                          // Local y id within padded grid
const int tzh = threadIdx.z + Halo;                                          // Local z id within padded grid

// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);

const int lid = gid(threadIdx.x,threadIdx.y,threadIdx.z,BX,BY,BZ);           // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);                             // Local id within padded block

// Fill centre volume source arrays & mapping coeff arrays
sx[pid] = f1[bid];
sy[pid] = f1[bid+NT];
sz[pid] = f1[bid+2*NT];

// Load particle displacements
const Real dx = d[bid		];
const Real dy = d[bid+NT	];
const Real dz = d[bid+2*NT	];
__syncthreads();

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
		const bool exy = (ghy<0 || ghy>=NY);      // Is y coordinate outside of the domain?
		const bool exz = (ghz<0 || ghz>=NZ);      // Is z coordinate outside of the domain?
		if (exx || exy || exz){                    // Catch: is id outside of domain?
			sx[lhid] = Real(0.);                   
			sy[lhid] = Real(0.);                   
			sz[lhid] = Real(0.); 
		}  
		else {      
			sx[lhid] = f1[bhid     ];
			sy[lhid] = f1[bhid+1*NT];
			sz[lhid] = f1[bhid+2*NT];
		}
	}
	__syncthreads();
}

// Now carry out interpolation using shared memory
// The displacement of the particles is used to calculate which nodes shall be used for mapping.
// The interpolation coefficients for this are then calculated and used to map the source field from the shared memory arrays.
Real m1x = Real(0.), m1y = Real(0.), m1z = Real(0.);     // Interpolated values
const bool mxx = (gx>(Halo-1) && gx<(NX-1-(Halo-1)));
const bool mxy = (gy>(Halo-1) && gy<(NY-1-(Halo-1)));
const bool mxz = (gz>(Halo-1) && gz<(NZ-1-(Halo-1)));
// if (!mxx && !mxy && !mxz){
if (mxx && mxy && mxz){
			
	int iix, iiy, iiz;                                // Interpolation id
	const Real dxh = dx/hx, dyh = dy/hy, dzh = dz/hz;    // Normalized distances

	// Calculate interpolation weights
	const int H2 = Halo*2;
	Real cx[H2], cy[H2], cz[H2];			// Interpolation weights 
	int NS;								// Shift for node id
	
	// M2 interpolation
	if (Mapping==2){
		NS = 0;	
		if (dxh>=Real(0.)){	iix = txh;      mapM2(dxh,cx[0]);         	mapM2(Real(1.)-dxh,cx[1]);   }
		else {				iix = txh-1;    mapM2(Real(1.)+dxh,cx[0]);	mapM2(-dxh,cx[1]);        }

		if (dyh>=Real(0.)){	iiy = tyh;      mapM2(dyh,cy[0]);			mapM2(Real(1.)-dyh,cy[1]);   }
		else {             	iiy = tyh-1;    mapM2(Real(1.)+dyh,cy[0]);	mapM2(-dyh,cy[1]);        }

		if (dzh>=Real(0.)){	iiz = tzh;      mapM2(dzh,cz[0]);			mapM2(Real(1.)-dzh,cz[1]);   }
		else {				iiz = tzh-1;    mapM2(Real(1.)+dzh,cz[0]);	mapM2(-dzh,cz[1]);        }
	}
	
	// M4 interpolation
	if (Mapping==4){
		NS = -1;	
		if (dxh>=Real(0.)){   iix = txh;      mapM4(Real(1.)+dxh,cx[0]);  mapM4(dxh,cx[1]);    		mapM4(Real(1.)-dxh,cx[2]);  	mapM4(Real(2.)-dxh,cx[3]);}
		else {             iix = txh-1;    mapM4(Real(2.)+dxh,cx[0]);  mapM4(Real(1.)+dxh,cx[1]);   	mapM4(-dxh,cx[2]);  		mapM4(Real(1.)-dxh,cx[3]);}

		if (dyh>=Real(0.)){   iiy = tyh;      mapM4(Real(1.)+dyh,cy[0]);  mapM4(dyh,cy[1]);   		mapM4(Real(1.)-dyh,cy[2]);  	mapM4(Real(2.)-dyh,cy[3]);}
		else {             iiy = tyh-1;    mapM4(Real(2.)+dyh,cy[0]);  mapM4(Real(1.)+dyh,cy[1]);   	mapM4(-dyh,cy[2]);  		mapM4(Real(1.)-dyh,cy[3]);}

		if (dzh>=Real(0.)){   iiz = tzh;      mapM4(Real(1.)+dzh,cz[0]);  mapM4(dzh,cz[1]);   		mapM4(Real(1.)-dzh,cz[2]);  	mapM4(Real(2.)-dzh,cz[3]);}
		else {             iiz = tzh-1;    mapM4(Real(2.)+dzh,cz[0]);  mapM4(Real(1.)+dzh,cz[1]);   	mapM4(-dzh,cz[2]);  		mapM4(Real(1.)-dzh,cz[3]);}     
	}
	
	// M4' interpolation
	if (Mapping==42){
		NS = -1;	
		if (dxh>=Real(0.)){   iix = txh;      mapM4D(Real(1.)+dxh,cx[0]);  mapM4D(dxh,cx[1]);    		mapM4D(Real(1.)-dxh,cx[2]);  	mapM4D(Real(2.)-dxh,cx[3]);}
		else {             iix = txh-1;    mapM4D(Real(2.)+dxh,cx[0]);  mapM4D(Real(1.)+dxh,cx[1]);   mapM4D(-dxh,cx[2]);  		mapM4D(Real(1.)-dxh,cx[3]);}

		if (dyh>=Real(0.)){   iiy = tyh;      mapM4D(Real(1.)+dyh,cy[0]);  mapM4D(dyh,cy[1]);   		mapM4D(Real(1.)-dyh,cy[2]);  	mapM4D(Real(2.)-dyh,cy[3]);}
		else {             iiy = tyh-1;    mapM4D(Real(2.)+dyh,cy[0]);  mapM4D(Real(1.)+dyh,cy[1]);   mapM4D(-dyh,cy[2]);  		mapM4D(Real(1.)-dyh,cy[3]);}

		if (dzh>=Real(0.)){   iiz = tzh;      mapM4D(Real(1.)+dzh,cz[0]);  mapM4D(dzh,cz[1]);   		mapM4D(Real(1.)-dzh,cz[2]);  	mapM4D(Real(2.)-dzh,cz[3]);}
		else {             iiz = tzh-1;    mapM4D(Real(2.)+dzh,cz[0]);  mapM4D(Real(1.)+dzh,cz[1]);   mapM4D(-dzh,cz[2]); 		mapM4D(Real(1.)-dzh,cz[3]);}
	}
	
	// M6' interpolation
	if (Mapping==6){
		NS = -2;	
		if (dxh>=Real(0.)){   iix = txh;      mapM6D(Real(2.)+dxh,cx[0]);  mapM6D(Real(1.)+dxh,cx[1]);  mapM6D(      dxh,cx[2]);  mapM6D(Real(1.)-dxh,cx[3]); 	mapM6D(Real(2.)-dxh,cx[4]);  	mapM6D(Real(3.)-dxh,cx[5]);}
		else {             iix = txh-1;    mapM6D(Real(3.)+dxh,cx[0]);  mapM6D(Real(2.)+dxh,cx[1]);  mapM6D(Real(1.)+dxh,cx[2]);  mapM6D(     -dxh,cx[3]); 	mapM6D(Real(1.)-dxh,cx[4]);  	mapM6D(Real(2.)-dxh,cx[5]);}
	
		if (dyh>=Real(0.)){   iiy = tyh;      mapM6D(Real(2.)+dyh,cy[0]);  mapM6D(Real(1.)+dyh,cy[1]);  mapM6D(      dyh,cy[2]);  mapM6D(Real(1.)-dyh,cy[3]); 	mapM6D(Real(2.)-dyh,cy[4]);  	mapM6D(Real(3.)-dyh,cy[5]);}
		else {             iiy = tyh-1;    mapM6D(Real(3.)+dyh,cy[0]);  mapM6D(Real(2.)+dyh,cy[1]);  mapM6D(Real(1.)+dyh,cy[2]);  mapM6D(     -dyh,cy[3]); 	mapM6D(Real(1.)-dyh,cy[4]);  	mapM6D(Real(2.)-dyh,cy[5]);}
	
		if (dzh>=Real(0.)){   iiz = tzh;      mapM6D(Real(2.)+dzh,cz[0]);  mapM6D(Real(1.)+dzh,cz[1]);  mapM6D(      dzh,cz[2]);  mapM6D(Real(1.)-dzh,cz[3]); 	mapM6D(Real(2.)-dzh,cz[4]);  	mapM6D(Real(3.)-dzh,cz[5]);}
		else {             iiz = tzh-1;    mapM6D(Real(3.)+dzh,cz[0]);  mapM6D(Real(2.)+dzh,cz[1]);  mapM6D(Real(1.)+dzh,cz[2]);  mapM6D(     -dzh,cz[3]); 	mapM6D(Real(1.)-dzh,cz[4]);  	mapM6D(Real(2.)-dzh,cz[5]);}
	}
	
	// printf(\'Global indices %iids %i %i %i \\n\', iix, iiy, iiz);\n
	
	// Carry out interpolation
	for (int i=0; i<H2; i++){                                       
	   for (int j=0; j<H2; j++){                                    
		   for (int k=0; k<H2; k++){                                
			   const int idsx =  iix + NS + i;                           
			   const int idsy =  iiy + NS + j;                           
			   const int idsz =  iiz + NS + k;                           
			   const int ids =  gid(idsx,idsy,idsz,NFDX,NFDY,NFDZ); 
			   const Real fac =  cx[i]*cy[j]*cz[k];                   
			   m1x =  fastma(fac,sx[ids],m1x);                       
			   m1y =  fastma(fac,sy[ids],m1y);                       
			   m1z =  fastma(fac,sz[ids],m1z);  
			   
		   }                                                       
	   }                                                           
	}
}
__syncthreads();

// Write output
interpf1[bid     ] += m1x;
interpf1[bid+1*NT] += m1y;
interpf1[bid+2*NT] += m1z;

}

template<int sax, int say, int saz, int nbxaux, int nbyaux, int nbzaux>    
__global__ void MapFromAuxiliaryVPMGrid(const Real* auxGrid, Real* Grid) {

	// Calculate source index (block-ordered)
	const int sgx = threadIdx.x + blockIdx.x*BX; 
	const int sgy = threadIdx.y + blockIdx.y*BY; 
	const int sgz = threadIdx.z + blockIdx.z*BZ; 
	const int sid = gidb(sgx,sgy,sgz,nbxaux,nbyaux,nbzaux);		
	const int NTA = nbxaux*nbyaux*nbzaux*BT;	

	// Calculate source index (block-ordered)
	const int rid = gidb(sgx+sax, sgy+say, sgz+saz);

	Grid[rid	 ] = auxGrid[sid  	   ];
	Grid[rid+1*NT] = auxGrid[sid+1*NTA];
	Grid[rid+2*NT] = auxGrid[sid+2*NTA];	
}

template<int sax, int say, int saz, int nbxaux, int nbyaux, int nbzaux>
__global__ void MapFromAuxiliaryGridVPM_toUnbounded(const Real* auxGrid, Real* dst1, Real* dst2, Real* dst3) {

	// Calculate source index (block)
	const int sgx = threadIdx.x + blockIdx.x*BX; 
	const int sgy = threadIdx.y + blockIdx.y*BY; 
	const int sgz = threadIdx.z + blockIdx.z*BZ; 
	const int did = gid(sgx+sax, sgy+say, sgz+saz, 2*NX, 2*NY, 2*NZ);		// Monolithic id of receiving grid
	const int bid = gidb(sgx,sgy,sgz,nbxaux,nbyaux,nbzaux);
	const int NTA = nbxaux*nbyaux*nbzaux*BT;	
	
	// printf("Grid constants: sax say saz %i %i %i, nbxaux nbyaux nbzaux %i %i %i,  \n", sax, say, saz, nbxaux, nbyaux, nbzaux);
	// printf("Grid constants: NTA %i, NT %i, sgx sgy sgz %i %i %i, did %i, bid %i,  \n", NTA, NT, sgx, sgy, sgz, did, bid);

	// __syncthreads();
	
	dst1[did] += auxGrid[bid      ];
    dst2[did] += auxGrid[bid+1*NTA];
    dst3[did] += auxGrid[bid+2*NTA];
}

// This function maps one box array wiht block dimensions NBX,NBY,NBZ, onto a second block array with block dimensions nbxaux nbyaux nbzaux
template<int sax, int say, int saz, int nbxaux, int nbyaux, int nbzaux>    
__global__ void MaptoAuxiliaryVPMGrid(const Real* Grid, Real* auxGrid) {

	// Calculate Receiver index (block-ordered)
	const int rgx = threadIdx.x + blockIdx.x*BX; 
	const int rgy = threadIdx.y + blockIdx.y*BY; 
	const int rgz = threadIdx.z + blockIdx.z*BZ; 	
	const int rid = gidb(rgx,rgy,rgz,nbxaux,nbyaux,nbzaux);		
	const int NTA = nbxaux*nbyaux*nbzaux*BT;	

	// Calculate source index (block-ordered)
	const int sid = gidb(rgx+sax, rgy+say, rgz+saz);

	const Real a = auxGrid[rid 	   ];
	const Real b = auxGrid[rid + 1*NTA];
	const Real c = auxGrid[rid + 2*NTA];
	// printf("Grid constants: NTA %i, NT %i, sgx sgy sgz %i %i %i, rid %i, sid %i,  \n", NTA, NT, rgx, rgy, rgz, rid, sid);
	
	__syncthreads();

	auxGrid[rid 	   ] = Grid[sid       ] - a;
	auxGrid[rid + 1*NTA] = Grid[sid + 1*NT] - b;
	auxGrid[rid + 2*NTA] = Grid[sid + 2*NT] - c;
}

//----------------------------------
// Block-data interpolation routines
//----------------------------------


template<int Halo, int Mapping, int NHT>                    		// Pass padded grid size as template argument
__global__ void Interp_Block(	const Real* fd, 									// Source grid
								const Real* dsx, const Real* dsy, const Real* dsz, 		// Positions of particles to be mapped (in local block coordinate system)
								// const int *NS, 									// NS: Number of source points for this block
								const int *blX, const int *blY, const int *blZ, // Block indices
								const int* hs, 									// Halo indices
								Real* ux, Real* uy, Real* uz) 							// Interpolated values from grid
{
// Declare shared memory vars (velocity field in shared memory)
__shared__ Real sx[NHT];
__shared__ Real sy[NHT];
__shared__ Real sz[NHT];

// We are still executing the blocks with blockdim [BX, BY, BZ], however are are using the Griddims in a 1d sense.
// So we need to modify the x-dim based on the number of active blocks. 

// Prepare relevant indices
const int BlockID = blockIdx.x;
const int gx0 = blX[BlockID]*BX; 
const int gy0 = blY[BlockID]*BY; 
const int gz0 = blZ[BlockID]*BZ; 
const int gx = threadIdx.x + gx0; 
const int gy = threadIdx.y + gy0; 
const int gz = threadIdx.z + gz0; 
const int txh = threadIdx.x + Halo;                                         // Local x id within padded grid
const int tyh = threadIdx.y + Halo;                                         // Local y id within padded grid
const int tzh = threadIdx.z + Halo;                                         // Local z id within padded grid
// const int tNS = NS[blockIdx.x];												// How many source points shalll be loaded for this block?	 

// Monolith structure
// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);
const int lid = gid(threadIdx.x,threadIdx.y,threadIdx.z,BX,BY,BZ);           // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);                             // Local id within padded block


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

__syncthreads();

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
			sx[lhid] = Real(0.);
			sy[lhid] = Real(0.);
			sz[lhid] = Real(0.);                  
		}  
		else {    
			sx[lhid] = fd[bhid       ];
			sy[lhid] = fd[bhid + 1*NT];
			sz[lhid] = fd[bhid + 2*NT];  
		}
	}
	__syncthreads();
} 

//------------------------------------------------------------------------------
// Step 2) Interpolate values from grid
//------------------------------------------------------------------------------

Real m1x = Real(0.), m1y = Real(0.), m1z = Real(0.);     // Interpolated values

// Calculate relative indices and displacement factors
Real pmx, pmy, pmz; 
if (Mapping==2){
	pmx = Real(0.5)*hx + pX;
	pmy = Real(0.5)*hy + pY;
	pmz = Real(0.5)*hz + pZ;
}
if (Mapping==4){
	pmx = Real(1.5)*hx + pX;
	pmy = Real(1.5)*hy + pY;
	pmz = Real(1.5)*hz + pZ;
}
if (Mapping==42){
	pmx = Real(1.5)*hx + pX;
	pmy = Real(1.5)*hy + pY;
	pmz = Real(1.5)*hz + pZ;
}

const int iix = int(pmx/hx);
const int iiy = int(pmy/hy);
const int iiz = int(pmz/hz);
const Real dxh = (pmx-hx*iix)/hx;
const Real dyh = (pmy-hy*iiy)/hy;
const Real dzh = (pmz-hz*iiz)/hz;

// Calculate interpolation weights
const int H2 = Halo*2;
Real cx[H2], cy[H2], cz[H2];			// Interpolation weights 
int NS;								// Shift for node id

// M2 interpolation
if (Mapping==2){
	NS = 0;	
	mapM2(dxh,cx[0]);         mapM2(Real(1.)-dxh,cx[1]);
	mapM2(dyh,cy[0]);         mapM2(Real(1.)-dyh,cy[1]); 
	mapM2(dzh,cz[0]);         mapM2(Real(1.)-dzh,cz[1]); 
}

// M4 interpolation
if (Mapping==4){
	NS = -1;	
	mapM4(Real(1.)+dxh,cx[0]);  mapM4(dxh,cx[1]);    		mapM4(Real(1.)-dxh,cx[2]);  	mapM4(Real(2.)-dxh,cx[3]);
	mapM4(Real(1.)+dyh,cy[0]);  mapM4(dyh,cy[1]);   		mapM4(Real(1.)-dyh,cy[2]);  	mapM4(Real(2.)-dyh,cy[3]);
	mapM4(Real(1.)+dzh,cz[0]);  mapM4(dzh,cz[1]);   		mapM4(Real(1.)-dzh,cz[2]);  	mapM4(Real(2.)-dzh,cz[3]);
}

// M4' interpolation
if (Mapping==42){
	NS = -1;	
	mapM4D(Real(1.)+dxh,cx[0]);  mapM4D(dxh,cx[1]);    	mapM4D(Real(1.)-dxh,cx[2]);  	mapM4D(Real(2.)-dxh,cx[3]);
	mapM4D(Real(1.)+dyh,cy[0]);  mapM4D(dyh,cy[1]);   		mapM4D(Real(1.)-dyh,cy[2]);  	mapM4D(Real(2.)-dyh,cy[3]);
	mapM4D(Real(1.)+dzh,cz[0]);  mapM4D(dzh,cz[1]);   		mapM4D(Real(1.)-dzh,cz[2]);  	mapM4D(Real(2.)-dzh,cz[3]);
}

// M6' interpolation
if (Mapping==6){
	NS = -2;	
	mapM6D(Real(2.)+dxh,cx[0]);  mapM6D(Real(1.)+dxh,cx[1]);  mapM6D(dxh,cx[2]);  mapM6D(Real(1.)-dxh,cx[3]); 	mapM6D(Real(2.)-dxh,cx[4]);  	mapM6D(Real(3.)-dxh,cx[5]);
	mapM6D(Real(2.)+dyh,cy[0]);  mapM6D(Real(1.)+dyh,cy[1]);  mapM6D(dyh,cy[2]);  mapM6D(Real(1.)-dyh,cy[3]); 	mapM6D(Real(2.)-dyh,cy[4]);  	mapM6D(Real(3.)-dyh,cy[5]);
	mapM6D(Real(2.)+dzh,cz[0]);  mapM6D(Real(1.)+dzh,cz[1]);  mapM6D(dzh,cz[2]);  mapM6D(Real(1.)-dzh,cz[3]); 	mapM6D(Real(2.)-dzh,cz[4]);  	mapM6D(Real(3.)-dzh,cz[5]);
}

// Carry out interpolation
for (int i=0; i<H2; i++){                                       
	for (int j=0; j<H2; j++){                                    
		for (int k=0; k<H2; k++){                                
			const int idsx =  iix + NS + i;                           
			const int idsy =  iiy + NS + j;                           
			const int idsz =  iiz + NS + k;                           
			const int ids =  gid(idsx,idsy,idsz,NFDX,NFDY,NFDZ); 
			const Real fac =  cx[i]*cy[j]*cz[k];                   
			m1x =  fastma(fac,sx[ids],m1x);                       
			m1y =  fastma(fac,sy[ids],m1y);                       
			m1z =  fastma(fac,sz[ids],m1z);
			
	   }                                                       
	}                                                           
}     

__syncthreads();

//------------------------------------------------------------------------------
// Step 3) Transfer interpolated values back to array
//------------------------------------------------------------------------------

ux[llid] = m1x;
uy[llid] = m1y;
uz[llid] = m1z;
}

//----------------------
// Visualisation kernels 
//----------------------
            
__global__ void ExtractPlaneX(const Real* f, Real* ux, Real* uy, Real* uz, const int PX) {

// Source and receiver indices
const int sy = blockIdx.x*BX + threadIdx.x; 
const int sz = blockIdx.y*BY + threadIdx.y;
const int sid = gidb(PX,sy,sz);		// Assume block-style data ordering
const int rid = sy*NZ + sz;			// Monolithic (output) ordering

// Retrieve source
ux[rid] = f[sid     ];
uy[rid] = f[sid+1*NT];
uz[rid] = f[sid+2*NT];
}
                
__global__ void ExtractPlaneY(const Real* f, Real* p, const int PY) {

// Source and receiver indices
const int sx = blockIdx.x*BX + threadIdx.x; 
const int sz = blockIdx.y*BZ + threadIdx.y;
const int sid = gidb(sx,PY,sz);		// Assume block-style data ordering
const int rid = sx*NZ + sz;			// Monolithic (output) ordering

// Retrieve source
const Real vx = f[sid     ];
const Real vy = f[sid+1*NT];
const Real vz = f[sid+2*NT];
const Real omag = sqrt(vx*vx + vy*vy + vz*vz);

__syncthreads();

// Write output
p[rid] = omag;
}

//-------------------------------
// Kernels for turbulence schemes
//-------------------------------

template<int Halo, int Order, int NFDGrid>   
__global__ void Laplacian_Operator(const Real *O, const int* hs, Real *L) { 

// Declare shared memory arrays
__shared__ Real wx[NFDGrid];
__shared__ Real wy[NFDGrid];
__shared__ Real wz[NFDGrid];

// Prepare relevant indices
const int gx0 = blockIdx.x*BX; 
const int gy0 = blockIdx.y*BY; 
const int gz0 = blockIdx.z*BZ; 
const int gx = threadIdx.x + gx0; 
const int gy = threadIdx.y + gy0; 
const int gz = threadIdx.z + gz0; 
const int txh = threadIdx.x + Halo;                                          // Local x id within padded grid
const int tyh = threadIdx.y + Halo;                                          // Local y id within padded grid
const int tzh = threadIdx.z + Halo;                                          // Local z id within padded grid
	
// Monolith structure
// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);

const int lid = gid(threadIdx.x,threadIdx.y,threadIdx.z,BX,BY,BZ);           // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);                             // Local id within padded block

// Fill centre volume
wx[pid] = O[bid];
wy[pid] = O[bid+NT];
wz[pid] = O[bid+2*NT];

__syncthreads();

//--- Fill Halo (with coalesced index read)
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
			wx[lhid] = Real(0.);                     
			wy[lhid] = Real(0.);                     
			wz[lhid] = Real(0.);                       
		}  
		else {      
			wx[lhid] = O[bhid     ];                     
			wy[lhid] = O[bhid+1*NT];                     
			wz[lhid] = O[bhid+2*NT];                      
		}  
   }                                           
   __syncthreads();
}

//------- Calculate finite differences

Real lx = Real(0.), ly = Real(0.), lz = Real(0.);                                                                                                         // Laplacian of Omega
const bool mxx = (gx>(Halo-1) && gx<(NX-Halo));
const bool mxy = (gy>(Halo-1) && gy<(NY-Halo));
const bool mxz = (gz>(Halo-1) && gz<(NZ-Halo));

if (mxx && mxy && mxz){
	
	int ids;	// Dummy integer
	  
	//--- Calculate Laplacian
	if (Order==2){ 
		ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[0]*wx[ids];  ly += D2C2[0]*wy[ids];  lz += D2C2[0]*wz[ids];  
		ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[1]*wx[ids];  ly += D2C2[1]*wy[ids];  lz += D2C2[1]*wz[ids];  
		ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[2]*wx[ids];  ly += D2C2[2]*wy[ids];  lz += D2C2[2]*wz[ids];  
		ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[0]*wx[ids];  ly += D2C2[0]*wy[ids];  lz += D2C2[0]*wz[ids];  
		ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[1]*wx[ids];  ly += D2C2[1]*wy[ids];  lz += D2C2[1]*wz[ids];  
		ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[2]*wx[ids];  ly += D2C2[2]*wy[ids];  lz += D2C2[2]*wz[ids];  
		ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    lx += D2C2[0]*wx[ids];  ly += D2C2[0]*wy[ids];  lz += D2C2[0]*wz[ids];  
		ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[1]*wx[ids];  ly += D2C2[1]*wy[ids];  lz += D2C2[1]*wz[ids];  
		ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    lx += D2C2[2]*wx[ids];  ly += D2C2[2]*wy[ids];  lz += D2C2[2]*wz[ids];  
	}
	if (Order==4){ 
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
	}
	
	if (Order==6){
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
	}
	if (Order==8){
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
	}

	// Scale laplacian for grid size
	lx *= Real(1.)/hx/hx; 
	ly *= Real(1.)/hy/hy; 
	lz *= Real(1.)/hz/hz;
}

__syncthreads();

// Assign output vars
L[bid]      = lx;
L[bid+NT]   = ly;
L[bid+2*NT] = lz;

} 

template<int Halo, int Order, int NFDGrid>   
__global__ void Hyperviscosity_Operator(const Real *L, const int* hs, const Real C_hv, Real *dodt) { 

// Declare shared memory arrays
__shared__ Real wx[NFDGrid];
__shared__ Real wy[NFDGrid];
__shared__ Real wz[NFDGrid];

// Prepare relevant indices
const int gx0 = blockIdx.x*BX; 
const int gy0 = blockIdx.y*BY; 
const int gz0 = blockIdx.z*BZ; 
const int gx = threadIdx.x + gx0; 
const int gy = threadIdx.y + gy0; 
const int gz = threadIdx.z + gz0; 
const int txh = threadIdx.x + Halo;                                          // Local x id within padded grid
const int tyh = threadIdx.y + Halo;                                          // Local y id within padded grid
const int tzh = threadIdx.z + Halo;                                          // Local z id within padded grid
	
// Monolith structure
// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);

const int lid = gid(threadIdx.x,threadIdx.y,threadIdx.z,BX,BY,BZ);           // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);                             // Local id within padded block

// Fill centre volume
wx[pid] = L[bid];
wy[pid] = L[bid+NT];
wz[pid] = L[bid+2*NT];

__syncthreads();

//-- Fill Halo (with coalesced index read)
for (int i=0; i<NHIT; i++){           
   const int hid = BT*i + lid;  
   const int hsx = hs[hid];                        // global x-shift relative to position
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
			wx[lhid] = Real(0.);                     
			wy[lhid] = Real(0.);                     
			wz[lhid] = Real(0.);                       
		}  
		else {      
			wx[lhid] = L[bhid     ];                     
			wy[lhid] = L[bhid+1*NT];                     
			wz[lhid] = L[bhid+2*NT];                      
		}  
   }                                           
   __syncthreads();
}

//------- Calculate finite differences

Real lx = Real(0.), ly = Real(0.), lz = Real(0.);                                                                                                         // Laplacian of Omega
const bool mxx = (gx>(Halo-1) && gx<(NX-Halo));
const bool mxy = (gy>(Halo-1) && gy<(NY-Halo));
const bool mxz = (gz>(Halo-1) && gz<(NZ-Halo));

if (mxx && mxy && mxz){
	
	int ids;	// Dummy integer
	  
	//--- Calculate Laplacian
	if (Order==2){ 
		ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[0]*wx[ids];  ly += D2C2[0]*wy[ids];  lz += D2C2[0]*wz[ids];  
		ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[1]*wx[ids];  ly += D2C2[1]*wy[ids];  lz += D2C2[1]*wz[ids];  
		ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[2]*wx[ids];  ly += D2C2[2]*wy[ids];  lz += D2C2[2]*wz[ids];  
		ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[0]*wx[ids];  ly += D2C2[0]*wy[ids];  lz += D2C2[0]*wz[ids];  
		ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[1]*wx[ids];  ly += D2C2[1]*wy[ids];  lz += D2C2[1]*wz[ids];  
		ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[2]*wx[ids];  ly += D2C2[2]*wy[ids];  lz += D2C2[2]*wz[ids];  
		ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    lx += D2C2[0]*wx[ids];  ly += D2C2[0]*wy[ids];  lz += D2C2[0]*wz[ids];  
		ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lx += D2C2[1]*wx[ids];  ly += D2C2[1]*wy[ids];  lz += D2C2[1]*wz[ids];  
		ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    lx += D2C2[2]*wx[ids];  ly += D2C2[2]*wy[ids];  lz += D2C2[2]*wz[ids];  
	}
 	if (Order==4){ 
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
 	}
 	if (Order==6){
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
 	}
 	if (Order==8){
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
 	}

	// Scale laplacian for grid size (the h*h factor comes from the fact that the laplacian was calculated with the h2 term)
	lx *= -C_hv*hx*hx;
    ly *= -C_hv*hy*hy;
    lz *= -C_hv*hz*hz;
}

__syncthreads();

// Assign output vars
dodt[bid]      += lx;
dodt[bid+NT]   += ly;
dodt[bid+2*NT] += lz;

}


template<int Halo, int NFDGrid>   
__global__ void SubGrid_DiscFilter(const Real* Om, const int* hs, const Real *O, Real *filtO, int option) { 

// Declare shared memory arrays
__shared__ Real ox[NFDGrid];
__shared__ Real oy[NFDGrid];
__shared__ Real oz[NFDGrid];

// Prepare relevant indices
const int gx0 = blockIdx.x*BX; 
const int gy0 = blockIdx.y*BY; 
const int gz0 = blockIdx.z*BZ; 
const int gx = threadIdx.x + gx0; 
const int gy = threadIdx.y + gy0; 
const int gz = threadIdx.z + gz0; 
const int txh = threadIdx.x + Halo;                                          // Local x id within padded grid
const int tyh = threadIdx.y + Halo;                                          // Local y id within padded grid
const int tzh = threadIdx.z + Halo;                                          // Local z id within padded grid
	
// Monolith structure
// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);

const int lid = gid(threadIdx.x,threadIdx.y,threadIdx.z,BX,BY,BZ);           // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);                             // Local id within padded block

// Fill centre volume
ox[pid] = O[bid		];
oy[pid] = O[bid+1*NT];
oz[pid] = O[bid+2*NT];

// Extract vorticity here 
const Real Omx = Om[bid	 ];
const Real Omy = Om[bid+1*NT];
const Real Omz = Om[bid+2*NT];

__syncthreads();

//--- Fill Halo (with coalesced index read)
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
			ox[lhid] = Real(0.);                     
			oy[lhid] = Real(0.);                     
			oz[lhid] = Real(0.);                       
		}  
		else {      
			ox[lhid] = O[bhid     ];                     
			oy[lhid] = O[bhid+1*NT];                     
			oz[lhid] = O[bhid+2*NT];                      
		}  
   }                                           
   __syncthreads();
}

//------- Calculate discrete filter

Real fx = Real(0.), fy = Real(0.), fz = Real(0.);                                                                                                         // Laplacian of Omega
const bool mxx = (gx>(Halo-1) && gx<(NX-Halo));
const bool mxy = (gy>(Halo-1) && gy<(NY-Halo));
const bool mxz = (gz>(Halo-1) && gz<(NZ-Halo));

if (mxx && mxy && mxz){

	switch (option){
	
		case (0): {		// Carry out discrete filtering operation in x direction
			fx = Real(0.25)*ox[gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.5)*ox[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.25)*ox[gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)];
			fy = Real(0.25)*oy[gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.5)*oy[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.25)*oy[gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)];
			fz = Real(0.25)*oz[gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.5)*oz[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.25)*oz[gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)];
			break;
		}
		
		case (1): {		// Carry out discrete filtering operation in y direction
			fx = Real(0.25)*ox[gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.5)*ox[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.25)*ox[gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ)];
			fy = Real(0.25)*oy[gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.5)*oy[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.25)*oy[gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ)];
			fz = Real(0.25)*oz[gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.5)*oz[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.25)*oz[gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ)];
			break;
		}
		case (2): {		// Carry out discrete filtering operation in z direction
			fx = Real(0.25)*ox[gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ)] + Real(0.5)*ox[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.25)*ox[gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ)];
			fy = Real(0.25)*oy[gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ)] + Real(0.5)*oy[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.25)*oy[gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ)];
			fz = Real(0.25)*oz[gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ)] + Real(0.5)*oz[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.25)*oz[gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ)];
			break;
		}
		case (3): {		// Subtract filtered scale 
			fx = Omx - ox[pid];
			fy = Omy - oy[pid];
			fz = Omz - oz[pid];
			break;
		}
		default: {break;}
	}

}

__syncthreads();

// Assign output vars
filtO[bid]      = fx;
filtO[bid+NT]   = fy;
filtO[bid+2*NT] = fz;

}  

template<int Halo, int NFDGrid>   
__global__ void SubGrid_DiscFilter2(const Real* Om, const int* hs, const Real *O, Real *filtO, int option) { 

// Declare shared memory arrays
__shared__ Real ox[NFDGrid];
__shared__ Real oy[NFDGrid];
__shared__ Real oz[NFDGrid];

// Prepare relevant indices
const int gx0 = blockIdx.x*BX; 
const int gy0 = blockIdx.y*BY; 
const int gz0 = blockIdx.z*BZ; 
const int gx = threadIdx.x + gx0; 
const int gy = threadIdx.y + gy0; 
const int gz = threadIdx.z + gz0; 
const int txh = threadIdx.x + Halo;                                          // Local x id within padded grid
const int tyh = threadIdx.y + Halo;                                          // Local y id within padded grid
const int tzh = threadIdx.z + Halo;                                          // Local z id within padded grid
	
// Monolith structure
// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);

const int lid = gid(threadIdx.x,threadIdx.y,threadIdx.z,BX,BY,BZ);           // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);                             // Local id within padded block

// Fill centre volume
ox[pid] = O[bid		];
oy[pid] = O[bid+1*NT];
oz[pid] = O[bid+2*NT];

// Extract vorticity here 
const Real Omx = Om[bid	 ];
const Real Omy = Om[bid+1*NT];
const Real Omz = Om[bid+2*NT];

__syncthreads();

//--- Fill Halo (with coalesced index read)
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
			ox[lhid] = Real(0.);                     
			oy[lhid] = Real(0.);                     
			oz[lhid] = Real(0.);                       
		}  
		else {      
			ox[lhid] = O[bhid     ];                     
			oy[lhid] = O[bhid+1*NT];                     
			oz[lhid] = O[bhid+2*NT];                      
		}  
   }                                           
   __syncthreads();
}

//------- Calculate discrete filter

Real fx = Real(0.), fy = Real(0.), fz = Real(0.);                                                                                                         // Laplacian of Omega
const bool mxx = (gx>(Halo-1) && gx<(NX-Halo));
const bool mxy = (gy>(Halo-1) && gy<(NY-Halo));
const bool mxz = (gz>(Halo-1) && gz<(NZ-Halo));

if (mxx && mxy && mxz){

	switch (option){
	
		case (0): {		// Carry out discrete filtering operation in x direction
			// X-sweep
			fx = Real(0.25)*ox[gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.5)*ox[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.25)*ox[gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)];
			fy = Real(0.25)*oy[gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.5)*oy[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.25)*oy[gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)];
			fz = Real(0.25)*oz[gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.5)*oz[gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.25)*oz[gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ)];
			
			// Y-sweep
			Real fx2 = Real(0.25)*ox[gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.5)*fx + Real(0.25)*ox[gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ)];
			Real fy2 = Real(0.25)*oy[gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.5)*fy + Real(0.25)*oy[gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ)];
			Real fz2 = Real(0.25)*oz[gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ)] + Real(0.5)*fz + Real(0.25)*oz[gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ)];
			
			// Z-sweep
			Real fx3 = Real(0.25)*ox[gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ)] + Real(0.5)*fx2 + Real(0.25)*ox[gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ)];
			Real fy3 = Real(0.25)*oy[gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ)] + Real(0.5)*fx2 + Real(0.25)*oy[gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ)];
			Real fz3 = Real(0.25)*oz[gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ)] + Real(0.5)*fx2 + Real(0.25)*oz[gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ)];
			
			fx = fx3;
			fy = fx3;
			fz = fx3;
			
			break;
		}
		
		case (1): {		// Subtract filtered scale 
			fx = Omx - ox[pid];
			fy = Omy - oy[pid];
			fz = Omz - oz[pid];
			break;
		}
		default: {break;}
	}

}

__syncthreads();

// Assign output vars
filtO[bid]      = fx;
filtO[bid+NT]   = fy;
filtO[bid+2*NT] = fz;

}  



template<int Halo, int Order, int NFDGrid>   
__global__ void RVM_turbulentstress(const Real* fO, const Real* sgs, const int* hs, const Real smag, Real *dOmdt) { 

// Declare shared memory arrays
__shared__ Real gsx[NFDGrid];
__shared__ Real gsy[NFDGrid];
__shared__ Real gsz[NFDGrid];
__shared__ Real sg[NFDGrid];

// Prepare relevant indices
const int gx0 = blockIdx.x*BX; 
const int gy0 = blockIdx.y*BY; 
const int gz0 = blockIdx.z*BZ; 
const int gx = threadIdx.x + gx0; 
const int gy = threadIdx.y + gy0; 
const int gz = threadIdx.z + gz0; 
const int txh = threadIdx.x + Halo;                                          // Local x id within padded grid
const int tyh = threadIdx.y + Halo;                                          // Local y id within padded grid
const int tzh = threadIdx.z + Halo;                                          // Local z id within padded grid
	
// Monolith structure
// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);

const int lid = gid(threadIdx.x,threadIdx.y,threadIdx.z,BX,BY,BZ);           // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);                             // Local id within padded block

// Fill centre volume
gsx[pid] = fO[bid];
gsy[pid] = fO[bid+NT];
gsz[pid] = fO[bid+2*NT];
sg[pid] = smag*sgs[bid];

__syncthreads();

//--- Fill Halo (with coalesced index read)
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
			gsx[lhid] = Real(0.);                     
			gsy[lhid] = Real(0.);                     
			gsz[lhid] = Real(0.);                       
			sg[lhid] = Real(0.); 
		}  
		else {      
			gsx[lhid] = fO[bhid     ];                     
			gsy[lhid] = fO[bhid+1*NT];                     
			gsz[lhid] = fO[bhid+2*NT];  	
			sg[lhid] = smag*sgs[bhid]; 	
		}  
   }                                           
   __syncthreads();
}

//------- Calculate discrete filter

Real lx = Real(0.), ly = Real(0.), lz = Real(0.);           	// Laplacian of Omega
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
	
	const Real Hlf = Real(0.5);

	if (Order==2){ 
		ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[0]*(gsx[ids]-gxm)*lfm;  ly += D2C2[0]*(gsy[ids]-gym)*lfm;  lz += D2C2[0]*(gsz[ids]-gzm)*lfm;  
		ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[1]*(gsx[ids]-gxm)*lfm;  ly += D2C2[1]*(gsy[ids]-gym)*lfm;  lz += D2C2[1]*(gsz[ids]-gzm)*lfm;  
		ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[2]*(gsx[ids]-gxm)*lfm;  ly += D2C2[2]*(gsy[ids]-gym)*lfm;  lz += D2C2[2]*(gsz[ids]-gzm)*lfm;  
		ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[0]*(gsx[ids]-gxm)*lfm;  ly += D2C2[0]*(gsy[ids]-gym)*lfm;  lz += D2C2[0]*(gsz[ids]-gzm)*lfm;  
		ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[1]*(gsx[ids]-gxm)*lfm;  ly += D2C2[1]*(gsy[ids]-gym)*lfm;  lz += D2C2[1]*(gsz[ids]-gzm)*lfm;  
		ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[2]*(gsx[ids]-gxm)*lfm;  ly += D2C2[2]*(gsy[ids]-gym)*lfm;  lz += D2C2[2]*(gsz[ids]-gzm)*lfm;  
		ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[0]*(gsx[ids]-gxm)*lfm;  ly += D2C2[0]*(gsy[ids]-gym)*lfm;  lz += D2C2[0]*(gsz[ids]-gzm)*lfm;  
		ids = gid(txh  ,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[1]*(gsx[ids]-gxm)*lfm;  ly += D2C2[1]*(gsy[ids]-gym)*lfm;  lz += D2C2[1]*(gsz[ids]-gzm)*lfm;  
		ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    lfm = Hlf*(sgm + sg[ids]);	lx += D2C2[2]*(gsx[ids]-gxm)*lfm;  ly += D2C2[2]*(gsy[ids]-gym)*lfm;  lz += D2C2[2]*(gsz[ids]-gzm)*lfm;  
	}
	if (Order==4){ 
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
	}
	if (Order==6){
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
	}
	if (Order==8){
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
	}
	
	// Scale for grid size 
	lx *= Real(1.)/hx/hx; 
	ly *= Real(1.)/hy/hy; 
	lz *= Real(1.)/hz/hz;
}

__syncthreads();

// Assign output vars
dOmdt[bid]      += lx;
dOmdt[bid+NT]   += ly;
dOmdt[bid+2*NT] += lz;

}  

template<int Halo, int Order, int NFDGrid>   
__global__ void RVM_DGC_turbulentstress(const Real* fO, const Real* sgs, const int* hs, const Real smag, Real *dOmdt, const bool AddSGS) { 

// Declare shared memory arrays
__shared__ Real ux[NFDGrid];
__shared__ Real uy[NFDGrid];
__shared__ Real uz[NFDGrid];

// Prepare relevant indices
const int gx0 = blockIdx.x*BX; 
const int gy0 = blockIdx.y*BY; 
const int gz0 = blockIdx.z*BZ; 
const int gx = threadIdx.x + gx0; 
const int gy = threadIdx.y + gy0; 
const int gz = threadIdx.z + gz0; 
const int txh = threadIdx.x + Halo;                                          // Local x id within padded grid
const int tyh = threadIdx.y + Halo;                                          // Local y id within padded grid
const int tzh = threadIdx.z + Halo;                                          // Local z id within padded grid
	
// Monolith structure
// if (Data==Monolith) Kernel.append(const int bid = gid(gx,gy,gz,NX,NY,NZ););
// if (Data==Block)    Kernel.append(const int bid = gidb(gx,gy,gz););
const int bid = gidb(gx,gy,gz);

const int lid = gid(threadIdx.x,threadIdx.y,threadIdx.z,BX,BY,BZ);           // Local id within block
const int pid = gid(txh,tyh,tzh,NFDX,NFDY,NFDZ);                             // Local id within padded block

// Fill centre volume
ux[pid] = fO[bid];
uy[pid] = fO[bid+NT];
uz[pid] = fO[bid+2*NT];
Real sg = smag*sgs[bid];
if (AddSGS) sg = Real(-1.);		// Flag to set curl negative

__syncthreads();

//--- Fill Halo (with coalesced index read)
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
 		const bool exy = (ghy<0 || ghy>=NY);      // Is y coordinate outside of the domain?
 		const bool exz = (ghz<0 || ghz>=NZ);      // Is z coordinate outside of the domain?
 		if (exx || exy || exz){                    // Catch: is id within domain?
			ux[lhid] = Real(0.);                     
			uy[lhid] = Real(0.);                     
			uz[lhid] = Real(0.);                       
//			sg[lhid] = Real(0.); 
//			if (sf) sg[lhid] = Real(-1.);
		}  
		else {      
			ux[lhid] = fO[bhid     ];                     
			uy[lhid] = fO[bhid+1*NT];                     
			uz[lhid] = fO[bhid+2*NT];  	
//			sg[lhid] = smag*sgs[bhid]; 	
//			if (sf) sg[lhid] = Real(-1.);
 		}  
    }                                           
   __syncthreads();
}

//------- Calculate discrete filter

Real curlx = Real(0.), curly = Real(0.), curlz = Real(0.);           	// Components of curl
const bool mxx = (gx>(Halo-1) && gx<(NX-Halo));
const bool mxy = (gy>(Halo-1) && gy<(NY-Halo));
const bool mxz = (gz>(Halo-1) && gz<(NZ-Halo));

if (mxx && mxy && mxz){

	int ids;  // Dummy value
	// Components of cross product
	Real duydx = Real(0.); 
	Real duzdx = Real(0.); 
	Real duxdy = Real(0.); 
	Real duzdy = Real(0.); 
	Real duxdz = Real(0.); 
	Real duydz = Real(0.); 

	// Calculate the cross product terms
	
	if (Order==2){
		ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C2[0]*uy[ids];  	duzdx += D1C2[0]*uz[ids];  
		ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C2[1]*uy[ids];  	duzdx += D1C2[1]*uz[ids];  
		ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C2[0]*ux[ids];   						  	duzdy += D1C2[0]*uz[ids];  
		ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C2[1]*ux[ids];   						  	duzdy += D1C2[1]*uz[ids];  
		ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    duxdz += D1C2[0]*ux[ids];   duydz += D1C2[0]*uy[ids];  
		ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    duxdz += D1C2[1]*ux[ids];   duydz += D1C2[1]*uy[ids];  
	}
 	if (Order==4){
 		ids = gid(txh-2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    				   			duydx += D1C4[0]*uy[ids];  	duzdx += D1C4[0]*uz[ids];  
 		ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    				   			duydx += D1C4[1]*uy[ids];  	duzdx += D1C4[1]*uz[ids];  
 		ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    				   			duydx += D1C4[2]*uy[ids];  	duzdx += D1C4[2]*uz[ids];  
 		ids = gid(txh+2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    				   			duydx += D1C4[3]*uy[ids];  	duzdx += D1C4[3]*uz[ids];  
 		ids = gid(txh  ,tyh-2,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C4[0]*ux[ids];   							duzdy += D1C4[0]*uz[ids];  
 		ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C4[1]*ux[ids];   							duzdy += D1C4[1]*uz[ids];  
 		ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C4[2]*ux[ids];   							duzdy += D1C4[2]*uz[ids];  
 		ids = gid(txh  ,tyh+2,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C4[3]*ux[ids];   							duzdy += D1C4[3]*uz[ids];  
 		ids = gid(txh  ,tyh  ,tzh-2,NFDX,NFDY,NFDZ);    duxdz += D1C4[0]*ux[ids];   duydz += D1C4[0]*uy[ids];  
 		ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    duxdz += D1C4[1]*ux[ids];   duydz += D1C4[1]*uy[ids];  
 		ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    duxdz += D1C4[2]*ux[ids];   duydz += D1C4[2]*uy[ids];  
 		ids = gid(txh  ,tyh  ,tzh+2,NFDX,NFDY,NFDZ);    duxdz += D1C4[3]*ux[ids];   duydz += D1C4[3]*uy[ids];  
 	}
 	if (Order==6){
 		ids = gid(txh-3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C6[0]*uy[ids];  	duzdx += D1C6[0]*uz[ids];  
 		ids = gid(txh-2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C6[1]*uy[ids];  	duzdx += D1C6[1]*uz[ids];  
 		ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C6[2]*uy[ids];  	duzdx += D1C6[2]*uz[ids];  
 		ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C6[3]*uy[ids];  	duzdx += D1C6[3]*uz[ids];  
 		ids = gid(txh+2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C6[4]*uy[ids];  	duzdx += D1C6[4]*uz[ids];  
 		ids = gid(txh+3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C6[5]*uy[ids];  	duzdx += D1C6[5]*uz[ids];  
 		ids = gid(txh  ,tyh-3,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C6[0]*ux[ids];   							duzdy += D1C6[0]*uz[ids];  
 		ids = gid(txh  ,tyh-2,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C6[1]*ux[ids];   							duzdy += D1C6[1]*uz[ids];  
 		ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C6[2]*ux[ids];   							duzdy += D1C6[2]*uz[ids];  
 		ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C6[3]*ux[ids];   							duzdy += D1C6[3]*uz[ids];  
 		ids = gid(txh  ,tyh+2,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C6[4]*ux[ids];   							duzdy += D1C6[4]*uz[ids];  
 		ids = gid(txh  ,tyh+3,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C6[5]*ux[ids];   							duzdy += D1C6[5]*uz[ids];  
 		ids = gid(txh  ,tyh  ,tzh-3,NFDX,NFDY,NFDZ);    duxdz += D1C6[0]*ux[ids];   duydz += D1C6[0]*uy[ids];  
 		ids = gid(txh  ,tyh  ,tzh-2,NFDX,NFDY,NFDZ);    duxdz += D1C6[1]*ux[ids];   duydz += D1C6[1]*uy[ids];  
 		ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    duxdz += D1C6[2]*ux[ids];   duydz += D1C6[2]*uy[ids];  
 		ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    duxdz += D1C6[3]*ux[ids];   duydz += D1C6[3]*uy[ids];  
 		ids = gid(txh  ,tyh  ,tzh+2,NFDX,NFDY,NFDZ);    duxdz += D1C6[4]*ux[ids];   duydz += D1C6[4]*uy[ids];  
 		ids = gid(txh  ,tyh  ,tzh+3,NFDX,NFDY,NFDZ);    duxdz += D1C6[5]*ux[ids];   duydz += D1C6[5]*uy[ids];  
 	}
 	if (Order==8){
 		ids = gid(txh-4,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C8[0]*uy[ids];  	duzdx += D1C8[0]*uz[ids];  
 		ids = gid(txh-3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C8[1]*uy[ids];  	duzdx += D1C8[1]*uz[ids];  
 		ids = gid(txh-2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C8[2]*uy[ids];  	duzdx += D1C8[2]*uz[ids];  
 		ids = gid(txh-1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C8[3]*uy[ids];  	duzdx += D1C8[3]*uz[ids];  
 		ids = gid(txh+1,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C8[4]*uy[ids];  	duzdx += D1C8[4]*uz[ids];  
 		ids = gid(txh+2,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C8[5]*uy[ids];  	duzdx += D1C8[5]*uz[ids];  
 		ids = gid(txh+3,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C8[6]*uy[ids];  	duzdx += D1C8[6]*uz[ids];  
 		ids = gid(txh+4,tyh  ,tzh  ,NFDX,NFDY,NFDZ);    							duydx += D1C8[7]*uy[ids];  	duzdx += D1C8[7]*uz[ids];  
 		ids = gid(txh  ,tyh-4,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[0]*ux[ids];   							duzdy += D1C8[0]*uz[ids];  
 		ids = gid(txh  ,tyh-3,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[1]*ux[ids];   							duzdy += D1C8[1]*uz[ids];  
 		ids = gid(txh  ,tyh-2,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[2]*ux[ids];   							duzdy += D1C8[2]*uz[ids];  
 		ids = gid(txh  ,tyh-1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[3]*ux[ids];   							duzdy += D1C8[3]*uz[ids];  
 		ids = gid(txh  ,tyh+1,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[4]*ux[ids];   							duzdy += D1C8[4]*uz[ids];  
 		ids = gid(txh  ,tyh+2,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[5]*ux[ids];   							duzdy += D1C8[5]*uz[ids];  
 		ids = gid(txh  ,tyh+3,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[6]*ux[ids];   							duzdy += D1C8[6]*uz[ids];  
 		ids = gid(txh  ,tyh+4,tzh  ,NFDX,NFDY,NFDZ);    duxdy += D1C8[7]*ux[ids];   							duzdy += D1C8[7]*uz[ids];  
 		ids = gid(txh  ,tyh  ,tzh-4,NFDX,NFDY,NFDZ);    duxdz += D1C8[0]*ux[ids];   duydz += D1C8[0]*uy[ids];  
 		ids = gid(txh  ,tyh  ,tzh-3,NFDX,NFDY,NFDZ);    duxdz += D1C8[1]*ux[ids];   duydz += D1C8[1]*uy[ids];  
 		ids = gid(txh  ,tyh  ,tzh-2,NFDX,NFDY,NFDZ);    duxdz += D1C8[2]*ux[ids];   duydz += D1C8[2]*uy[ids];  
 		ids = gid(txh  ,tyh  ,tzh-1,NFDX,NFDY,NFDZ);    duxdz += D1C8[3]*ux[ids];   duydz += D1C8[3]*uy[ids];  
 		ids = gid(txh  ,tyh  ,tzh+1,NFDX,NFDY,NFDZ);    duxdz += D1C8[4]*ux[ids];   duydz += D1C8[4]*uy[ids];  
 		ids = gid(txh  ,tyh  ,tzh+2,NFDX,NFDY,NFDZ);    duxdz += D1C8[5]*ux[ids];   duydz += D1C8[5]*uy[ids];  
 		ids = gid(txh  ,tyh  ,tzh+3,NFDX,NFDY,NFDZ);    duxdz += D1C8[6]*ux[ids];   duydz += D1C8[6]*uy[ids];  
 		ids = gid(txh  ,tyh  ,tzh+4,NFDX,NFDY,NFDZ);    duxdz += D1C8[7]*ux[ids];   duydz += D1C8[7]*uy[ids];  
 	}
	
	// Scale terms
	const Real invhx = Real(1.)/hx, invhy = Real(1.)/hy, invhz = Real(1.)/hz;  
	duydx *= invhx;  
	duzdx *= invhx;  
	duxdy *= invhy;   
	duzdy *= invhy;  
	duxdz *= invhz;  
	duydz *= invhz;  
	curlx = sg*(duzdy-duydz);
	curly = sg*(duxdz-duzdx);
	curlz = sg*(duydx-duxdy);
}

__syncthreads();

// Assign output vars
dOmdt[bid]      += curlx;
dOmdt[bid+NT]   += curly;
dOmdt[bid+2*NT] += curlz;

}

)CLC";

}
