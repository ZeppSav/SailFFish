#ifndef VPML_TYPES_H
#define VPML_TYPES_H

//#include <Eigen/Eigen>          // Eigen data types
#include <tr1/cmath>            // Special functions
#include <memory>               // Shared ptr.
//#include <cstdio>
//#include <cstdlib>
#include <cstring>              // Memset
#include <iostream>
#include <iomanip>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>              // ofstream

using namespace std;

#define OpenMPfor _Pragma("omp parallel for")
#define csp <<" "<<

#ifdef SinglePrec

//--------Single Precision----------
typedef float                       Real;
typedef std::complex<float>         CReal;
typedef std::vector<float>          RVector;
typedef std::vector<CReal>          CVector;

#endif

#ifdef DoublePrec

//--------Double Precision----------
typedef double                      Real;
typedef std::complex<double>        CReal;
typedef std::vector<double>         RVector;
typedef std::vector<CReal>          CVector;

#endif

//---- Mathematical Constants

#define M_GAMMA   0.57721566490153286060
static Real const M_2PI         = 2.0 * M_PI;
static Real const M_INV2PI      = 1.0 / M_2PI;
static Real const M_INV2PISQ    = 1.0 / 2.0 / M_PI / M_PI;
static Real const M_INV4PI      = 1.0 / 2.0 / M_2PI;
static Real const M_INVSQRT2    = 1.0 / M_SQRT2;
static Real const M_4PI         =  4.0*M_PI;
static Real const  FourPIinv    =  1.0 / 4.0 / M_PI;
static Real const  Rt2oPi       =  sqrt(2.0/M_PI);
static Real const  M_INVSQRT2PI =  sqrt(M_INV2PI);

static CReal const ComplexNull(0.0,0.0);
static CReal const Unity(1.0,0.0);
static CReal const Im(0.0,1.0);

//---- Conversions

static Real const  D2R          =  M_PI/180;
static Real const  R2D          =  180/M_PI;
static Real const  RPMtoOmega   = M_PI/30.0;
static Real const  OmegatoRPM   = 1.0/RPMtoOmega;

//--- Create directory

#include <sys/types.h>  // required for stat.h
#include <sys/stat.h>   // no clue why required -- man pages say so

inline int CreateDirectory(std::string Path)
{
    #if defined(_WIN32)
        return mkdir(Path.c_str());
    #else
        mode_t nMode = 0733; // UNIX style permissions
        return mkdir(Path.c_str(),nMode);
    #endif
}

#endif // VPML_TYPES_H
