#ifndef VPML_TYPES_H
#define VPML_TYPES_H

#include <complex>
#include <vector>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>              // ofstream
#include <algorithm>
#include <chrono>
#include <numeric>

#define OpenMPfor _Pragma("omp parallel for")
#define csp <<" "<<

#ifdef SinglePrec
    //--------Single Precision----------
    typedef float                       Real;
    typedef std::complex<float>         CReal;
    typedef std::vector<float>          RVector;
    typedef std::vector<CReal>          CVector;
    #define fmadd fmaf
#endif

#ifdef DoublePrec
    //--------Double Precision----------
    typedef double                      Real;
    typedef std::complex<double>        CReal;
    typedef std::vector<double>         RVector;
    typedef std::vector<CReal>          CVector;
    #define fmadd fma
#endif

//--- Macros to improve readability

namespace SailFFish {

// #define OpenMPfor _Pragma("omp parallel for")
// #define csp <<" "<<

#define Parallel_Kernel(X) OpenMPfor for (int i=0; i<(X); i++)
#define Serial_Kernel(X) for (int i=0; i<(X); i++)

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

//-------- Std::vector helper functions

template <class T>          // Hack to avoid using insert everytime
static void StdAppend(std::vector<T>& lhs, const std::vector<T>& rhs)       {lhs.insert(lhs.end(),rhs.begin(), rhs.end());}

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

//---- Input string handling

inline std::vector<std::string> SplitUp(const std::string &str)
{
    // Used to split string around spaces.
    std::istringstream ss(str);
    std::string word;
    std::vector<std::string> tokens;
    while (ss >> word)  tokens.push_back(word);
    return tokens;
}

}

#endif // VPML_TYPES_H
