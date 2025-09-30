//-----------------------------------------------------------------------------
//-------------------------DataType_Base Functions-----------------------------
//-----------------------------------------------------------------------------

#include "DataType_Base.h"

namespace SailFFish
{

//--- Problem Setup (plan creation and destruction)

SFStatus DataType::Setup_1D(int iNX)
{
    // This function prepares the SailFFish and FFTW objects for the case of 1D solver

    // Setup status
    SFStatus Stat = NoError;

    //--- Prepare grid and scaling data
    NX = iNX;
    NT = NX;
    Set_NTM1D();
    if (Transform==DFT_R2C)    Set_NTM1D_R2C();

    //--- Memory allocation
    Stat = Allocate_Arrays();

    //--- Setup Plan
    Stat = Specify_1D_Plan();

    return Stat;
}

SFStatus DataType::Setup_2D(int iNX, int iNY)
{
    // This function prepares the SailFFish and FFTW objects for the case of a 2D solver

    // Setup status
    SFStatus Stat = NoError;

    //--- Prepare grid and scaling data
    NX = iNX;
    NY = iNY;
    NT = NX*NY;
    Set_NTM2D();
    if (Transform==DFT_R2C)    Set_NTM2D_R2C();

    //--- Memory allocation
    Stat = Allocate_Arrays();

    //--- Setup Plan
    Stat = Specify_2D_Plan();

    return Stat;
}

SFStatus DataType::Setup_3D(int iNX, int iNY, int iNZ)
{
    // This function prepares the SailFFish and FFTW objects for the case of a 3D solver

    // Setup status
    SFStatus Stat = NoError;

    //--- Prepare grid and scaling data
    NX = iNX;
    NY = iNY;
    NZ = iNZ;
    NT = NX*NY*NZ;
    Set_NTM3D();
    if (Transform==DFT_R2C) Set_NTM3D_R2C();

    //--- Memory allocation
    Stat = Allocate_Arrays();

    //--- Setup Plan
    Stat = Specify_3D_Plan();

    return Stat;
}

}
