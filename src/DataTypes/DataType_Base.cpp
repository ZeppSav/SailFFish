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

//--- Mapping functions

void DataType::Map_C2F_2D(const RVector &Src, RVector &Dest)
{
    // Map C-Style array to F-Style array
    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++){
            Dest[GID(j,i,NY,NX)] = Src[GID(i,j,NX,NY)];
        }
    }
}

void DataType::Map_F2C_2D(const RVector &Src, RVector &Dest)
{
    // Map F-Style array to C-Style array
    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++){
            Dest[GID(i,j,NX,NY)] = Src[GID(j,i,NY,NX)];
        }
    }
}

void DataType::Map_C2F_3D(const RVector &Src, RVector &Dest)
{
    // Map C-Style array to F-Style array
    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++){
            for (int k=0; k<NZ; k++){
                Dest[GID(k,j,i,NZ,NY,NX)] = Src[GID(i,j,k,NX,NY,NZ)];
            }
        }
    }
}

void DataType::Map_F2C_3D(const RVector &Src, RVector &Dest)
{
    // Map F-Style array to C-Style array
    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++){
            for (int k=0; k<NZ; k++){
                Dest[GID(i,j,k,NX,NY,NZ)] = Src[GID(k,j,i,NZ,NY,NX)];
            }
        }
    }
}

//--- Map grid

void DataType::Map_C2F_3DV(const RVector &Src1, const RVector &Src2, const RVector &Src3,
                                RVector &Dest1, RVector &Dest2, RVector &Dest3)
{
    // Map C-Style array to F-Style array
    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++){
            for (int k=0; k<NZ; k++){
                int cid = GID(i,j,k,NX,NY,NZ);
                int fid = GID(k,j,i,NZ,NY,NX);
                Dest1[fid] = Src1[cid];
                Dest2[fid] = Src2[cid];
                Dest3[fid] = Src3[cid];
            }
        }
    }
}

void DataType::Map_F2C_3DV( const RVector &Src1, const RVector &Src2, const RVector &Src3,
                                RVector &Dest1, RVector &Dest2, RVector &Dest3)
{
    // Map F-Style array to C-Style array
    OpenMPfor
    for (int i=0; i<NX; i++){
        for (int j=0; j<NY; j++){
            for (int k=0; k<NZ; k++){
                int cid = GID(i,j,k,NX,NY,NZ);
                int fid = GID(k,j,i,NZ,NY,NX);
                Dest1[cid] = Src1[fid];
                Dest2[cid] = Src2[fid];
                Dest3[cid] = Src3[fid];
            }
        }
    }
}

}
