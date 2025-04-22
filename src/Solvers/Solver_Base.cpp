//-----------------------------------------------------------------------------
//-------------------------Solver Base Functions-------------------------------
//-----------------------------------------------------------------------------

#include "Solver_Base.h"

namespace SailFFish
{

//---------------------------
//--- Parent Solver class ---
//---------------------------

void Solver::X_Grid_Setup(Real X[2], int iNX)
{
    // This specifies the X grid properties

    NCX = iNX;                      // Number of grid cells
    if (Grid==REGULAR)       gNX = NCX + 1;
    if (Grid==STAGGERED)     gNX = NCX;
    Xl = X[0];
    Xu = X[1];
    Lx = Xu-Xl;
    Hx = Lx/NCX;
    gX.assign(gNX,0);
    if (Grid==REGULAR)       for (int i=0; i<gNX; i++) {gX[i] = Xl + i*Hx;           }
    if (Grid==STAGGERED)     for (int i=0; i<gNX; i++) {gX[i] = Xl + 0.5*Hx + i*Hx;  }
}

void Solver::Y_Grid_Setup(Real Y[2], int iNY)
{
    // This specifies the X grid properties

    NCY = iNY;                      // Number of grid cells
    if (Grid==REGULAR)       gNY = NCY + 1;
    if (Grid==STAGGERED)     gNY = NCY;
    Yl = Y[0];
    Yu = Y[1];
    Ly = Yu-Yl;
    Hy = Ly/NCY;
    gY.assign(gNY,0);
    if (Grid==REGULAR)       for (int i=0; i<gNY; i++) {gY[i] = Yl + i*Hy;           }
    if (Grid==STAGGERED)     for (int i=0; i<gNY; i++) {gY[i] = Yl + 0.5*Hy + i*Hy;  }
}

void Solver::Z_Grid_Setup(Real Z[2], int iNZ)
{
    // This specifies the X grid properties

    NCZ = iNZ;                      // Number of grid cells
    if (Grid==REGULAR)       gNZ = NCZ + 1;
    if (Grid==STAGGERED)     gNZ = NCZ;
    Zl = Z[0];
    Zu = Z[1];
    Lz = Zu-Zl;
    Hz = Lz/NCZ;
    gZ.assign(gNZ,0);
    if (Grid==REGULAR)       for (int i=0; i<gNZ; i++) {gZ[i] = Zl + i*Hz;           }
    if (Grid==STAGGERED)     for (int i=0; i<gNZ; i++) {gZ[i] = Zl + 0.5*Hz + i*Hz;  }
}

//--- Destructor

Solver::~Solver()
{

}

//---------------------------
//--- 1D Scalar solver ------
//---------------------------

//--- Solver setup

SFStatus Solver_1D_Scalar::Setup(Real X[2], int iNX)
{
    // Solver setup for 1D solver types

    // Setup status
    SFStatus Stat = NoError;

    X_Grid_Setup(X,iNX);
    gNT = gNX;                  // Total number of grid points
    GridDim = dim3(iNX);

    // Prepare data
    Datatype_Setup();
    Stat = FFT_Data_Setup();
    Specify_Greens_Function();
    if (Operator!=NONE) Prepare_Dif_Operators_1D(Hx);

    std::cout << "Solver_1D_Scalar Grid Setup Complete" << std::endl;
    std::cout << "Xl = " << Xl << " Xu = " << Xu << " Lx = " << Lx << " Hx = " << Hx << std::endl;
    std::cout << "NX = " << NX << std::endl;

    return Stat;
}

//---------------------------
//--- 2D Scalar solver ------
//---------------------------

//--- Solver setup

SFStatus Solver_2D_Scalar::Setup(Real X[2], Real Y[2], int iNX, int iNY)
{
    // Solver setup for 2D scalar solver types

    // Setup status
    SFStatus Stat = NoError;

    X_Grid_Setup(X,iNX);
    Y_Grid_Setup(Y,iNY);
    gNT = gNX*gNY;
    GridDim = dim3(iNX,iNY);

    // Prepare data
    Datatype_Setup();
    Stat = FFT_Data_Setup();
    Specify_Greens_Function();
    if (Operator!=NONE) Prepare_Dif_Operators_2D(Hx,Hy);

    std::cout << "Solver_2D_Scalar Grid Setup Complete" << std::endl;
    std::cout << "Xl = " << Xl << " Xu = " << Xu << " Lx = " << Lx << " Hx = " << Hx << std::endl;
    std::cout << "Yl = " << Yl << " Yu = " << Yu << " Ly = " << Ly << " Hy = " << Hy << std::endl;
    std::cout << "NX = " << NX << " NY = " << NY << std::endl;

    return Stat;
}

//---------------------------
//--- 3D Scalar solver ------
//---------------------------

//--- Solver setup

SFStatus Solver_3D_Scalar::Setup(Real X[2], Real Y[2], Real Z[2], int iNX, int iNY, int iNZ)
{
    // Solver setup for 3D scalar solver types

    // Setup status
    SFStatus Stat = NoError;

    X_Grid_Setup(X,iNX);
    Y_Grid_Setup(Y,iNY);
    Z_Grid_Setup(Z,iNZ);
    gNT = gNX*gNY*gNZ;                  // Total number of grid points
    GridDim = dim3(iNX, iNY, iNZ);

    // Prepare data
    Datatype_Setup();
    Stat = FFT_Data_Setup();
    Specify_Greens_Function();
    if (Operator!=NONE) Prepare_Dif_Operators_3D(Hx,Hy,Hz);

    std::cout << "Bounded_Poisson_Dirichlet_3D Grid Setup Complete" << std::endl;
    std::cout << "Xl = " << Xl << " Xu = " << Xu << " Lx = " << Lx << " Hx = " << Hx << std::endl;
    std::cout << "Yl = " << Yl << " Yu = " << Yu << " Ly = " << Ly << " Hy = " << Hy << std::endl;
    std::cout << "Zl = " << Zl << " Zu = " << Zu << " Zy = " << Lz << " Hz = " << Hz << std::endl;
    std::cout << "NX = " << NX << " NY = " << NY << " NZ = " << NZ << std::endl;

    return Stat;
}

//---------------------------
//--- 3D Vector solver ------
//---------------------------

//--- Solver setup

SFStatus Solver_3D_Vector::Setup(Real X[2], Real Y[2], Real Z[2], int iNX, int iNY, int iNZ)
{
    // Solver setup for 3D vector solver types

    // Setup status
    SFStatus Stat = NoError;

    X_Grid_Setup(X,iNX);
    Y_Grid_Setup(Y,iNY);
    Z_Grid_Setup(Z,iNZ);
    gNT = gNX*gNY*gNZ;                  // Total number of grid points
    GridDim = dim3(iNX, iNY, iNZ);

    // Prepare data
    Datatype_Setup();
    Stat = FFT_Data_Setup();
    Specify_Greens_Function();
    if (Operator!=NONE) Prepare_Dif_Operators_3D(Hx,Hy,Hz);

    std::cout << "3D Vector Grid Setup Complete" << std::endl;
    std::cout << "Xl = " << Xl << " Xu = " << Xu << " Lx = " << Lx << " Hx = " << Hx << std::endl;
    std::cout << "Yl = " << Yl << " Yu = " << Yu << " Ly = " << Ly << " Hy = " << Hy << std::endl;
    std::cout << "Zl = " << Zl << " Zu = " << Zu << " Zy = " << Lz << " Hz = " << Hz << std::endl;
    std::cout << "NX = " << NX << " NY = " << NY << " NZ = " << NZ << std::endl;

    return Stat;
}

}
