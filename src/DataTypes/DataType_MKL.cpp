//-----------------------------------------------------------------------------
//-------------------------DataType_MKL Functions------------------------------
//-----------------------------------------------------------------------------

#include "DataType_MKL.h"

namespace SailFFish
{

#ifdef MKL

void DataType_MKL::Setup_2D_DFT(int iNX, int iNY)
{
    // This prepares and allocates data for the discrete sin transform (periodic with boundaries zero)
    // Boundary positions

    // The values nx, ny refer to the number of cells in total.
    nx = iNX;
    ny = iNY;

    // This is simply a hack for testing on unit square

    ax = 0.0;
    bx = 1.0;

    ay = 0.0;
    by = 1.0;

//    ax = 0.0;
//    bx = 3.0;

//    ay = 0.0;
//    by = 7.0;

    Real Lx = bx-ax;
    Real Ly = by-ay;

    Hx = Lx/nx;
    Hy = Ly/ny;

    // Allocate memory blocks -- spar, f, bd_a..... ( Internal var Helmholtz Solver, RHS, BCs)

    spar=(Real*)malloc((13*nx/2+7)*sizeof(Real));

    BD_AX = new Real*[1];
    BD_BX = new Real*[1];
    BD_AY = new Real*[1];
    BD_BY = new Real*[1];

    BD_AX[0] = (Real*)calloc((ny+1)*sizeof(Real),sizeof(Real));
    BD_BX[0] = (Real*)calloc((ny+1)*sizeof(Real),sizeof(Real));
    BD_AY[0] = (Real*)calloc((nx+1)*sizeof(Real),sizeof(Real));
    BD_BY[0] = (Real*)calloc((nx+1)*sizeof(Real),sizeof(Real));

    nt = (nx+1)*(ny+1);
    r_Input1 = (Real*)calloc(nt*sizeof(Real),sizeof(Real));

    N_Points = (nx+1)*(ny+1);
    N_FaceX = (nx+1);
    N_FaceY = (ny+1);
    N_BPoints = 2*N_FaceX + 2*N_FaceY;
    N_Cells = nx*ny;

    // Set q value (0 for Poisson problem)

    q = 0;

    // Initializing ipar array to make it free from garbage

    for (int i=0;i<128;i++)  ipar[i]=0;

    // Reset initial arrays
    Reset_Array(BD_AX[0],N_FaceX);
    Reset_Array(BD_BX[0],N_FaceX);
    Reset_Array(BD_AY[0],N_FaceY);
    Reset_Array(BD_BY[0],N_FaceY);
    Reset_Array(r_Input1,N_Points);

    // Dirichlet
    for (int i=0; i<N_FaceX; i++) BD_AX[0][i] = 1.0;
    for (int i=0; i<N_FaceX; i++) BD_BX[0][i] = 1.0;
    for (int i=0; i<N_FaceY; i++) BD_AY[0][i] = 1.0;
    for (int i=0; i<N_FaceY; i++) BD_BY[0][i] = 1.0;

//    for (int i=0; i<N_FaceX; i++) BD_AX[0][i] = -1.0;
//    for (int i=0; i<N_FaceX; i++) BD_BX[0][i] = 1.0;
//    for (int i=0; i<N_FaceY; i++) BD_AY[0][i] = -1.0+2.0*i/(N_FaceY-1);
//    for (int i=0; i<N_FaceY; i++) BD_BY[0][i] = -1.0+2.0*i/(N_FaceY-1);

//    // Neumann
//    for (int i=0; i<N_FaceX; i++) BD_AX[0][i] = -1.0;
//    for (int i=0; i<N_FaceX; i++) BD_BX[0][i] = -1.0;
//    for (int i=0; i<N_FaceY; i++) BD_AY[0][i] = -1.0;
//    for (int i=0; i<N_FaceY; i++) BD_BY[0][i] = 1.0;

    // Set f on grid
    for (int i=0; i<nx+1; i++){
        for (int j=0; j<ny+1; j++){
            Real x = ax + i*Hx;
            Real y = ay + j*Hy;
            Real r2 = x*x+y*y;
//            Input[i*NY+j][0] = 1.0;  Input[i][1] = 0.0;               // Constant
//            Input[i*NY+j][0] = sin(M_PI*x)*cos(M_PI*y);                   // Sin-cos curve
//            if (r2<=1.0) r_Input1[i*NY+j] = exp(-1.0/(1.0-r2));        // Bump function
//            if (r2<=1.0) r_Input1[i*(ny+1)+j] = exp(-1.0/(1.0-r2));      // Bump function
//            cout << x <<  " " << y csp r_Input1[i*(ny+1)+j] << endl;
//            r_Input1[i*(ny+1)+j] = 2.0*y*(Ly-y) + 2.0*x*(Lx-x);
            r_Input1[i+j*(nx+1)] = 2.0*y*(Ly-y) + 2.0*x*(Lx-x);

//            cout << x <<  " " << y csp r_Input1[i*(ny+1)+j] << endl;
        }
    }

    //--- Debug Input (remember MKL ordering....)
//    for (int i=0; i<nx+1; i++){
//        for (int j=0; j<ny+1; j++){
//            cout << x <<  " " << y csp r_Input1[i*(ny+1)+j] << endl;
//        }
//    }

    char *BCtype;
    BCtype = "DDDD";    // Set BC Type (Dirichlet---zeros)
//    BCtype = "NNNN";    // Set BC Type (Neumann---zero gradient)

    //--- Initialize and execute solver
    s_init_Helmholtz_2D(&ax, &bx, &ay, &by, &nx, &ny, BCtype, &q, ipar, spar, &stat);
    s_commit_Helmholtz_2D(r_Input1, BD_AX[0], BD_BX[0], BD_AY[0], BD_BY[0], &xhandle, ipar, spar, &stat);
    s_Helmholtz_2D(r_Input1, BD_AX[0], BD_BX[0], BD_AY[0], BD_BY[0], &xhandle, ipar, spar, &stat);

    //--- Debug Output1 (remember MKL ordering...)
    for (int i=0; i<nx+1; i++){
        for (int j=0; j<ny+1; j++){
            Real x = ax + i*Hx;
            Real y = ay + j*Hy;
            cout << x csp y csp r_Input1[i+j*(nx+1)] << endl;
        }
    }
}

#endif

}
