#include "Tests/Test_1D.h"
#include "Tests/Test_2D.h"
#include "Tests/Test_3D.h"
#include "Tests/Test_3DV.h"

static const bool Export_VTI = true;

int main()
{

    int nx = 128;
    int ny = 128;
    int nz = 128;

    //--- 1D solver tests
//    Test_Dirichlet_1D(nx);
//    Test_Dirichlet_1D_IHBC(nx);
//    Test_Neumann_1D(nx);
//    Test_Periodic_1D(nx);
//    Test_Unbounded_1D(nx);

    //--- 2D scalar solver tests
//    Test_Dirichlet_2D(nx,ny);
//    Test_Dirichlet_2D_IHBC(nx,ny);
//    Test_Neumann_2D(nx,ny);
//    Test_Neumann_2D_IHBC(nx,ny);
    Test_Periodic_2D(nx,ny);
//    Test_Unbounded_2D(nx,ny);
//    Test_Unbounded_2D_Grad(nx,ny);

    //--- 3D scalar solver tests
//    Test_Dirichlet_3D(nx,ny,nz);
//    Test_Dirichlet_3D_IHBC(nx,ny,nz);
//    Test_Neumann_3D(nx,ny,nz);
//    Test_Neumann_3D_IHBC(nx,ny,nz);
//    Test_Periodic_3D(nx,ny,nz);
//    Test_Unbounded_3D(nx,ny,nz);

    //--- 3D vector solver tests
//    Test_Dirichlet_3DV(nx,ny,nz);
//    Test_Neumann_3DV(nx,ny,nz);
//    Test_Periodic_3DV(nx,ny,nz);
//    Test_Unbounded_3DV_Curl(nx,ny,nz);

    return 0;
}
