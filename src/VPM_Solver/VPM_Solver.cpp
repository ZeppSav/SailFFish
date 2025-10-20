//-----------------------------------------------------------------------
//----------------------- VPM Solver Functions --------------------------
//-----------------------------------------------------------------------

#include "VPM_Solver.h"
#include "../Solvers/Greens_Functions.h"

namespace SailFFish
{

//--- Constructor
VPM_3D_Solver::VPM_3D_Solver(Grid_Type G, Unbounded_Kernel B)
{
    // Basis constructor

    // Specify grid and Ker types
    Grid = G;
    Greens_Kernel = B;

    // Specify operator
    Specify_Operator(SailFFish::CURL);
    // InPlace = true;     // Specify that transforms should occur either in place or out of place (reduced memory footprint)
    // c_dbf_1 = true;     // This dummy buffer is required as I have not included custom Kernels into Datatype_Cuda yet

    // Extract forcing nodes and subtract this solution (rather than getting the solution from the auxiliary grid)
    switch (Greens_Kernel)
    {
    case (HEJ_G2):  {BS = &SailFFish::Kernel_HEJ3_K2;   break;}
    case (HEJ_G4):  {BS = &SailFFish::Kernel_HEJ3_K4;   break;}
    case (HEJ_G6):  {BS = &SailFFish::Kernel_HEJ3_K6;   break;}
    case (HEJ_G8):  {BS = &SailFFish::Kernel_HEJ3_K8;   break;}
    case (HEJ_G10): {BS = &SailFFish::Kernel_HEJ3_K10;  break;}
    default:   break;
    }
}

//--- Grid setup

void VPM_3D_Solver::Set_Grid_Positions()
{
    //--- Specify grid positions
    OpenMPfor
    for (int i=0; i<NNX; i++){
        for (int j=0; j<NNY; j++){
            for (int k=0; k<NNZ; k++){
                int id_dest = GID(i,j,k,NNX,NNY,NNZ);
                eu_d[0][id_dest] = i*H_Grid + XN1;
                eu_d[1][id_dest] = j*H_Grid + YN1;
                eu_d[2][id_dest] = k*H_Grid + ZN1;
            }
        }
    }

    //    for (int i=0; i<NNT; i++) std::cout << g_Array[0][i] csp g_Array[1][i] csp g_Array[2][i] << std::endl;
}

SFStatus VPM_3D_Solver::Define_Grid_Nodes(VPM_Input *I)
{
    // The solver grid is specified based on the grid size and the specified domain limits

    std::cout << "Defining grid with node format" << std::endl;

    H_Grid = I->H_Grid;
    Hx = H_Grid;
    Hy = H_Grid;
    Hz = H_Grid;

    NNX = I->NX;
    NNY = I->NY;
    NNZ = I->NZ;
    NNT = NNX*NNY*NNZ;
    NBT = NNT/NBlock3 + 1;

    X0 = I->Cx;
    Y0 = I->Cy;
    Z0 = I->Cz;
    XN1 = X0 + 0.5*H_Grid;
    YN1 = Y0 + 0.5*H_Grid;
    ZN1 = Z0 + 0.5*H_Grid;

    std::cout << "Grid lower bounds [xl,yl,zl] = [" << I->Cx csp I->Cy csp I->Cz                                   << "]" << std::endl;
    std::cout << "Grid upper bounds [xh,yh,zh] = [" << I->Cx+H_Grid*NNX csp I->Cy+H_Grid*NNY csp I->Cz+H_Grid*NNZ  << "]" << std::endl;

    //--- Initialize SailFFish solver.
    Real Xg[2] = {I->Cx,I->Cx+H_Grid*NNX};
    Real Yg[2] = {I->Cy,I->Cy+H_Grid*NNY};
    Real Zg[2] = {I->Cz,I->Cz+H_Grid*NNZ};
    return Setup(Xg, Yg, Zg, NNX, NNY, NNZ);
}

SFStatus VPM_3D_Solver::Define_Grid_Blocks(VPM_Input *I)
{
    // The solver grid is specified based on the block limits provided.

    std::cout << "Defining grid with block format" << std::endl;

    H_Grid = I->H_Grid;
    Hx = H_Grid;
    Hy = H_Grid;
    Hz = H_Grid;

    // Set block dimensions
    BX = I->BX;
    BY = I->BY;
    BZ = I->BZ;
    BT = BX*BY*BZ;

    iBX0 = I->iBX0;
    iBY0 = I->iBY0;
    iBZ0 = I->iBZ0;

    NBX = I->NBX;
    NBY = I->NBY;
    NBZ = I->NBZ;
    NBT = NBX*NBY*NBZ;
    // NBT = NNT/NBlock3 + 1;

    // Set upper & lower bounds which include requested boundary
    HLx = BX*H_Grid;
    HLy = BY*H_Grid;
    HLz = BZ*H_Grid;
    Real lowblockx = I->Cx + Real(iBX0)*HLx;
    Real lowblocky = I->Cy + Real(iBY0)*HLy;
    Real lowblockz = I->Cz + Real(iBZ0)*HLz;
    Real highblockx = lowblockx + Real(NBX)*HLx;
    Real highblocky = lowblocky + Real(NBY)*HLy;
    Real highblockz = lowblockz + Real(NBZ)*HLz;

    // std::cout << "Block side lengths [lx,ly,lz] = [ " << HLx csp HLy csp HLz  << "]" << std::endl;
    // std::cout << "ID 0 = [ " << I->iBX0 csp I->iBY0 csp I->iBZ0  << "]" << std::endl;
    std::cout << "Grid lower bounds [xl,yl,zl] = [" << lowblockx csp lowblocky csp lowblockz    << "]" << std::endl;
    std::cout << "Grid upper bounds [xh,yh,zh] = [" << highblockx csp highblocky csp highblockz << "]" << std::endl;

    NNX = NBX*BX;
    NNY = NBY*BY;
    NNZ = NBZ*BZ;
    NNT = NNX*NNY*NNZ;

    //--- Initialize SailFFish solver.
    Real Xg[2] = {lowblockx,highblockx};
    Real Yg[2] = {lowblocky,highblocky};
    Real Zg[2] = {lowblockz,highblockz};

    XN1 = Xg[0]+0.5*H_Grid;
    YN1 = Yg[0]+0.5*H_Grid;
    ZN1 = Zg[0]+0.5*H_Grid;

    X0 = lowblockx;
    Y0 = lowblocky;
    Z0 = lowblockz;
    XN1 = X0 + 0.5*H_Grid;
    YN1 = Y0 + 0.5*H_Grid;
    ZN1 = Z0 + 0.5*H_Grid;

    return Setup(Xg, Yg, Zg, NNX, NNY, NNZ);
}

SFStatus VPM_3D_Solver::Define_Grid_Blocks_Adaptive(VPM_Input *I)
{
    // The solver grid is specified based on the block limits provided.

    std::cout << "Defining grid with adaptive block format." << std::endl;

    H_Grid = I->H_Grid;
    Hx = H_Grid;
    Hy = H_Grid;
    Hz = H_Grid;

    // Set block dimensions
    BX = I->BX;
    BY = I->BY;
    BZ = I->BZ;
    BT = BX*BY*BZ;

    // Set upper & lower bounds which include requested boundary
    // Note: the minimum length of
    HLx = BX*H_Grid;
    HLy = BY*H_Grid;
    HLz = BZ*H_Grid;

    int lowblockx = floor(I->AdaptMinX/HLx);        if (lowblockx >0)   lowblockx--;
    int lowblocky = floor(I->AdaptMinY/HLy);        if (lowblocky >0)   lowblocky--;
    int lowblockz = floor(I->AdaptMinZ/HLz);        if (lowblockz >0)   lowblockz--;
    int highblockx = ceil(I->AdaptMaxX/HLx);        if (highblockx>0)   highblockx--;
    int highblocky = ceil(I->AdaptMaxY/HLy);        if (highblocky>0)   highblocky--;
    int highblockz = ceil(I->AdaptMaxZ/HLz);        if (highblockz>0)   highblockz--;

    NBX = highblockx - lowblockx + 1;
    NBY = highblocky - lowblocky + 1;
    NBZ = highblockz - lowblockz + 1;
    NBT = NBX*NBY*NBZ;

    NNX = NBX*BX;
    NNY = NBY*BY;
    NNZ = NBZ*BZ;
    NNT = NNX*NNY*NNZ;

    // Specify these values within the VPM_Input file
    I->iBX0 = lowblockx;
    I->iBY0 = lowblocky;
    I->iBZ0 = lowblockz;
    I->NBX = NBX;
    I->NBY = NBY;
    I->NBZ = NBZ;

    //--- Initialize SailFFish solver.
    Real Xg[2] = {I->Cx + lowblockx*HLx, I->Cx + (highblockx+1)*HLx};
    Real Yg[2] = {I->Cy + lowblocky*HLy, I->Cy + (highblocky+1)*HLy};
    Real Zg[2] = {I->Cz + lowblockz*HLz, I->Cz + (highblockz+1)*HLz};

    X0 = lowblockx*HLx;
    Y0 = lowblocky*HLy;
    Z0 = lowblockz*HLz;
    XN1 = X0 + 0.5*H_Grid;
    YN1 = Y0 + 0.5*H_Grid;
    ZN1 = Z0 + 0.5*H_Grid;

    std::cout << "Grid domain will be adaptively set to [xl,yl,zl] = [" << Xg[0] <<","<< Yg[0] <<","<< Zg[0] << "]." << std::endl;
    std::cout << "Grid domain will be adaptively set to [xu,yu,zu] = [" << Xg[1] <<","<< Yg[1] <<","<< Zg[1] << "]." << std::endl;

    return Setup(Xg, Yg, Zg, NNX, NNY, NNZ);
}

//--- Grid Mapping operation

int VPM_3D_Solver::Set_Map_Shift(Mapping Map)
{
    int sh = 0;
    switch (Map)
    {
    case (M2):  {sh = 0;  break;}
    case (M4):  {sh = -1; break;}
    case (M4D): {sh = -1; break;}
    case (M6D): {sh = -2; break;}
    default:    {std::cout << "Set_Map_Shift: Mapping unknown" << std::endl;    break;}
    }
    return sh;
}

int VPM_3D_Solver::Set_Map_Stencil_Width(Mapping Map)
{
    int st = 0;
    switch (Map)
    {
    case (M2):  {st = 2;  break;}
    case (M4):  {st = 4; break;}
    case (M4D): {st = 4; break;}
    case (M6D): {st = 6; break;}
    default:    {std::cout << "Set_Map_Stencil: Mapping unknown" << std::endl;    break;}
    }
    return st;
}

void VPM_3D_Solver::Grid_Interp_Coeffs(const RVector &Px, const RVector &Py, const RVector &Pz,
                                       std::vector<dim3s> &IDs, std::vector<bool> &Flags,
                                       Matrix &Mx, Matrix &My, Matrix &Mz, Mapping Map)
{

    // Loop over source positions & specify source id
    int NP = Px.size();

    // Set buffer for mapping/interpolation
    int b = abs(Set_Map_Shift(Map));
    int nc = Set_Map_Stencil_Width(Map);

    Parallel_Kernel(NP) {
        IDs[i] = dim3s(int((Px[i] - XN1)/H_Grid),
                       int((Py[i] - YN1)/H_Grid),
                       int((Pz[i] - ZN1)/H_Grid));
        bool xbuf = (IDs[i].x < b || IDs[i].x >= NNX-b-1);
        bool ybuf = (IDs[i].y < b || IDs[i].y >= NNY-b-1);
        bool zbuf = (IDs[i].z < b || IDs[i].z >= NNZ-b-1);
        if (xbuf || ybuf || zbuf){      // Position out of bounds
            Flags[i] = false;
            Status = GridError;      // HERE! THIS FLAG SHOULD BE ACTIVATED- The IGNORE OOB FLag is only to avoid the catch in QBlade
        }
    }

    // Specify mapping coefficient matrices
    Mx = Matrix(NP,nc), My = Matrix(NP,nc), Mz = Matrix(NP,nc);
    switch (Map)
    {
        case (M2):      {
        Parallel_Kernel(NP)  {
            Real fx = (Px[i] - XN1 - H_Grid*IDs[i].x)/H_Grid;
            Real fy = (Py[i] - YN1 - H_Grid*IDs[i].y)/H_Grid;
            Real fz = (Pz[i] - ZN1 - H_Grid*IDs[i].z)/H_Grid;
            mapM2(fx, Mx(i,0)); mapM2(1.0-fx, Mx(i,1));
            mapM2(fy, My(i,0)); mapM2(1.0-fy, My(i,1));
            mapM2(fz, Mz(i,0)); mapM2(1.0-fz, Mz(i,1));
        }
        break;
        }
        case (M4):      {
            Parallel_Kernel(NP)  {
                Real fx = (Px[i] - XN1 - H_Grid*IDs[i].x)/H_Grid;
                Real fy = (Py[i] - YN1 - H_Grid*IDs[i].y)/H_Grid;
                Real fz = (Pz[i] - ZN1 - H_Grid*IDs[i].z)/H_Grid;
                mapM4(1.0+fx,Mx(i,0));     mapM4(fx,Mx(i,1));  mapM4(1.0-fx,Mx(i,2));  mapM4(2.0-fx,Mx(i,3));
                mapM4(1.0+fy,My(i,0));     mapM4(fy,My(i,1));  mapM4(1.0-fy,My(i,2));  mapM4(2.0-fy,My(i,3));
                mapM4(1.0+fz,Mz(i,0));     mapM4(fz,Mz(i,1));  mapM4(1.0-fz,Mz(i,2));  mapM4(2.0-fz,Mz(i,3));
            }
            break;
        }
        case (M4D):     {
            Parallel_Kernel(NP)  {
                Real fx = (Px[i] - XN1 - H_Grid*IDs[i].x)/H_Grid;
                Real fy = (Py[i] - YN1 - H_Grid*IDs[i].y)/H_Grid;
                Real fz = (Pz[i] - ZN1 - H_Grid*IDs[i].z)/H_Grid;
                mapM4D(1.0+fx,Mx(i,0));    mapM4D(fx,Mx(i,1)); mapM4D(1.0-fx,Mx(i,2)); mapM4D(2.0-fx,Mx(i,3));
                mapM4D(1.0+fy,My(i,0));    mapM4D(fy,My(i,1)); mapM4D(1.0-fy,My(i,2)); mapM4D(2.0-fy,My(i,3));
                mapM4D(1.0+fz,Mz(i,0));    mapM4D(fz,Mz(i,1)); mapM4D(1.0-fz,Mz(i,2)); mapM4D(2.0-fz,Mz(i,3));
            }
            break;
        }
        case (M6D):     {
            Parallel_Kernel(NP)  {
                Real fx = (Px[i] - XN1 - H_Grid*IDs[i].x)/H_Grid;
                Real fy = (Py[i] - YN1 - H_Grid*IDs[i].y)/H_Grid;
                Real fz = (Pz[i] - ZN1 - H_Grid*IDs[i].z)/H_Grid;
                mapM6D(2.0+fx,Mx(i,0)); mapM6D(1.0+fx,Mx(i,1)); mapM6D(fx,Mx(i,2)); mapM6D(1.0-fx,Mx(i,3)); mapM6D(2.0-fx,Mx(i,4)); mapM6D(3.0-fx,Mx(i,5));
                mapM6D(2.0+fy,My(i,0)); mapM6D(1.0+fy,My(i,1)); mapM6D(fy,My(i,2)); mapM6D(1.0-fy,My(i,3)); mapM6D(2.0-fy,My(i,4)); mapM6D(3.0-fy,My(i,5));
                mapM6D(2.0+fz,Mz(i,0)); mapM6D(1.0+fz,Mz(i,1)); mapM6D(fz,Mz(i,2)); mapM6D(1.0-fz,Mz(i,3)); mapM6D(2.0-fz,Mz(i,4)); mapM6D(3.0-fz,Mz(i,5));
            }
            break;
        }
        default:        {std::cout << "VPM_3D_Solver::Map_Grid_Sources: Mapping unknown" << std::endl;    break;}
    }
}

void VPM_3D_Solver::Map_from_Grid(const RVector &Px, const RVector &Py, const RVector &Pz,
                                  const RVector &Gx, const RVector &Gy, const RVector &Gz,
                                  RVector &uX, RVector &uY, RVector &uZ, Mapping Map)
{
    // This function maps the input particle to the specified destination grid g_i

    if (Px.empty() || Py.empty() || Pz.empty()) return;
    int NP = Px.size();
    std::vector<dim3s> ID(NP);       // Array of grid ids
    std::vector<bool> Flags(NP,true);    // Array of boundary flag
    Matrix Mx, My, Mz;
    Grid_Interp_Coeffs(Px, Py, Pz, ID, Flags, Mx, My, Mz, Map);

    // Specify constants for mapping
    int idsh = Set_Map_Shift(Map);
    int nc = Set_Map_Stencil_Width(Map);

    // Extract grid values
    Parallel_Kernel(NP) {
        Real tUx = 0., tUy = 0., tUz = 0.;
        if (Flags[i]){                         // Jump out in case node is out of bounds
            for (int x=0; x<nc; x++){
                for (int y=0; y<nc; y++){
                    for (int z=0; z<nc; z++){
                        int ids = GID(ID[i].x+idsh+x, ID[i].y+idsh+y, ID[i].z+idsh+z, NNX, NNY, NNZ);
                        Real Fac =  Mx(i,x) * My(i,y) * Mz(i,z);
                        tUx += Fac*Gx[ids];
                        tUy += Fac*Gy[ids];
                        tUz += Fac*Gz[ids];
                    }
                }
            }
        }
        uX[i] = tUx;
        uY[i] = tUy;
        uZ[i] = tUz;
    }
}


void VPM_3D_Solver::Map_to_Grid(const RVector &Px, const RVector &Py, const RVector &Pz,
                                const RVector &Ox, const RVector &Oy, const RVector &Oz,
                                RVector &gX, RVector &gY, RVector &gZ, Mapping Map)
{
    // This function maps the input particle to the specified destination grid g_i

    if (Px.empty() || Py.empty() || Pz.empty()) return;
    int NP = Px.size();
    std::vector<dim3s> ID(NP);          // Array of grid ids
    std::vector<bool> Flags(NP,true);   // Array of boundary flag
    Matrix Mx, My, Mz;
    Grid_Interp_Coeffs(Px, Py, Pz, ID, Flags, Mx, My, Mz, Map);

    // Specify constants for mapping
    int idsh = Set_Map_Shift(Map);
    int nc = Set_Map_Stencil_Width(Map);

    // Map to p_Array grid (serial- to avoid race conditions)
    for (int i=0; i<NP; i++){
        if (Flags[i]){
            for (int x=0; x<nc; x++){
                for (int y=0; y<nc; y++){
                    for (int z=0; z<nc; z++){
                        int idr = GID(ID[i].x+idsh+x, ID[i].y+idsh+y, ID[i].z+idsh+z, NNX, NNY, NNZ);
                        Real Fac =  Mx(i,x) * My(i,y) * Mz(i,z);
                        gX[idr] += Fac*Ox[i];
                        gY[idr] += Fac*Oy[i];
                        gZ[idr] += Fac*Oz[i];
                    }
                }
            }
        }
    }
}

//--- New approach to avoid auxiliary grid.

void VPM_3D_Solver::Get_External_Forcing_Nodes(std::vector<Vector3> &Om, std::vector<Vector3> &Pos)
{
    // For the hybrid solver, we need to know at each timestep the influence of the
    // temporary mapped vorticity on the field. We export here the nodes for this.
    Real dV = Hx*Hy*Hz;
    // std::cout << dV  << std::endl;
    // for (size_t i=0; i<size(Ext_Forcing); i++){
    for (ParticleMap p : Ext_Forcing){
        dim3 nid = p.cartid;    //std::get<0>(Ext_Forcing[i]);
        Vector3 no = p.Vort;    //std::get<1>(Ext_Forcing[i]);
        Om.push_back(no*dV);
        Pos.push_back(Vector3(XN1 + nid.x*Hx, YN1 + nid.y*Hy, ZN1 + nid.z*Hz));
    }
}

void VPM_3D_Solver::Process_Cells(const RVector &Px, const RVector &Py, const RVector &Pz,
                                  const RVector &Ox, const RVector &Oy, const RVector &Oz,
                                  std::vector<CellMap> &IDs, Mapping Map)
{

    // Loop over source positions & specify source id
    int NP = Px.size();

    // Set buffer for mapping/interpolation
    uint b = abs(Set_Map_Shift(Map));

    std::vector<CellMap> Cells(NP);             // Full array of cells

    // Step 1: Specify node ids
    Parallel_Kernel(NP) {

        dim3 tID = dim3(int((Px[i] - XN1)/H_Grid),
                        int((Py[i] - YN1)/H_Grid),
                        int((Pz[i] - ZN1)/H_Grid));
        bool xbuf = (tID.x < b || tID.x >= NNX-b-1);
        bool ybuf = (tID.y < b || tID.y >= NNY-b-1);
        bool zbuf = (tID.z < b || tID.z >= NNZ-b-1);
        Cells[i].idin = i;
        Cells[i].Pos = Vector3(Px[i],Py[i],Pz[i]);
        Cells[i].id = GID(tID.x,tID.y,tID.z,NNX,NNY,NNZ);   // Set particle cell global id
        Cells[i].id3 = tID;                                 // Set particle cell cartesian id
        Cells[i].Weight = Vector3(Ox[i],Oy[i],Oz[i]);       // Set particle cell weight
        if (xbuf || ybuf || zbuf)   Cells[i].Valid = false; // Mark flag to check whether cell is within the domain
        else                        Cells[i].Valid = true;
    }

    // Step 2: Remove invalid cell mappings and generate list of pointers to cell objects
    std::vector<CellMap> InvCells, ValCells;
    for (CellMap C : Cells){
        if (C.Valid)    ValCells.push_back(C);
        else            InvCells.push_back(C);
    }

    // Step 3: Sort remaining cell mappings based on global id
    // std::sort(IDs.begin(), IDs.end(), [](const CellMap& a, const CellMap& b) { return a.id < b.id; });
    std::sort(ValCells.begin(), ValCells.end(), [](const CellMap& a, const CellMap& b) { return a.id < b.id; });

    // Step 2: Remove invalid cell mappings and generate list of pointers to cell objects
    // std::vector<CellMap*> unsortedIDs;
    // for (size_t i=0; i<size(Cells); i++)   {if (Valid[i])   unsortedIDs.push_back(&Cells[i]);}

    // Step 3: Sort remaining cell mappings based on global id
    // std::sort(IDs.begin(), IDs.end(), [](const CellMap& a, const CellMap& b) { return a.id < b.id; });

    // // Step 3: Sort objects (use pointers!)
    // std::vector<CellMap*> sortedpIDs;
    // for (auto& cell : IDs) sortedPtrs.push_back(&cell);
    // std::sort(unsortedIDs.begin(), unsortedIDs.end(), [](const CellMap* a, const CellMap* b) { return a->id < b->id; });
    // for (CellMap* p : unsortedIDs) IDs.push_back(*p);
    // for (auto& cell : IDs) IDs.push_back(&cell);

    // Specify mapping coefficient matrices
    NP = int(ValCells.size());
    switch (Map)
    {
        case (M2):      {
            Parallel_Kernel(NP)  {
                Real fx = (ValCells[i].Pos(0) - XN1 - H_Grid*ValCells[i].id3.x)/H_Grid;
                Real fy = (ValCells[i].Pos(1) - YN1 - H_Grid*ValCells[i].id3.y)/H_Grid;
                Real fz = (ValCells[i].Pos(2) - ZN1 - H_Grid*ValCells[i].id3.z)/H_Grid;
                Matrix mMat = Matrix(3,2);
                mapM2(fx, mMat(0,0)); mapM2(1.0-fx, mMat(0,1));
                mapM2(fy, mMat(1,0)); mapM2(1.0-fy, mMat(1,1));
                mapM2(fz, mMat(2,0)); mapM2(1.0-fz, mMat(2,1));
                ValCells[i].Coeff = mMat;
            }
            break;
        }
        case (M4):      {
            Parallel_Kernel(NP)  {
                Real fx = (ValCells[i].Pos(0) - XN1 - H_Grid*ValCells[i].id3.x)/H_Grid;
                Real fy = (ValCells[i].Pos(1) - YN1 - H_Grid*ValCells[i].id3.y)/H_Grid;
                Real fz = (ValCells[i].Pos(2) - ZN1 - H_Grid*ValCells[i].id3.z)/H_Grid;
                Matrix mMat = Matrix(3,4);
                mapM4(1.0+fx,mMat(0,0));     mapM4(fx,mMat(0,1));  mapM4(1.0-fx,mMat(0,2));  mapM4(2.0-fx,mMat(0,3));
                mapM4(1.0+fy,mMat(1,0));     mapM4(fy,mMat(1,1));  mapM4(1.0-fy,mMat(1,2));  mapM4(2.0-fy,mMat(1,3));
                mapM4(1.0+fz,mMat(2,0));     mapM4(fz,mMat(2,1));  mapM4(1.0-fz,mMat(2,2));  mapM4(2.0-fz,mMat(2,3));
                ValCells[i].Coeff = mMat;
            }
            break;
        }
        case (M4D):     {
            Parallel_Kernel(NP)  {
                Real fx = (ValCells[i].Pos(0) - XN1 - H_Grid*ValCells[i].id3.x)/H_Grid;
                Real fy = (ValCells[i].Pos(1) - YN1 - H_Grid*ValCells[i].id3.y)/H_Grid;
                Real fz = (ValCells[i].Pos(2) - ZN1 - H_Grid*ValCells[i].id3.z)/H_Grid;
                Matrix mMat = Matrix(3,4);
                mapM4D(1.0+fx,mMat(0,0));    mapM4D(fx,mMat(0,1)); mapM4D(1.0-fx,mMat(0,2)); mapM4D(2.0-fx,mMat(0,3));
                mapM4D(1.0+fy,mMat(1,0));    mapM4D(fy,mMat(1,1)); mapM4D(1.0-fy,mMat(1,2)); mapM4D(2.0-fy,mMat(1,3));
                mapM4D(1.0+fz,mMat(2,0));    mapM4D(fz,mMat(2,1)); mapM4D(1.0-fz,mMat(2,2)); mapM4D(2.0-fz,mMat(2,3));
                ValCells[i].Coeff = mMat;
            }
            break;
        }
        case (M6D):     {
            Parallel_Kernel(NP)  {
                Real fx = (ValCells[i].Pos(0) - XN1 - H_Grid*ValCells[i].id3.x)/H_Grid;
                Real fy = (ValCells[i].Pos(1) - YN1 - H_Grid*ValCells[i].id3.y)/H_Grid;
                Real fz = (ValCells[i].Pos(2) - ZN1 - H_Grid*ValCells[i].id3.z)/H_Grid;
                Matrix mMat = Matrix(3,6);
                mapM6D(2.0+fx,mMat(0,0)); mapM6D(1.0+fx,mMat(0,1)); mapM6D(fx,mMat(0,2)); mapM6D(1.0-fx,mMat(0,3)); mapM6D(2.0-fx,mMat(0,4)); mapM6D(3.0-fx,mMat(0,5));
                mapM6D(2.0+fy,mMat(1,0)); mapM6D(1.0+fy,mMat(1,1)); mapM6D(fy,mMat(1,2)); mapM6D(1.0-fy,mMat(1,3)); mapM6D(2.0-fy,mMat(1,4)); mapM6D(3.0-fy,mMat(1,5));
                mapM6D(2.0+fz,mMat(2,0)); mapM6D(1.0+fz,mMat(2,1)); mapM6D(fz,mMat(2,2)); mapM6D(1.0-fz,mMat(2,3)); mapM6D(2.0-fz,mMat(2,4)); mapM6D(3.0-fz,mMat(2,5));
                ValCells[i].Coeff = mMat;
            }
            break;
        }
        default:        {std::cout << "VPM_3D_Solver::Map_Grid_Sources: Mapping unknown" << std::endl;    break;}
    }

    // Now fill output array with invalid and valid cells.
    StdAppend(IDs, InvCells);
    StdAppend(IDs, ValCells);
}

void VPM_3D_Solver::Domain_Bounds(std::vector<CellMap> &CD, dim3 &Lower, dim3 &Upper)
{
    CellMap min_x = *std::min_element(CD.begin(), CD.end(), [](const CellMap& a, const CellMap& b) { return a.id3.x < b.id3.x; });
    CellMap max_x = *std::max_element(CD.begin(), CD.end(), [](const CellMap& a, const CellMap& b) { return a.id3.x < b.id3.x; });
    CellMap min_y = *std::min_element(CD.begin(), CD.end(), [](const CellMap& a, const CellMap& b) { return a.id3.y < b.id3.y; });
    CellMap max_y = *std::max_element(CD.begin(), CD.end(), [](const CellMap& a, const CellMap& b) { return a.id3.y < b.id3.y; });
    CellMap min_z = *std::min_element(CD.begin(), CD.end(), [](const CellMap& a, const CellMap& b) { return a.id3.z < b.id3.z; });
    CellMap max_z = *std::max_element(CD.begin(), CD.end(), [](const CellMap& a, const CellMap& b) { return a.id3.z < b.id3.z; });
    Lower = dim3(min_x.id3.x, min_y.id3.y, min_z.id3.z);
    Upper = dim3(max_x.id3.x, max_y.id3.y, max_z.id3.z);
    // std::cout << Lower.x csp Lower.y csp Lower.z csp  Upper.x csp Upper.y csp Upper.z << std::endl;
}

void VPM_3D_Solver::Map_Source_Nodes(const RVector &Px, const RVector &Py, const RVector &Pz,
                                     const RVector &Ox, const RVector &Oy, const RVector &Oz, std::vector<ParticleMap> &GP, Mapping Map)
{
    // This function takes the array of nodes specified at Position Px,y,z with strength Ox,y,z and returns
    // the representative grid ids and weights at OrderedParticles in GP.

    if (Px.empty() || Py.empty() || Pz.empty()) return;

    // Process nodes
    std::vector<CellMap> CellData;
    Process_Cells(Px, Py, Pz, Ox, Oy, Oz, CellData, Map);

    // The Process_Cells function returns a list of both valid and invalid cell locations.
    // These are ignored for invalid domain IDs.
    std::vector<CellMap> ValCells;
    for (CellMap C : CellData) {if (C.Valid) ValCells.push_back(C);}

    if (ValCells.empty()) return;

    // Accumulate equivalent node positions
    std::vector<CellMap> Maps;      // Array of sorted maps
    Maps.emplace_back(ValCells[0]);       // Add first element
    Maps[0].Weights.push_back(ValCells[0].Weight);
    Maps[0].Coeffs.push_back(ValCells[0].Coeff);
    for (size_t i=1; i<ValCells.size(); ++i) {
        if (ValCells[i].id == Maps.back().id){
            Maps.back().Weights.push_back(ValCells[i].Weight);
            Maps.back().Coeffs.push_back(ValCells[i].Coeff);
        }
        else {
            Maps.emplace_back(ValCells[i]);       // Add next element
            Maps.back().Weights.push_back(ValCells[i].Weight);
            Maps.back().Coeffs.push_back(ValCells[i].Coeff);
        }
    }

    // Check domain limits
    dim3 dLow,dUpp;
    Domain_Bounds(Maps,dLow,dUpp);
    dLow.x -= 3; dLow.y -= 3; dLow.z -= 3;
    dUpp.x += 3; dUpp.y += 3; dUpp.z += 3;
    uint nx = dUpp.x-dLow.x+1, ny = dUpp.y-dLow.y+1, nz = dUpp.z-dLow.z+1, nt = nx*ny*nz;

    // Generate grid of nodes
    RVector mox(nt,0.), moy(nt,0.), moz(nt,0.);

    // Map particles to temp grid
    int idsh = Set_Map_Shift(Map);
    int nc = Set_Map_Stencil_Width(Map);
    for (size_t p=0; p<Maps.size(); p++){

        dim3 rid(Maps[p].id3.x-dLow.x, Maps[p].id3.y-dLow.y, Maps[p].id3.z-dLow.z); // Local position of cell within sparse grid

        // Append onto temp grid
        #pragma omp parallel for collapse(3)
        for (int i=0; i<nc; i++){
            for (int j=0; j<nc; j++){
                for (int k=0; k<nc; k++){
                    dim3 mid(rid.x+idsh+i, rid.y+idsh+j, rid.z+idsh+k);     // Local dim3 id of node within sparse grid
                    int gid = GID(mid.x,mid.y,mid.z,nx,ny,nz);              // Local id of node within sparse grid
                    for (size_t pm=0; pm<Maps[p].Weights.size(); pm++){
                        Matrix M = Maps[p].Coeffs[pm];
                        Vector3 Om = Maps[p].Weights[pm];
                        Real Fac = M(0,i)*M(1,j)*M(2,k);
                        mox[gid] += Fac*Om(0);
                        moy[gid] += Fac*Om(1);
                        moz[gid] += Fac*Om(2);
                    }
                }
            }
        }
    }

    //  std::cout << "Now grid"<< std::endl;
    // Approach 1: Fullgrid
    // std::vector<OrdPart> Parts;
    for (uint i=0; i<nx; i++) {
        for (uint j=0; j<ny; j++) {
            for (uint k=0; k<nz; k++) {
                uint lid = GID(i,j,k,nx,ny,nz);
                Vector3 v(mox[lid],moy[lid],moz[lid]);
                if (v.norm()>0){    // GP.emplace_back(OrdPart(dim3(i+dLow.x,j+dLow.y,k+dLow.z), v, 0));
                    ParticleMap p;
                    p.Vort = v;
                    p.cartid = dim3(i+dLow.x,j+dLow.y,k+dLow.z)  ;
                    GP.emplace_back(p);
                }

            }
        }
    }
}

void VPM_3D_Solver::Store_Grid_Node_Sources(const RVector &Px, const RVector &Py, const RVector &Pz, const RVector &Ox, const RVector &Oy, const RVector &Oz, Mapping Map)
{
    // Processes the external forcing nodes.
    if (Px.empty() || Py.empty() || Pz.empty()) return;

    Ext_Forcing.clear();
    Map_Source_Nodes(Px, Py, Pz, Ox, Oy, Oz, Ext_Forcing, Map);
    // std::cout << Ext_Forcing.size() << " Nodes mapped with " << Px.size() << " input nodes." << std::endl;
}

void VPM_3D_Solver::Get_Ext_Velocity(const RVector &Px, const RVector &Py, const RVector &Pz,
                                     RVector &Ux, RVector &Uy, RVector &Uz, Mapping Map)
{
    // This function uses the mapped source particles to calculate the mapped velocity.

    if (Px.empty() || Py.empty() || Pz.empty()) return;

    // Process nodes
    std::vector<CellMap> CellData;
    RVector dumOx(Px.size(),0), dumOy(Px.size(),0), dumOz(Px.size(),0);    // Dummy source arrays
    Process_Cells(Px, Py, Pz, dumOx, dumOy, dumOz, CellData, Map);

    // The Process_Cells function returns a list of both valid and invalid cell locations.
    // These are ignored for invalid domain IDs.
    std::vector<CellMap> ValCells;
    for (CellMap C : CellData){
        if (C.Valid) ValCells.push_back(C);
        else {
            Ux[C.idin] = 0.0;
            Uy[C.idin] = 0.0;
            Uz[C.idin] = 0.0;
        }
    }

    if (ValCells.empty()) return;

    // Accumulate equivalent receiver node positions
    std::vector<CellMap> Maps;      // Array of sorted maps
    Maps.emplace_back(ValCells[0]);       // Add first element
    Maps[0].Coeffs.push_back(ValCells[0].Coeff);
    for (size_t i=1; i<ValCells.size(); ++i) {
        if (ValCells[i].id == Maps.back().id){
            Maps.back().Coeffs.push_back(ValCells[i].Coeff);
        }
        else {
            Maps.emplace_back(ValCells[i]);       // Add next element
            Maps.back().Coeffs.push_back(ValCells[i].Coeff);
        }
    }

    // Check domain limits and add buffer
    dim3 dLow,dUpp;
    Domain_Bounds(Maps,dLow,dUpp);
    dLow.x -= 3; dLow.y -= 3; dLow.z -= 3;
    dUpp.x += 3; dUpp.y += 3; dUpp.z += 3;
    uint nx = dUpp.x-dLow.x+1, ny = dUpp.y-dLow.y+1, nz = dUpp.z-dLow.z+1, nt = nx*ny*nz;

    // Generate grid of flags
    std::vector<bool> fox(nt,false);

    // Specify locations where influence needs to be calculated
    int idsh = Set_Map_Shift(Map);
    int nc = Set_Map_Stencil_Width(Map);
    for (size_t p=0; p<Maps.size(); p++){

        dim3 rid(Maps[p].id3.x-dLow.x, Maps[p].id3.y-dLow.y, Maps[p].id3.z-dLow.z); // Local position of cell within sparse grid

        // Set activity
        #pragma omp parallel for collapse(3)
        for (int i=0; i<nc; i++){
            for (int j=0; j<nc; j++){
                for (int k=0; k<nc; k++){
                    dim3 mid(rid.x+idsh+i, rid.y+idsh+j, rid.z+idsh+k);     // Local dim3 id of node within sparse grid
                    int gid = GID(mid.x,mid.y,mid.z,nx,ny,nz);              // Local id of node within sparse grid
                    fox[gid] = true;
                }
            }
        }
    }

    // Identify probe points
    std::vector<ParticleMap> RcvrNodes;
    for (uint i=0; i<nx; i++) {
        for (uint j=0; j<ny; j++) {
            for (uint k=0; k<nz; k++) {
                uint lid = GID(i,j,k,nx,ny,nz);
                if (fox[lid]){// RcvrNodes.emplace_back(OrdPart(dim3(i+dLow.x,j+dLow.y,k+dLow.z), Vector3::Zero(), 0));
                    ParticleMap p;
                    // p.Vort = v;
                    p.cartid = dim3(i+dLow.x,j+dLow.y,k+dLow.z);
                    RcvrNodes.emplace_back(p);


                    // dim3(i+dLow.x,j+dLow.y,k+dLow.z)
                }
            }
        }
    }

    // Generate grid of potential/velocity
    RVector ugx(nt,0), ugy(nt,0), ugz(nt,0);

    // Calculate influence of source terms
    Real dV = Hx*Hy*Hz;
    Real Sigma = 2.0*Hx;
    OpenMPfor
    for (size_t p=0; p<RcvrNodes.size(); p++){                               // Loop over evaluation points
        dim3 rcvr = RcvrNodes[p].cartid; //std::get<0>(RcvrNodes[p]);           // Receiver global id
        Vector3 prcvr(XN1+rcvr.x*Hx, YN1+rcvr.y*Hy, ZN1+rcvr.z*Hz);             // Receiver global position

        Vector3 tPhi = Vector3::Zero();
        for (size_t s=0; s<Ext_Forcing.size(); s++){
            dim3 sid = Ext_Forcing[s].cartid;   //std::get<0>(Ext_Forcing[s]);                     // Global id of source node
            Vector3 psrc(XN1+sid.x*Hx, YN1+sid.y*Hy, ZN1+sid.z*Hz);     // Global position of source node
            Vector3 alpha = Ext_Forcing[s].Vort*dV;    //std::get<1>(Ext_Forcing[s])*dV;             // Circulation of source node
            Vector3 r = prcvr-psrc;
            Real rn = r.norm();
            if (rn!=0.) tPhi += BS(rn,Sigma)*r.cross(alpha);
            // if (rn>0.1*Hx) tPhi += BS(rn,Sigma)*r.cross(alpha);         // Modified influence... ensure no node self-influence
        }

        dim3 idl3 = dim3(rcvr.x-dLow.x, rcvr.y-dLow.y, rcvr.z-dLow.z);  // Receiver local id
        int idl = GID(idl3.x, idl3.y, idl3.z, nx, ny, nz);
        ugx[idl] = tPhi(0);
        ugy[idl] = tPhi(1);
        ugz[idl] = tPhi(2);
    }

    // std::cout << "Source nodes" << std::endl;
    // for (size_t i=0; i<size(Ext_Forcing); i++) {
    //     dim3 sid = std::get<0>(Ext_Forcing[i]);
    //     Vector3 psrc(XN1+sid.x*Hx, YN1+sid.y*Hy, ZN1+sid.z*Hz);     // Global position of source node
    //     std::cout << psrc(0) csp psrc(1) csp psrc(2) << std::endl;
    // }

    // std::cout << "Probe nodes" << std::endl;
    // for (size_t i=0; i<size(Px); i++) {
    //     std::cout << Px[i] csp Py[i] csp Pz[i] << std::endl;
    // }

    // std::cout << "Receiver nodes" << std::endl;
    // for (size_t i=0; i<size(RcvrNodes); i++) {
    //     dim3 rcvr = std::get<0>(RcvrNodes[i]);                                  // Receiver global id
    //     Vector3 prcvr(XN1+rcvr.x*Hx, YN1+rcvr.y*Hy, ZN1+rcvr.z*Hz);         // Receiver global position
    //     std::cout << prcvr(0) csp prcvr(1) csp prcvr(2) << std::endl;
    // }

    // Now carry out interpolation on the local grid and store outputs
    OpenMPfor
    for (size_t p=0; p<ValCells.size(); p++){                 // Loop over evaluation points
        dim3 rid = ValCells[p].id3;                              // Receiver global id
        Matrix M = ValCells[p].Coeff;
        Vector3 Phi = Vector3::Zero();
        for (int i=0; i<nc; i++){
            for (int j=0; j<nc; j++){
                for (int k=0; k<nc; k++){
                    // int pid = GID(rid.x+idsh-dLow.x, rid.y+idsh-dLow.y, rid.z+idsh-dLow.z, nx, ny, nz); // Local id of node within sparse grid
                    dim3 sidg(rid.x+idsh+i,rid.y+idsh+j,rid.z+idsh+k);      // Global id of source node
                    dim3 sidl(sidg.x-dLow.x,sidg.y-dLow.y,sidg.z-dLow.z);   // Local id of source node
                    int pid = GID(sidl.x, sidl.y, sidl.z, nx, ny, nz);      // Local index of source
                    Real Fac = M(0,i)*M(1,j)*M(2,k);
                    Phi(0) += Fac*ugx[pid];
                    Phi(1) += Fac*ugy[pid];
                    Phi(2) += Fac*ugz[pid];
                }
            }
        }
        Ux[ValCells[p].idin] = Phi(0);
        Uy[ValCells[p].idin] = Phi(1);
        Uz[ValCells[p].idin] = Phi(2);
    }
}


//--- Output

void VPM_3D_Solver::Generate_Summary(std::string Filename)
{
    // This function generates a summary of the simulation

    const auto now = std::chrono::system_clock::now();
    const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    std::cout << "The system clock is currently at " << std::ctime(&t_c);

    //    chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    //    chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    //    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;

    //--- Create output director if non existent
    std::string OutputDirectory = "Output/" + OutputFolder;
    Create_Directory(OutputDirectory);
    Output_Filename = Filename;
    std::string FilePath = OutputDirectory + "/" + Output_Filename;
    std::ofstream file;
    file.open(FilePath, std::ofstream::out | std::ofstream::trunc); // Clear!
    file.close();

    // Print simulation summary to file.
    file.open(FilePath, std::ios_base::app);
    if (file.is_open())
    {
        file << "#-------------------------------------------------------------------------#" << std::endl;
        file << "#-----------------VoldeVort VPM Solver Summary File-----------------------#" << std::endl;
        file << "#-------------------------------------------------------------------------#" << std::endl;
        //        file << "\n" << std::endl;
        file << "Simulation began: " << std::ctime(&t_c) << std::endl;
        //        file << "\n" << std::endl;
        file << "#-----------Grid Variables-----------------#" << std::endl;
        file << "XLOW       " << Xl     << "    (Lower X Limit)" <<std::endl;
        file << "YLOW       " << Yl     << "    (Lower Y Limit)" <<std::endl;
        file << "ZLOW       " << Zl     << "    (Lower Z Limit)" <<std::endl;
        file << "XUPPER     " << Xu     << "    (Upper X Limit)" <<std::endl;
        file << "YUPPER     " << Yu     << "    (Upper Y Limit)" <<std::endl;
        file << "ZUPPER     " << Zu     << "    (Upper Z Limit)" <<std::endl;
        file << "XLENGTH    " << Lx     << "    (Domain X length)" <<std::endl;
        file << "YLENGTH    " << Ly     << "    (Domain Y length)" <<std::endl;
        file << "ZLENGTH    " << Lz     << "    (Domain Z length)" <<std::endl;
        file << "NX         " << NNX    << "    (Number of cells in X direction)" <<std::endl;
        file << "NY         " << NNY    << "    (Number of cells in Y direction)" <<std::endl;
        file << "NZ         " << NNZ    << "    (Number of cells in Z direction)" <<std::endl;
        file << "NTot       " << NNT    << "    (Total number of cells)" <<std::endl;
        file << "HX         " << Hx     << "    (Cell width in X direction)" <<std::endl;
        file << "HY         " << Hy     << "    (Cell width in Y direction)" <<std::endl;
        file << "HZ         " << Hz     << "    (Cell width in Z direction)" <<std::endl;
        file << "dV         " << Hx*Hy*Hz<< "    (Cell volume)" <<std::endl;
        if (Grid==REGULAR){     file << "GRID       " << "REGULAR       (Grid type)" <<std::endl;}
        if (Grid==STAGGERED){   file << "GRID       " << "STAGGERED     (Grid type)" <<std::endl;}
        if (SolverMap==M2)  file << "SOLVER MAPPING SCHEME: M2" <<std::endl;
        if (SolverMap==M4)  file << "SOLVER MAPPING SCHEME: M4" <<std::endl;
        if (SolverMap==M4D) file << "SOLVER MAPPING SCHEME: M4D" <<std::endl;
        if (SolverMap==M6D) file << "SOLVER MAPPING SCHEME: M6D" <<std::endl;
        if (RemeshMap==M2)  file << "REMESH MAPPING SCHEME: M2" <<std::endl;
        if (RemeshMap==M4)  file << "REMESH MAPPING SCHEME: M4" <<std::endl;
        if (RemeshMap==M4D) file << "REMESH MAPPING SCHEME: M4D" <<std::endl;
        if (RemeshMap==M6D) file << "REMESH MAPPING SCHEME: M6D" <<std::endl;
        file << "\n" << std::endl;
        file << "#-----------Solver Variables-----------------#" << std::endl;
        if (FDOrder==CD2)   file << "FIN. DIFF  2    (Finite difference order)" <<std::endl;
        if (FDOrder==CD4)   file << "FIN. DIFF  4    (Finite difference order)" <<std::endl;
        if (FDOrder==CD6)   file << "FIN. DIFF  6    (Finite difference order)" <<std::endl;
        if (FDOrder==CD8)   file << "FIN. DIFF  8    (Finite difference order)" <<std::endl;
        std::string KER;
        switch (Greens_Kernel)
        {
        case (HEJ_S0):  {KER = "HEJ_S0"; break;}
        case (HEJ_G2):  {KER = "HEJ_G2"; break;}
        case (HEJ_G4):  {KER = "HEJ_G4"; break;}
        case (HEJ_G6):  {KER = "HEJ_G6"; break;}
        case (HEJ_G8):  {KER = "HEJ_G8"; break;}
        case (HEJ_G10): {KER = "HEJ_G10"; break;}
        default:        {KER = "NONE"; break;}
        }
        file << "KER        "   << KER                  <<"    (Kernel type)" <<std::endl;
        file << "REMESH     "   << NRemesh              << "    (Remeshing Frequency)" <<std::endl;
        if (DivFilt)        file << "REPROJECT  "   << "TRUE"   << "    (Divergence Filtering Flag)" <<std::endl;
        else                file << "REPROJECT  "   << "FALSE"   << "   (Divergence Filtering Flag)" <<std::endl;
        if (MagFiltFac>0)   file << "MAGFILT    "   << "TRUE"   << "    (Magnitude Filtering Flag)" <<std::endl;
        else                file << "MAGFILT    "   << "FALSE"   << "   (Magnitude Filtering Flag)" <<std::endl;
        file << "MAGFILTFAC "   << MagFiltFac           << "    (Magnitude filtering factor)" <<std::endl;
        file << "\n" << std::endl;
        file << "#-----------Environmental Variables-----------------#" << std::endl;
        file << "UINFX      "   << Ux       << "    (Freestream velocity in x direction)" <<std::endl;
        file << "UINFY      "   << Uy       << "    (Freestream velocity in x direction)" <<std::endl;
        file << "UINFZ      "   << Uz       << "    (Freestream velocity in x direction)" <<std::endl;
        file << "KINVISC    "   << KinVisc  << "    (Kinematic viscosity of fluid)" <<std::endl;
        file << "DENSITY    "   << Rho      << "    (Density of fluid)" <<std::endl;
        file << "\n" << std::endl;
        file << "#-----------Timestepping Variables-----------------#" << std::endl;
        std::string TS;
        switch (Integrator)
        {
        case (EF):      {TS = "EF"; break;}
        case (EM):      {TS = "EM"; break;}
        case (RK2):     {TS = "RK2"; break;}
        case (AB2LF):   {TS = "AB2LF"; break;}
        case (RK3):     {TS = "RK3"; break;}
        case (RK4):     {TS = "RK4"; break;}
        default:        {TS = "NONE"; break;}
        }
        file << "INT        "   << TS                  <<"    (Integration scheme)" <<std::endl;
        file << "dT        "   << dT                  <<"    (Timestep size [s])" <<std::endl;
        file << "\n" << std::endl;
    }

    file.close();

    // Now begin timer.
    Sim_begin = std::chrono::steady_clock::now();    // Begin clock
}

void VPM_3D_Solver::Generate_Summary_End()
{
    // This function generates a summary of the simulation

    // End simulation clock & calculate total time
    Sim_end = std::chrono::steady_clock::now();
    auto durmil = std::chrono::duration_cast<std::chrono::milliseconds> (Sim_end - Sim_begin).count();
    auto dursec = std::chrono::duration_cast<std::chrono::seconds> (Sim_end - Sim_begin).count();
    auto durmin = std::chrono::duration_cast<std::chrono::minutes> (Sim_end - Sim_begin).count();

    const auto now = std::chrono::system_clock::now();
    const std::time_t t_c = std::chrono::system_clock::to_time_t(now);  // Define vars for date & time display below

    //--- Create output director if non existent
    std::string OutputDirectory = "Output/" + OutputFolder;
    Create_Directory(OutputDirectory);
    std::string FilePath = OutputDirectory + "/" + Output_Filename;
    std::ofstream file;

    // Print simulation summary to file.
    file.open(FilePath, std::ios_base::app);
    if (file.is_open())
    {
        file << "#----------- Simulation Summary -----------------#" << std::endl;
        file << "Simulation completed: " << std::ctime(&t_c) << std::endl;
        file << "Simulation wall clock duration: " << durmil << " milliseconds." << std::endl;
        file << "Simulation wall clock duration: " << dursec << " seconds." << std::endl;
        file << "Simulation wall clock duration: " << durmin << " minutes." << std::endl;
    }

    file.close();

}

}
