//-----------------------------------------------------------------------------
//-------------------------Export Grid Functions-------------------------------
//-----------------------------------------------------------------------------

#include "Solver_Base.h"

// Additions to compensate for old MinGW & c++17
// #include <iomanip>
// #include <direct.h>
// static void create_directory(const char *path)  {_mkdir(path);}

namespace SailFFish
{
template <typename T>
void convertToBigEndian(T& value) {
    char* ptr = reinterpret_cast<char*>(&value);
    std::reverse(ptr, ptr + sizeof(T));
}

// static int const vtkPrecision = 8;     // High Precision, high memory
// static int const vtkWidth = 17;          // Width for parsing high precision number

static int const vtkPrecision = 3;     // Low Precision, low of memory
static int const vtkWidth = 11;        // Width for parsing low precision number

//---------------------------
//--- 2D Scalar solver ------
//---------------------------

void Solver_2D_Scalar::Create_vtk()
{
    // When the solver has run, the grid values are stored as a vtk file.
    // This allows the user to visualise the grid in Paraview.

    // Grid sizes
    int CX = gNX;
    int CY = gNY;
    if (gNX>NX) CX = NX;
    if (gNY>NY) CY = NY;

    std::ostringstream str;
    std::string ifilename;

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Individual .vtk files
    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

    //--- Create output director if not existing
    std::string OutputDirectory = "Output";
    Create_Directory(OutputDirectory);
    std::string filename = "./Output/" + vtk_Name;
    str.str(""); // clear str
    std::ofstream vtkfile( filename.c_str() );
    vtkfile.precision(vtkPrecision);
    if(!vtkfile.is_open())
    {
        std::cerr << "ERROR: cannot open vtkfile." << std::endl;
        return;
    }

    // Generate text for file
    Real minX = Xl, minY = Yl;
    if (Grid==STAGGERED)
    {
        minX += 0.5*Hx;
        minY += 0.5*Hy;
    }
    vtkfile << "<?xml version='1.0'?>" << "\n";
    vtkfile << "<VTKFile type='ImageData' version='0.1' byte_order='LittleEndian'>" << std::endl;
    vtkfile <<"  <ImageData WholeExtent='"  csp 0       csp CX-1    csp 0 csp CY-1  csp 0 csp 0
            <<" ' Ghostlevel='0' Origin='"   csp minX    csp minY    csp 0
            <<" ' Spacing='"                 csp Hx      csp Hy      csp 0 << " '>" << std::endl;
    vtkfile <<"   <Piece Extent='"        csp 0       csp CX-1    csp 0 csp CY-1  csp 0 csp 0 << " '>" << "\n";

    // Store point data
    vtkfile << "      <PointData>" << "\n";

    // Omega
    vtkfile << "        <DataArray type='Float64' Name='Omega' NumberOfComponents='1'  format='ascii'>" << "\n";
    for(int j=0; j<CY; j++)
    {
        for(int i=0; i<CX; i++)    vtkfile << std::scientific << std::setw(vtkWidth) << r_Input1[GID(i,j,NX,NY)];
    }
    vtkfile << "\n        </DataArray>" << "\n";

    // Additional solver outputs based on solver type
    if (Operator==NONE)
    {
        vtkfile << "        <DataArray type='Float64' Name='Phi' NumberOfComponents='1'  format='ascii'>" << "\n";
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++)    vtkfile << std::scientific << std::setw(vtkWidth) << r_Output1[GID(i,j,NX,NY)];
        }
        vtkfile << "\n        </DataArray>" << "\n";
    }

    if (Operator==GRAD)
    {
        vtkfile << "        <DataArray type='Float64' Name='dPhi_dx' NumberOfComponents='1'  format='ascii'>" << "\n";
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++)    vtkfile << std::scientific << std::setw(vtkWidth) << r_Output1[GID(i,j,NX,NY)];
        }
        vtkfile << "\n        </DataArray>" << "\n";

        vtkfile << "        <DataArray type='Float64' Name='dPhi_dy' NumberOfComponents='1'  format='ascii'>" << "\n";
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++)    vtkfile << std::scientific << std::setw(vtkWidth) << r_Output2[GID(i,j,NX,NY)];
        }
        vtkfile << "\n        </DataArray>" << "\n";
    }

    if (Operator==CURL)
    {
        vtkfile << "        <DataArray type='Float64' Name='Velocity' NumberOfComponents='2'  format='ascii'>" << "\n";
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++)    vtkfile << std::scientific  << std::setw(vtkWidth) << r_Output1[GID(i,j,NX,NY)]
                        << std::setw(vtkWidth) << r_Output2[GID(i,j,NX,NY)];
        }
        vtkfile << "\n        </DataArray>" << "\n";
    }

    if (Operator==DIV)
    {
        vtkfile << "        <DataArray type='Float64' Name='DivOmega' NumberOfComponents='1'  format='ascii'>" << "\n";
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++)    vtkfile << std::scientific << std::setw(vtkWidth) << r_Output1[GID(i,j,NX,NY)];
        }
        vtkfile << "\n        </DataArray>" << "\n";
    }

    if (Operator==NABLA)
    {
        vtkfile << "        <DataArray type='Float64' Name='dPhi2_dx2' NumberOfComponents='1'  format='ascii'>" << "\n";
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++)    vtkfile << std::scientific << std::setw(vtkWidth) << r_Output1[GID(i,j,NX,NY)];
        }
        vtkfile << "\n        </DataArray>" << "\n";

        vtkfile << "        <DataArray type='Float64' Name='dPhi2_dy2' NumberOfComponents='1'  format='ascii'>" << "\n";
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++)    vtkfile << std::scientific << std::setw(vtkWidth) << r_Output2[GID(i,j,NX,NY)];
        }
        vtkfile << "\n        </DataArray>" << "\n";
    }

    vtkfile << "      </PointData>" << "\n";
    vtkfile << "      <CellData>" << "\n";
    vtkfile << "      </CellData>" << "\n";

    vtkfile << "    </Piece>" << "\n";
    vtkfile << "  </ImageData>" << "\n";
    vtkfile << "</VTKFile>" << "\n";

    vtkfile.close();

    std::cout << "Grid data has been exportet in .vtk format to: " << filename << std::endl;
}

//---------------------------
//--- 3D Scalar solver ------
//---------------------------

void Solver_3D_Scalar::Create_vtk()
{
    // When the solver has run, the grid values are stored as a vtk file.
    // This allows the user to visualise the grid in Paraview.

    // Grid sizes
    int CX = gNX;
    int CY = gNY;
    int CZ = gNZ;
    if (gNX>NX) CX = NX;
    if (gNY>NY) CY = NY;
    if (gNZ>NZ) CZ = NZ;

    Real minX = Xl, minY = Yl, minZ = Zl;
    if (Grid==STAGGERED)
    {
        minX += 0.5*Hx;
        minY += 0.5*Hy;
        minZ += 0.5*Hz;
    }

    const int nx = CX, ny = CY, nz = CZ; // Number of points in each direction
    const Real origin[3] = {minX, minY, minZ}; // Grid origin
    const Real spacing[3] = {Hx, Hy, Hz}; // Grid spacing

    // Generate vector data for the grid
    const int numPoints = nx * ny * nz;

    // Prepare binary buffer
    std::vector<Real> binaryBuffer(numPoints);  // Flattened array for binary data
    std::vector<Real> binaryBuffer2(numPoints); // Flattened array for binary data
    OpenMPfor
    for(int k=0; k<nz; k++){
        for(int j=0; j<ny; j++){
            for(int i=0; i<nx; i++){
                int id = GID(i,j,k,NX,NY,NZ);           // Global id of unbounded box
                // int id = GID(k,j,i,nz,ny,nx);        // Row-major ordering of paraview input
                int idb = GID(k,j,i,nz,ny,nx);          // Row-major ordering of paraview input
                binaryBuffer[idb] = r_Input1[id];       // x-component
                convertToBigEndian(binaryBuffer[idb]);  // Convert each value to big-endian format
                binaryBuffer2[idb] = r_Output1[id]; // x-component
                convertToBigEndian(binaryBuffer2[idb]);
            }
        }
    }

    // Open file for binary writing
    std::string OutputDirectory = "Output/" + OutputFolder;
    Create_Directory(OutputDirectory);
    std::string filename = OutputDirectory + "/" + vtk_Name;
    std::ofstream file(filename.c_str(), std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file for writing.\n";
        return;
    }

    // Write VTK header
    file << "# vtk DataFile Version 2.0\n";
    file << "Binary VTK file with vector data\n";
    file << "BINARY\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    file << "ORIGIN " << origin[0] << " " << origin[1] << " " << origin[2] << "\n";
    file << "SPACING " << spacing[0] << " " << spacing[1] << " " << spacing[2] << "\n";

    // Write binary data array to file
    file << "POINT_DATA " << numPoints << "\n";
    if (std::is_same<Real,float>::value)    file << "SCALARS Omega float\n"; // Specify vector data
    if (std::is_same<Real,double>::value)   file << "SCALARS Omega double\n"; // Specify vector data
    file << "LOOKUP_TABLE default\n";
    file.write(reinterpret_cast<const char*>(binaryBuffer.data()), binaryBuffer.size() * sizeof(Real));

    file << " " << "\n";
    if (std::is_same<Real,float>::value)    file << "SCALARS Phi float\n"; // Specify vector data
    if (std::is_same<Real,double>::value)   file << "SCALARS Phi double\n"; // Specify vector data
    file << "LOOKUP_TABLE default\n";
    file.write(reinterpret_cast<const char*>(binaryBuffer2.data()), binaryBuffer2.size() * sizeof(Real));

    // Close file
    file.close();
    std::cout << "Grid data has been exportet in binary .vtk format to: " << filename << std::endl;
    return;
}

//---------------------------
//--- 3D Vector solver ------
//---------------------------

void Solver_3D_Vector::Create_vtk()
{
    // When the solver has run, the grid values are stored as a vtk file.
    // This allows the user to visualise the grid in ParaVIEW.

    // Grid sizes
    int CX = gNX;
    int CY = gNY;
    int CZ = gNZ;
    if (gNX>NX) CX = NX;
    if (gNY>NY) CY = NY;
    if (gNZ>NZ) CZ = NZ;

    Real minX = Xl, minY = Yl, minZ = Zl;
    if (Grid==STAGGERED)
    {
        minX += 0.5*Hx;
        minY += 0.5*Hy;
        minZ += 0.5*Hz;
    }

    const int nx = CX, ny = CY, nz = CZ; // Number of points in each direction
    const Real origin[3] = {minX, minY, minZ}; // Grid origin
    const Real spacing[3] = {Hx, Hy, Hz}; // Grid spacing

    // Generate vector data for the grid
    const int numPoints = nx * ny * nz;

    // Prepare binary buffer
    std::vector<Real> binaryBuffer(numPoints*3); // Flattened array for binary data
    std::vector<Real> binaryBuffer2(numPoints*3); // Flattened array for binary data
    // std::string Buffer = " "; // Flattened array for binary data        // ASCII FORMAT
    // std::string Buffer2; // Flattened array for binary data
    OpenMPfor           // DEACTIVATE FOR ASCII FORMAT
    for(int k=0; k<nz; k++){
        for(int j=0; j<ny; j++){
            for(int i=0; i<nx; i++){
                int id = GF_GID3(i,j,k,NX,NY,NZ);       // F_Style Global id of unbounded box (VkFFT)
                int idb = GID(k,j,i,nz,ny,nx);          // Row-major ordering of paraview input
                binaryBuffer[idb * 3 + 0] = r_Input1[id]; // x-component
                binaryBuffer[idb * 3 + 1] = r_Input2[id]; // y-component
                binaryBuffer[idb * 3 + 2] = r_Input3[id]; // z-component
                // Convert each value to big-endian format
                convertToBigEndian(binaryBuffer[idb * 3 + 0]);
                convertToBigEndian(binaryBuffer[idb * 3 + 1]);
                convertToBigEndian(binaryBuffer[idb * 3 + 2]);

                binaryBuffer2[idb * 3 + 0] = r_Output1[id]; // x-component
                binaryBuffer2[idb * 3 + 1] = r_Output2[id]; // y-component
                binaryBuffer2[idb * 3 + 2] = r_Output3[id]; // z-component
                // Convert each value to big-endian format
                convertToBigEndian(binaryBuffer2[idb * 3 + 0]);
                convertToBigEndian(binaryBuffer2[idb * 3 + 1]);
                convertToBigEndian(binaryBuffer2[idb * 3 + 2]);

                // // ASCII FORMAT
                // std::ostringstream is1, is2, is3;
                // is1 << std::scientific << std::setw(17) << r_Input2[id]; Buffer += is1.str();
                // is2 << std::scientific << std::setw(17) << r_Input2[id]; Buffer += is2.str();
                // is3 << std::scientific << std::setw(17) << r_Input3[id]; Buffer += is3.str();
                // std::ostringstream os1, os2, os3;
                // os1 << std::scientific << std::setw(17) << r_Output1[id];  Buffer2 += os1.str();
                // os2 << std::scientific << std::setw(17) << r_Output2[id];  Buffer2 += os2.str();
                // os3 << std::scientific << std::setw(17) << r_Output3[id];  Buffer2 += os3.str();
            }
        }
    }

    // Open file for binary writing
    std::string OutputDirectory = "Output/" + OutputFolder;
    Create_Directory(OutputDirectory);
    std::string filename = OutputDirectory + "/" + vtk_Name;
    std::ofstream file(filename.c_str(), std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file for writing.\n";
        return;
    }

    // Write VTK header
    file << "# vtk DataFile Version 2.0\n";
    file << "Binary VTK file with vector data\n";
    file << "BINARY\n";
    // file << "ASCII VTK file with vector data\n";        // ASCII FORMAT
    // file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    file << "ORIGIN " << origin[0] << " " << origin[1] << " " << origin[2] << "\n";
    file << "SPACING " << spacing[0] << " " << spacing[1] << " " << spacing[2] << "\n";

    // Write binary data array to file
    file << "POINT_DATA " << numPoints << "\n";
    if (std::is_same<Real,float>::value)    file << "VECTORS Omega float\n"; // Specify vector data
    if (std::is_same<Real,double>::value)   file << "VECTORS Omega double\n"; // Specify vector data
    file.write(reinterpret_cast<const char*>(binaryBuffer.data()), binaryBuffer.size() * sizeof(Real));
    // file << Buffer;             // ASCII FORMAT

    file << " " << "\n";
    if (std::is_same<Real,float>::value)    file << "VECTORS Velocity float\n"; // Specify vector data
    if (std::is_same<Real,double>::value)   file << "VECTORS Velocity double\n"; // Specify vector data
    file.write(reinterpret_cast<const char*>(binaryBuffer2.data()), binaryBuffer2.size() * sizeof(Real));
    // file << Buffer2;            // ASCII FORMAT

    // Close file
    file.close();
    std::cout << "Grid data has been exportet in binary .vtk format to: " << filename << std::endl;
    return;
}

}
