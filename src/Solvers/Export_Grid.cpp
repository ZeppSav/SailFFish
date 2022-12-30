//-----------------------------------------------------------------------------
//-------------------------Export Grid Functions-------------------------------
//-----------------------------------------------------------------------------

#include "Solver_Base.h"

namespace SailFFish
{

//---------------------------
//--- 2D Scalar solver ------
//---------------------------

void Solver_2D_Scalar::Create_vti()
{
    // When the solver has run, the grid values are stored as a vti file.
    // This allows the user to visualise the grid in LabVIEW.

    // Grid sizes
    int CX = gNX;
    int CY = gNY;
    if (gNX>NX) CX = NX;
    if (gNY>NY) CY = NY;

    cout << CX csp CY << endl;

    std::ostringstream str;
    std::string ifilename;

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Individual .vti files
    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

    //--- Create output director if not existing
    std::string OutputDirectory = "Output";
    CreateDirectory(OutputDirectory);
    std::string filename = "./Output/Mesh_2D.vti";
    str.str(""); // clear str
    std::ofstream vtifile( filename.c_str() );
//	vtifile.precision(16);
    vtifile.precision(8);
    if(!vtifile.is_open())
    {
        std::cerr << "ERROR: cannot open vtifile." << std::endl;
        return;
    }

    // Generate text for file
    Real minX = Xl, minY = Yl;
    if (Grid==STAGGERED)
    {
        minX += 0.5*Hx;
        minY += 0.5*Hy;
    }
    vtifile << "<?xml version='1.0'?>" << "\n";
    vtifile << "<VTKFile type='ImageData' version='0.1' byte_order='LittleEndian'>" << endl;
    vtifile <<"  <ImageData WholeExtent='"  csp 0       csp CX-1    csp 0 csp CY-1  csp 0 csp 0
            <<"' Ghostlevel='0' Origin='"   csp minX    csp minY    csp 0
            <<"' Spacing='"                 csp Hx      csp Hy      csp 0 << "'>" << endl;
    vtifile << "    <Piece Extent='"        csp 0       csp CX-1    csp 0 csp CY-1  csp 0 csp 0 << "'>" << "\n";

    // Store point data
    vtifile << "      <PointData>" << "\n";

    // Omega
    vtifile << "        <DataArray type='Float64' Name='Omega' NumberOfComponents='1'  format='ascii'>" << "\n";
    for(int j=0; j<CY; j++)
    {
        for(int i=0; i<CX; i++)    vtifile << std::scientific << std::setw(17) << r_Input1[GID(i,j,NX,NY,D2D)];
    }
    vtifile << "\n        </DataArray>" << "\n";

    // Additional solver outputs based on solver type
    if (Operator==NONE)
    {
        vtifile << "        <DataArray type='Float64' Name='Phi' NumberOfComponents='1'  format='ascii'>" << "\n";
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++)    vtifile << std::scientific << std::setw(17) << r_Output1[GID(i,j,NX,NY,D2D)];
        }
        vtifile << "\n        </DataArray>" << "\n";
    }

    if (Operator==GRAD)
    {
        vtifile << "        <DataArray type='Float64' Name='dPhi_dx' NumberOfComponents='1'  format='ascii'>" << "\n";
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++)    vtifile << std::scientific << std::setw(17) << r_Output1[GID(i,j,NX,NY,D2D)];
        }
        vtifile << "\n        </DataArray>" << "\n";

        vtifile << "        <DataArray type='Float64' Name='dPhi_dy' NumberOfComponents='1'  format='ascii'>" << "\n";
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++)    vtifile << std::scientific << std::setw(17) << r_Output2[GID(i,j,NX,NY,D2D)];
        }
        vtifile << "\n        </DataArray>" << "\n";
    }

    if (Operator==CURL)
    {
        vtifile << "        <DataArray type='Float64' Name='Velocity' NumberOfComponents='2'  format='ascii'>" << "\n";
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++)    vtifile << std::scientific  << std::setw(17) << r_Output1[GID(i,j,NX,NY,D2D)]
                                                                    << std::setw(17) << r_Output2[GID(i,j,NX,NY,D2D)];
        }
        vtifile << "\n        </DataArray>" << "\n";
    }

    if (Operator==DIV)
    {
        vtifile << "        <DataArray type='Float64' Name='DivOmega' NumberOfComponents='1'  format='ascii'>" << "\n";
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++)    vtifile << std::scientific << std::setw(17) << r_Output1[GID(i,j,NX,NY,D2D)];
        }
        vtifile << "\n        </DataArray>" << "\n";
    }

    if (Operator==NABLA)
    {
        vtifile << "        <DataArray type='Float64' Name='dPhi2_dx2' NumberOfComponents='1'  format='ascii'>" << "\n";
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++)    vtifile << std::scientific << std::setw(17) << r_Output1[GID(i,j,NX,NY,D2D)];
        }
        vtifile << "\n        </DataArray>" << "\n";

        vtifile << "        <DataArray type='Float64' Name='dPhi2_dy2' NumberOfComponents='1'  format='ascii'>" << "\n";
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++)    vtifile << std::scientific << std::setw(17) << r_Output2[GID(i,j,NX,NY,D2D)];
        }
        vtifile << "\n        </DataArray>" << "\n";
    }

    vtifile << "      </PointData>" << "\n";
    vtifile << "      <CellData>" << "\n";
    vtifile << "      </CellData>" << "\n";

    vtifile << "    </Piece>" << "\n";
    vtifile << "  </ImageData>" << "\n";
    vtifile << "</VTKFile>" << "\n";

    vtifile.close();

    std::cout << "Grid data has been exportet in .vti format to: " << filename << std::endl;
}

//---------------------------
//--- 3D Scalar solver ------
//---------------------------

void Solver_3D_Scalar::Create_vti()
{
    // When the solver has run, the grid values are stored as a vti file.
    // This allows the user to visualise the grid in LabVIEW.

    // Grid sizes
    int CX = gNX;
    int CY = gNY;
    int CZ = gNZ;
    if (gNX>NX) CX = NX;
    if (gNY>NY) CY = NY;
    if (gNZ>NZ) CZ = NZ;

    std::ostringstream str;
    std::string ifilename;

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Individual .vti files
    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

    //--- Create output director if not existing
    std::string OutputDirectory = "Output";
    CreateDirectory(OutputDirectory);
    std::string filename = "./Output/Mesh_3D.vti";
    str.str(""); // clear str
    std::ofstream vtifile( filename.c_str() );
//	vtifile.precision(16);
    vtifile.precision(8);
    if(!vtifile.is_open())
    {
        std::cerr << "ERROR: cannot open vtifile." << std::endl;
        return;
    }

    // Generate text for file
    Real minX = Xl, minY = Yl, minZ = Zl;
    if (Grid==STAGGERED)
    {
        minX += 0.5*Hx;
        minY += 0.5*Hy;
        minZ += 0.5*Hz;
    }
    vtifile << "<?xml version='1.0'?>" << "\n";
    vtifile << "<VTKFile type='ImageData' version='0.1' byte_order='LittleEndian'>" << endl;
    vtifile << "<VTKFile type='ImageData' version='0.1' byte_order='LittleEndian'>" << endl;
    vtifile <<"  <ImageData WholeExtent='"  csp 0       csp CX-1    csp 0   csp CY-1   csp 0 csp CZ-1
            <<"' Ghostlevel='0' Origin='"   csp minX    csp minY    csp minY
            <<"' Spacing='"                 csp Hx      csp Hy      csp Hz << "'>" << endl;
    vtifile << "    <Piece Extent='"        csp 0       csp CX-1    csp 0   csp CY-1   csp 0 csp CZ-1 << "'>" << "\n";

    // Store point data
    vtifile << "      <PointData>" << "\n";

    // Omega
    vtifile << "        <DataArray type='Float64' Name='Omega' NumberOfComponents='1'  format='ascii'>" << "\n";
    for(int k=0; k<CZ; k++)
    {
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++)    vtifile << std::scientific << std::setw(17) << r_Input1[GID(i,j,k,NX,NY,NZ)];
        }
    }
    vtifile << "\n        </DataArray>" << "\n";

    if (Operator==NONE)
    {
        vtifile << "        <DataArray type='Float64' Name='Phi' NumberOfComponents='1'  format='ascii'>" << "\n";
        for(int k=0; k<CZ; k++)
        {
            for(int j=0; j<CY; j++)
            {
                for(int i=0; i<CX; i++)    vtifile << std::scientific << std::setw(17) << r_Output1[GID(i,j,k,NX,NY,NZ)];
            }
        }
        vtifile << "\n        </DataArray>" << "\n";
    }

    if (Operator==DIV)
    {
        vtifile << "        <DataArray type='Float64' Name='Div_Phi' NumberOfComponents='1'  format='ascii'>" << "\n";
        for(int k=0; k<CZ; k++)
        {
            for(int j=0; j<CY; j++)
            {
                for(int i=0; i<CX; i++)    vtifile << std::scientific << std::setw(17) << r_Output1[GID(i,j,k,NX,NY,NZ)];
            }
        }
        vtifile << "\n        </DataArray>" << "\n";
    }

    if (Operator==GRAD)
    {
        vtifile << "        <DataArray type='Float64' Name='Grad_Phi' NumberOfComponents='3'  format='ascii'>" << "\n";
        for(int k=0; k<CZ; k++)
        {
            for(int j=0; j<CY; j++)
            {
                for(int i=0; i<CX; i++){
                    vtifile << std::scientific  << std::setw(17) << r_Output1[GID(i,j,k,NX,NY,NZ)]
                                                << std::setw(17) << r_Output2[GID(i,j,k,NX,NY,NZ)]
                                                << std::setw(17) << r_Output3[GID(i,j,k,NX,NY,NZ)];
                }
            }
        }
        vtifile << "\n        </DataArray>" << "\n";
    }

    if (Operator==CURL)
    {
        // Ignoring this case for now
    }

    vtifile << "      </PointData>" << "\n";
    vtifile << "      <CellData>" << "\n";
    vtifile << "      </CellData>" << "\n";

    vtifile << "    </Piece>" << "\n";
    vtifile << "  </ImageData>" << "\n";
    vtifile << "</VTKFile>" << "\n";

    vtifile.close();

    std::cout << "Grid data has been exportet in .vti format to: " << filename << std::endl;
}

//---------------------------
//--- 3D Vector solver ------
//---------------------------

void Solver_3D_Vector::Create_vti()
{
    // When the solver has run, the grid values are stored as a vti file.
    // This allows the user to visualise the grid in LabVIEW.

    // Grid sizes
    int CX = gNX;
    int CY = gNY;
    int CZ = gNZ;
    if (gNX>NX) CX = NX;
    if (gNY>NY) CY = NY;
    if (gNZ>NZ) CZ = NZ;

    std::ostringstream str;
    std::string ifilename;

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Individual .vti files
    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

    //--- Create output director if not existing
    std::string OutputDirectory = "Output";
    CreateDirectory(OutputDirectory);
    std::string filename = "./Output/Mesh_3DV.vti";
    str.str(""); // clear str
    std::ofstream vtifile( filename.c_str() );
//	vtifile.precision(16);
    vtifile.precision(8);
    if(!vtifile.is_open())
    {
        std::cerr << "ERROR: cannot open vtifile." << std::endl;
        return;
    }

    // Generate text for file
    Real minX = Xl, minY = Yl, minZ = Zl;
    if (Grid==STAGGERED)
    {
        minX += 0.5*Hx;
        minY += 0.5*Hy;
        minZ += 0.5*Hz;
    }
    vtifile << "<?xml version='1.0'?>" << "\n";
    vtifile << "<VTKFile type='ImageData' version='0.1' byte_order='LittleEndian'>" << endl;
    vtifile <<"  <ImageData WholeExtent='"  csp 0       csp CX-1   csp 0 csp CY-1   csp 0 csp CZ-1
            <<"' Ghostlevel='0' Origin='"   csp minX    csp minY    csp minY
            <<"' Spacing='"                 csp Hx      csp Hy      csp Hz << "'>" << endl;
    vtifile << "    <Piece Extent='"        csp 0       csp CX-1   csp 0 csp CY-1   csp 0 csp CZ-1 << "'>" << "\n";

    // Store point data
    vtifile << "      <PointData>" << "\n";

    // Omega
    vtifile << "        <DataArray type='Float64' Name='Omega' NumberOfComponents='3'  format='ascii'>" << "\n";
    for(int k=0; k<CZ; k++)
    {
        for(int j=0; j<CY; j++)
        {
            for(int i=0; i<CX; i++){
                vtifile << std::scientific  << std::setw(17) << r_Input1[GID(i,j,k,NX,NY,NZ)]
                                            << std::setw(17) << r_Input2[GID(i,j,k,NX,NY,NZ)]
                                            << std::setw(17) << r_Input3[GID(i,j,k,NX,NY,NZ)];
            }
        }
    }
    vtifile << "\n        </DataArray>" << "\n";

    if (Operator==NONE)
    {
        vtifile << "        <DataArray type='Float64' Name='Phi' NumberOfComponents='3'  format='ascii'>" << "\n";
        for(int k=0; k<CZ; k++)
        {
            for(int j=0; j<CY; j++)
            {
                for(int i=0; i<CX; i++){
                    vtifile << std::scientific  << std::setw(17) << r_Output1[GID(i,j,k,NX,NY,NZ)]
                                                << std::setw(17) << r_Output2[GID(i,j,k,NX,NY,NZ)]
                                                << std::setw(17) << r_Output3[GID(i,j,k,NX,NY,NZ)];
                }
            }
        }
        vtifile << "\n        </DataArray>" << "\n";
    }

    if (Operator==CURL)
    {
        vtifile << "        <DataArray type='Float64' Name='Velocity' NumberOfComponents='3'  format='ascii'>" << "\n";
        for(int k=0; k<CZ; k++)
        {
            for(int j=0; j<CY; j++)
            {
                for(int i=0; i<CX; i++){
                    vtifile << std::scientific  << std::setw(17) << r_Output1[GID(i,j,k,NX,NY,NZ)]
                                                << std::setw(17) << r_Output2[GID(i,j,k,NX,NY,NZ)]
                                                << std::setw(17) << r_Output3[GID(i,j,k,NX,NY,NZ)];
                }
            }
        }
        vtifile << "\n        </DataArray>" << "\n";
    }

    if (Operator==DIV)
    {
        vtifile << "        <DataArray type='Float64' Name='Div_Phi' NumberOfComponents='1'  format='ascii'>" << "\n";
        for(int k=0; k<CZ; k++)
        {
            for(int j=0; j<CY; j++)
            {
                for(int i=0; i<CX; i++){
                    vtifile << std::scientific  << std::setw(17) << r_Output1[GID(i,j,k,NX,NY,NZ)];
                }
            }
        }
        vtifile << "\n        </DataArray>" << "\n";
    }

    if (Operator==GRAD)
    {
        // Skip this option for now. Creates lots of output arrays!
    }

    if (Operator==NABLA)
    {
        vtifile << "        <DataArray type='Float64' Name='Nabla_Phi' NumberOfComponents='3'  format='ascii'>" << "\n";
        for(int k=0; k<CZ; k++)
        {
            for(int j=0; j<CY; j++)
            {
                for(int i=0; i<CX; i++){
                    vtifile << std::scientific  << std::setw(17) << r_Output1[GID(i,j,k,NX,NY,NZ)]
                                                << std::setw(17) << r_Output2[GID(i,j,k,NX,NY,NZ)]
                                                << std::setw(17) << r_Output3[GID(i,j,k,NX,NY,NZ)];
                }
            }
        }
        vtifile << "\n        </DataArray>" << "\n";
    }

    vtifile << "      </PointData>" << "\n";
    vtifile << "      <CellData>" << "\n";
    vtifile << "      </CellData>" << "\n";

    vtifile << "    </Piece>" << "\n";
    vtifile << "  </ImageData>" << "\n";
    vtifile << "</VTKFile>" << "\n";

    vtifile.close();

    std::cout << "Grid data has been exportet in .vti format to: " << filename << std::endl;
}

}
