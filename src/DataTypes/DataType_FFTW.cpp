//-----------------------------------------------------------------------------
//-------------------------DataType_FFTW Functions-----------------------------
//-----------------------------------------------------------------------------

#include "DataType_FFTW.h"
#include "omp.h"

#ifdef FFTW

namespace SailFFish
{

//--- Datatype setup

void DataType_FFTW::Datatype_Setup()
{
    // This is a datatype-specific setup. This allows allocation / preparation based on the dataype being used.

    // For FFTW, I will initialize multihtreading here.
    int j = fftwf_init_threads();
    int nt = omp_get_max_threads();
    fftwf_plan_with_nthreads(8);
    cout << "FFTW is being executed in OpenMP mode: NThreads = " << nt << "." <<  endl;
}

//--- Array allocation

SFStatus DataType_FFTW::Allocate_Arrays()
{
    // Depending on the type of solver and the chosen operator/outputs, the arrays which must be allocated vary.
    // This is simply controlled here by specifying the necessary flags during solver initialization

    // Real-valued arrays
    if (r_in1)      r_Input1 = (Real*)malloc(NT*sizeof(Real));
    if (r_in2)      r_Input2 = (Real*)malloc(NT*sizeof(Real));
    if (r_in3)      r_Input3 = (Real*)malloc(NT*sizeof(Real));

    if (r_ft_in1)   r_FTInput1 = (Real*)malloc(NT*sizeof(Real));
    if (r_ft_in2)   r_FTInput2 = (Real*)malloc(NT*sizeof(Real));
    if (r_ft_in3)   r_FTInput3 = (Real*)malloc(NT*sizeof(Real));

    if (r_out_1)    r_Output1 = (Real*)malloc(NT*sizeof(Real));
    if (r_out_2)    r_Output2 = (Real*)malloc(NT*sizeof(Real));
    if (r_out_3)    r_Output3 = (Real*)malloc(NT*sizeof(Real));

    // Arrays for real Green's function
    if (r_fg)       r_FG = (Real*)malloc(NT*sizeof(Real));

    // Complex-valued arrays
    if (c_in1)      c_Input1 = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NT);
    if (c_in2)      c_Input2 = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NT);
    if (c_in3)      c_Input3 = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NT);

    if (c_ft_in1)   c_FTInput1 = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NTM);
    if (c_ft_in2)   c_FTInput2 = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NTM);
    if (c_ft_in3)   c_FTInput3 = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NTM);

    if (c_out_1)    c_Output1 = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NT);
    if (c_out_2)    c_Output2 = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NT);
    if (c_out_3)    c_Output3 = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NT);

    // Arrays for complex Green's function & spectral operators arrays
    if (c_fg)       c_FG  = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NTM);
    if (c_fg_i)     c_FGi = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NX);
    if (c_fg_j)     c_FGj = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NY);
    if (c_fg_k)     c_FGk = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NZ);

    return NoError;
}

SFStatus DataType_FFTW::Deallocate_Arrays()
{
    // Depending on the type of solver and the chosen operator/outputs, the arrays which must be allocated vary.
    // This is simply controlled here by specifying the necessary flags during solver initialization

    // Real-valued arrays
    if (r_in1)      free(r_Input1);
    if (r_in2)      free(r_Input2);
    if (r_in3)      free(r_Input3);

    if (r_ft_in1)   free(r_FTInput1);
    if (r_ft_in2)   free(r_FTInput2);
    if (r_ft_in3)   free(r_FTInput3);

    if (r_out_1)    free(r_Output1);
    if (r_out_2)    free(r_Output2);
    if (r_out_3)    free(r_Output3);

    // Arrays for real Green's function
    if (r_fg)       free(r_FG);

    // Complex-valued arrays
    if (c_in1)      FFTW_Free(c_Input1);
    if (c_in2)      FFTW_Free(c_Input2);
    if (c_in3)      FFTW_Free(c_Input3);

    if (c_ft_in1)   FFTW_Free(c_FTInput1);
    if (c_ft_in2)   FFTW_Free(c_FTInput2);
    if (c_ft_in3)   FFTW_Free(c_FTInput3);

    if (c_out_1)    FFTW_Free(c_Output1);
    if (c_out_2)    FFTW_Free(c_Output2);
    if (c_out_3)    FFTW_Free(c_Output3);

    // Arrays for complex Green's function & spectral operators arrays
    if (c_fg)       FFTW_Free(c_FG);
    if (c_fg_i)     FFTW_Free(c_FGi);
    if (c_fg_j)     FFTW_Free(c_FGj);
    if (c_fg_k)     FFTW_Free(c_FGk);

    return NoError;
}

//--- Specify Plan

SFStatus DataType_FFTW::Specify_1D_Plan()
{
    // This specifies the forward and backward plans
    if (Transform==DCT1)
    {
        BFac = 1.0/(2.0*(NX-1));
        Forward_Plan =  FFTW_Plan_1D_DCT(NX, r_Input1, r_FTInput1, FFTW_REDFT00, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_1D_DCT(NX, r_FTInput1, r_Output1, FFTW_REDFT00, FFTW_PLANSETUP);
    }
    if (Transform==DCT2)
    {
        BFac = 1.0/(2.0*NX);
        Forward_Plan =  FFTW_Plan_1D_DCT(NX, r_Input1, r_FTInput1, FFTW_REDFT10 , FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_1D_DCT(NX, r_FTInput1, r_Output1, FFTW_REDFT01, FFTW_PLANSETUP);
    }
    if (Transform==DST1)
    {
        BFac = 1.0/(2.0*(NX+1));
        Forward_Plan =  FFTW_Plan_1D_DCT(NX, r_Input1, r_FTInput1, FFTW_RODFT00, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_1D_DCT(NX, r_FTInput1, r_Output1, FFTW_RODFT00, FFTW_PLANSETUP);
    }
    if (Transform==DST2)
    {
        BFac = 1.0/(2.0*NX);
        Forward_Plan =  FFTW_Plan_1D_DCT(NX, r_Input1, r_FTInput1,  FFTW_RODFT10, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_1D_DCT(NX, r_FTInput1, r_Output1, FFTW_RODFT01, FFTW_PLANSETUP);
    }
    if (Transform==DFT_C2C)
    {
        BFac = 1.0/NT;
        Forward_Plan =  FFTW_Plan_1D(NX, c_Input1, c_FTInput1, FFTW_FORWARD, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_1D(NX, c_FTInput1, c_Output1, FFTW_BACKWARD, FFTW_PLANSETUP);
    }
    if (Transform==DFT_R2C)
    {
        BFac = 1.0/NT;
        Forward_Plan =  FFTW_Plan_1D_R2C(NX, r_Input1, c_FTInput1, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_1D_C2R(NX, c_FTInput1, r_Output1, FFTW_PLANSETUP);
    }
    return NoError;
}

SFStatus DataType_FFTW::Specify_2D_Plan()
{
    // This specifies the forward and backward plans
    if (Transform==DCT1)
    {
        BFac = 1.0/(2.0*(NX-1)*2.0*(NY-1));
        Forward_Plan = FFTW_Plan_2D_DCT(NX, NY, r_Input1, r_FTInput1, FFTW_REDFT00, FFTW_REDFT00, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_2D_DCT(NX, NY, r_FTInput1, r_Output1, FFTW_REDFT00, FFTW_REDFT00, FFTW_PLANSETUP);
    }
    if (Transform==DCT2)
    {
        BFac = 1.0/(2.0*NX*2.0*NY);
        Forward_Plan = FFTW_Plan_2D_DCT(NX, NY, r_Input1, r_FTInput1, FFTW_REDFT10, FFTW_REDFT10, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_2D_DCT(NX, NY, r_FTInput1, r_Output1, FFTW_REDFT01, FFTW_REDFT01, FFTW_PLANSETUP);
    }
    if (Transform==DST1)
    {
        BFac = 1.0/(2.0*(NX+1)*2.0*(NY+1));
        Forward_Plan = FFTW_Plan_2D_DCT(NX, NY, r_Input1, r_FTInput1, FFTW_RODFT00 , FFTW_RODFT00, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_2D_DCT(NX, NY, r_FTInput1, r_Output1, FFTW_RODFT00 , FFTW_RODFT00, FFTW_PLANSETUP);
    }
    if (Transform==DST2)
    {
        BFac = 1.0/(2.0*NX*2.0*NY);
        Forward_Plan =  FFTW_Plan_2D_DCT(NX, NY, r_Input1, r_FTInput1,  FFTW_RODFT10, FFTW_RODFT10, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_2D_DCT(NX, NY, r_FTInput1, r_Output1, FFTW_RODFT01, FFTW_RODFT01, FFTW_PLANSETUP);
    }
    if (Transform==DFT_C2C)
    {
        BFac = 1.0/NT;
        Forward_Plan = FFTW_Plan_2D(NX, NY, c_Input1, c_FTInput1, FFTW_FORWARD, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_2D(NX, NY, c_FTInput1, c_Output1, FFTW_BACKWARD, FFTW_PLANSETUP);
    }
    if (Transform==DFT_R2C)
    {
        BFac = 1.0/NT;
        Forward_Plan =  FFTW_Plan_2D_R2C(NX, NY, r_Input1, c_FTInput1, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_2D_C2R(NX, NY, c_FTInput1, r_Output1, FFTW_PLANSETUP);
    }
    return NoError;
}

SFStatus DataType_FFTW::Specify_3D_Plan()
{
    // This specifies the forward and backward plans
    if (Transform==DCT1)
    {
        BFac = 1.0/(2.0*(NX-1)*2.0*(NY-1)*2.0*(NZ-1));
        Forward_Plan = FFTW_Plan_3D_DCT(NX, NY, NZ, r_Input1, r_FTInput1, FFTW_REDFT00, FFTW_REDFT00, FFTW_REDFT00, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_3D_DCT(NX, NY, NZ, r_FTInput1, r_Output1, FFTW_REDFT00, FFTW_REDFT00, FFTW_REDFT00, FFTW_PLANSETUP);
    }
    if (Transform==DCT2)
    {
        BFac = 1.0/(2.0*NX*2.0*NY*2.0*NZ);
        Forward_Plan = FFTW_Plan_3D_DCT(NX, NY, NZ, r_Input1, r_FTInput1, FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_3D_DCT(NX, NY, NZ, r_FTInput1, r_Output1, FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01, FFTW_PLANSETUP);
    }
    if (Transform==DST1)
    {
        BFac = 1.0/(2.0*(NX+1)*2.0*(NY+1)*2.0*(NZ+1));
        Forward_Plan = FFTW_Plan_3D_DCT(NX, NY, NZ, r_Input1, r_FTInput1, FFTW_RODFT00, FFTW_RODFT00, FFTW_RODFT00, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_3D_DCT(NX, NY, NZ, r_FTInput1, r_Output1, FFTW_RODFT00, FFTW_RODFT00, FFTW_RODFT00, FFTW_PLANSETUP);
    }
    if (Transform==DST2)
    {
        BFac = 1.0/(2.0*NX*2.0*NY*2.0*NZ);
        Forward_Plan =  FFTW_Plan_3D_DCT(NX, NY, NZ, r_Input1, r_FTInput1,  FFTW_RODFT10, FFTW_RODFT10, FFTW_RODFT10, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_3D_DCT(NX, NY, NZ, r_FTInput1, r_Output1, FFTW_RODFT01, FFTW_RODFT01, FFTW_RODFT01, FFTW_PLANSETUP);
    }
    if (Transform==DFT_C2C)
    {
        BFac = 1.0/NT;
        Forward_Plan = FFTW_Plan_3D(NX, NY, NZ, c_Input1, c_FTInput1, FFTW_FORWARD, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_3D(NX, NY, NZ, c_FTInput1, c_Output1, FFTW_BACKWARD, FFTW_PLANSETUP);
    }
    if (Transform==DFT_R2C)
    {
        BFac = 1.0/NT;
        Forward_Plan = FFTW_Plan_3D_R2C(NX, NY, NZ, r_Input1, c_FTInput1, FFTW_PLANSETUP);
        Backward_Plan = FFTW_Plan_3D_C2R(NX, NY, NZ, c_FTInput1, r_Output1, FFTW_PLANSETUP);
    }
    return NoError;
}

//--- Input specification

SFStatus DataType_FFTW::Set_Input(RVector &I)
{
    // This function takes the input vector and stores this in the appropriate input array for the FFT
    if (int(I.size())!=NT){
        cout << "Input array has incorrect dimension." << endl;
        return DimError;
    }
    if (r_in1)   std::memcpy(r_Input1, I.data(), NT*sizeof(Real));   // Just copy over
    if (c_in1)   for (int i=0; i<NT; i++) {c_Input1[i][0] = I[i]; c_Input1[i][1] = 0.;}
    return NoError;
}

SFStatus DataType_FFTW::Set_Input(RVector &I1, RVector &I2, RVector &I3)
{
    // This function takes the input vector and stores this in the appropriate input array for the FFT
    if (int(I1.size())!=NT || int(I2.size())!=NT || int(I3.size())!=NT){
        cout << "Input array has incorrect dimension." << endl;
        return DimError;
    }

    if (r_in1) std::memcpy(r_Input1, I1.data(), NT*sizeof(Real));
    if (r_in2) std::memcpy(r_Input2, I2.data(), NT*sizeof(Real));
    if (r_in3) std::memcpy(r_Input3, I3.data(), NT*sizeof(Real));
    if (c_in1)  {for (int i=0; i<NT; i++) {c_Input1[i][0] = I1[i]; c_Input1[i][1] = 0.;}}
    if (c_in2)  {for (int i=0; i<NT; i++) {c_Input2[i][0] = I2[i]; c_Input2[i][1] = 0.;}}
    if (c_in3)  {for (int i=0; i<NT; i++) {c_Input3[i][0] = I3[i]; c_Input3[i][1] = 0.;}}
    return NoError;
}

SFStatus DataType_FFTW::Set_Input_Unbounded_1D(RVector &I)
{
    // This function takes the input vector and stores this in the appropriate input array for the FFT
    int NXH = NX/2;
    cout << NX csp NT csp NXM csp NTM csp NXH << endl;
    if (int(I.size())!=NXH){
        cout << "Input array has incorrect dimension." << endl;
        return DimError;
    }
    if (r_in1) memset(r_Input1, 0, NT*sizeof(Real));   // Pad entire domain with zeros
    if (c_in1) Set_Null(c_Input1,NXH);
    for (int i=0; i<NXH; i++){
            if (r_in1) r_Input1[i] = I[i];
            if (c_in1) c_Input1[i][0] = I[i];
    }
    return NoError;
}

SFStatus DataType_FFTW::Set_Input_Unbounded_2D(RVector &I)
{
    // This function takes the input vector and stores this in the appropriate input array for the FFT
    int NXH = NX/2;
    int NYH = NY/2;
    if (int(I.size())!=NXH*NYH){
        cout << "Input array has incorrect dimension." << endl;
        return DimError;
    }
    if (r_in1) memset(r_Input1, 0, NT*sizeof(Real));   // Pad entire domain with zeros
    if (c_in1) Set_Null(c_Input1,NT);
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            int idg = i*NY + j;
            int idl = i*NYH + j;
            if (r_in1) r_Input1[idg] = I[idl];
            if (c_in1) c_Input1[idg][0] = I[idl];
        }
    }
    return NoError;
}

SFStatus DataType_FFTW::Set_Input_Unbounded_3D(RVector &I)
{
    // This function takes the input vector and stores this in the appropriate input array for the FFT
    int NXH = NX/2;
    int NYH = NY/2;
    int NZH = NZ/2;
    if (int(I.size())!=NXH*NYH*NZH){
        cout << "Input array has incorrect dimension." << endl;
        return DimError;
    }
    if (r_in1) memset(r_Input1, 0, NT*sizeof(Real));   // Pad entire domain with zeros
    if (c_in1) Set_Null(c_Input1,NT);
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            for (int k=0; k<NZH; k++){
                int idg = i*NY*NZ + j*NZ + k;
                int idl = i*NYH*NZH + j*NZH + k;
                if (r_in1) r_Input1[idg] = I[idl];
                if (c_in1) c_Input1[idg][0] = I[idl];
            }
        }
    }
    return NoError;
}

SFStatus DataType_FFTW::Set_Input_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3)
{
    // This function takes the input vector and stores this in the appropriate input array for the FFT
    int NXH = NX/2;
    int NYH = NY/2;
    int NZH = NZ/2;
    if (int(I1.size())!=NXH*NYH*NZH || int(I2.size())!=NXH*NYH*NZH || int(I3.size())!=NXH*NYH*NZH){
        cout << "Input array has incorrect dimension." << endl;
        return DimError;
    }
    if (r_in1) memset(r_Input1, 0, NT*sizeof(Real));   // Pad entire domain with zeros
    if (r_in2) memset(r_Input2, 0, NT*sizeof(Real));   // Pad entire domain with zeros
    if (r_in3) memset(r_Input3, 0, NT*sizeof(Real));   // Pad entire domain with zeros
    if (c_in1) Set_Null(c_Input1,NT);
    if (c_in2) Set_Null(c_Input2,NT);
    if (c_in3) Set_Null(c_Input3,NT);
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            for (int k=0; k<NZH; k++){
                int idg = i*NY*NZ + j*NZ + k;
                int idl = i*NYH*NZH + j*NZH + k;
                if (r_in1) r_Input1[idg] = I1[idl];
                if (r_in2) r_Input2[idg] = I2[idl];
                if (r_in3) r_Input3[idg] = I3[idl];
                if (c_in1) c_Input1[idg][0] = I1[idl];
                if (c_in2) c_Input2[idg][0] = I2[idl];
                if (c_in3) c_Input3[idg][0] = I3[idl];
            }
        }
    }
    return NoError;
}

//--- Output specification

void DataType_FFTW::Get_Output(RVector &I)
{
    // This function converts the output array into an easily accesible format
    if (I.empty()) I.assign(NT,0);
    if (r_out_1)    std::memcpy(I.data(), r_Output1, NT*sizeof(Real));   // Just copy over
    if (c_out_1)    for (int i=0; i<NT; i++)    I[i]= c_Output1[i][0];
}

void DataType_FFTW::Get_Output(RVector &I1, RVector &I2, RVector &I3)
{
    // This function converts the output array into an easily accesible format
    if (I1.empty()) I1.assign(NT,0);
    if (I2.empty()) I2.assign(NT,0);
    if (I3.empty()) I3.assign(NT,0);
    if (r_out_1)  std::memcpy(I1.data(), r_Output1, NT*sizeof(Real));   // Just copy over
    if (r_out_2)  std::memcpy(I2.data(), r_Output2, NT*sizeof(Real));   // Just copy over
    if (r_out_3)  std::memcpy(I3.data(), r_Output3, NT*sizeof(Real));   // Just copy over
    if (c_out_1)    for (int i=0; i<NT; i++)    I1[i]= c_Output1[i][0];
    if (c_out_2)    for (int i=0; i<NT; i++)    I2[i]= c_Output2[i][0];
    if (c_out_3)    for (int i=0; i<NT; i++)    I3[i]= c_Output3[i][0];
}

void DataType_FFTW::Get_Output_Unbounded_1D(RVector &I)
{
    // Extracts appropriate output
    int NXH = NX/2;
    if (I.empty()) I.assign(NXH,0);
    for (int i=0; i<NXH; i++){
        if (r_out_1) I[i] = r_Output1[i];
        if (c_out_1) I[i] = c_Output1[i][0];
    }
}

void DataType_FFTW::Get_Output_Unbounded_2D(RVector &I)
{
    // Extracts appropriate output
    int NXH = NX/2;
    int NYH = NY/2;
    if (I.empty()) I.assign(NXH*NYH,0);
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            int idg = i*NY + j;
            int idl = i*NYH + j;
            if (r_out_1) I[idl] = r_Output1[idg];
            if (c_out_1) I[idl] = c_Output1[idg][0];
        }
    }
}

void DataType_FFTW::Get_Output_Unbounded_3D(RVector &I)
{
    // Extracts appropriate output
    int NXH = NX/2;
    int NYH = NY/2;
    int NZH = NZ/2;
    if (I.empty()) I.assign(NXH*NYH*NZH,0);
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            for (int k=0; k<NZH; k++){
                int idg = i*NY*NZ + j*NZ + k;
                int idl = i*NYH*NZH + j*NZH + k;
                if (r_out_1) I[idl] = r_Output1[idg];
                if (c_out_1) I[idl] = c_Output1[idg][0];
            }
        }
    }
}

void DataType_FFTW::Get_Output_Unbounded_2D(RVector &I1, RVector &I2)
{
    // This function get the output of the
    int NXH = NX/2;
    int NYH = NY/2;
    if (I1.empty()) I1.assign(NXH*NYH,0);
    if (I2.empty()) I2.assign(NXH*NYH,0);
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            int idg = i*NY + j;
            int idl = i*NYH + j;
            if (r_out_1) I1[idl] = r_Output1[idg];
            if (r_out_2) I2[idl] = r_Output2[idg];
            if (c_out_1) I1[idl] = c_Output1[idg][0];
            if (c_out_2) I2[idl] = c_Output2[idg][0];
        }
    }
}

void DataType_FFTW::Get_Output_Unbounded_3D(RVector &I1, RVector &I2, RVector &I3)
{
    // Extracts appropriate output
    int NXH = NX/2;
    int NYH = NY/2;
    int NZH = NZ/2;
    if (I1.empty()) I1.assign(NXH*NYH*NZH,0);
    if (I2.empty()) I2.assign(NXH*NYH*NZH,0);
    if (I3.empty()) I3.assign(NXH*NYH*NZH,0);
    OpenMPfor
    for (int i=0; i<NXH; i++){
        for (int j=0; j<NYH; j++){
            for (int k=0; k<NZH; k++){
                int idg = i*NY*NZ + j*NZ + k;
                int idl = i*NYH*NZH + j*NZH + k;
                if (r_out_1) I1[idl] = r_Output1[idg];
                if (r_out_2) I2[idl] = r_Output2[idg];
                if (r_out_3) I3[idl] = r_Output3[idg];
                if (c_out_1) I1[idl] = c_Output1[idg][0];
                if (c_out_2) I2[idl] = c_Output2[idg][0];
                if (c_out_3) I3[idl] = c_Output3[idg][0];
            }
        }
    }
}

//--- Convolution

inline void Multiply(const FFTWReal &A, const FFTWReal &B, FFTWReal &Out)
{
    // Naiv multiplication algorithm for improved readability
    Real Re = A[0]*B[0] - A[1]*B[1];
    Real Im = A[0]*B[1] + A[1]*B[0];
    Out[0] = Re;
    Out[1] = Im;
}

void DataType_FFTW::Convolution_Real()
{
    // The convolution is carried out by multiplying the RHS in the frequency space with the chosen Greens function
    // This is done in-place to save memory and transfer time
    if (!Kernel_Convolution) return;

    OpenMPfor
    for (int i=0; i<NTM; i++)    r_FTInput1[i] *= r_FG[i];
}

void DataType_FFTW::Convolution_Real3()
{
    // The convolution is carried out by multiplying the RHS in the frequency space with the chosen Greens function
    // This is done in-place to save memory and transfer time
    if (!Kernel_Convolution) return;

    OpenMPfor
    for (int i=0; i<NTM; i++){
        r_FTInput1[i] *= r_FG[i];
        r_FTInput2[i] *= r_FG[i];
        r_FTInput3[i] *= r_FG[i];
    }
}

void DataType_FFTW::Convolution_Complex()
{
    // The convolution is carried out by multiplying the RHS in the frequency space with the chosen Greens function
    // This is done in-place to save memory and transfer time
    if (!Kernel_Convolution) return;

    OpenMPfor
    for (int i=0; i<NTM; i++)    Multiply(c_FTInput1[i],c_FG[i],c_FTInput1[i]);
}

void DataType_FFTW::Convolution_Complex3()
{
    // The convolution is carried out by multiplying the RHS in the frequency space with the chosen Greens function
    // This is done in-place to save memory and transfer time
    if (!Kernel_Convolution) return;

    OpenMPfor
    for (int i=0; i<NTM; i++){
        Multiply(c_FTInput1[i],c_FG[i],c_FTInput1[i]);
        Multiply(c_FTInput2[i],c_FG[i],c_FTInput2[i]);
        Multiply(c_FTInput3[i],c_FG[i],c_FTInput3[i]);
    }
}

//--- Greens functions prep

void DataType_FFTW::Prep_Greens_Function_C2C()
{
    // This is a hack to avoid using c_FG array within the solver class.
    // The Green's function is specified with the r_FG array and then converted here.

    OpenMPfor
    for (int i=0; i<NT; i++){
        c_FG[i][0] = r_FG[i];
        c_FG[i][1] = 0.0;
    }
}

void DataType_FFTW::Prep_Greens_Function_R2C()
{
    // Prepares the Fourier transform
//    FFTW_Execute_DFT(Forward_Plan,c_FG,c_FG);
    FFTW_Execute_R2C(Forward_Plan,r_FG,c_FG);
}

void DataType_FFTW::Prepare_Dif_Operators_1D(Real Hx)
{
    // This prepares the differential operators in frequency space

    c_FGi = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NX);

    for (int i=0; i<NX; i++) {
        if (2*i<NX)     c_FGi[i][1] = M_2PI*i/Hx/NX;
        else            c_FGi[i][1] = M_2PI*(i-NX)/Hx/NX;
        c_FGi[i][0] = 0.0;
    }
}

void DataType_FFTW::Prepare_Dif_Operators_2D(Real Hx, Real Hy)
{
    // This prepares the differential operators in frequency space

    c_FGi = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NX);
    c_FGj = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NY);

    for (int i=0; i<NX; i++) {
        if (2*i<NX)     c_FGi[i][1] = M_2PI*i/Hx/NX;
        else            c_FGi[i][1] = M_2PI*(i-NX)/Hx/NX;
        c_FGi[i][0] = 0.0;
    }
    for (int j=0; j<NY; j++) {
        if (2*j<NY)     c_FGj[j][1] = M_2PI*j/Hy/NY;
        else            c_FGj[j][1] = M_2PI*(j-NY)/Hy/NY;
        c_FGj[j][0] = 0.0;
    }
}

void DataType_FFTW::Prepare_Dif_Operators_3D(Real Hx, Real Hy, Real Hz)
{
    // This prepares the differential operators in frequency space

    c_FGi = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NX);
    c_FGj = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NY);
    c_FGk = (FFTWReal*) FFTW_malloc(sizeof(FFTWReal) * NZ);

    for (int i=0; i<NX; i++) {
        if (2*i<NX)     c_FGi[i][1] = M_2PI*i/Hx/NX;
        else            c_FGi[i][1] = M_2PI*(i-NX)/Hx/NX;
        c_FGi[i][0] = 0.0;
    }
    for (int j=0; j<NY; j++) {
        if (2*j<NY)     c_FGj[j][1] = M_2PI*j/Hy/NY;
        else            c_FGj[j][1] = M_2PI*(j-NY)/Hy/NY;
        c_FGj[j][0] = 0.0;
    }
    for (int k=0; k<NZ; k++) {
        if (2*k<NZ)     c_FGk[k][1] = M_2PI*k/Hz/NZ;
        else            c_FGk[k][1] = M_2PI*(k-NZ)/Hz/NZ;
        c_FGk[k][0] = 0.0;
    }
}

void DataType_FFTW::Spectral_Gradients_1D()
{
    // The convolution of the input signal with the greens function was carried out in a previous step (DataType_FFTW::Convolution_Complex()).
    // If the gradients are to be calculated, the spectral differentiation is carried out here.
    // This is done in-place

    if (Operator==NONE)     return;
    if (Operator==CURL )    return; // Invalid in 1D
    if (Operator==DIV)      return; // Just gradient in 1D

    if (Operator==GRAD)
    {
        OpenMPfor
        for (int i=0; i<NXM; i++)   Multiply(c_FGi[i],c_FTInput1[i],c_FTInput1[i]);
    }

    if (Operator==NABLA)
    {
        OpenMPfor
        for (int i=0; i<NXM; i++){
                Multiply(c_FGi[i],c_FTInput1[i],c_FTInput1[i]);
                Multiply(c_FGi[i],c_FTInput1[i],c_FTInput1[i]);
        }
    }
}

void DataType_FFTW::Spectral_Gradients_2D()
{
    // The convolution of the input signal with the greens function was carried out in a previous step (DataType_FFTW::Convolution_Complex()).
    // If the gradients are to be calculated, the spectral differentiation is carried out here.
    // This is done in-place

    if (Operator==NONE) return;

    if (Operator==DIV)
    {
        OpenMPfor
        for (int i=0; i<NXM; i++){
            for (int j=0; j<NYM; j++){
                int id = i*NYM + j;
                FFTWReal GTX, GTY;
                Multiply(c_FGi[i],c_FTInput1[id],GTX);
                Multiply(c_FGj[j],c_FTInput1[id],GTY);
                c_FTInput1[id][0] = GTX[0]+GTY[0];
                c_FTInput1[id][1] = GTX[1]+GTY[1];
            }
        }
    }

    if (Operator==GRAD)
    {
        OpenMPfor
        for (int i=0; i<NXM; i++){
            for (int j=0; j<NYM; j++){
                int id = i*NYM + j;
                Multiply(c_FGj[j],c_FTInput1[id],c_FTInput2[id]);
                Multiply(c_FGi[i],c_FTInput1[id],c_FTInput1[id]);
            }
        }
    }

    if (Operator==CURL)
    {
        OpenMPfor
        for (int i=0; i<NXM; i++){
            for (int j=0; j<NYM; j++){
                int id = i*NYM + j;
                Multiply(c_FGi[i],c_FTInput1[id],c_FTInput2[id]);
                Multiply(c_FGj[j],c_FTInput1[id],c_FTInput1[id]);
                c_FTInput2[id][0] *= -1.0;
                c_FTInput2[id][1] *= -1.0;
            }
        }
    }

    if (Operator==NABLA)
    {
        OpenMPfor
        for (int i=0; i<NXM; i++){
            for (int j=0; j<NYM; j++){
                int id = i*NYM + j;
                Multiply(c_FGj[j],c_FTInput1[id],c_FTInput2[id]);
                Multiply(c_FGj[j],c_FTInput2[id],c_FTInput2[id]);
                Multiply(c_FGi[i],c_FTInput1[id],c_FTInput1[id]);
                Multiply(c_FGi[i],c_FTInput1[id],c_FTInput1[id]);
            }
        }
    }
}

void DataType_FFTW::Spectral_Gradients_3D()
{
    // The convolution of the input signal with the greens function was carried out in a previous step (DataType_FFTW::Convolution_Complex()).
    // If the gradients are to be calculated, the spectral differentiation is carried out here.
    // This is done in-place

    if (Operator==NONE) return;

    if (Operator==DIV)
    {
        OpenMPfor
        for (int i=0; i<NXM; i++){
            for (int j=0; j<NYM; j++){
                for (int k=0; k<NZM; k++){
                    int id = i*NYM*NZM + j*NZM + k;
                    FFTWReal GTX, GTY, GTZ;
                    Multiply(c_FGi[i],c_FTInput1[id],GTX);
                    Multiply(c_FGj[j],c_FTInput1[id],GTY);
                    Multiply(c_FGk[k],c_FTInput1[id],GTZ);
                    c_FTInput1[id][0] = GTX[0]+GTY[0]+GTZ[0];
                    c_FTInput1[id][1] = GTX[1]+GTY[1]+GTZ[1];
                }
            }
        }
    }

    if (Operator==GRAD)
    {
        OpenMPfor
        for (int i=0; i<NXM; i++){
            for (int j=0; j<NYM; j++){
                for (int k=0; k<NZM; k++){
                    int id = i*NYM*NZM + j*NZM + k;
                    Multiply(c_FGk[k],c_FTInput1[id],c_FTInput3[id]);
                    Multiply(c_FGj[j],c_FTInput1[id],c_FTInput2[id]);
                    Multiply(c_FGi[i],c_FTInput1[id],c_FTInput1[id]);
                }
            }
        }
    }

    if (Operator==CURL)
    {
        // Ignore this case...
        //we would need to scan through and check which vector component is actually being filled ()
    }
}

void DataType_FFTW::Spectral_Gradients_3DV()
{
    // The convolution of the input signal with the greens function was carried out in a previous step (DataType_FFTW::Convolution_Complex()).
    // If the gradients are to be calculated, the spectral differentiation is carried out here.
    // This is done in-place

    if (Operator==NONE) return;

    if (Operator==DIV)
    {
        OpenMPfor
        for (int i=0; i<NXM; i++){
            for (int j=0; j<NYM; j++){
                for (int k=0; k<NZM; k++){
                    int id = i*NYM*NZM + j*NZM + k;
                    FFTWReal GTX, GTY, GTZ;
                    Multiply(c_FGi[i],c_FTInput1[id],GTX);
                    Multiply(c_FGj[j],c_FTInput2[id],GTY);
                    Multiply(c_FGk[k],c_FTInput3[id],GTZ);
                    c_FTInput1[id][0] = GTX[0]+GTY[0]+GTZ[0];
                    c_FTInput1[id][1] = GTX[1]+GTY[1]+GTZ[1];
                }
            }
        }
    }

    if (Operator==GRAD)
    {
        // Ignore this case for now. We would require 9 arrays to store the outputs
        // Will be useful at some point, but not right now!
    }

    if (Operator==CURL)
    {
        // Velocity field extraction for 3D case! The holy grail!

        OpenMPfor
        for (int i=0; i<NXM; i++){
            for (int j=0; j<NYM; j++){
                for (int k=0; k<NZM; k++){

                    int id = i*NYM*NZM + j*NZM + k;

                    // Calculate the frequency space representations of the curl operator
                    FFTWReal XT1, XT2, YT1, YT2, ZT1, ZT2;
                    Multiply(c_FGj[j],c_FTInput3[id],XT1);  // 1
                    Multiply(c_FGk[k],c_FTInput2[id],XT2);
                    Multiply(c_FGk[k],c_FTInput1[id],YT1);  // 2
                    Multiply(c_FGi[i],c_FTInput3[id],YT2);
                    Multiply(c_FGi[i],c_FTInput2[id],ZT1);  // 3
                    Multiply(c_FGj[j],c_FTInput1[id],ZT2);

                    // Store in arrays for backward transform
                    c_FTInput1[id][0] = XT2[0]-XT1[0];
                    c_FTInput1[id][1] = XT2[1]-XT1[1];
                    c_FTInput2[id][0] = YT2[0]-YT1[0];
                    c_FTInput2[id][1] = YT2[1]-YT1[1];
                    c_FTInput3[id][0] = ZT2[0]-ZT1[0];
                    c_FTInput3[id][1] = ZT2[1]-ZT1[1];
                }
            }
        }
    }

    if (Operator==NABLA)
    {
        OpenMPfor
        for (int i=0; i<NXM; i++){
            for (int j=0; j<NYM; j++){
                for (int k=0; k<NZM; k++){
                    int id = i*NYM*NZM + j*NZM + k;
                    Multiply(c_FGi[i],c_FTInput1[id],c_FTInput1[id]);
                    Multiply(c_FGi[i],c_FTInput1[id],c_FTInput1[id]);
                    Multiply(c_FGj[j],c_FTInput2[id],c_FTInput2[id]);
                    Multiply(c_FGj[j],c_FTInput2[id],c_FTInput2[id]);
                    Multiply(c_FGk[k],c_FTInput3[id],c_FTInput3[id]);
                    Multiply(c_FGk[k],c_FTInput3[id],c_FTInput3[id]);
                }
            }
        }
    }
}

//--- Destructor

DataType_FFTW::~DataType_FFTW()
{
    //--- This clears the data associated with this FFTW object

    // Clear arrays
    Deallocate_Arrays();

    // Clear plans
    FFTW_Destroy_Plan(Forward_Plan);
    FFTW_Destroy_Plan(Backward_Plan);

//    fftw_cleanup_threads();
//    fftw_cleanup();
}

}

#endif
