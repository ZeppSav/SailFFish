# SailFFish - A lightweight fast Poisson solver for execution on both CPU & GPU.

The purpose of the SailFFish is to provide an open source, easily linked fast Poisson solver with minimal 
dependencies. The software is configured for shared memory machines. 
At the heart of the solver is the fast fourier transform (FFT), which allows us to integrate the Poisson equation in frequency space.
Transforms to and from the frequency space via FFTs are not carried out by SailFFish, 
but rather call existing (and very optimised) libraries through the inherited `DataType` class. 
Currently two compilation options exists: The first is the (deservedly) popular library [FFTW](https://www.fftw.org/) for calculation on a CPU. 
The second is the high-performance NVIDIA FFT implementation [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) for calculation on a GPU. 
Solvers exist for 1D, 2D and 3D scalar and 3D vector input fields. A range of differential operators may be applied to modify the form of the Poisson equation being solved.

<p align="center">
<img src="/img/Ring_Omega_Vel.png" width=65% height=65%> 
</p>
A 3D_Vector style solver with a curl differential operator has been used to solve for the velocity distribution (right half: x-velocity field) of a vortex ring (left half: voriticity contours).
<br />

A [preprint](https://arxiv.org/abs/2301.01145) has been prepared for SailFFish which contains a detailed overview of the software architecture along with a full set of validation cases.

## Why should you use SailFFish?
* You want a relatively simple library for the calculation of the Poisson equation on a regular grid;
* You are not carrying out calculations with a distributed memory architecture;
* You want to use the GPU to optimise your calculations;
* You want to compile a lightweight executable for a specific form of the fast Poisson equation;
* You want an easy library or namespace which you can link to in your own software.

## What does SailFFish do?
SailFish solves the Poisson equation
```math
\nabla^2\psi = f
```
where the right-hand side $f$ is defined on a rectangular grid with regular grid spacing. 
The input/solution vector can be specified for either *regular* (cell boundary) or *staggered* (cell centre) grid configurations and can be defined as either 
scalar data or 3-vector data.  
### Which boundary conditions are available?
The solver can be executed as a *bounded* solver, whereby the boundary conditions (BC) are specified.
In this case three types of solver are available:
* Periodic BC
* Dirichlet (even-symmetric) or Neumann (odd-symmetric) BC
	* Homogeneous BC: Two options exist for the eigenvalues: pseudo-spectral (`SailFFish::PS`) or finite-difference second-order (`SailFFish::FD2`) 
	* Inhomogeneous BC: then arbitrary BCs may be specified however only with use of the (`SailFFish::FD2`) type eigenvalue.

Alternatively an *unbounded* solver may be applied, whereby the BCs need not be specified and the potential is treated as if the problem were unbounded.
In this case a range of options exist for the calculation of the free-space Green's function. 

## What is the data architecture of SailFFish?
The functionality required for FFT plan setup, execution & destruction along with frequency space operations are wrapped up inside the `DataType` class. 
The base `Solver` class is derived from the choice of `DataType` class specified at compile time. This inheritence is described in the figure below.   
<img src="/img/Arch.png" width=100% height=100%>
Data architecture within SailFFish. <br />
For users who wish to use another FFT library, it is simply a matter of specifying a new `DataType` class, linking to the library and ensuring the `Solver` class inherits this.
For users who wish to implement specialized solver classes, it is simply a matter of creating a new derived class from the existing solvers. 

## Licensing and authorship
SailFFish is developed by Joseph Saverin and is distributed under the GNU General Public License Version 2.0, copyright © Joseph Saverin 2022.

## How to use SailFFish
In order to use SailFFish you simply need to link to the `Solvers.h` header in the source directory and link to the appropriate libraries for the chosen `DataType` class.
### Creating a bounded type solver 
This is achieved by generating the associated `Solver` type and then carrying out the grid setup:
```
SailFFish::Grid_Type G = SailFFish::STAGGERED;
SailFFish::Bounded_Kernel K = SailFFish::FD2;
SailFFish::Poisson_Periodic_2D *Solver2DP = new SailFFish::Poisson_Periodic_2D(G,K);
int NX = 128;
int NY = 256;
float UnitX[2] = {-1.0, 1.0};
float UnitY2[2] = {-2.0, 2.0};
SailFFish::SFStatus Stat = Solver2DP->Setup(UnitX,UnitY2,NX,NY);
```
This will generate a 2D solver for the Poisson equation with periodic BCs. 
In the $x$ direction the grid extends between $-1$ and $1$ and has $128$ cells.
In the $y$ direction the grid extends between $-2$ and $2$ and has $256$ cells. 
The input must be specified on a staggered grid, so that the input is specified at $128$ x $256$ cell-centred grid positions. 
The solution will be found on this same grid (with the same ordering) by using the finite-difference second-order eigenvalues.
The options for grid type  and bounded kernel are:
* `Grid_Type`          	{REGULAR, STAGGERED};
* `Bounded_Kernel`     	{PS, FD2};
### Creating an unbounded type solver 
This is achieved by generating the associated `Solver` type and then carrying out the grid setup:
```
SailFFish::Grid_Type G = SailFFish::REGULAR;
SailFFish::Unbounded_Kernel K = SailFFish::HEJ_G8;
SailFFish::Unbounded_Solver_3DV *Solver3DU = new SailFFish::Unbounded_Solver_3DV(G,K);
Solver3DU->Specify_Operator(SailFFish::CURL);
int NX = 128;
int NY = 256;
int NZ = 512;
float Unit[2] = {-1.0, 1.0};
SailFFish::SFStatus Stat = Solver3DU->Setup(Unit,Unit,Unit,NX,NY,NZ);
```
This will generate a 3D solver for the Poisson equation with unbounded BCs. 
This also illustrates where and how the differential operator is specified.
In the $x$ direction the grid extends between $-1$ and $1$ and has $128$ cells.
In the $y$ direction the grid extends between $-1$ and $1$ and has $256$ cells. 
In the $z$ direction the grid extends between $-1$ and $1$ and has $512$ cells. 
The input must be specified on a regular grid, so that the input is specified at $129$ x $257$ x $513$ cell-boundary grid positions. 
The solution will be found on this same grid (with the same ordering) by using the Hejlesen eighth-order Gaussian kernel.
The options for the unbounded kernel and the differential operators are:
* `Unbounded_Kernel` 	{HEJ_S0, HEJ_G2, HEJ_G4, HEJ_G6, HEJ_G8, HEJ_G10};
* `OperatorType` 		{NONE, DIV, CURL, GRAD, NABLA};
### Specifying input vector - f
In order to pass the input (right-hand side, or $f$) values, you need to construct a `std::vector` of the chosen floating point precision. 
```
std::vector<float> Input = std::vector<float>(NT,1.0);
SailFFish::SFStatus Stat = Solver2DP->Set_Input(Input);
```
In this example (given for the 2D periodic case above) an array of a very uninteresting input field is constructed. The spatial ordering of the grid in SailFFish is *row-major*. 
This implies that as as you step through adjacent memory locations, the first dimension’s index varies most slowly and the last dimension’s index varies most quickly. 
This is then passed to the solver using the `Set_Input` function. This function has an `SFStatus` output type as a check is carried out to ensure that the correct number of grid node values are being passed. 
The following values are defined:
* `SFStatus`       {NoError, DimError, MemError, SetupError, ExecError};
This can be used to abort calculation if an error type is thrown (this is done in the test cases).
### Executing
This is carried out by calling the following three functions in this order:
```
Solver3DU->Forward_Transform();
Solver3DU->Convolution();
Solver3DU->Backward_Transform();
```
where the `Solver3DU` solver from above has been used as an example. This could be wrapped together into a single function, however as the user may wish 
to perform additional operations before or after the convolution, for flexibility it shall be left like this for now.
### Extracting results
The solution $\psi$ on the grid can be extracted as follows:
```
std::vector<float> Output;
Solver2DP->Get_Output(Output);
```
The ordering is again row-major, as with the input. The output vector is automatically allocated with the correct size within the `Get_Output` function.
### Exporting the source and solution on a grid
A method is defined within each 2D and 3D solver base classes which creates a `.vti` file. 
This allows visualisation of the source and solution fields with [Paraview](https://www.paraview.org/). 
The image of the vortex ring above was generated automatically with this function. 

## Citation information		
An ArXiv preprint has been prepared which contains an overview of the solver, data architecture and a range of validation cases. 
This can be found at the [following link](https://arxiv.org/abs/2301.01145).

## Compilation
The compilation of SailFFish has been tested with GCC (v7.3).
Two options are available for compiling:
- qmake: SailFFish was prepared with the cross-platform development environment [Qt Creator](https://www.qt.io/product/development-tools). 
The .pro file required for compiling with qmake has been provided. 
- CMake: The `CMakeLists.txt` file has been provided. Note: This is not thoroughly tested!
### Floating point precision	
SailFFish can be compiled to use either single or double floating point precision. 
Simply specify with the appropriate compiler flags: `SinglePrec` or `DoublePrec`
### Choice of datatype
As described above, there are two native options for `DataType` in SailFFish. These are specified with the appropriate compiler flags:
* `FFTW` The code will compile such that the FFTW3 library is used. 
* `CUDA` The code will compile such that the cuFFT library is used. 
In either case, it should be clear from the `.pro` or `CMakeLists.txt` files where you need to point to the corresponding directories and the libraries which must be linked. 