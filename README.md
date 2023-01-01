# SailFFish - A simple, optimised fast Poisson solver in c++.

The purpose of the SailFFish is to provide an open source, easily linked fast Poisson solver with minimal 
dependencies. The software is configured for shared memory machines. 
At the heart of the solver is the fast fourier transform (FFT), which allows us to integrate the Poisson equation in frequency space.
Transforms to and from the frequency space via FFTs are not carried out by SailFFish, 
but rather call existing (and very optimised) libraries through the inherited `DataType` class. 
Currently two options exists: The first is the (deservedly) popular library [FFTW](https://www.fftw.org/) for calculation on a CPU. 
The second is the high-performance NVIDIA FFT implementation [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) for calculation on a GPU. 
Solvers exist for 1D, 2D and 3D scalar and 3D vector input fields. A range of differential operators may be applied to modify the form of the Poisson equation being solved.
<img align="center" src="/img/Ring_Omega_Vel.png" width=65% height=65%> <br />
*A 3D_Vector style solver with a curl differential operator has been used to solve for the velocity distribution (right half: x-velocity field) of a vortex ring (left half: voriticity contours).*
<br />

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
The input/solution vector can be specified for either *regular* (cell boundary) or *staggered* (cell centre) grid configurations. 
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
*Data architecture within SailFFish.*
For users who wish to use another FFT library, it is simply a matter of specifying a new `DataType` class, linking to the library and ensuring the `Solver` class inherits this.
For users to with to implement specialized solver classes, it is simply a matter of creating a new derived class from the existing solvers. 
## Licensing and authorship
SailFFish is developed by Joseph Saverin and is distributed under the GNU General Public License Version 2.0, copyright © Joseph Saverin 2022.

## How to use SailFFish
In order to use SailFFish you simply need to link to the `Solvers.h` header in the source directory.

### Creating a solver for a specified grid
This is achieved by generating the associated `Solver` type and then carrying out the grid setup:
```
SailFFish::Grid_Type G = SailFFish::Staggered;
SailFFish::Bounded_Kernel K = SailFFish::FD2;
SailFFish::Poisson_Periodic_2D *Solver = new SailFFish::Poisson_Periodic_2D(G,K);
int NX = 128;
int NY = 256;
float UnitX[2] = {-1.0, 1.0};
float UnitY2[2] = {-2.0, 2.0};
SailFFish::SFStatus Stat = Solver->Setup(UnitX,UnitY2,NX,NY);
```
This will generate a 2D solver for the Poisson equation with periodic BCs. 
In the $x$ direction the grid extends between $-1$ and $1$ and has $128$ cells.
In the $y$ direction the grid extends between $-2$ and $2$ and has $256$ cells. 
The input must be specified on a staggered grid, so that the input is specified at $[128x256]$ cell-centred grid positions. 
The solution will be found on this same grid (with the same ordering) by using the finite-difference second-order eigenvalues.
### Inputing grid values
Do that
### Executing
Know this 
### Extracting results
Know that

## SailFFish DataType
Blah

### FFTW
Bee bup
### cuFFT
Bup bup bo bup

## Citation information		
Point to Preprint!

## Compilation


Two options are available for compiling:
- qmake: SailFFish was prepared with the cross-platform development environment [Qt Creator](https://www.qt.io/product/development-tools). 
The .pro file required for compiling with qmake has been provided. 

- CMake: The `CMakeLists.txt` file has been provided. 

## Documentation

A [readthedocs] page is currently being prepared. 
