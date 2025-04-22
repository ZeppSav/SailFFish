TEMPLATE = app
CONFIG += console c++20
CONFIG -= app_bundle
CONFIG -= qt

#----------------------------------------------
# Specify floating point precision
#----------------------------------------------

DEFINES += SinglePrec
#DEFINES += DoublePrec

#----------------------------------------------
# Configuration flags
#----------------------------------------------

# The following define makes your compiler emit warnings if you useA
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

#---Optimise flags

QMAKE_CXXFLAGS += -O3               # Optimisations for eigen
QMAKE_CXXFLAGS += -march=native     # Activate all optimisation flags

#----------------------------------------------
# OpenMP support
#----------------------------------------------

QMAKE_CXXFLAGS += -fopenmp  # Compiler support for openMP
LIBS += -fopenmp
INCLUDEPATH += "C:\Qt\Qt5.12.12\Tools\mingw730_64\lib\gcc\x86_64-w64-mingw32\7.3.0\include"

##----------------------------------------------
## DataType FFTW
##----------------------------------------------
DEFINES += FFTW
##----------------------------------------------
## include path to FFTW & FFTW Libraries
##----------------------------------------------
INCLUDEPATH += $$PWD/../FFTW/fftw-3.3.5-dll64
#INCLUDEPATH += $$PWD/../FFTW
LIBS += -L$$PWD/../FFTW/fftw-3.3.5-dll64/ -llibfftw3-3
LIBS += -L$$PWD/../FFTW/fftw-3.3.5-dll64/ -llibfftw3f-3
#LIBS += -L$$PWD/../FFTW/ -llibfftw3-3
#LIBS += -L$$PWD/../FFTW/ -llibfftw3f-3     # Precompiled from FFTW
##----------------------------------------------


##----------------------------------------------
## DataType cuFFT
##----------------------------------------------
#DEFINES += CUFFT
##----------------------------------------------
## include path to cuda & cuda Libraries
##----------------------------------------------
#INCLUDEPATH += "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include"       #Laptop
#CUDAPFAD = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
#LIBS += -L$$CUDAPFAD\bin
#LIBS += -lcudart64_110
#LIBS += -lcufft64_10
#LIBS += -lcufftw64_10
#LIBS += -lcublas64_11
##----------------------------------------------

#----------------------------------------------
# Eigen support
#----------------------------------------------
INCLUDEPATH += ..\Eigen

#-----------------------------------
# Source and header files SailFFish
#-----------------------------------

SOURCES += \
    main.cpp \
    src/DataTypes/DataType_Base.cpp \
    src/DataTypes/DataType_CUDA.cpp \
    src/DataTypes/DataType_FFTW.cpp \
    src/DataTypes/DataType_MKL.cpp \
    src/Solvers/Dirichlet_Solver.cpp \
    src/Solvers/Export_Grid.cpp \
    src/Solvers/Neumann_Solver.cpp \
    src/Solvers/Periodic_Solver.cpp \
    src/Solvers/Solver_Base.cpp \
    src/Solvers/Unbounded_Solver.cpp

HEADERS += \
    src/DataTypes/DataType_Base.h \
    src/DataTypes/DataType_CUDA.h \
    src/DataTypes/DataType_FFTW.h \
    src/DataTypes/DataType_MKL.h \
    src/SailFFish.h \
    src/Solvers/Dirichlet_Solver.h \
    src/Solvers/Greens_Functions.h \
    src/Solvers/Neumann_Solver.h \
    src/Solvers/Periodic_Solver.h \
    src/Solvers/Solver_Base.h \
    src/Solvers/Solvers.h \
    src/Solvers/Unbounded_Solver.h \
    src/VPM_Solver/VPM3D_kernels_cpu.h \
    src/VPM_Solver/VPM_Solver.h

# #------------------------------------------
# # Source and header files VPM Solver - FFTW
# #------------------------------------------

# SOURCES += src/VPM_Solver/VPM3D_cpu.cpp \
#     src/VPM_Solver/VPM_Solver.cpp

# HEADERS += src/VPM_Solver/VPM3D_cpu.h
