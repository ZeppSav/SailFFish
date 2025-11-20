TEMPLATE = app
CONFIG += console c++20
CONFIG -= app_bundle
CONFIG -= qt

#----------------------------------------------
# Specify backend
#----------------------------------------------

# CONFIG += FFTWBackend
CONFIG += CUDABackend
# CONFIG += VKFFTBackend

# CONFIG += VPMSolver

#----------------------------------------------
# Specify floating point precision
#----------------------------------------------

DEFINES += SinglePrec
# DEFINES += DoublePrec

#----------------------------------------------
# Configuration flags
#----------------------------------------------

# The following define makes your compiler emit warnings if you use
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
QMAKE_CXXFLAGS += -fopenmp
LIBS += -fopenmp


#-----------------------------------
# Source and header files SailFFish
#-----------------------------------

SOURCES += \
    main.cpp \
    src/DataTypes/DataType_Base.cpp \
    src/DataTypes/DataType_CUDA.cpp \
    src/DataTypes/DataType_FFTW.cpp \
    src/DataTypes/DataType_MKL.cpp \
    src/DataTypes/DataType_VkFFT.cpp \
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
    src/DataTypes/DataType_VkFFT.h \
    src/SailFFish.h \
    src/Solvers/Dirichlet_Solver.h \
    src/Solvers/Greens_Functions.h \
    src/Solvers/Neumann_Solver.h \
    src/Solvers/Periodic_Solver.h \
    src/Solvers/Solver_Base.h \
    src/Solvers/Unbounded_Solver.h \
    src/VPM_Solver/VPM3D_kernels_ocl.h

#----------------------------------------------
# DataType FFTW
#----------------------------------------------
CONFIG(FFTWBackend){
    DEFINES += FFTW
    #----------------------------------------------
    # include path to FFTW & FFTW Libraries
    #----------------------------------------------
    win32{
        FFTW_DIR = $$PWD/../FFTW/fftw-3.3.5-dll64
        INCLUDEPATH += $$FFTW_DIR
        LIBS += -L$$FFTW_DIR -llibfftw3-3 -llibfftw3f-3
    }
    unix: LIBS += -lfftw3f -lfftw3f_threads -lfftw3 -lfftw3_threads
    #----------------------------------------------
}

#----------------------------------------------
# DataType cuFFT
#----------------------------------------------
CONFIG(CUDABackend) {
    DEFINES += CUFFT
    #----------------------------------------------
    # include path to cuda & cuda Libraries
    #----------------------------------------------
    INCLUDEPATH += "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include"
    CUDAPFAD = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
    win32: LIBS += -L$$CUDAPFAD\bin -lcudart64_110 -lcufft64_10 -lcufftw64_10 -lcublas64_11
    win32: LIBS += -L$$CUDAPFAD\lib\x64 -lcuda -lnvrtc
    # win32: LIBS += -L$$CUDAPFAD\bin -lcudart64_12 -lcufft64_11 -lcufftw64_11 -lcublas64_12
    # win32: LIBS += -L$$CUDAPFAD\lib\x64 -lcuda -lnvrtc
    unix: LIBS += -L$$CUDAPFAD\bin -lcudart64_110 -lcufft64_10 -lcufftw64_10 -lcublas64_11
    unix: LIBS += -L$$CUDAPFAD\lib\x64 -lcuda -lnvrtc
    #----------------------------------------------
}

#----------------------------------------------
# DataType VkFFT
#----------------------------------------------
CONFIG(VKFFTBackend) {
    DEFINES += VKFFT
    #----------------------------------------------
    # include path to OpenCL headers & VkFFT
    #----------------------------------------------
    INCLUDEPATH += $$PWD/..                     # Add path for CL headers
    INCLUDEPATH += $$PWD/../vkFFT               # Add path for vkFFT library
    INCLUDEPATH += $$PWD/../vkFFT/vkFFT         # Add path for vkFFT library
    # INCLUDEPATH += $$PWD/../VkFFT/half_lib    # Add path for half precision library
    # INCLUDEPATH += $$PWD/../VkFFT/benchmark_scripts/vkFFT_scripts/src      # Add path for tests and utilities sources

    #--- Include OpenCL lib
    win32: LIBS += -L$$PWD/libswin64 -lOpenCL
    DEFINES += VKFFT_BACKEND=3
}

CONFIG(VPMSolver) {

    #----------------------------------------------
    # Eigen support
    #----------------------------------------------
    INCLUDEPATH += ..\Eigen

    #-----------------------------------
    # Source and header files VPM Solver
    #-----------------------------------

    SOURCES += src/VPM_Solver/VPM_Solver.cpp
    HEADERS += src/VPM_Solver/VPM_Solver.h \
        src/VPM_Solver/VPM3D_kernels_cpu.h

    # Source and header files VPM Solver - FFTW backend
    CONFIG(FFTWBackend) {
        SOURCES += src/VPM_Solver/VPM3D_cpu.cpp
        HEADERS += src/VPM_Solver/VPM3D_cpu.h
    }

    # Source and header files VPM Solver - CUDA backend
    CONFIG(CUDABackend) {
        INCLUDEPATH += $$PWD/../jitify      #Add Jitify path for custom kernels
        SOURCES += src/VPM_Solver/VPM3D_cuda.cpp
        HEADERS += src/VPM_Solver/VPM3D_cuda.h \
            src/VPM_Solver/VPM3D_kernels_cuda.h
    }

    # Source and header files VPM Solver - VKFFT backend
    CONFIG(VKFFTBackend) {
        SOURCES += src/VPM_Solver/VPM3D_ocl.cpp
        HEADERS += src/VPM_Solver/VPM3D_ocl.h
    }
}
