cmake_minimum_required(VERSION 3.10)

project(SailFFish VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)                                      # Set c++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)                             # Enforce
set(CMAKE_CXX_FLAGS "-std=gnu++11 -O3 -march=native -fopenmp")  # Set compiler optimisation flags

find_package(OpenMP REQUIRED)

#--- Preprocessor Directives
add_compile_definitions(SinglePrec)     # Single precision compile
#add_compile_definitions(DoublePrec)    # Double precision compile

#--- Compile definitions
add_compile_definitions(FFTW)           # Uses DataType_FFTW
#add_compile_definitions(CUFFT)         # Uses DataType_CUDA

#--- Specify target executable
add_executable(SailFFish main.cpp)

#--- Link OpenMP
target_link_libraries(SailFFish PRIVATE OpenMP::OpenMP_CXX)

get_directory_property(MYDEFS COMPILE_DEFINITIONS)

#--- If using FFTW DataType: Link FFTW directory
if (MYDEFS MATCHES "FFTW")
    set(FFTW_DIR C:/Development/FFTW/fftw-3.3.5-dll64)  #--- Specify your FFTW path here ---#
    target_include_directories(SailFFish PRIVATE ${FFTW_DIR})
    target_link_libraries(SailFFish PRIVATE ${FFTW_DIR}/libfftw3f-3.lib)
    target_link_libraries(SailFFish PRIVATE ${FFTW_DIR}/libfftw3-3.lib)
endif()

#--- If using FFTW DataType: Link FFTW directory
if (MYDEFS MATCHES "CUFFT")
    set(CUDA_DIR ---DIR---) #--- Specify your CUDA path here ---#
    target_include_directories(SailFFish PRIVATE ${CUDA_DIR}/include)
    target_include_directories(SailFFish PRIVATE ${CUDA_DIR}/bin)
    target_include_directories(SailFFish PRIVATE ${CUDA_DIR}/lib/x64)
    target_link_libraries(SailFFish PRIVATE ${CUDA_DIR}/cudart64_110.lib, ${FFTW_DIR}/cufft64_10.lib, ${FFTW_DIR}/cublas64_11.lib)
endif()

target_sources(SailFFish
                PRIVATE

                # Sources (DataTypes)
                "${PROJECT_SOURCE_DIR}/src/DataTypes/DataType_Base.cpp"
                "${PROJECT_SOURCE_DIR}/src/DataTypes/DataType_FFTW.cpp"
                "${PROJECT_SOURCE_DIR}/src/DataTypes/DataType_CUDA.cpp"

                # Sources (Solvers)
                "${PROJECT_SOURCE_DIR}/src/Solvers/Solver_Base.cpp"
                "${PROJECT_SOURCE_DIR}/src/Solvers/Dirichlet_Solver.cpp"
                "${PROJECT_SOURCE_DIR}/src/Solvers/Neumann_Solver.cpp"
                "${PROJECT_SOURCE_DIR}/src/Solvers/Periodic_Solver.cpp"
                "${PROJECT_SOURCE_DIR}/src/Solvers/Unbounded_Solver.cpp"
                "${PROJECT_SOURCE_DIR}/src/Solvers/Greens_Functions.h"
                "${PROJECT_SOURCE_DIR}/src/Solvers/Special_functions.cpp"
                "${PROJECT_SOURCE_DIR}/src/Solvers/Export_Grid.cpp"
                )

