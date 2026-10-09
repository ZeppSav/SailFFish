#--- Options (visible both standalone and when added via add_subdirectory(.../SailFFish/src))
set(SAILFFISH_BACKEND "FFTW" CACHE STRING "FFT backend: FFTW, CUFFT or VKFFT")
set_property(CACHE SAILFFISH_BACKEND PROPERTY STRINGS FFTW CUFFT VKFFT)
set(SAILFFISH_PRECISION "Single" CACHE STRING "Floating-point precision: Single or Double")
set_property(CACHE SAILFFISH_PRECISION PROPERTY STRINGS Single Double)
option(SAILFFISH_BUILD_VPM "Build the vortex particle mesh solver" OFF)

if(NOT SAILFFISH_BACKEND MATCHES "^(FFTW|CUFFT|VKFFT)$")
    message(FATAL_ERROR "Unknown SAILFFISH_BACKEND: ${SAILFFISH_BACKEND}")
endif()
if(NOT SAILFFISH_PRECISION MATCHES "^(Single|Double)$")
    message(FATAL_ERROR "Unknown SAILFFISH_PRECISION: ${SAILFFISH_PRECISION}")
endif()
message(STATUS "SailFFish: backend=${SAILFFISH_BACKEND}, precision=${SAILFFISH_PRECISION}, VPM=${SAILFFISH_BUILD_VPM}")

#--- Check that an external path variable is set and contains one of the expected files
function(sailffish_require_path var)
    foreach(file IN LISTS ARGN)
        if(${var} AND EXISTS "${${var}}/${file}")
            return()
        endif()
    endforeach()
    message(FATAL_ERROR "SailFFish: set ${var} (looked for ${ARGN} in '${${var}}')")
endfunction()

# Dependencies
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(SailFFish_lib)  # Create a library for SailFFish
add_library(SailFFish::SailFFish ALIAS SailFFish_lib)
set_target_properties(SailFFish_lib PROPERTIES OUTPUT_NAME SailFFish)
target_compile_definitions(SailFFish_lib PUBLIC ${SAILFFISH_PRECISION}Prec)   # SinglePrec / DoublePrec

if(SAILFFISH_BACKEND STREQUAL "FFTW")       # FFTW headers and libraries

    if(WIN32)
        sailffish_require_path(FFTW_DIR fftw3.h)
        target_include_directories(SailFFish_lib PUBLIC ${FFTW_DIR})
        target_link_libraries(SailFFish_lib PRIVATE "${FFTW_DIR}/libfftw3-3.lib" "${FFTW_DIR}/libfftw3f-3.lib")
    else()
        target_link_libraries(SailFFish_lib PRIVATE fftw3f fftw3f_threads fftw3 fftw3_threads )
    endif()
    target_compile_definitions(SailFFish_lib PUBLIC FFTW)

elseif(SAILFFISH_BACKEND STREQUAL "CUFFT")  # CUFFT headers and libraries

    if(WIN32)
        sailffish_require_path(CUDA_DIR include/cuda.h)
        target_include_directories(SailFFish_lib PUBLIC "${CUDA_DIR}/include")
        target_link_directories(SailFFish_lib PUBLIC "${CUDA_DIR}/lib/x64" "${CUDA_DIR}/bin")
        target_link_libraries(SailFFish_lib PRIVATE cuda cudart64_12 cufft64_11 cufftw64_11 cublas64_12)
    else()
        target_link_libraries(SailFFish_lib PRIVATE cuda cudart64_12 cufft64_11 cufftw64_11 cublas64_12)
    endif()
    target_compile_definitions(SailFFish_lib PUBLIC CUFFT)

elseif(SAILFFISH_BACKEND STREQUAL "VKFFT")  # VKFFT headers and libraries

    sailffish_require_path(VKFFT_DIR vkFFT.h vkFFT/vkFFT.h)
    if(WIN32)
        sailffish_require_path(OPENCL_HEADERS CL/cl.h)
        sailffish_require_path(OPENCL_LIB OpenCL.lib OpenCL.dll libOpenCL.dll.a libOpenCL.a)
    endif()
    target_include_directories(SailFFish_lib PUBLIC "${VKFFT_DIR}" "${VKFFT_DIR}/vkFFT")    # Paths for VkFFT headers
    target_include_directories(SailFFish_lib PUBLIC "${OPENCL_HEADERS}")                            # Path for CL headers
    target_link_directories(SailFFish_lib PUBLIC "${OPENCL_LIB}")                               # Path to CL lib

    if(WIN32)
        target_link_libraries(SailFFish_lib PRIVATE OpenCL)
    else()
        target_link_libraries(SailFFish_lib PRIVATE OpenCL)                                 # System ICD loader (libOpenCL.so)
    endif()
    target_compile_definitions(SailFFish_lib PUBLIC VKFFT VKFFT_BACKEND=3)
endif()

# Sources
target_sources(SailFFish_lib PRIVATE
    DataTypes/DataType_Base.cpp
    DataTypes/DataType_CUDA.cpp
    DataTypes/DataType_FFTW.cpp
    DataTypes/DataType_MKL.cpp
    DataTypes/DataType_VkFFT.cpp
    Solvers/Dirichlet_Solver.cpp
    Solvers/Export_Grid.cpp
    Solvers/Neumann_Solver.cpp
    Solvers/Periodic_Solver.cpp
    Solvers/Unbounded_Solver.cpp
    Solvers/Solver_Base.cpp
)

# Headers
target_sources(SailFFish_lib PUBLIC
    FILE_SET HEADERS
    FILES
    SailFFish_Math_Types.h
    DataTypes/DataType_Base.h
    DataTypes/DataType_CUDA.h
    DataTypes/DataType_FFTW.h
    DataTypes/DataType_MKL.h
    DataTypes/DataType_VkFFT.h
    SailFFish.h
    Solvers/Dirichlet_Solver.h
    Solvers/Greens_Functions.h
    Solvers/Neumann_Solver.h
    Solvers/Periodic_Solver.h
    Solvers/Unbounded_Solver.h
    Solvers/Solver_Base.h
)

#--- Add source if compiling VPM solver
if(SAILFFISH_BUILD_VPM)

    target_sources(SailFFish_lib PRIVATE VPM_Solver/VPM_Solver.cpp )
    target_sources(SailFFish_lib PUBLIC FILE_SET HEADERS FILES VPM_Solver/VPM_Solver.h VPM_Solver/VPM3D_kernels_cpu.h )

    if(SAILFFISH_BACKEND STREQUAL "FFTW")
        target_sources(SailFFish_lib PRIVATE VPM_Solver/VPM3D_cpu.cpp )
        target_sources(SailFFish_lib PUBLIC FILE_SET HEADERS FILES VPM_Solver/VPM3D_cpu.h )
    elseif(SAILFFISH_BACKEND STREQUAL "CUFFT")
        sailffish_require_path(JITIFY_DIR jitify.hpp)
        target_include_directories(SailFFish_lib PUBLIC "${JITIFY_DIR}")                            # Path for jitify headers
        target_link_libraries(SailFFish_lib PRIVATE nvrtc)                                          # Link to nvrtc lib for compiling cuda kernels
        target_sources(SailFFish_lib PRIVATE VPM_Solver/VPM3D_cuda.cpp )
        target_sources(SailFFish_lib PUBLIC FILE_SET HEADERS FILES VPM_Solver/VPM3D_cuda.h )
    elseif(SAILFFISH_BACKEND STREQUAL "VKFFT")
        target_sources(SailFFish_lib PRIVATE VPM_Solver/VPM3D_ocl.cpp )
        target_sources(SailFFish_lib PUBLIC FILE_SET HEADERS FILES VPM_Solver/VPM3D_ocl.h )
    endif()

endif()

target_link_libraries(SailFFish_lib PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)

include(GNUInstallDirs)
install(TARGETS SailFFish_lib FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/SailFFish)
