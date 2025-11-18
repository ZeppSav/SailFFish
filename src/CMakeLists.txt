# Dependencies
find_package(OpenMP REQUIRED)

add_library(SailFFish_lib)  # Create a library for SailFFish
set_target_properties(SailFFish_lib PROPERTIES OUTPUT_NAME SailFFish)

if(BUILD_FFTW)      # Link FFTW headers and libraries

    if(WIN32)
        set(FFTW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../FFTW/fftw-3.3.5-dll64")
        target_include_directories(SailFFish_lib PUBLIC ${FFTW_DIR})
        target_link_libraries(SailFFish_lib PRIVATE "${FFTW_DIR}/libfftw3-3.lib" "${FFTW_DIR}/libfftw3f-3.lib")
    else()
        target_link_libraries(SailFFish_lib PRIVATE fftw3f fftw3f_threads fftw3 fftw3_threads )
    endif()
    target_compile_definitions(SailFFish_lib PUBLIC FFTW)

elseif(BUILD_CUFFT) # Link CUFFT headers and libraries

    if(WIN32)
        target_include_directories(SailFFish_lib PUBLIC "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include")
        target_link_directories(SailFFish_lib PUBLIC "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64")
        target_link_directories(SailFFish_lib PUBLIC "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
        target_link_libraries(SailFFish_lib PRIVATE cuda cudart64_110 cufft64_10 cufftw64_10 cublas64_11)
    else()
        find_package(CUDA REQUIRED)
        target_include_directories(SailFFish_lib PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include")
        target_link_directories(SailFFish_lib PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
        target_link_libraries(SailFFish_lib PRIVATE cudart64_110 cufft64_10 cufftw64_10 cublas64_11)
    endif()
    target_compile_definitions(SailFFish_lib PUBLIC CUFFT)

elseif(BUILD_VKFFT)
    if(WIN32)
        target_include_directories(SailFFish_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../CL")                # Path for Cl headers
        target_include_directories(SailFFish_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../vkFFT")          # Path for VkFFT headers
        target_include_directories(SailFFish_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../vkFFT/vkFFT")    # Path for VkFFT headers
        target_link_libraries(SailFFish_lib PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../libswin64/OpenCL.dll")  # Link OpenCL library
    else()
        # Linux!
    endif()
    target_compile_definitions(SailFFish_lib PUBLIC VKFFT VKFFT_BACKEND=3)
endif()

#--- Add source if compiling VPM solver
if (BUILD_VPM)
    add_subdirectory(VPM_Solver) # Add the VPM directory
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

target_link_libraries(SailFFish_lib PRIVATE OpenMP::OpenMP_CXX)

include(GNUInstallDirs)
install(TARGETS SailFFish_lib FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/SailFFish)
