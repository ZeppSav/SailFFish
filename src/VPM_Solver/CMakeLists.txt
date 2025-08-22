# Dependencies
# find_package(Eigen3 REQUIRED)       #Note this isn't working..

set(EIGEN3_INCLUDE_DIR "C:/Development/eigen")
target_include_directories(SailFFish_lib PRIVATE ${EIGEN3_INCLUDE_DIR})

# Sources
target_sources(SailFFish_lib PRIVATE
    VPM_Solver/VPM_Solver.cpp
    VPM_Solver/VPM3D_kernels_cpu.h
)

# Headers
target_sources(SailFFish_lib PUBLIC
    FILE_SET HEADERS
    FILES
    VPM_Solver/VPM_Solver.h
)


if(BUILD_FFTW && BUILD_VPM)

    # Sources
    target_sources(SailFFish_lib PRIVATE
        VPM_Solver/VPM3D_cpu.cpp
    )
    # Headers
    target_sources(SailFFish_lib PUBLIC
        FILE_SET HEADERS
        FILES
        VPM_Solver/VPM3D_cpu.h
    )

elseif(BUILD_CUFFT && BUILD_VPM) # Link CUFFT headers and libraries

    # Sources
    target_sources(SailFFish_lib PRIVATE
    )

elseif(BUILD_VKFFT)

    # Sources
    target_sources(SailFFish_lib PRIVATE
    )

endif()

# target_link_libraries(SailFFish_lib PUBLIC Eigen3::Eigen)

include(GNUInstallDirs)
install(TARGETS SailFFish_lib FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/SailFFish)
