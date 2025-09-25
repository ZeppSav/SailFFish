# Dependencies
find_package(Eigen3 REQUIRED)       #Note this isn't working..

add_library(SailFFish_vpm)  # Create a static library for BEMUse
set_target_properties(SailFFish_vpm PROPERTIES OUTPUT_NAME SailFFish_VPM)

# Sources
target_sources(SailFFish_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/VPM_Solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VPM3D_kernels_cpu.h
)

# Headers
target_sources(SailFFish_vpm PUBLIC
    FILE_SET HEADERS
    FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/VPM_Solver.h
)


if(BUILD_FFTW AND BUILD_VPM)

    # Sources
    target_sources(SailFFish_vpm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/VPM3D_cpu.cpp
    )
    # Headers
    target_sources(SailFFish_vpm PUBLIC
        FILE_SET HEADERS
        FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/VPM3D_cpu.h
    )

elseif(BUILD_CUFFT AND BUILD_VPM) # Link CUFFT headers and libraries

    # Sources
    target_sources(SailFFish_vpm PRIVATE
    )

elseif(BUILD_VKFFT)

    # Sources
    target_sources(SailFFish_vpm PRIVATE
    )

endif()

target_link_libraries(SailFFish_vpm PUBLIC Eigen3::Eigen)

include(GNUInstallDirs)
install(TARGETS SailFFish_vpm FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/SailFFish)
