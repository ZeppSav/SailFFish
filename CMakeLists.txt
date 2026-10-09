cmake_minimum_required(VERSION 3.23)

project(SailFFish VERSION 1.0 LANGUAGES CXX)

option(BUILD_SHARED_LIBS "Build shared library" OFF)
option(SAILFFISH_NATIVE_ARCH "Compile with -march=native" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)                      # Set c++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)             # Enforce

#--- RPATH tweaks for relocatable installation
include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)     # Also add external library paths to RPATH
set(CMAKE_SKIP_BUILD_RPATH FALSE)               # Do not skip build RPATH
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)       # Use build RPATH in build tree, install RPATH on install
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")     # Not tested
    list(APPEND CMAKE_INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
endif()

#--- Add library directory (options and backend selection live in src/CMakeLists.txt)
add_subdirectory(src)

if(SAILFFISH_NATIVE_ARCH)
    target_compile_options(SailFFish_lib PRIVATE -march=native)
endif()

add_executable(SailFFish main.cpp)
target_link_libraries(SailFFish PRIVATE SailFFish_lib)
if(SAILFFISH_NATIVE_ARCH)
    target_compile_options(SailFFish PRIVATE -march=native)
endif()

install(TARGETS SailFFish)
