cmake_minimum_required(VERSION 3.25)
project(hipster)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(__HIP_PLATFORM_AMD__)

find_package(hip REQUIRED)
find_package(rocblas REQUIRED)

find_package(spdlog REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
find_package(fmt REQUIRED)

set(SPDLOG_FMT_EXTERNAL ON)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++20" HAS_CPP20_FLAG)
if(NOT HAS_CPP20_FLAG)
    message(FATAL_ERROR "Compiler does not support C++20")
endif()

find_path(HSA_INCLUDE_DIR
    NAMES hsa.h hsa_ext_finalize.h
    PATHS
        /opt/hsa/include
        /opt/rocm/include/hsa
        /opt/rocm/include
        /usr/include/hsa
        /usr/local/include/hsa
)

find_library(HSA_RUNTIME_LIBRARY
    NAMES hsa-runtime64 libhsa-runtime64.so
    PATHS
        /opt/hsa/lib
        /opt/rocm/lib
        /opt/rocm/lib64
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
)

find_library(HSA_IMAGE_LIBRARY
    NAMES hsa-runtime-image64 libhsa-runtime-image64.so
    PATHS
        /opt/hsa/lib
        /opt/rocm/lib
        /opt/rocm/lib64
)

find_library(HSA_FINALIZE_LIBRARY
    NAMES hsa-runtime-finalize64 libhsa-runtime-finalize64.so
    PATHS
        /opt/hsa/lib
        /opt/rocm/lib
        /opt/rocm/lib64
)

find_library(HSA_TOOLS_LIBRARY
    NAMES hsa-runtime-tools64 libhsa-runtime-tools64.so
    PATHS
        /opt/hsa/lib
        /opt/rocm/lib
        /opt/rocm/lib64
)

find_library(HSA_AMD_EXT_LIBRARY
    NAMES hsa-amd-aqlprofile64 libhsa-amd-aqlprofile64.so
    PATHS
        /opt/rocm/lib
        /opt/rocm/lib64
        /opt/rocm-7.2.0/lib
)

find_library(HIP_DMABUF_LIBRARY
    NAMES amdhip64 libamdhip64.so
    PATHS
        /opt/rocm/lib
        /opt/rocm/lib64
        /opt/rocm-7.2.0/lib
)

find_path(DRM_INCLUDE_DIR
    NAMES drm.h
    PATHS
        /usr/include/drm
        /usr/include/libdrm
        /usr/local/include/drm
        /opt/rocm/include/drm
)

find_library(DRM_LIBRARY
    NAMES drm libdrm.so
    PATHS
        /usr/lib
        /usr/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        /opt/rocm/lib
)

find_library(DRM_AMDGPU_LIBRARY
    NAMES drm_amdgpu libdrm_amdgpu.so
    PATHS
        /usr/lib
        /usr/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        /opt/rocm/lib
)

find_path(LINUX_DMA_BUF_INCLUDE_DIR
    NAMES linux/dma-buf.h
    PATHS
        /usr/include
        /usr/local/include
)

if(HSA_INCLUDE_DIR AND HSA_RUNTIME_LIBRARY)
    set(HSA_FOUND TRUE)
    message(STATUS "HSA found: ${HSA_INCLUDE_DIR}")
    message(STATUS "HSA runtime library: ${HSA_RUNTIME_LIBRARY}")
    
    add_library(hsa_interface INTERFACE)
    target_include_directories(hsa_interface INTERFACE ${HSA_INCLUDE_DIR})
    target_link_libraries(hsa_interface INTERFACE ${HSA_RUNTIME_LIBRARY})
    
    if(HSA_IMAGE_LIBRARY)
        target_link_libraries(hsa_interface INTERFACE ${HSA_IMAGE_LIBRARY})
        message(STATUS "HSA image library found")
    endif()
    
    if(HSA_FINALIZE_LIBRARY)
        target_link_libraries(hsa_interface INTERFACE ${HSA_FINALIZE_LIBRARY})
        message(STATUS "HSA finalize library found")
    endif()
    
    if(HSA_TOOLS_LIBRARY)
        target_link_libraries(hsa_interface INTERFACE ${HSA_TOOLS_LIBRARY})
        message(STATUS "HSA tools library found")
    endif()
    
    if(HSA_AMD_EXT_LIBRARY)
        target_link_libraries(hsa_interface INTERFACE ${HSA_AMD_EXT_LIBRARY})
        message(STATUS "HSA AMD extensions found")
    endif()
    
    add_compile_definitions(HAVE_HSA)
else()
    set(HSA_FOUND FALSE)
    message(WARNING "HSA not found. Install HSA runtime from https://github.com/HSAFoundation/HSA-Runtime-AMD")
endif()

if(DRM_INCLUDE_DIR AND DRM_LIBRARY)
    set(DRM_FOUND TRUE)
    message(STATUS "DRM found: ${DRM_INCLUDE_DIR}")
    
    add_library(drm_interface INTERFACE)
    target_include_directories(drm_interface INTERFACE ${DRM_INCLUDE_DIR})
    target_link_libraries(drm_interface INTERFACE ${DRM_LIBRARY})
    
    if(DRM_AMDGPU_LIBRARY)
        target_link_libraries(drm_interface INTERFACE ${DRM_AMDGPU_LIBRARY})
        message(STATUS "DRM AMDGPU library found")
    endif()
    
    add_compile_definitions(HAVE_DRM)
else()
    set(DRM_FOUND FALSE)
    message(WARNING "DRM not found")
endif()

if(LINUX_DMA_BUF_INCLUDE_DIR)
    message(STATUS "DMA-BUF headers found: ${LINUX_DMA_BUF_INCLUDE_DIR}")
    add_compile_definitions(HAVE_DMA_BUF)
endif()

add_library(hipster INTERFACE)

target_include_directories(hipster INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hipster>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hipster/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3dparty/boost>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3dparty/cppcore>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3dparty/eio>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3dparty/spdlog/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>
    ${HIP_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
    ${fmt_INCLUDE_DIRS}
)

target_compile_definitions(hipster INTERFACE
    BOOST_HANA_CONFIG_ENABLE_STRING_UDL
)

target_link_libraries(hipster INTERFACE
    cppcore
    eio
    uring
    spdlog::spdlog
    fmt::fmt
    hip::host
    roc::rocblas
    ${JSONCPP_LIBRARIES}
    pthread
    dl
    rt
    uring
)

if(HSA_FOUND)
    target_link_libraries(hipster INTERFACE hsa_interface)
    
    if(HIP_DMABUF_LIBRARY)
        target_link_libraries(hipster INTERFACE ${HIP_DMABUF_LIBRARY})
        message(STATUS "HIP library linked for dmabuf support")
    endif()
    
    message(STATUS "HSA support enabled")
    
    if(EXISTS "/dev/kfd")
        message(STATUS "/dev/kfd found - HSA device available")
    else()
        message(WARNING "/dev/kfd not found - HSA device may not be available")
    endif()
endif()

if(DRM_FOUND)
    target_link_libraries(hipster INTERFACE drm_interface)
    message(STATUS "DRM support enabled")
endif()

set_target_properties(hipster PROPERTIES
    INTERFACE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)