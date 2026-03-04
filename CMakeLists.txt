cmake_minimum_required(VERSION 3.16)

add_library(hipster INTERFACE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH /opt/rocm)

find_package(hip REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocprim REQUIRED)
find_package(rocrand REQUIRED)
find_package(rocsparse REQUIRED)
find_package(rocsolver REQUIRED)
find_package(rocfft REQUIRED)
find_package(hipblas REQUIRED)
find_package(hipsparse REQUIRED)
find_package(hipfft REQUIRED)
find_package(hiprand REQUIRED)
find_package(hipsolver REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

find_package(fmt REQUIRED)

set(SPDLOG_FMT_EXTERNAL ON)

target_include_directories(hipster INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3dparty/boost
    ${CMAKE_SOURCE_DIR}/3dparty/cppcore
    ${CMAKE_SOURCE_DIR}/3dparty/cppcore/infrastructure
    ${CMAKE_SOURCE_DIR}/3dparty/spdlog
    ${JSONCPP_INCLUDE_DIRS}
    ${HIP_INCLUDE_DIRS}
    ${ROCBLAS_INCLUDE_DIRS}
    ${ROCPRIM_INCLUDE_DIRS}
    ${ROCRAND_INCLUDE_DIRS}
    ${ROCSPARSE_INCLUDE_DIRS}
    ${ROCSOLVER_INCLUDE_DIRS}
    ${ROCFFT_INCLUDE_DIRS}
    ${MIOPEN_INCLUDE_DIRS}
)

target_compile_definitions(hipster INTERFACE 
    BOOST_HANA_CONFIG_ENABLE_STRING_UDL
    __HIP_PLATFORM_AMD__
)

target_link_libraries(hipster INTERFACE 
    ${JSONCPP_LIBRARIES}
    fmt::fmt
    hip::host
    roc::rocblas
    roc::rocprim
    roc::rocrand
    roc::rocsparse
    roc::rocsolver
    roc::rocfft
    ${ROCPROFILER_LIBRARIES}
    ${ROCTRACER_LIBRARIES}
)