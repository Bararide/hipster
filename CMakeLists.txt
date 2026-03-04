cmake_minimum_required(VERSION 3.16)
project(hipster)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(hip REQUIRED)
find_package(rocblas REQUIRED)

add_library(hipster INTERFACE)

target_include_directories(hipster INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hipster>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/3dparty/boost>
)

target_compile_definitions(hipster INTERFACE
    BOOST_HANA_CONFIG_ENABLE_STRING_UDL
)

target_link_libraries(hipster INTERFACE
    cppcore
    spdlog::spdlog
    fmt::fmt
    hip::host
    roc::rocblas
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

target_include_directories(hipster INTERFACE
    ${JSONCPP_INCLUDE_DIRS}
    ${HIP_INCLUDE_DIRS}
    ${ROCBLAS_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/3dparty/cppcore
    ${CMAKE_SOURCE_DIR}/3dparty/spdlog
)