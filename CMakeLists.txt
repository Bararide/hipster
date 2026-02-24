cmake_minimum_required(VERSION 3.16)
project(hipster)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

find_package(fmt REQUIRED)

set(SPDLOG_FMT_EXTERNAL ON)

target_include_directories(valito PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3dparty/boost
    ${CMAKE_SOURCE_DIR}/3dparty/cppcore
    ${CMAKE_SOURCE_DIR}/3dparty/cppcore/infrastructure
    ${CMAKE_SOURCE_DIR}/3dparty/spdlog
    ${JSONCPP_INCLUDE_DIRS}
)

target_compile_definitions(hipster PRIVATE 
    BOOST_HANA_CONFIG_ENABLE_STRING_UDL
)

target_link_libraries(hipster PRIVATE 
    ${JSONCPP_LIBRARIES}
    fmt::fmt
)