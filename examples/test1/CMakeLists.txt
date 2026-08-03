cmake_minimum_required(VERSION 3.21)
project(hipster_test LANGUAGES CXX)

set(EXE_NAME test1)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(cmake/hipEmbed.cmake)
include(cmake/hipLaunch.cmake)
include(cmake/hipSetup.cmake)
include(cmake/projectOptions.cmake)

setup_hip_environment()
ensure_rocm_available()

add_hip_kernel(test_kernel 
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel.hip.cpp
    HSACO_NAME graph_kernel
    EMBED_NAME kernel_embed
    ARCH gfx1103
)

add_executable(${EXE_NAME} main.cpp)

set_project_options(${EXE_NAME})
add_project_includes(${EXE_NAME})
add_hip_definitions(${EXE_NAME})

target_link_libraries(${EXE_NAME} PRIVATE
    eio
    uring
    hipster
    test_kernel_obj
    kernel_embed
)

add_dependencies(${EXE_NAME} graph_kernel)