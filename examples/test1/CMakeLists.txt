cmake_minimum_required(VERSION 3.21)
project(hipster_test LANGUAGES CXX)

set(EXE_NAME test1)

find_program(HIPCC_EXECUTABLE hipcc)
if(NOT HIPCC_EXECUTABLE)
    message(FATAL_ERROR "hipcc not found")
endif()

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel.o
    COMMAND ${HIPCC_EXECUTABLE} -c -std=c++26 -fPIC ${CMAKE_CURRENT_SOURCE_DIR}/kernel.hip.cpp -o ${CMAKE_CURRENT_BINARY_DIR}/kernel.o
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel.hip.cpp
    COMMENT "Compiling HIP kernel with hipcc"
)
add_library(kernel_lib STATIC ${CMAKE_CURRENT_BINARY_DIR}/kernel.o)
set_target_properties(kernel_lib PROPERTIES LINKER_LANGUAGE CXX)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/graph_kernel.hsaco
    COMMAND ${HIPCC_EXECUTABLE} --genco --offload-arch=gfx1103 -O3 
            -o ${CMAKE_CURRENT_BINARY_DIR}/graph_kernel.hsaco
            ${CMAKE_CURRENT_SOURCE_DIR}/kernel.hip.cpp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernel.hip.cpp
    COMMENT "Compiling HIP kernel to .hsaco"
    VERBATIM
)
add_custom_target(graph_kernel_hsaco ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/graph_kernel.hsaco)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernel_embed.cpp
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_BINARY_DIR} xxd -i graph_kernel.hsaco kernel_embed.cpp
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/graph_kernel.hsaco
    COMMENT "Embedding .hsaco into binary with clean variable name"
)
add_library(kernel_embed STATIC ${CMAKE_CURRENT_BINARY_DIR}/kernel_embed.cpp)

add_executable(${EXE_NAME} main.cpp)

add_dependencies(${EXE_NAME} graph_kernel_hsaco kernel_embed)

target_link_libraries(${EXE_NAME} PRIVATE
    eio
    uring
    hipster
    kernel_lib
    kernel_embed
)

set_target_properties(${EXE_NAME} PROPERTIES 
    CXX_STANDARD 26 
    CXX_STANDARD_REQUIRED ON
)

target_compile_definitions(${EXE_NAME} PRIVATE 
    __HIP_PLATFORM_AMD__
)