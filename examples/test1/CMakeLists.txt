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

set_target_properties(kernel_lib PROPERTIES
    LINKER_LANGUAGE CXX
)

add_executable(${EXE_NAME} main.cpp)

target_link_libraries(${EXE_NAME} PRIVATE
    eio
    uring
    hipster
    kernel_lib
)

set_target_properties(${EXE_NAME} PROPERTIES 
    CXX_STANDARD 26 
    CXX_STANDARD_REQUIRED ON
)

target_compile_definitions(${EXE_NAME} PRIVATE 
    __HIP_PLATFORM_AMD__
)