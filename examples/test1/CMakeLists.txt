cmake_minimum_required(VERSION 3.21)
project(hipster_test LANGUAGES CXX HIP)

set(EXE_NAME test1)

add_executable(${EXE_NAME} main.hip)

target_link_libraries(${EXE_NAME} PRIVATE
    eio
    uring
    hipster
)

set_target_properties(${EXE_NAME} PROPERTIES CXX_STANDARD 26 CXX_STANDARD_REQUIRED ON)