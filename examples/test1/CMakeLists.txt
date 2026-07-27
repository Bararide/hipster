set(EXE_NAME test1)

add_executable(${EXE_NAME} main.cpp)

target_link_libraries(${EXE_NAME} PRIVATE
    eio
    uring
    hipster
)
