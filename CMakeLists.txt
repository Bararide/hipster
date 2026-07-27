set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(3dparty/eio)
add_subdirectory(3dparty/cppcore)

add_subdirectory(hipster)
add_subdirectory(examples)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/hipster_config.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/hipster_config.h
)