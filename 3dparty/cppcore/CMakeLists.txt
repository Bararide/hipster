add_library(cppcore INTERFACE)

target_include_directories(cppcore INTERFACE
    ${CMAKE_SOURCE_DIR}/3dparty/spdlog
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/infrastructure
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/memory
)

target_link_libraries(cppcore INTERFACE spdlog)