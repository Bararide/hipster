macro(set_project_options TARGET)
    set_target_properties(${TARGET} PROPERTIES 
        CXX_STANDARD 26
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    
    target_compile_definitions(${TARGET} PRIVATE 
        __HIP_PLATFORM_AMD__
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${TARGET} PRIVATE -g -O0)
    else()
        target_compile_options(${TARGET} PRIVATE -O3)
    endif()
endmacro()

macro(add_project_includes TARGET)
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/3dparty
        ${CMAKE_SOURCE_DIR}/hipster
        ${CMAKE_SOURCE_DIR}/hipster/include
        ${CMAKE_SOURCE_DIR}/hipster/core
    )
endmacro()

function(add_hip_definitions TARGET)
    target_compile_definitions(${TARGET} PRIVATE
        __HIP_PLATFORM_AMD__
        USE_PROF_API=1
    )
endfunction()