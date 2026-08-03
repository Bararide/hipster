macro(setup_hip_environment)
    find_program(HIPCC_EXECUTABLE hipcc)
    if(NOT HIPCC_EXECUTABLE)
        message(FATAL_ERROR "hipcc not found. Please install ROCm.")
    endif()
    
    message(STATUS "HIP compiler: ${HIPCC_EXECUTABLE}")
    
    execute_process(
        COMMAND ${HIPCC_EXECUTABLE} --version
        OUTPUT_VARIABLE HIP_VERSION_OUTPUT
        ERROR_QUIET
    )
    string(REGEX MATCH "HIP version: ([0-9.]+)" _ "${HIP_VERSION_OUTPUT}")
    if(CMAKE_MATCH_1)
        set(HIP_VERSION ${CMAKE_MATCH_1})
        message(STATUS "HIP version: ${HIP_VERSION}")
    endif()
endmacro()

function(ensure_rocm_available)
    if(NOT DEFINED ENV{ROCM_PATH})
        set(ENV{ROCM_PATH} "/opt/rocm")
    endif()
    
    if(NOT EXISTS $ENV{ROCM_PATH})
        message(WARNING "ROCM_PATH not found: $ENV{ROCM_PATH}")
    else()
        message(STATUS "ROCM path: $ENV{ROCM_PATH}")
    endif()
endfunction()