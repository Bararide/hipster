function(add_hip_kernel TARGET_NAME SOURCE_FILE)
    set(options NO_HSACO NO_OBJECT NO_EMBED)
    set(oneValueArgs HSACO_NAME EMBED_NAME ARCH)
    set(multiValueArgs "")
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_ARCH)
        set(ARG_ARCH "gfx1103")
    endif()

    if(NOT ARG_HSACO_NAME)
        set(ARG_HSACO_NAME "${TARGET_NAME}_kernel")
    endif()

    if(NOT ARG_EMBED_NAME)
        set(ARG_EMBED_NAME "${ARG_HSACO_NAME}_embed")
    endif()

    find_program(HIPCC_EXECUTABLE hipcc)
    if(NOT HIPCC_EXECUTABLE)
        message(FATAL_ERROR "hipcc not found")
    endif()

    if(NOT ARG_NO_OBJECT)
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.o
            COMMAND ${HIPCC_EXECUTABLE} -c -std=c++26 -fPIC 
                    ${SOURCE_FILE} -o ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.o
            DEPENDS ${SOURCE_FILE}
            COMMENT "Compiling HIP kernel object: ${TARGET_NAME}"
            VERBATIM
        )
        
        add_library(${TARGET_NAME}_obj STATIC ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.o)
        set_target_properties(${TARGET_NAME}_obj PROPERTIES LINKER_LANGUAGE CXX)
        set(${TARGET_NAME}_OBJECT ${TARGET_NAME}_obj PARENT_SCOPE)
    endif()

    if(NOT ARG_NO_HSACO)
        set(HSACO_FILE ${CMAKE_CURRENT_BINARY_DIR}/${ARG_HSACO_NAME}.hsaco)
        
        add_custom_command(
            OUTPUT ${HSACO_FILE}
            COMMAND ${HIPCC_EXECUTABLE} --genco --offload-arch=${ARG_ARCH} -O3 
                    -o ${HSACO_FILE} ${SOURCE_FILE}
            DEPENDS ${SOURCE_FILE}
            COMMENT "Compiling HIP kernel to .hsaco: ${ARG_HSACO_NAME}"
            VERBATIM
        )
        
        add_custom_target(${ARG_HSACO_NAME} ALL DEPENDS ${HSACO_FILE})
        set(${TARGET_NAME}_HSACO ${ARG_HSACO_NAME} PARENT_SCOPE)
    endif()

    if(NOT ARG_NO_EMBED)
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ARG_EMBED_NAME}.cpp
            COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_BINARY_DIR} 
                    xxd -i ${ARG_HSACO_NAME}.hsaco ${ARG_EMBED_NAME}.cpp
            DEPENDS ${HSACO_FILE}
            COMMENT "Embedding .hsaco into binary: ${ARG_EMBED_NAME}"
        )
        
        add_library(${ARG_EMBED_NAME} STATIC 
                    ${CMAKE_CURRENT_BINARY_DIR}/${ARG_EMBED_NAME}.cpp)
        set(${TARGET_NAME}_EMBED ${ARG_EMBED_NAME} PARENT_SCOPE)
    endif()

endfunction()