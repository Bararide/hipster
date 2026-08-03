function(embed_hsaco_file HSACO_INPUT EMBED_OUTPUT)
    get_filename_component(HSACO_NAME ${HSACO_INPUT} NAME_WE)
    
    if(EXISTS ${HSACO_INPUT})
        set(HSACO_FILE ${HSACO_INPUT})
    else()
        set(HSACO_FILE ${CMAKE_CURRENT_BINARY_DIR}/${HSACO_INPUT})
    endif()
    
    if(EMBED_OUTPUT)
        set(OUTPUT_NAME ${EMBED_OUTPUT})
    else()
        set(OUTPUT_NAME ${HSACO_NAME}_embed)
    endif()
    
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_NAME}.cpp
        COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_BINARY_DIR} 
                xxd -i ${HSACO_FILE} ${OUTPUT_NAME}.cpp
        DEPENDS ${HSACO_FILE}
        COMMENT "Embedding ${HSACO_FILE} as ${OUTPUT_NAME}"
    )
    
    add_library(${OUTPUT_NAME} STATIC 
                ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_NAME}.cpp)
    set(${OUTPUT_NAME}_LIB ${OUTPUT_NAME} PARENT_SCOPE)
endfunction()