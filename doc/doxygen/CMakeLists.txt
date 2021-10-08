
MESSAGE(STATUS "Generating options.dox for Doxygen")
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/options.dox.in
  ${CMAKE_CURRENT_BINARY_DIR}/options.dox
)

# file(READ "options.dox.in" input_buffer)
# string(REGEX REPLACE "([^\n]+)\n" "Hello : \\1\n" input_buffer ${output_buffer})
# file(WRITE "options.dox" ${output_buffer})

SET(INPUT  "${CMAKE_CURRENT_BINARY_DIR}/options.dox")
SET(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/options.dox")

STRING(REPLACE "\@CMAKE_SOURCE_DIR\@" "${CMAKE_CURRENT_SOURCE_DIR}" ${OUTPUT} ${INPUT})
STRING(REPLACE "\@CMAKE_CURRENT_BINARY_DIR\@" "${CMAKE_CURRENT_BINARY_DIR}" ${OUTPUT} ${INPUT})

#
# Finalize the doxygen configuration:
#

SET(_doxygen_input
  ${CMAKE_CURRENT_SOURCE_DIR}/headers/
)

LIST(APPEND _doxygen_input
  ${CMAKE_SOURCE_DIR}/include/
  ${CMAKE_SOURCE_DIR}/source/
  )

MACRO(TO_STRING _variable)
  SET(${_variable} "")
  FOREACH(_var  ${ARGN})
    SET(${_variable} "${${_variable}} ${_var}")
  ENDFOREACH()
  STRING(STRIP "${${_variable}}" ${_variable})
ENDMACRO()

TO_STRING(_doxygen_input_string ${_doxygen_input})


FILE(APPEND "${CMAKE_CURRENT_BINARY_DIR}/options.dox"
  "
  INPUT=${_doxygen_input_string}
  IMAGE_PATH=${_doxygen_image_path_string}
  "
  )