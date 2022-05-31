##
# CMake script for the project:
##

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

##
# Set the name of the project and target:
##
SET(WEAK_FORMS_PROJECT_NAME "weak forms")
SET(WEAK_FORMS_PARENT_PROJECT_NAME "deal.II")
SET(WEAK_FORMS_PACKAGE_NAME "${WEAK_FORMS_PARENT_PROJECT_NAME}-${WEAK_FORMS_PROJECT_NAME}")

# Match deal.II library name: libdeal_II <--> libdeal_II-weak_forms
STRING(REPLACE " " "_" WEAK_FORMS_LIB "${WEAK_FORMS_PACKAGE_NAME}")
STRING(REPLACE "." "_" WEAK_FORMS_LIB "${WEAK_FORMS_LIB}")

# Match GitHub project name: dealii-weak_forms
STRING(REPLACE " " "_" WEAK_FORMS_SANITIZED_PACKAGE_NAME "${WEAK_FORMS_PACKAGE_NAME}")
STRING(REPLACE "." "" WEAK_FORMS_SANITIZED_PACKAGE_NAME "${WEAK_FORMS_SANITIZED_PACKAGE_NAME}")
STRING(TOLOWER "${WEAK_FORMS_SANITIZED_PACKAGE_NAME}" WEAK_FORMS_SANITIZED_PACKAGE_NAME)

##
# Declare project version
##
SET(WEAK_FORMS_VERSION_MAJOR    0)
SET(WEAK_FORMS_VERSION_MINOR    1)
SET(WEAK_FORMS_VERSION_SUBMINOR 0)
SET(WEAK_FORMS_VERSION ${WEAK_FORMS_VERSION_MAJOR}.${WEAK_FORMS_VERSION_MINOR}.${WEAK_FORMS_VERSION_SUBMINOR})


##
# Information about paths, install locations and names
##
SET(WEAK_FORMS_CMAKE_EXPORT_NAME "${WEAK_FORMS_SANITIZED_PACKAGE_NAME}")
SET(WEAK_FORMS_PROJECT_CONFIG_NAME "${WEAK_FORMS_CMAKE_EXPORT_NAME}")
SET(WEAK_FORMS_PROJECT_TARGET_NAME "${WEAK_FORMS_PROJECT_CONFIG_NAME}Targets")

SET(WEAK_FORMS_EXECUTABLE_RELDIR "bin")
SET(WEAK_FORMS_INCLUDE_RELDIR "include")
SET(WEAK_FORMS_CONFIGURED_FILES_RELDIR "configured_files")
SET(WEAK_FORMS_LIBRARY_RELDIR "lib")
SET(WEAK_FORMS_PROJECT_CONFIG_RELDIR "${WEAK_FORMS_LIBRARY_RELDIR}/cmake/${WEAK_FORMS_PROJECT_CONFIG_NAME}")
SET(WEAK_FORMS_SHARE_RELDIR "share/${WEAK_FORMS_SANITIZED_PACKAGE_NAME}")
SET(WEAK_FORMS_DOCHTML_RELDIR "doc")

##
# Follow the official guidelines for setting up CMake projects:
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/tutorials/How-to-create-a-ProjectConfig.cmake-file
##

##
# Offer the user the choice of overriding the installation directories
##
SET(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
SET(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
SET(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
IF(WIN32 AND NOT CYGWIN)
  SET(DEF_INSTALL_CMAKE_DIR CMake)
ELSE()
  SET(DEF_INSTALL_CMAKE_DIR ${WEAK_FORMS_PROJECT_CONFIG_RELDIR})
ENDIF()
SET(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "Installation directory for CMake files")

IF(NOT IS_ABSOLUTE "${CMAKE_INSTALL_PREFIX}")
  SET(CMAKE_INSTALL_PREFIX_IS_ABSOLUTE FALSE)

  MESSAGE(STATUS "CMake install path is not absolute. Correcting...")
  SET(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_PREFIX}")
ELSE()
  SET(CMAKE_INSTALL_PREFIX_IS_ABSOLUTE TRUE)
ENDIF()

# Make relative paths absolute (needed later on)
FOREACH(p LIB BIN INCLUDE CMAKE)
  SET(var INSTALL_${p}_DIR)
  IF(NOT IS_ABSOLUTE "${${var}}")
    SET(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  ENDIF()
ENDFOREACH()


##
# Configuration files for this project:
##
CONFIGURE_FILE(
  "${CMAKE_SOURCE_DIR}/include/weak_forms/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${WEAK_FORMS_CONFIGURED_FILES_RELDIR}/weak_forms/config.h"
)

# Declare all header files the target consists of:
INCLUDE_DIRECTORIES(
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}/${WEAK_FORMS_CONFIGURED_FILES_RELDIR}"
)

# Declare all header files the target consists of
FILE(GLOB_RECURSE FILE_LIST_INC 
  "${CMAKE_SOURCE_DIR}/include/*.h"
  "${CMAKE_SOURCE_DIR}/include/weak_forms/*.h"
)

# Declare all source files the target consists of:
FILE(GLOB_RECURSE FILE_LIST_SRC  
  "${CMAKE_SOURCE_DIR}/source/*.cc"
  "${CMAKE_SOURCE_DIR}/source/weak_forms/*.cc"
)

# Declare all test files:
FILE(GLOB_RECURSE FILE_LIST_TESTS
  "${CMAKE_SOURCE_DIR}/tests/*.h"
  "${CMAKE_SOURCE_DIR}/tests/*.cc"
)

# Declare all benchmark test files:
FILE(GLOB_RECURSE FILE_LIST_BENCHMARKS
  "${CMAKE_SOURCE_DIR}/benchmarks/*.h"
  "${CMAKE_SOURCE_DIR}/benchmarks/*.cc"
)

# Declare all example files:
FILE(GLOB_RECURSE FILE_LIST_EXAMPLES
  "${CMAKE_SOURCE_DIR}/examples/*.h"
  "${CMAKE_SOURCE_DIR}/examples/*.cc"
)

##
# Setup deal.II
##
FIND_PACKAGE(deal.II 9.4 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ELSE()
  MESSAGE(STATUS 
  "Found deal.II ${DEAL_II_VERSION} installation at location: ${DEAL_II_DIR}"
  )
ENDIF()

IF(DEAL_II_WITH_ADOLC OR DEAL_II_WITH_TRILINOS)
  IF(DEAL_II_WITH_TRILINOS)
    MESSAGE(STATUS 
    "DEAL_II_WITH_TRILINOS = ON"
    )
  ENDIF()
  IF(DEAL_II_WITH_ADOLC)
    MESSAGE(STATUS 
    "DEAL_II_WITH_ADOLC = ON"
    )
  ENDIF()
  MESSAGE(STATUS 
  "Found deal.II with AD library support. Auto-differentiation support enabled."
  )
ENDIF()

IF(DEAL_II_WITH_SYMENGINE)
  MESSAGE(STATUS 
  "DEAL_II_WITH_SYMENGINE = ON"
  )
  MESSAGE(STATUS 
  "Found deal.II with SD library support. Symbolic differentiation support enabled."
  )
ENDIF()

##
# Initialise this project
##
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${WEAK_FORMS_LIB} CXX)

##
# Reset the module path
##
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

##
# Load CMake macros
##
MESSAGE(STATUS "Include ${CMAKE_SOURCE_DIR}/cmake/setup_external_macros.cmake")
INCLUDE(${CMAKE_SOURCE_DIR}/cmake/setup_external_macros.cmake)

##
# Run system checks
# (These might require the deal.II project to be configured first)
##
FILE(GLOB _check_files "cmake/checks/*.cmake")
LIST(SORT _check_files)
FOREACH(_file ${_check_files})
  MESSAGE(STATUS "Include ${_file}")
  INCLUDE(${_file})
ENDFOREACH()

##
# Setup the desired compiler flags. Currently, for debugging purposes only.
#
# This is done after the system checks, just in case we've set some
# pedantic build flags (e.g. -Werror) and this triggers unexpected
# in the build checks.
##
OPTION(STRICT_BUILD_FLAGS "Developer mode: Add strict flags for building the library and tests" OFF)
IF(STRICT_BUILD_FLAGS)
  MESSAGE(STATUS "Adding strict build flags")
  INCLUDE(${CMAKE_SOURCE_DIR}/cmake/setup_compiler_flags.cmake)
ENDIF()

##
# Setup Clang format:
# https://github.com/zemasoft/clangformat-cmake
# https://github.com/zemasoft/clangformat-cmake-example
##
OPTION(CLANGFORMAT "Setup clangformat target" OFF)

IF(CLANGFORMAT)
  INCLUDE(ClangFormat)

  # List sources to format
  SET(CLANG_FORMAT_SRCS 
    ${FILE_LIST_INC}
    ${FILE_LIST_SRC}
    ${FILE_LIST_TESTS}
    ${FILE_LIST_BENCHMARKS}
    ${FILE_LIST_EXAMPLES}
  )

  # Setup clangformat target
  clangformat_setup("${CLANG_FORMAT_SRCS}")
  MESSAGE(STATUS "ClangFormat found and configured")
  MESSAGE(STATUS "Path to clang format executable: ${CLANGFORMAT_EXECUTABLE}")
ENDIF()

##
# Setup project installation
##

# Set installation path
# https://stackoverflow.com/a/16076855
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}" 
       CACHE PATH "default install path" FORCE
      )
ENDIF()
MESSAGE(STATUS 
  "Target installation path: ${CMAKE_INSTALL_PREFIX}"
)

##
# Manage RPaths
# https://cmake.org/Wiki/CMake_RPATH_handling#Mac_OS_X_and_the_RPATH
##

# Configure RPath handling 
# https://cmake.org/Wiki/CMake_RPATH_handling
# https://cmake.org/pipermail/cmake/2010-April/036618.html // <----- Setting a configure-time option
IF(APPLE)
  IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    ## cmake  --help-policy CMP0042
    set(CMAKE_MACOSX_RPATH ON)
  ENDIF()

  # use, i.e. don't skip the full RPATH for the build tree
  SET(CMAKE_SKIP_BUILD_RPATH FALSE)

  # when building, don't use the install RPATH already
  # (but later on when installing)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${WEAK_FORMS_LIBRARY_RELDIR}")

  # add the automatically determined parts of the RPATH
  # which point to directories outside the build tree to the install RPATH
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

  # the RPATH to be used when installing, but only if it's not a system directory
  LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/${WEAK_FORMS_LIBRARY_RELDIR}" isSystemDir)
  IF("${isSystemDir}" STREQUAL "-1")
     SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${WEAK_FORMS_LIBRARY_RELDIR}")
  ENDIF("${isSystemDir}" STREQUAL "-1")

  # make sure RPATHs are used on macOS
  SET(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/${WEAK_FORMS_LIBRARY_RELDIR}")
ELSE() # Linux
  SET(CMAKE_SKIP_BUILD_RPATH FALSE)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${WEAK_FORMS_LIBRARY_RELDIR})
ENDIF()

##
# Create library and executable
##
ADD_LIBRARY(${WEAK_FORMS_LIB} SHARED
  ${FILE_LIST_SRC}
)

##
# Build a different debug and release library
# https://cmake.org/cmake/help/git-stage/guide/tutorial/Packaging%20Debug%20and%20Release.html
##
SET(CMAKE_DEBUG_POSTFIX ".g")

SET_TARGET_PROPERTIES(${WEAK_FORMS_LIB} PROPERTIES 
  PUBLIC_HEADER "${FILE_LIST_INC};${CMAKE_CURRENT_BINARY_DIR}/${WEAK_FORMS_CONFIGURED_FILES_RELDIR}/weak_forms/config.h"
  DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX}
)

DEAL_II_SETUP_TARGET(${WEAK_FORMS_LIB})

##
# Install generated headers
##
INSTALL(
  DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${WEAK_FORMS_CONFIGURED_FILES_RELDIR}/weak_forms"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/${WEAK_FORMS_INCLUDE_RELDIR}"
  COMPONENT library
  FILES_MATCHING PATTERN "*.h"
)

##
# Install binary
##
INSTALL(
  TARGETS ${WEAK_FORMS_LIB}
  EXPORT ${WEAK_FORMS_PROJECT_TARGET_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/${WEAK_FORMS_INCLUDE_RELDIR}/weak_forms COMPONENT dev
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${WEAK_FORMS_LIBRARY_RELDIR} COMPONENT shlib
)

##
# Build documentation
##
OPTION(BUILD_DOCUMENTATION "Build the Doxygen documentation" OFF)
OPTION(DOXYGEN_EXECUTABLE "Path of the doxygen executable" "doxygen")

IF(BUILD_DOCUMENTATION)
  MESSAGE(STATUS "Configuring with Doxygen documentation")
  ADD_SUBDIRECTORY(doc/doxygen)

  ADD_CUSTOM_TARGET(documentation
    COMMAND doxygen ${CMAKE_CURRENT_BINARY_DIR}/doc/doxygen/options.dox
    COMMENT "Build Doxygen documentation"
  )
  FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html")
  INSTALL(
    DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/${WEAK_FORMS_DOCHTML_RELDIR}"
  )
ENDIF()

##
# Enable testing and descent into tests/ subdirectory
##
OPTION(BUILD_TESTS "Configure test environment" ON)

##
# Enable testing and descent into tests/ subdirectory
##
OPTION(BUILD_BENCHMARKS "Configure benchmark tests" OFF)

##
# Ensure that we don't allow tests and benchmarks at the same time
# This is because running 'ctest' would invoke the both of them, which
# is probably not what is desired.
##
IF(BUILD_BENCHMARKS AND BUILD_TESTS)
  MESSAGE(FATAL_ERROR "For your own sanity, the benchmark tests cannot be "
                      "enabled at the same time as the testsuite.")
ENDIF()

##
# Configure the test / benchmark environment
##
IF(BUILD_TESTS OR BUILD_BENCHMARKS)
  MESSAGE(STATUS "CTest environment enabled")
  ENABLE_TESTING()
  INCLUDE(CTest)

  FIND_FILE(DEAL_II_TESTS_H tests.h
    HINTS ${deal.II_SOURCE_DIR}/tests ${DEAL_II_SOURCE_DIR} $ENV{DEAL_II_SOURCE_DIR}
    PATH_SUFFIXES tests
    NO_DEFAULT_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_PATH
    NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH NO_CMAKE_FIND_ROOT_PATH
  )

  IF(EXISTS ${DEAL_II_TESTS_H})
    STRING(REPLACE "/tests.h" "" DEAL_II_TESTS_DIR "${DEAL_II_TESTS_H}")
    MESSAGE(STATUS "Path to deal.II test header: ${DEAL_II_TESTS_H}")
    MESSAGE(STATUS "Path to deal.II test directory: ${DEAL_II_TESTS_DIR}")
  ELSE()
    MESSAGE(STATUS "Tests could not be enabled: deal.II's tests.h file could not be found.")
  ENDIF()

  IF(BUILD_TESTS AND EXISTS ${DEAL_II_TESTS_H})
    MESSAGE(STATUS "Tests enabled")
    INCLUDE_DIRECTORIES(${DEAL_II_TESTS_DIR})
    ADD_SUBDIRECTORY(tests)
  ENDIF()

  IF(BUILD_BENCHMARKS AND EXISTS ${DEAL_II_TESTS_H})
    IF(${CMAKE_BUILD_TYPE} MATCHES "Release")
      MESSAGE(STATUS "Benchmarks enabled")

      # Set the time limit for the benchmark tests
      MATH(EXPR CUSTOM_TEST_TIME_LIMIT "24 * 60 * 60") # Seconds
      SET(TEST_TIME_LIMIT ${CUSTOM_TEST_TIME_LIMIT})

      INCLUDE_DIRECTORIES(${DEAL_II_TESTS_DIR})
      ADD_SUBDIRECTORY(benchmarks)
    ELSE()
      MESSAGE(STATUS "Benchmarks disabled when not in release mode")
    ENDIF()
  ENDIF()
ENDIF()



##
# Exports
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/tutorials/How-to-create-a-ProjectConfig.cmake-file
# https://stackoverflow.com/a/49857699
##

# Add all targets to the build-tree export set
EXPORT(TARGETS ${WEAK_FORMS_LIB}
  FILE "${PROJECT_BINARY_DIR}/${WEAK_FORMS_CONFIGURED_FILES_RELDIR}/${WEAK_FORMS_PROJECT_CONFIG_NAME}Targets.cmake"
)

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
EXPORT(PACKAGE ${WEAK_FORMS_CMAKE_EXPORT_NAME})

##
# Install CMake configuration files
##

# Create the ${WEAK_FORMS_PROJECT_CONFIG_NAME}Config.cmake and ${WEAK_FORMS_PROJECT_CONFIG_NAME}ConfigVersion files
FILE(
  RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
  "${INSTALL_INCLUDE_DIR}"
)
    
# ... for the build tree
SET(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config/Config.cmake.in
  "${PROJECT_BINARY_DIR}/${WEAK_FORMS_CONFIGURED_FILES_RELDIR}/${WEAK_FORMS_PROJECT_CONFIG_NAME}Config.cmake" 
  @ONLY
)

# ... for the install tree
SET(CONF_INCLUDE_DIRS "\${WEAK_FORMS_CMAKE_DIR}/${REL_INCLUDE_DIR}")
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config/Config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${WEAK_FORMS_PROJECT_CONFIG_NAME}Config.cmake" 
  @ONLY
)

# ... for both
CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config/ConfigVersion.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${WEAK_FORMS_PROJECT_CONFIG_NAME}ConfigVersion.cmake" 
  @ONLY
)

# Install the ${WEAK_FORMS_PROJECT_CONFIG_NAME}Config.cmake and ${WEAK_FORMS_PROJECT_CONFIG_NAME}ConfigVersion.cmake
INSTALL(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${WEAK_FORMS_PROJECT_CONFIG_NAME}Config.cmake"
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${WEAK_FORMS_PROJECT_CONFIG_NAME}ConfigVersion.cmake"
  DESTINATION "${INSTALL_CMAKE_DIR}" 
  COMPONENT dev
)

##
# Install the export set for use with the install-tree
##
INSTALL(
  EXPORT ${WEAK_FORMS_PROJECT_TARGET_NAME}
  DESTINATION "${INSTALL_CMAKE_DIR}"
  COMPONENT dev
)
  
