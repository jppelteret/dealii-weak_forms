##
#  CMake script for the project:
##

# Set the name of the project and target:
# SET(TARGET "weak_forms-evaluation_exec")
SET(TARGET_LIB "dealii_weakforms")

# Declare all header files the target consists of:
INCLUDE_DIRECTORIES(include)

# Declare all source files the target consists of.
# FILE(GLOB_RECURSE TARGET_SRC  
#   )
FILE(GLOB_RECURSE TARGET_LIB_INC 
  "${CMAKE_SOURCE_DIR}/include/*.h"
  "${CMAKE_SOURCE_DIR}/include/weak_forms/*.h"
)

FILE(GLOB_RECURSE TARGET_LIB_SRC  
  "${CMAKE_SOURCE_DIR}/source/*.cc"
  "${CMAKE_SOURCE_DIR}/source/weak_forms/*.cc"
  )

# Declare all include files:
FILE(GLOB_RECURSE TARGET_ALL_INC  
  "${CMAKE_SOURCE_DIR}/include/*.h"
  "${CMAKE_SOURCE_DIR}/include/weak_forms/*.h"
  )

# Declare all source files:
FILE(GLOB_RECURSE TARGET_ALL_SRC  
  "${CMAKE_SOURCE_DIR}/source/*.cc"
  "${CMAKE_SOURCE_DIR}/source/weak_forms/*.cc"
  )

# Declare all test files:
FILE(GLOB_RECURSE TARGET_ALL_TESTS
  "${CMAKE_SOURCE_DIR}/tests/*.h"
  "${CMAKE_SOURCE_DIR}/tests/*.cc"
)

# Declare all benchmark test files:
FILE(GLOB_RECURSE TARGET_ALL_BENCHMARKS
  "${CMAKE_SOURCE_DIR}/benchmarks/*.h"
  "${CMAKE_SOURCE_DIR}/benchmarks/*.cc")


# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 10.0 QUIET
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

SET(CLEAN_UP_FILES
  *.log *.vtu *.pvtu *.pvd *.vtk
)

DEAL_II_INITIALIZE_CACHED_VARIABLES()
# PROJECT(${TARGET})
PROJECT(${TARGET_LIB})

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# DEAL_II_INVOKE_AUTOPILOT()

#
# Are all dependencies fulfilled?
#
IF(NOT DEAL_II_WITH_ADOLC OR NOT DEAL_II_WITH_TRILINOS OR NOT DEAL_II_WITH_SYMENGINE)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_ADOLC = ON
    DEAL_II_WITH_TRILINOS = ON
    DEAL_II_WITH_SYMENGINE = ON
One or all of these are OFF in your installation but are required for this project."
    )
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
      ${TARGET_ALL_INC}
      ${TARGET_ALL_SRC}
      ${TARGET_ALL_TESTS}
      ${TARGET_ALL_BENCHMARKS}
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
  SET (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "default install path" FORCE )
ENDIF()

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

  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

  # add the automatically determined parts of the RPATH
  # which point to directories outside the build tree to the install RPATH
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

  # the RPATH to be used when installing, but only if it's not a system directory
  LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  IF("${isSystemDir}" STREQUAL "-1")
     SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  ENDIF("${isSystemDir}" STREQUAL "-1")

  # make sure RPATHs are used on macOS
  SET(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")
ELSE() # Linux
  SET(CMAKE_SKIP_BUILD_RPATH FALSE)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE) # tried also with TRUE
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
ENDIF()

##
# Create library and executable
##
ADD_LIBRARY(${TARGET_LIB} SHARED
    ${TARGET_LIB_SRC}
  )
SET_TARGET_PROPERTIES(${TARGET_LIB} PROPERTIES PUBLIC_HEADER "${TARGET_LIB_INC}")

# ADD_EXECUTABLE(${TARGET}
#     ${TARGET_SRC}
#   )

DEAL_II_SETUP_TARGET(${TARGET_LIB})
# DEAL_II_SETUP_TARGET(${TARGET})

# TARGET_LINK_LIBRARIES(${TARGET}
#     ${TARGET_LIB}
#   )

# Install binary
INSTALL(
  TARGETS 
    # ${TARGET} 
    ${TARGET_LIB}
    # RUNTIME  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
    LIBRARY  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

##
# Generate custom "debug" and "release" make targets
##
# ADD_CUSTOM_TARGET(debug
# COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
# COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
# COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
# )

# ADD_CUSTOM_TARGET(release
# COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
# COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
# COMMENT "Switch CMAKE_BUILD_TYPE to Release"
# )

##
# Build documentation
##
OPTION(BUILD_DOCUMENTATION "Build the Doxygen documentation" OFF)
OPTION(DOXYGEN_EXECUTABLE "Path of the doxygen executable" "doxygen")

IF(BUILD_DOCUMENTATION)
  MESSAGE(STATUS "Configuring with Doxygen documentation")
  ADD_SUBDIRECTORY(doc/doxygen)

  ADD_CUSTOM_TARGET(docs
    COMMAND doxygen doc/doxygen/options.dox
    COMMENT "Build Doxygen documentation"
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
    MESSAGE(STATUS "Path to deal.II test header: ${DEAL_II_TESTS_H}")
  ELSE()
    MESSAGE(STATUS "Tests could not be enabled: deal.II's test.h file could not be found.")
  ENDIF()

  IF(BUILD_TESTS AND EXISTS ${DEAL_II_TESTS_H})
    MESSAGE(STATUS "Tests enabled")
    INCLUDE_DIRECTORIES(${DEAL_II_SOURCE_DIR}/tests)
    ADD_SUBDIRECTORY(tests)
  ENDIF()

  IF(BUILD_BENCHMARKS AND EXISTS ${DEAL_II_TESTS_H})
    IF(${CMAKE_BUILD_TYPE} MATCHES "Release")
      MESSAGE(STATUS "Benchmarks enabled")

      # Set the time limit for the benchmark tests
      MATH(EXPR CUSTOM_TEST_TIME_LIMIT "24 * 60 * 60") # Seconds
      SET(TEST_TIME_LIMIT ${CUSTOM_TEST_TIME_LIMIT})

      INCLUDE_DIRECTORIES(${DEAL_II_SOURCE_DIR}/tests)
      ADD_SUBDIRECTORY(benchmarks)
    ELSE()
      MESSAGE(STATUS "Benchmarks disabled when not in release mode")
    ENDIF()
  ENDIF()
ENDIF()
