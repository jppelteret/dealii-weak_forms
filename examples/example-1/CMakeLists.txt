##
#  CMake script for deal.II weak forms programs
#  This follows closely the scripts provided with the deal.II step-X
#  tutorial programs:
##

# Set the name of the project and target:
SET(TARGET "example-1")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#    FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#    FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#    SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
SET(TARGET_SRC
  ${TARGET}.cc
  )

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

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

DEAL_II_INITIALIZE_CACHED_VARIABLES()

##
# The project must be initialised after the deal.II cached variables are
# initialised, but before the Weak Forms package is imported.
##
PROJECT(${TARGET} CXX)

##
# Setup deal.II Weak Forms
##
FIND_PACKAGE(dealii-weak_forms 0.1.0
  HINTS ${weak_forms_DIR} ${WEAK_FORMS_DIR} ../ ../../ $ENV{WEAK_FORMS_DIR}
  )
IF(NOT ${weak_forms_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II Weak Forms. ***\n\n"
    "You may want to either pass a flag -DWEAK_FORMS_DIR=/path/to/<deal.II Weak Forms> to cmake\n"
    "or set an environment variable \"WEAK_FORMS_DIR\" that contains this path."
    )
ELSE()
  MESSAGE(STATUS "Found deal.II Weak Forms installation at location: ${WEAK_FORMS_DIR}")
ENDIF()

##
# Allow deal.II Weak Forms headers to be picked up
INCLUDE_DIRECTORIES(${WEAK_FORMS_INCLUDE_DIRS})

##
# Setup the binary that is to be built
##
ADD_EXECUTABLE(${TARGET} "${TARGET_SRC}")

##
# Link in the dependent libraries
##
DEAL_II_SETUP_TARGET(${TARGET})
TARGET_LINK_LIBRARIES(${TARGET} ${WEAK_FORMS_LIBRARIES})
