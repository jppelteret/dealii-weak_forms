## ---------------------------------------------------------------------
##
## Copyright (C) 2021 - 2022 by Jean-Paul Pelteret
##
## This file is part of the Weak forms for deal.II library.
##
## The Weak forms for deal.II library is free software; you can use it,
## redistribute it, and/or modify it under the terms of the GNU Lesser
## General Public License as published by the Free Software Foundation;
## either version 3.0 of the License, or (at your option) any later
## version. The full text of the license can be found in the file LICENSE
## at the top level of the Weak forms for deal.II distribution.
##
## ---------------------------------------------------------------------


##
# A macro that inverts the logic of CMake's CHECK_CXX_SOURCE_RUNS
#
# We want the flipped as the macro should only be definied in the event
# of a failure.
##
MACRO(WF_CHECK_CXX_SOURCE_RUNS _source _var)
  IF(NOT DEFINED ${_var}_OK)
  CHECK_CXX_SOURCE_RUNS(
      "${_source}"
      ${_var}_OK
      )
  ENDIF()

  IF(${_var}_OK)
    # MESSAGE(STATUS "Execution of source successful: do not define ${_var}")
    SET(${_var} FALSE)
  ELSE()
    MESSAGE(STATUS "Execution of source failed: define ${_var}")
    SET(${_var} TRUE)
  ENDIF()
ENDMACRO()