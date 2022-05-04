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
# CMake macros that are used to add compiler flags
#
# Source: https://stackoverflow.com/a/33266748
##
include(CheckCXXCompilerFlag)

function(enable_cxx_compiler_flag_if_supported flag)
    string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)
    if(flag_already_set EQUAL -1)
        MESSAGE(STATUS "Compiler flag check: ${flag}")
        check_cxx_compiler_flag("${flag}" flag_supported)
        if(flag_supported)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
        endif()
        unset(flag_supported CACHE)
    endif()
endfunction()


##
# Add desired flags
##
enable_cxx_compiler_flag_if_supported("-Wall")
enable_cxx_compiler_flag_if_supported("-Werror")
enable_cxx_compiler_flag_if_supported("-Wextra")
enable_cxx_compiler_flag_if_supported("-pedantic")
