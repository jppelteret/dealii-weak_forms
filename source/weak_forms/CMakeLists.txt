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

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

SET(_src
    assembler_base.cc
    assembler_matrix_based.cc
    symbolic_decorations.cc
    utilities.cc
  )

SET(_inst
    assembler_base.inst.in
    assembler_matrix_based.inst.in
  )

FILE(GLOB _header
  ${CMAKE_SOURCE_DIR}/include/weak_forms/*.h
  )

DEAL_II_ADD_LIBRARY(obj_weakforms OBJECT ${_src} ${_header} ${_inst})
EXPAND_INSTANTIATIONS(obj_weakforms "${_inst}")
