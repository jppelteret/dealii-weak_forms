// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2022 by Jean-Paul Pelteret
//
// This file is part of the Weak forms for deal.II library.
//
// The Weak forms for deal.II library is free software; you can use it,
// redistribute it, and/or modify it under the terms of the GNU Lesser
// General Public License as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later
// version. The full text of the license can be found in the file LICENSE
// at the top level of the Weak forms for deal.II distribution.
//
// ---------------------------------------------------------------------

#include <weak_forms/symbolic_decorations.h>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Decorations
  {
    Discretization::Discretization(const std::string solution_field,
                                   const std::string test_function,
                                   const std::string trial_solution,
                                   const std::string shape_function,
                                   const std::string dof_value,
                                   const std::string JxW)
      : solution_field(solution_field)
      , test_function(test_function)
      , trial_solution(trial_solution)
      , shape_function(shape_function)
      , dof_value(dof_value)
      , JxW(JxW)
    {}


    DifferentialOperators::DifferentialOperators(
      const std::string value,
      const std::string gradient,
      const std::string symmetric_gradient,
      const std::string divergence,
      const std::string curl,
      const std::string hessian,
      const std::string laplacian,
      const std::string third_derivative)
      : value(value)
      , gradient(gradient)
      , symmetric_gradient(symmetric_gradient)
      , divergence(divergence)
      , curl(curl)
      , hessian(hessian)
      , laplacian(laplacian)
      , third_derivative(third_derivative)
    {}


    Geometry::Geometry(const std::string position,
                       const std::string normal,
                       const std::string basis,
                       const std::string volume,
                       const std::string boundary,
                       const std::string interface)
      : position(position)
      , normal(normal)
      , basis(basis)
      , volume(volume)
      , boundary(boundary)
      , interface(interface)
    {}


    DifferentialGeometry::DifferentialGeometry(
      const std::string infinitesimal_element_volume,
      const std::string infinitesimal_element_boundary_area,
      const std::string infinitesimal_element_interface_area)
      : infinitesimal_element_volume(infinitesimal_element_volume)
      , infinitesimal_element_boundary_area(infinitesimal_element_boundary_area)
      , infinitesimal_element_interface_area(
          infinitesimal_element_interface_area)
    {}


    ContinuumMechanics::ContinuumMechanics(const std::string free_energy,
                                           const std::string stored_energy,
                                           const std::string energy_functional,
                                           const std::string residual)
      : free_energy(free_energy)
      , stored_energy(stored_energy)
      , energy_functional(energy_functional)
      , residual(residual)
    {}



    Discretization
    make_symbolic_names_discretization(const SymbolicNameType &type)
    {
      if (type == SymbolicNameType::ascii)
        {
          const std::string solution_field = "U";
          const std::string test_function  = "d";
          const std::string trial_solution = "D";
          const std::string shape_function = "Nx";
          const std::string dof_value      = "c";
          const std::string JxW            = "JxW";

          return Discretization(solution_field,
                                test_function,
                                trial_solution,
                                shape_function,
                                dof_value,
                                JxW);
        }
      else
        {
          Assert(type == SymbolicNameType::latex, ExcInternalError());

          const std::string solution_field = "\\varphi";
          const std::string test_function  = "\\delta";
          const std::string trial_solution = "\\Delta";
          const std::string shape_function = "N";
          const std::string dof_value      = "c";
          const std::string JxW            = "\\int";

          return Discretization(solution_field,
                                test_function,
                                trial_solution,
                                shape_function,
                                dof_value,
                                JxW);
        }
    }


    DifferentialOperators
    make_symbolic_names_differential_operators(const SymbolicNameType &type)
    {
      if (type == SymbolicNameType::ascii)
        {
          const std::string value              = "";
          const std::string gradient           = "Grad";
          const std::string symmetric_gradient = "symm_Grad";
          const std::string divergence         = "Div";
          const std::string curl               = "Curl";
          const std::string hessian            = "Hessian";
          const std::string laplacian          = "Laplacian";
          const std::string third_derivative   = "3rd_Derivative";

          return DifferentialOperators(value,
                                       gradient,
                                       symmetric_gradient,
                                       divergence,
                                       curl,
                                       hessian,
                                       laplacian,
                                       third_derivative);
        }
      else
        {
          Assert(type == SymbolicNameType::latex, ExcInternalError());

          const std::string value              = "";
          const std::string gradient           = "\\nabla";
          const std::string symmetric_gradient = "\\nabla^{S}";
          const std::string divergence         = "\\nabla \\cdot";
          const std::string curl               = "\\nabla \\times";
          const std::string hessian            = "\\nabla\\nabla";
          const std::string laplacian          = "\\nabla^{2}";
          const std::string third_derivative   = "\\nabla\\nabla\\nabla";

          return DifferentialOperators(value,
                                       gradient,
                                       symmetric_gradient,
                                       divergence,
                                       curl,
                                       hessian,
                                       laplacian,
                                       third_derivative);
        }
    }


    Geometry
    make_symbolic_names_geometry(const SymbolicNameType &type)
    {
      if (type == SymbolicNameType::ascii)
        {
          const std::string position  = "X";
          const std::string normal    = "N";
          const std::string basis     = "e";
          const std::string volume    = "V";
          const std::string area      = "A";
          const std::string interface = "I";

          return Geometry(position, normal, basis, volume, area, interface);
        }
      else
        {
          Assert(type == SymbolicNameType::latex, ExcInternalError());

          const std::string position  = "\\mathbf{X}";
          const std::string normal    = "\\mathbf{N}";
          const std::string basis     = "\\mathbf{e}";
          const std::string volume    = "\\textrm{V}";
          const std::string area      = "\\textrm{A}";
          const std::string interface = "\\textrm{I}";

          return Geometry(position, normal, basis, volume, area, interface);
        }
    }


    DifferentialGeometry
    make_symbolic_names_differential_geometry(const SymbolicNameType &type)
    {
      if (type == SymbolicNameType::ascii)
        {
          const std::string infinitesimal_element_volume         = "dV";
          const std::string infinitesimal_element_boundary_area  = "dA";
          const std::string infinitesimal_element_interface_area = "dI";

          return DifferentialGeometry(infinitesimal_element_volume,
                                      infinitesimal_element_boundary_area,
                                      infinitesimal_element_interface_area);
        }
      else
        {
          Assert(type == SymbolicNameType::latex, ExcInternalError());

          const std::string infinitesimal_element_volume = "\\textrm{dV}";
          const std::string infinitesimal_element_boundary_area =
            "\\textrm{dA}";
          const std::string infinitesimal_element_interface_area =
            "\\textrm{dI}";

          return DifferentialGeometry(infinitesimal_element_volume,
                                      infinitesimal_element_boundary_area,
                                      infinitesimal_element_interface_area);
        }
    }


    ContinuumMechanics
    make_symbolic_names_continuum_mechanics(const SymbolicNameType &type)
    {
      if (type == SymbolicNameType::ascii)
        {
          const std::string free_energy       = "e";
          const std::string stored_energy     = "e";
          const std::string energy_functional = "E";
          const std::string residual          = "R";

          return ContinuumMechanics(free_energy,
                                    stored_energy,
                                    energy_functional,
                                    residual);
        }
      else
        {
          Assert(type == SymbolicNameType::latex, ExcInternalError());

          const std::string free_energy       = "\\psi";
          const std::string stored_energy     = "\\phi";
          const std::string energy_functional = "\\Psi";
          const std::string residual          = "\\text{R}";

          return ContinuumMechanics(free_energy,
                                    stored_energy,
                                    energy_functional,
                                    residual);
        }
    }

  } // namespace Decorations


  SymbolicNames::SymbolicNames(
    const Decorations::Discretization        discretization,
    const Decorations::Geometry              geometry,
    const Decorations::DifferentialGeometry  differential_geometry,
    const Decorations::DifferentialOperators differential_operators,
    const Decorations::ContinuumMechanics    continuum_mechanics)
    : discretization(discretization)
    , geometry(geometry)
    , differential_geometry(differential_geometry)
    , differential_operators(differential_operators)
    , continuum_mechanics(continuum_mechanics)
  {}



  SymbolicNamesAscii::SymbolicNamesAscii(
    const Decorations::Discretization        discretization,
    const Decorations::Geometry              geometry,
    const Decorations::DifferentialGeometry  differential_geometry,
    const Decorations::DifferentialOperators differential_operators,
    const Decorations::ContinuumMechanics    continuum_mechanics)
    : SymbolicNames(discretization,
                    geometry,
                    differential_geometry,
                    differential_operators,
                    continuum_mechanics)
  {}



  SymbolicNamesLaTeX::SymbolicNamesLaTeX(
    const Decorations::Discretization        discretization,
    const Decorations::Geometry              geometry,
    const Decorations::DifferentialGeometry  differential_geometry,
    const Decorations::DifferentialOperators differential_operators,
    const Decorations::ContinuumMechanics    continuum_mechanics)
    : SymbolicNames(discretization,
                    geometry,
                    differential_geometry,
                    differential_operators,
                    continuum_mechanics)
  {}



  FormattingLaTeX::FormattingLaTeX(
    const FormattingLaTeX::IntegralFormat &integral_format)
    : integral_format(integral_format)
  {}



  SymbolicDecorations::SymbolicDecorations(
    const SymbolicNamesAscii &naming_ascii,
    const SymbolicNamesLaTeX &naming_latex,
    const FormattingLaTeX &   formatting_latex)
    : naming_ascii(naming_ascii)
    , naming_latex(naming_latex)
    , formatting_latex(formatting_latex)
  {}

} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE
