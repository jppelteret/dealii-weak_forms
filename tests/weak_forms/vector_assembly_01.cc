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


// Check assembly of a vector over an entire triangulation
// - Volume and boundary vector contributions (scalar-valued finite element)
// - Check source terms, boundary terms

#include <deal.II/base/function_lib.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function_parser.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/assembler_matrix_based.h>
#include <weak_forms/binary_operators.h>
#include <weak_forms/cell_face_subface_operators.h>
#include <weak_forms/functors.h>
#include <weak_forms/linear_forms.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/spaces.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>

#include "../weak_forms_tests.h"
#include "wf_common_tests/utilities.h"


template <int dim, int spacedim = dim>
void
run(const unsigned int n_subdivisions)
{
  LogStream::Prefix prefix("Dim " + Utilities::to_string(dim));
  std::cout << "Dim: " << dim << std::endl;

  const FE_Q<dim, spacedim>  fe(1);
  const QGauss<spacedim>     qf_cell(fe.degree + 1);
  const QGauss<spacedim - 1> qf_face(fe.degree + 1);

  Triangulation<dim, spacedim> triangulation;
  GridGenerator::subdivided_hyper_cube(triangulation, n_subdivisions, 0.0, 1.0);

  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  Vector<double> system_rhs_std;
  Vector<double> system_rhs_wf;

  std::map<std::string, double> constants;
  constants["pi"]                  = numbers::PI;
  const std::string variable_names = (dim == 2 ? "x, y" : "x, y, z");
  const std::string expression_source =
    (dim == 2 ? "1 - x^2 - y^2" : "1 - x^2 - y^2 - z^2");
  const std::string expression_traction =
    (dim == 2 ? "sin(pi/2*x); sin(pi/2*y)" :
                "sin(pi/2*x); sin(pi/2*y); sin(pi/2*z)");

  FunctionParser<spacedim> source_function;
  source_function.initialize(variable_names, expression_source, constants);
  TensorFunctionParser<1, spacedim> traction_function;
  traction_function.initialize(variable_names, expression_traction, constants);

  const UpdateFlags update_flags_cell =
    update_values | update_quadrature_points | update_JxW_values;
  const UpdateFlags update_flags_face =
    update_values | update_quadrature_points | update_normal_vectors |
    update_JxW_values;

  {
    system_rhs_std.reinit(dof_handler.n_dofs());
    system_rhs_wf.reinit(dof_handler.n_dofs());
  }

  auto verify_assembly = [](const Vector<double> &system_rhs_std,
                            const Vector<double> &system_rhs_wf)
  {
    constexpr double tol = 1e-12;

    Assert(system_rhs_std.size() == system_rhs_wf.size(),
           ExcDimensionMismatch(system_rhs_std.size(), system_rhs_wf.size()));
    for (unsigned int r = 0; r < system_rhs_std.size(); ++r)
      {
        AssertThrow(std::abs(system_rhs_std[r] - system_rhs_wf[r]) < tol,
                    ExcVectorEntriesNotEqual(r,
                                             system_rhs_std[r],
                                             system_rhs_wf[r]));
      }
  };

  // Blessed vector
  {
    std::cout << "Standard assembly" << std::endl;
    system_rhs_std = 0;

    FEValues<dim, spacedim>     fe_values(fe, qf_cell, update_flags_cell);
    FEFaceValues<dim, spacedim> fe_face_values(fe, qf_face, update_flags_face);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);

    for (auto &cell : dof_handler.active_cell_iterators())
      {
        cell_rhs = 0;
        fe_values.reinit(cell);

        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            const double s_q =
              source_function.value(fe_values.quadrature_point(q));
            for (const unsigned int i : fe_values.dof_indices())
              {
                cell_rhs(i) +=
                  fe_values.shape_value(i, q) * s_q * fe_values.JxW(q);
              }
          }

        for (auto face : GeometryInfo<dim>::face_indices())
          if (cell->face(face)->at_boundary())
            {
              fe_face_values.reinit(cell, face);

              for (const unsigned int q :
                   fe_face_values.quadrature_point_indices())
                {
                  const Tensor<1, dim> t_q =
                    traction_function.value(fe_face_values.quadrature_point(q));

                  for (const unsigned int i : fe_values.dof_indices())
                    {
                      cell_rhs(i) += fe_face_values.shape_value(i, q) *
                                     (fe_face_values.normal_vector(q) * t_q) *
                                     fe_face_values.JxW(q);
                    }
                }
            }


        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_rhs,
                                               local_dof_indices,
                                               system_rhs_std);
      }

    // system_rhs_std.print(std::cout);
  }

  // Expanded form of blessed matrix
  {
    using namespace WeakForms;

    std::cout << "Exemplar weak form assembly" << std::endl;
    system_rhs_wf = 0;

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);

    MeshWorker::ScratchData<dim, spacedim> scratch_data(
      fe, qf_cell, update_flags_cell, qf_face, update_flags_face);

    for (auto &cell : dof_handler.active_cell_iterators())
      {
        cell_rhs = 0;
        const FEValuesBase<dim, spacedim> &fe_values =
          scratch_data.reinit(cell);

        // Cell contributions
        {
          const std::vector<double> &      JxW = fe_values.get_JxW_values();
          std::vector<std::vector<double>> Nx(fe_values.dofs_per_cell,
                                              std::vector<double>(
                                                fe_values.n_quadrature_points));
          std::vector<double>              s(fe_values.n_quadrature_points);

          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int q : fe_values.quadrature_point_indices())
              Nx[i][q] = fe_values.shape_value(i, q);
          for (const unsigned int q : fe_values.quadrature_point_indices())
            s[q] = source_function.value(fe_values.quadrature_point(q));

          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int q : fe_values.quadrature_point_indices())
              {
                cell_rhs(i) += Nx[i][q] * s[q] * JxW[q];
              }
        }

        // Face contributions
        for (auto face : GeometryInfo<dim>::face_indices())
          if (cell->face(face)->at_boundary())
            {
              const FEValuesBase<dim, spacedim> &fe_face_values =
                scratch_data.reinit(cell, face);

              const std::vector<double> &JxW = fe_face_values.get_JxW_values();
              std::vector<std::vector<double>> Nx(
                fe_values.dofs_per_cell,
                std::vector<double>(fe_face_values.n_quadrature_points));
              std::vector<Tensor<1, dim, double>> t(
                fe_face_values.n_quadrature_points);
              const std::vector<Tensor<1, spacedim>> &N =
                fe_face_values.get_normal_vectors();

              for (const unsigned int i : fe_values.dof_indices())
                for (const unsigned int q :
                     fe_face_values.quadrature_point_indices())
                  Nx[i][q] = fe_face_values.shape_value(i, q);

              for (const unsigned int q :
                   fe_face_values.quadrature_point_indices())
                t[q] =
                  traction_function.value(fe_face_values.quadrature_point(q));

              for (const unsigned int i : fe_values.dof_indices())
                for (const unsigned int q :
                     fe_face_values.quadrature_point_indices())
                  {
                    cell_rhs(i) += Nx[i][q] * (N[q] * t[q]) * JxW[q];
                  }
            }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_rhs,
                                               local_dof_indices,
                                               system_rhs_wf);
      }

    // system_rhs_wf.print(std::cout);
    verify_assembly(system_rhs_std, system_rhs_wf);
  }

  {
    using namespace WeakForms;

    const std::string test_name = "Weak form assembly (bilinear form)";
    std::cout << test_name << std::endl;
    deallog << test_name << std::endl;
    system_rhs_wf = 0;

    // Symbolic types for test function, trial solution and a coefficient.
    const TestFunction<dim, spacedim> test;
    const Normal<spacedim>            normal{};

    const ScalarFunctionFunctor<dim> source("f_pillow", "f_{s}");
    const VectorFunctionFunctor<dim> traction("f_cosine", "\\mathbf{f}_{t}");

    const auto test_val      = test.value();
    const auto normal_val    = normal.value();
    const auto src_func      = source.value(source_function);
    const auto traction_func = traction.value(traction_function);

    // Non-vectorized assembler
    {
      // Still no concrete definitions
      // NB: Linear forms change sign when RHS is assembled.
      constexpr bool use_vectorization = false;
      MatrixBasedAssembler<dim, spacedim, double, use_vectorization> assembler;
      assembler -= linear_form(test_val, src_func).dV();
      assembler -= linear_form(test_val, normal_val * traction_func).dA();

      // Look at what we're going to compute
      const SymbolicDecorations decorator;
      deallog << "Weak form (ascii):\n"
              << assembler.as_ascii(decorator) << std::endl;
      deallog << "Weak form (LaTeX):\n"
              << assembler.as_latex(decorator) << std::endl;

      // Now we pass in concrete objects to get data from
      // and assemble into.
      assembler.assemble_rhs_vector(
        system_rhs_wf, constraints, dof_handler, qf_cell, qf_face);

      // system_rhs_wf.print(std::cout);
      verify_assembly(system_rhs_std, system_rhs_wf);
    }

    system_rhs_wf = 0;

    // Vectorized assembler
    {
      constexpr bool use_vectorization = true;
      MatrixBasedAssembler<dim, spacedim, double, use_vectorization> assembler;
      assembler -= linear_form(test_val, src_func).dV();
      assembler -= linear_form(test_val, normal_val * traction_func).dA();

      // Now we pass in concrete objects to get data from
      // and assemble into.
      assembler.assemble_rhs_vector(
        system_rhs_wf, constraints, dof_handler, qf_cell, qf_face);

      // system_rhs_wf.print(std::cout);
      verify_assembly(system_rhs_std, system_rhs_wf);
    }
  }

  deallog << "OK" << std::endl;
}


int
main(int argc, char *argv[])
{
  initlog();
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  // With one subdivision, we test face integration with all
  // faces contributing to the local vector.
  deallog.push("Divisions = 1");
  {
    const unsigned int n_subdivisions = 1;
    run<2>(n_subdivisions);
    run<3>(n_subdivisions);
  }
  deallog.pop();

  deallog.push("Divisions = 4");
  {
    const unsigned int n_subdivisions = 4;
    run<2>(n_subdivisions);
    run<3>(n_subdivisions);
  }
  deallog.pop();

  deallog << "OK" << std::endl;
}
