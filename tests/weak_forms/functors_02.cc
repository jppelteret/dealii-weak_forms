// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


// Check that cache functors work

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/vector_tools.h>

#include <weak_forms/cache_functors.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  deallog << "Dim: " << dim << std::endl;

  using namespace WeakForms;

  // Customise the naming convensions, if we wish to.
  const SymbolicDecorations decorator;

  const ScalarCacheFunctor                  scalar("s", "s");
  const VectorCacheFunctor<dim>             vector("v", "v");
  const TensorCacheFunctor<2, dim>          tensor2("T2", "T");
  const TensorCacheFunctor<3, dim>          tensor3("T3", "P");
  const TensorCacheFunctor<4, dim>          tensor4("T4", "K");
  const SymmetricTensorCacheFunctor<2, dim> symm_tensor2("S2", "T");
  const SymmetricTensorCacheFunctor<4, dim> symm_tensor4("S4", "K");


  const FE_Q<dim>                  fe_cell(3);
  const QGauss<dim>                qf_cell(3);
  const FEValuesExtractors::Scalar extractor(0);

  Triangulation<dim, spacedim> triangulation;
  GridGenerator::hyper_cube(triangulation);

  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe_cell);

  Vector<double> solution(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler,
                           Functions::CosineFunction<spacedim>(
                             fe_cell.n_components()),
                           solution);

  const UpdateFlags update_flags =
    update_values | update_gradients | update_hessians | update_3rd_derivatives;
  MeshWorker::ScratchData<dim, spacedim> scratch_data(fe_cell,
                                                      qf_cell,
                                                      update_flags);

  const auto                         cell      = dof_handler.begin_active();
  const FEValuesBase<dim, spacedim> &fe_values = scratch_data.reinit(cell);
  (void)fe_values;

  const WeakForms::SolutionStorage<Vector<double>> solution_storage(solution);
  solution_storage.extract_local_dof_values(scratch_data);
  const std::vector<std::string> &solution_names =
    solution_storage.get_solution_names();


  const auto s_func =
    [&extractor](MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &        solution_names,
                 const unsigned int                      q_point) {
      return scratch_data.get_values(solution_names[0], extractor)[q_point];
    };
  const auto s = value<NumberType, dim, spacedim>(scalar, s_func, update_flags);

  const auto v_func =
    [&extractor](MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &        solution_names,
                 const unsigned int                      q_point) {
      return scratch_data.get_gradients(solution_names[0], extractor)[q_point];
    };
  const auto v = value<NumberType, dim>(vector, v_func, update_flags);

  const auto T2_func =
    [&extractor](MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &        solution_names,
                 const unsigned int                      q_point) {
      return scratch_data.get_hessians(solution_names[0], extractor)[q_point];
    };
  const auto T2 = value<NumberType, dim>(tensor2, T2_func, update_flags);

  const auto T3_func =
    [&extractor](MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &        solution_names,
                 const unsigned int                      q_point) {
      return scratch_data.get_third_derivatives(solution_names[0],
                                                extractor)[q_point];
    };
  const auto T3 = value<NumberType, dim>(tensor3, T3_func, update_flags);

  const auto T4_func =
    [&extractor](MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &        solution_names,
                 const unsigned int                      q_point) {
      return outer_product(
        scratch_data.get_gradients(solution_names[0], extractor)[q_point],
        scratch_data.get_third_derivatives(solution_names[0],
                                           extractor)[q_point]);
    };
  const auto T4 = value<NumberType, dim>(tensor4, T4_func, update_flags);

  const auto S2_func =
    [&extractor](MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &        solution_names,
                 const unsigned int                      q_point) {
      return symmetrize(
        scratch_data.get_hessians(solution_names[0], extractor)[q_point]);
    };
  const auto S2 = value<NumberType, dim>(tensor2, S2_func, update_flags);

  const auto S4_func =
    [&extractor](MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &        solution_names,
                 const unsigned int                      q_point) {
      const auto S2 = symmetrize(
        scratch_data.get_hessians(solution_names[0], extractor)[q_point]);
      return outer_product(S2, S2);
    };
  const auto S4 = value<NumberType, dim>(tensor4, S4_func, update_flags);

  const unsigned int q_point = 0;


  // Test values
  {
    deallog << "Scalar: "
            << s.template operator()<NumberType>(scratch_data,
                                                 solution_names)[q_point]
            << std::endl;
    deallog << "Vector: "
            << v.template operator()<NumberType>(scratch_data,
                                                 solution_names)[q_point]
            << std::endl;
    deallog << "Tensor (rank 2): "
            << T2.template operator()<NumberType>(scratch_data,
                                                  solution_names)[q_point]
            << std::endl;
    deallog << "Tensor (rank 3): "
            << T3.template operator()<NumberType>(scratch_data,
                                                  solution_names)[q_point]
            << std::endl;
    deallog << "Tensor (rank 4): "
            << T4.template operator()<NumberType>(scratch_data,
                                                  solution_names)[q_point]
            << std::endl;
    deallog << "SymmetricTensor (rank 2): "
            << S2.template operator()<NumberType>(scratch_data,
                                                  solution_names)[q_point]
            << std::endl;
    deallog << "SymmetricTensor (rank 4): "
            << S4.template operator()<NumberType>(scratch_data,
                                                  solution_names)[q_point]
            << std::endl;
  }

  deallog << "OK" << std::endl << std::endl;
}


int
main()
{
  initlog();

  run<2>();
  run<3>();

  deallog << "OK" << std::endl;
}
