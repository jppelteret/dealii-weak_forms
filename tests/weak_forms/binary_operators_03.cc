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


// Check that binary operators work
// - Cell face subface operators

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/vector_tools.h>

#include <weak_forms/binary_operators.h>
#include <weak_forms/cell_face_subface_operators.h>
#include <weak_forms/functors.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/symbolic_operators.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  LogStream::Prefix prefix("Dim " + Utilities::to_string(dim));
  std::cout << "Dim: " << dim << std::endl;

  const FE_Q<dim, spacedim>  fe(1);
  const QGauss<spacedim>     qf_cell(fe.degree + 1);
  const QGauss<spacedim - 1> qf_face(fe.degree + 1);

  Triangulation<dim, spacedim> triangulation;
  GridGenerator::hyper_cube(triangulation);

  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  Vector<double> solution(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler,
                           Functions::CosineFunction<spacedim>(
                             fe.n_components()),
                           solution);

  const UpdateFlags update_flags_cell =
    update_quadrature_points | update_values | update_gradients |
    update_hessians | update_3rd_derivatives;
  const UpdateFlags           update_flags_face = update_normal_vectors;
  FEValues<dim, spacedim>     fe_values(fe, qf_cell, update_flags_cell);
  FEFaceValues<dim, spacedim> fe_face_values(fe, qf_face, update_flags_face);
  fe_values.reinit(dof_handler.begin_active());
  fe_face_values.reinit(dof_handler.begin_active(), 0);

  const unsigned int q_point = 0;

  {
    const std::string title = "Normal";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const ScalarFunctor c1("c1", "c1");
    const auto          f1 = c1.template value<double, dim, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return 2.0; });

    const VectorFunctor<dim> v1("v1", "v1");
    const auto               f2 = v1.template value<double, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      {
        Tensor<1, dim> t;
        for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
          *it = 2.0;
        return t;
      });

    const TensorFunctor<2, dim> T1("T1", "T1");
    const auto                  f3 = T1.template value<double, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      {
        Tensor<2, dim> t;
        for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
          *it = 2.0;
        return t;
      });

    const SymmetricTensorFunctor<2, dim> S1("S1", "S1");
    const auto f4 = S1.template value<double, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      {
        SymmetricTensor<2, dim> t;
        for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
          *it = 2.0;
        return t;
      });

    const Normal<spacedim> normal{};
    const auto             functor = normal.value();

    std::cout << "Scalar * normal: "
              << ((f1 * functor)
                    .template operator()<NumberType>(fe_face_values))[q_point]
              << std::endl;
    std::cout << "Vector * normal: "
              << ((f2 * functor)
                    .template operator()<NumberType>(fe_face_values))[q_point]
              << std::endl;
    std::cout << "Tensor * normal: "
              << ((f3 * functor)
                    .template operator()<NumberType>(fe_face_values))[q_point]
              << std::endl;
    std::cout << "SymmetricTensor * normal: "
              << ((f4 * functor)
                    .template operator()<NumberType>(fe_face_values))[q_point]
              << std::endl;

    std::cout << "Vector + normal: "
              << ((f2 + functor)
                    .template operator()<NumberType>(fe_face_values))[q_point]
              << std::endl;
    std::cout << "Vector - normal: "
              << ((f2 - functor)
                    .template operator()<NumberType>(fe_face_values))[q_point]
              << std::endl;

    deallog << "OK" << std::endl;
  }

  deallog << "OK" << std::endl;
}


int
main(int argc, char *argv[])
{
  initlog();
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  run<2>();
  run<3>();

  deallog << "OK" << std::endl;
}
