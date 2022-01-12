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


// Check that binary operators work (vectorised)
// - Functors

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <weak_forms/binary_operators.h>
#include <weak_forms/functors.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/types.h>

#include "../weak_forms_tests.h"


template <int dim, int spacedim = dim, typename NumberType = double>
void
run()
{
  LogStream::Prefix prefix("Dim " + Utilities::to_string(dim));
  std::cout << "Dim: " << dim << std::endl;

  const FE_Q<dim, spacedim> fe(1);
  const QGauss<spacedim>    qf_cell(fe.degree + 1);

  Triangulation<dim, spacedim> triangulation;
  GridGenerator::hyper_cube(triangulation);

  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const UpdateFlags       update_flags = update_quadrature_points;
  FEValues<dim, spacedim> fe_values(fe, qf_cell, update_flags);
  fe_values.reinit(dof_handler.begin_active());

  constexpr std::size_t width =
    dealii::internal::VectorizedArrayWidthSpecifier<double>::max_width;
  const WeakForms::types::vectorized_qp_range_t q_point_range(0, width);

  {
    const std::string title = "Scalar functor";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const ScalarFunctor c1("c1", "c1");
    const auto          f1 = c1.template value<double, dim, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return 1.0; });

    const ScalarFunctor c2("c2", "c2");
    const auto          f2 = c2.template value<double, dim, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return 2.0; });

    const ScalarFunctor c3("c3", "c3");
    const auto          f3 = c3.template value<double, dim, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return 3.0; });

    const ScalarFunctor c4("c4", "c4");
    const auto          f4 = c4.template value<double, dim, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return 4.0; });

    std::cout << "Addition 1: "
              << ((f1 + f2).template operator()<NumberType, width>(
                   fe_values, q_point_range))
              << std::endl;
    std::cout << "Addition 2: "
              << ((f1 + (f2 + f3))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Addition 3: "
              << (((f1 + f2) + f3)
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Addition 4: "
              << (((f1 + f2) + (f3 + f4))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Subtraction 1: "
              << ((f1 - f2).template operator()<NumberType, width>(
                   fe_values, q_point_range))
              << std::endl;
    std::cout << "Subtraction 2: "
              << ((f1 - (f2 - f3))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Subtraction 3: "
              << (((f1 - f2) - f3)
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Subtraction 4: "
              << (((f1 - f2) - (f3 - f4))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Multiplication 1: "
              << ((f1 * f2).template operator()<NumberType, width>(
                   fe_values, q_point_range))
              << std::endl;
    std::cout << "Multiplication 2: "
              << ((f1 * (f2 * f3))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Multiplication 3: "
              << (((f1 * f2) * f3)
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Multiplication 4: "
              << (((f1 * f2) * (f3 * f4))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    const std::string title = "Tensor functor";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const TensorFunctor<2, spacedim> t1("C1", "C1");
    const auto                       tf1 = t1.template value<double, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return Tensor<2, dim, double>(unit_symmetric_tensor<spacedim>()); });

    const TensorFunctor<2, spacedim> t2("C1", "C1");
    const auto                       tf2 = t2.template value<double, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int) {
        return Tensor<2, dim, double>(2.0 * unit_symmetric_tensor<spacedim>());
      });

    const TensorFunctor<2, spacedim> t3("C3", "C3");
    const auto                       tf3 = t3.template value<double, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int) {
        return Tensor<2, dim, double>(3.0 * unit_symmetric_tensor<spacedim>());
      });

    const TensorFunctor<2, spacedim> t4("C4", "C4");
    const auto                       tf4 = t4.template value<double, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int) {
        return Tensor<2, dim, double>(4.0 * unit_symmetric_tensor<spacedim>());
      });

    const ScalarFunctor c1("c1", "c1");
    const auto          f1 = c1.template value<double, dim, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return 1.0; });

    const ScalarFunctor c2("c2", "c2");
    const auto          f2 = c2.template value<double, dim, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return 2.0; });

    std::cout << "Addition 1: "
              << ((tf1 + tf2).template operator()<NumberType, width>(
                   fe_values, q_point_range))
              << std::endl;
    std::cout << "Addition 2: "
              << ((tf1 + (tf2 + tf3))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Addition 3: "
              << (((tf1 + tf2) + tf3)
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Addition 4: "
              << (((tf1 + tf2) + (tf3 + tf4))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Addition 5: "
              << (((f1 * tf1) + (tf2 * f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Subtraction 1: "
              << ((tf1 - tf2).template operator()<NumberType, width>(
                   fe_values, q_point_range))
              << std::endl;
    std::cout << "Subtraction 2: "
              << ((tf1 - (tf2 - tf3))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Subtraction 3: "
              << (((tf1 - tf2) - tf3)
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Subtraction 4: "
              << (((tf1 - tf2) - (tf3 - tf4))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Subtraction 5: "
              << (((f1 * tf1) - (tf2 * f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Single contraction 1: "
              << ((tf1 * tf2).template operator()<NumberType, width>(
                   fe_values, q_point_range))
              << std::endl;
    std::cout << "Single contraction 2: "
              << ((tf1 * (tf2 * tf3))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Single contraction 3: "
              << (((tf1 * tf2) * tf3)
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Single contraction 4: "
              << (((tf1 * tf2) * (tf3 * tf4))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Single contraction 5: "
              << (((f1 * tf1) * (tf2 * f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    const std::string title = "Scalar function functor";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const ScalarFunctionFunctor<spacedim> c1("c1", "c1");
    const Functions::ConstantFunction<spacedim, double>
               constant_scalar_function_1(1.0);
    const auto f1 = c1.template value<double, dim>(constant_scalar_function_1);

    const ScalarFunctionFunctor<spacedim> c2("c2", "c2");
    const Functions::ConstantFunction<spacedim, double>
               constant_scalar_function_2(2.0);
    const auto f2 = c2.template value<double, dim>(constant_scalar_function_2);

    const ScalarFunctionFunctor<spacedim> c3("c3", "c3");
    const Functions::ConstantFunction<spacedim, double>
               constant_scalar_function_3(3.0);
    const auto f3 = c3.template value<double, dim>(constant_scalar_function_3);

    const ScalarFunctionFunctor<spacedim> c4("c4", "c4");
    const Functions::ConstantFunction<spacedim, double>
               constant_scalar_function_4(4.0);
    const auto f4 = c4.template value<double, dim>(constant_scalar_function_4);

    std::cout << "Addition 1: "
              << ((f1 + f2).template operator()<NumberType, width>(
                   fe_values, q_point_range))
              << std::endl;
    std::cout << "Addition 2: "
              << ((f1 + (f2 + f3))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Addition 3: "
              << (((f1 + f2) + f3)
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Addition 4: "
              << (((f1 + f2) + (f3 + f4))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Subtraction 1: "
              << ((f1 - f2).template operator()<NumberType, width>(
                   fe_values, q_point_range))
              << std::endl;
    std::cout << "Subtraction 2: "
              << ((f1 - (f2 - f3))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Subtraction 3: "
              << (((f1 - f2) - f3)
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Subtraction 4: "
              << (((f1 - f2) - (f3 - f4))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Multiplication 1: "
              << ((f1 * f2).template operator()<NumberType, width>(
                   fe_values, q_point_range))
              << std::endl;
    std::cout << "Multiplication 2: "
              << ((f1 * (f2 * f3))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Multiplication 3: "
              << (((f1 * f2) * f3)
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Multiplication 4: "
              << (((f1 * f2) * (f3 * f4))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    const std::string title = "Tensor function functor";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const TensorFunctionFunctor<2, spacedim>     t1("C1", "C1");
    const ConstantTensorFunction<2, dim, double> constant_tensor_function_1(
      unit_symmetric_tensor<dim>());
    const auto tf1 = t1.template value<double, dim>(constant_tensor_function_1);

    const TensorFunctionFunctor<2, spacedim>     t2("C1", "C1");
    const ConstantTensorFunction<2, dim, double> constant_tensor_function_2(
      2.0 * unit_symmetric_tensor<dim>());
    const auto tf2 = t2.template value<double, dim>(constant_tensor_function_2);

    const TensorFunctionFunctor<2, spacedim>     t3("C3", "C3");
    const ConstantTensorFunction<2, dim, double> constant_tensor_function_3(
      3.0 * unit_symmetric_tensor<dim>());
    const auto tf3 = t3.template value<double, dim>(constant_tensor_function_3);

    const TensorFunctionFunctor<2, spacedim>     t4("C4", "C4");
    const ConstantTensorFunction<2, dim, double> constant_tensor_function_4(
      4.0 * unit_symmetric_tensor<dim>());
    const auto tf4 = t4.template value<double, dim>(constant_tensor_function_4);

    const ScalarFunctor c1("c1", "c1");
    const auto          f1 = c1.template value<double, dim, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return 1.0; });

    const ScalarFunctor c2("c2", "c2");
    const auto          f2 = c2.template value<double, dim, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return 2.0; });

    std::cout << "Addition 1: "
              << ((tf1 + tf2).template operator()<NumberType, width>(
                   fe_values, q_point_range))
              << std::endl;
    std::cout << "Addition 2: "
              << ((tf1 + (tf2 + tf3))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Addition 3: "
              << (((tf1 + tf2) + tf3)
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Addition 4: "
              << (((tf1 + tf2) + (tf3 + tf4))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Addition 5: "
              << (((f1 * tf1) + (tf2 * f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Subtraction 1: "
              << ((tf1 - tf2).template operator()<NumberType, width>(
                   fe_values, q_point_range))
              << std::endl;
    std::cout << "Subtraction 2: "
              << ((tf1 - (tf2 - tf3))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Subtraction 3: "
              << (((tf1 - tf2) - tf3)
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Subtraction 4: "
              << (((tf1 - tf2) - (tf3 - tf4))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Subtraction 5: "
              << (((f1 * tf1) - (tf2 * f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Single contraction 1: "
              << ((tf1 * tf2).template operator()<NumberType, width>(
                   fe_values, q_point_range))
              << std::endl;
    std::cout << "Single contraction 2: "
              << ((tf1 * (tf2 * tf3))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Single contraction 3: "
              << (((tf1 * tf2) * tf3)
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Single contraction 4: "
              << (((tf1 * tf2) * (tf3 * tf4))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
    std::cout << "Single contraction 5: "
              << (((f1 * tf1) * (tf2 * f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
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
