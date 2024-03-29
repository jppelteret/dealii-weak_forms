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


// Check that unary operators work (vectorised)

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/numerics/vector_tools.h>

#include <weak_forms/functors.h>
#include <weak_forms/mixed_operators.h>
#include <weak_forms/solution_storage.h>
#include <weak_forms/spaces.h>
#include <weak_forms/subspace_extractors.h>
#include <weak_forms/subspace_views.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/types.h>
#include <weak_forms/unary_operators.h>

#include "../weak_forms_tests.h"


// Work around the issue that the cross product is not available
// in 2d.
template <int spacedim>
struct CrossProductCheck;

template <>
struct CrossProductCheck<2>
{
  template <typename NumberType, std::size_t width, typename... Args>
  static void
  run(const Args &...)
  {}
};

template <>
struct CrossProductCheck<3>
{
  template <typename NumberType,
            std::size_t width,
            typename F1,
            typename F2,
            typename FEValuesType,
            typename QPointRangeType>
  static void
  run(const F1 &             f1,
      const F2 &             f2,
      const FEValuesType &   fe_values,
      const QPointRangeType &q_point_range)
  {
    std::cout << "Vector cross product: "
              << ((cross_product(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;
  }
};


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
  MeshWorker::ScratchData<dim, spacedim> scratch_data(fe,
                                                      qf_cell,
                                                      update_flags_cell);

  const auto                         cell      = dof_handler.begin_active();
  const FEValuesBase<dim, spacedim> &fe_values = scratch_data.reinit(cell);
  const unsigned int                 q_point   = 0;

  const WeakForms::SolutionStorage<Vector<double>> solution_storage(solution);
  solution_storage.extract_local_dof_values(scratch_data, dof_handler);
  const std::vector<WeakForms::SolutionExtractionData<dim, spacedim>>
    &solution_extraction_data =
      solution_storage.get_solution_extraction_data(scratch_data, dof_handler);

  constexpr std::size_t width =
    dealii::internal::VectorizedArrayWidthSpecifier<double>::max_width;
  const WeakForms::types::vectorized_qp_range_t q_point_range(0, width);

  {
    const std::string title = "Scalar";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const ScalarFunctor c1("c1", "c1");
    const auto          f1 = c1.template value<double, dim, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return 2.0; });

    const ScalarFunctor c2("c2", "c2");
    const auto          f2 = c2.template value<double, dim, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return 3.0; });

    std::cout << "Power: "
              << ((pow(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Maximum: "
              << ((max(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Minimum: "
              << ((min(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    const std::string title = "Vector";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const VectorFunctor<dim> v1("v1", "v1");
    const auto               f1 = v1.template value<double, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      {
        Tensor<1, dim> t;
        unsigned int   i = 0;
        for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
          *it = 2.0 + (i++);
        return t;
      });

    const VectorFunctor<dim> v2("v2", "v2");
    const auto               f2 = v2.template value<double, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      {
        Tensor<1, dim> t;
        unsigned int   i = 0;
        for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
          *it = 3.0 + (i++);
        return t;
      });

    CrossProductCheck<spacedim>::template run<NumberType, width>(f1,
                                                                 f2,
                                                                 fe_values,
                                                                 q_point_range);

    std::cout << "Outer product: "
              << ((outer_product(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "VSchur product: "
              << ((schur_product(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Scalar product: "
              << ((scalar_product(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Contract: "
              << ((contract<0, 0>(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    const std::string title = "Tensor";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const TensorFunctor<2, dim> T1("T1", "T1");
    const auto                  f1 = T1.template value<double, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      {
        Tensor<2, dim> t;
        for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
          *it = 2.0;
        for (unsigned int i = 0; i < dim; ++i)
          t[i][i] += 1.0;
        return t;
      });

    const TensorFunctor<2, dim> T2("T2", "T2");
    const auto                  f2 = T2.template value<double, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      {
        Tensor<2, dim> t;
        for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
          *it = 3.0;
        for (unsigned int i = 0; i < dim; ++i)
          t[i][i] += 1.0;
        return t;
      });

    std::cout << "Outer product: "
              << ((outer_product(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Schur product: "
              << ((schur_product(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Scalar product: "
              << ((scalar_product(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Contract: "
              << ((contract<1, 0>(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Doubel contract: "
              << ((double_contract<1, 0, 0, 1>(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    const std::string title = "SymmetricTensor";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const SymmetricTensorFunctor<2, dim> S1("S1", "S1");
    const auto f1 = S1.template value<double, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      {
        SymmetricTensor<2, dim> t;
        for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
          *it = 2.0;
        for (unsigned int i = 0; i < dim; ++i)
          t[i][i] += 1.0;
        return t;
      });

    const SymmetricTensorFunctor<2, dim> S2("S2", "S2");
    const auto f2 = S2.template value<double, spacedim>(
      [](const FEValuesBase<dim, spacedim> &, const unsigned int)
      {
        SymmetricTensor<2, dim> t;
        for (auto it = t.begin_raw(); it != t.end_raw(); ++it)
          *it = 3.0;
        for (unsigned int i = 0; i < dim; ++i)
          t[i][i] += 1.0;
        return t;
      });

    std::cout << "Outer product: "
              << ((outer_product(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    // std::cout << "Schur product: "
    //           << ((schur_product(f1,f2)).template operator()<NumberType,
    //           width>(fe_values,
    //                                                            q_point_range))
    //           << std::endl;

    std::cout << "Scalar product: "
              << ((scalar_product(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    std::cout << "Doubel contract: "
              << ((double_contract(f1, f2))
                    .template operator()<NumberType, width>(fe_values,
                                                            q_point_range))
              << std::endl;

    deallog << "OK" << std::endl;
  }

  {
    const std::string title = "Field solution";
    std::cout << title << std::endl;
    deallog << title << std::endl;

    using namespace WeakForms;

    const FieldSolution<dim, spacedim> field_solution;
    const SubSpaceExtractors::Scalar   subspace_extractor(0, "s", "s");
    const auto field_solution_ss = field_solution[subspace_extractor];

    const auto value            = field_solution_ss.value();
    const auto gradient         = field_solution_ss.gradient();
    const auto laplacian        = field_solution_ss.laplacian();
    const auto hessian          = field_solution_ss.hessian();
    const auto third_derivative = field_solution_ss.third_derivative();

    // std::cout << "value negation: "
    //           << ((-value).template operator()<NumberType, width>(
    //                fe_values, scratch_data, solution_extraction_data,
    //                q_point_range))
    //           << std::endl;

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
