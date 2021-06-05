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

#ifndef dealii_weakforms_assembler_base_h
#define dealii_weakforms_assembler_base_h

#include <deal.II/base/config.h>

#include <weak_forms/config.h>

// #include <deal.II/algorithms/general_data_storage.h>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>

#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/bilinear_forms.h>
#include <weak_forms/binary_integral_operators.h>
#include <weak_forms/binary_operators.h>
#include <weak_forms/linear_forms.h>
#include <weak_forms/solution_storage.h>
#include <weak_forms/symbolic_integral.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/unary_integral_operators.h>
#include <weak_forms/unary_operators.h>

#include <functional>
#include <type_traits>


WEAK_FORMS_NAMESPACE_OPEN


// Forward declarations
namespace WeakForms
{
  // namespace AutoDifferentiation
  // {
  //   template <int                                   dim,
  //             enum Differentiation::NumberTypes ADScalarTypeCode,
  //             typename ScalarType>
  //   class EnergyFunctional;
  // } // namespace AutoDifferentiation

  // namespace SelfLinearization
  // {
  //   template <typename... SymbolicOpsSubSpaceFieldSolution>
  //   class EnergyFunctional;
  // }
} // namespace WeakForms


namespace WeakForms
{
  namespace internal
  {
    enum class AccumulationSign
    {
      plus,
      minus
    };

    // template<typename ReturnType, typename T1, typename T2, typename T =
    // void> struct FullContraction;

    // /**
    //  * Generic contraction
    //  *
    //  * Type T1 is a scalar
    //  */
    // template<typename ReturnType, typename T1, typename T2>
    // struct FullContraction<ReturnType,T1,T2, typename
    // std::enable_if<std::is_arithmetic<T1>::value ||
    // std::is_arithmetic<T2>::value>::type>
    // {
    //   static ReturnType
    //   contract(const T1 &t1, const T2 &t2)
    //   {
    //     return t1*t2;
    //   }
    // };


    // /**
    //  * Generic contraction
    //  *
    //  * Type T2 is a scalar
    //  */
    // template<typename T1, typename T2>
    // struct FullContraction<T1,T2, typename
    // std::enable_if<std::is_arithmetic<T2>::value &&
    // !std::is_arithmetic<T1>::value>::type>
    // {
    //   static ReturnType
    //   contract(const T1 &t1, const T2 &t2)
    //   {
    //     // Call other implementation
    //     return FullContraction<ReturnType,T2,T1>::contract(t2,t1);
    //   }
    // };


    template <typename T1, typename T2, typename T = void>
    struct FullContraction;

    /**
     * Contraction with a scalar or complex scalar
     *
     * At least one of the templated types is an arithmetic type
     */
    template <typename T1, typename T2>
    struct FullContraction<
      T1,
      T2,
      typename std::enable_if<std::is_arithmetic<T1>::value ||
                              std::is_arithmetic<T2>::value>::type>
    {
      static auto
      contract(const T1 &t1, const T2 &t2) -> decltype(t1 * t2)
      {
        return t1 * t2;
      }
    };
    template <typename T1, typename T2>
    struct FullContraction<
      std::complex<T1>,
      T2,
      typename std::enable_if<std::is_arithmetic<T1>::value ||
                              std::is_arithmetic<T2>::value>::type>
    {
      static auto
      contract(const std::complex<T1> &t1, const T2 &t2) -> decltype(t1 * t2)
      {
        return t1 * t2;
      }
    };
    template <typename T1, typename T2>
    struct FullContraction<
      T1,
      std::complex<T2>,
      typename std::enable_if<std::is_arithmetic<T1>::value ||
                              std::is_arithmetic<T2>::value>::type>
    {
      static auto
      contract(const T1 &t1, const std::complex<T2> &t2) -> decltype(t1 * t2)
      {
        return t1 * t2;
      }
    };
    template <typename T1, typename T2>
    struct FullContraction<
      std::complex<T1>,
      std::complex<T2>,
      typename std::enable_if<std::is_arithmetic<T1>::value ||
                              std::is_arithmetic<T2>::value>::type>
    {
      static auto
      contract(const std::complex<T1> &t1, const std::complex<T2> &t2)
        -> decltype(t1 * t2)
      {
        return t1 * t2;
      }
    };

    /**
     * Contraction with a vectorized scalar
     *
     * At least one of the templated types is a VectorizedArray
     */
    template <typename T1, typename T2>
    struct FullContraction<VectorizedArray<T1>, T2>
    {
      static auto
      contract(const VectorizedArray<T1> &t1, const T2 &t2) -> decltype(t1 * t2)
      {
        return t1 * t2;
      }
    };
    template <typename T1, typename T2>
    struct FullContraction<T1, VectorizedArray<T2>>
    {
      static auto
      contract(const T1 &t1, const VectorizedArray<T2> &t2) -> decltype(t1 * t2)
      {
        return t1 * t2;
      }
    };
    template <typename T1, typename T2>
    struct FullContraction<VectorizedArray<T1>, VectorizedArray<T2>>
    {
      static auto
      contract(const VectorizedArray<T1> &t1, const VectorizedArray<T2> &t2)
        -> decltype(t1 * t2)
      {
        return t1 * t2;
      }
    };

    /**
     * Contraction with a tensor
     *
     * Here we recognise that the shape functions can only be
     * scalar valued (dealt with in the above specializations),
     * vector valued (Tensors of rank 1), rank-2 tensor valued or
     * rank-2 symmetric tensor valued. For the rank 1 and rank 2
     * case, we already have full contraction operations that we
     * can leverage.
     */
    template <int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<
      Tensor<rank_1, dim, T1>,
      Tensor<rank_2, dim, T2>,
      typename std::enable_if<(rank_1 == 0 || rank_2 == 0)>::type>
    {
      static Tensor<rank_1 + rank_2, dim, typename ProductType<T1, T2>::type>
      contract(const Tensor<rank_1, dim, T1> &t1,
               const Tensor<rank_2, dim, T2> &t2)
      {
        return t1 * t2;
      }
    };

    template <int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<
      Tensor<rank_1, dim, T1>,
      Tensor<rank_2, dim, T2>,
      typename std::enable_if<((rank_1 == 1 && rank_2 >= 1) ||
                               (rank_2 == 1 && rank_1 >= 1))>::type>
    {
      static Tensor<rank_1 + rank_2 - 2,
                    dim,
                    typename ProductType<T1, T2>::type>
      contract(const Tensor<rank_1, dim, T1> &t1,
               const Tensor<rank_2, dim, T2> &t2)
      {
        return dealii::contract<rank_1 - 1, 0>(t1, t2);
      }
    };

    template <int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<
      Tensor<rank_1, dim, T1>,
      Tensor<rank_2, dim, T2>,
      typename std::enable_if<((rank_1 == 2 && rank_2 >= 2) ||
                               (rank_2 == 2 && rank_1 >= 2))>::type>
    {
      static Tensor<rank_1 + rank_2 - 4,
                    dim,
                    typename ProductType<T1, T2>::type>
      contract(const Tensor<rank_1, dim, T1> &t1,
               const Tensor<rank_2, dim, T2> &t2)
      {
        return dealii::double_contract<rank_1 - 2, 0, rank_1 - 1, 1>(t1, t2);
      }
    };

    template <int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<Tensor<rank_1, dim, T1>,
                           Tensor<rank_2, dim, T2>,
                           typename std::enable_if<(rank_1 > 2 && rank_2 > 2 &&
                                                    rank_1 == rank_2)>::type>
    {
      static typename ProductType<T1, T2>::type
      contract(const Tensor<rank_1, dim, T1> &t1,
               const Tensor<rank_2, dim, T2> &t2)
      {
        return scalar_product(t1, t2);
      }
    };

    template <int dim, typename T1, typename T2>
    struct FullContraction<SymmetricTensor<2, dim, T1>,
                           SymmetricTensor<2, dim, T2>>
    {
      static typename ProductType<T1, T2>::type
      contract(const SymmetricTensor<2, dim, T1> &t1,
               const SymmetricTensor<2, dim, T2> &t2)
      {
        // Always a double contraction
        return t1 * t2;
      }
    };

    template <int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<
      SymmetricTensor<rank_1, dim, T1>,
      SymmetricTensor<rank_2, dim, T2>,
      typename std::enable_if<(rank_1 == 2 && rank_2 > 2) ||
                              (rank_2 == 2 && rank_1 > 2)>::type>
    {
      static SymmetricTensor<rank_1 + rank_2 - 4,
                             dim,
                             typename ProductType<T1, T2>::type>
      contract(const SymmetricTensor<rank_1, dim, T1> &t1,
               const SymmetricTensor<rank_2, dim, T2> &t2)
      {
        // Always a double contraction
        return t1 * t2;
      }
    };

    template <int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<Tensor<rank_1, dim, T1>,
                           SymmetricTensor<rank_2, dim, T2>,
                           typename std::enable_if<(rank_1 == 1)>::type>
    {
      static Tensor<rank_1 + rank_2 - 2,
                    dim,
                    typename ProductType<T1, T2>::type>
      contract(const Tensor<rank_1, dim, T1> &         t1,
               const SymmetricTensor<rank_2, dim, T2> &t2)
      {
        return t1 * t2;
      }
    };

    template <int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<SymmetricTensor<rank_1, dim, T1>,
                           Tensor<rank_2, dim, T2>,
                           typename std::enable_if<(rank_2 == 1)>::type>
    {
      static Tensor<rank_1 + rank_2 - 2,
                    dim,
                    typename ProductType<T1, T2>::type>
      contract(const SymmetricTensor<rank_1, dim, T1> &t1,
               const Tensor<rank_2, dim, T2> &         t2)
      {
        return t1 * t2;
      }
    };

    template <int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<Tensor<rank_1, dim, T1>,
                           SymmetricTensor<rank_2, dim, T2>,
                           typename std::enable_if<(rank_1 > 1)>::type>
    {
      // With mixed tensor types, its easier just to be defensive and not worry
      // about the symmetries of one of the tensors. The main issue comes in
      // when there are mixed ranks for the two arguments. Also, it might be
      // more expensive to do the symmetrization and subsequent contraction, as
      // opposed to this conversion and standard contraction.
      static auto
      contract(const Tensor<rank_1, dim, T1> &         t1,
               const SymmetricTensor<rank_2, dim, T2> &t2)
        -> decltype(
          FullContraction<Tensor<rank_1, dim, T1>, Tensor<rank_2, dim, T2>>::
            contract(Tensor<rank_1, dim, T1>(), Tensor<rank_2, dim, T2>()))
      {
        using Contraction_t =
          FullContraction<Tensor<rank_1, dim, T1>, Tensor<rank_2, dim, T2>>;
        return Contraction_t::contract(t1, Tensor<rank_2, dim, T2>(t2));
      }
    };

    template <int rank_1, int rank_2, int dim, typename T1, typename T2>
    struct FullContraction<SymmetricTensor<rank_1, dim, T1>,
                           Tensor<rank_2, dim, T2>,
                           typename std::enable_if<(rank_2 > 1)>::type>
    {
      static auto
      contract(const SymmetricTensor<rank_1, dim, T1> &t1,
               const Tensor<rank_2, dim, T2> &         t2)
        -> decltype(
          FullContraction<Tensor<rank_1, dim, T1>, Tensor<rank_2, dim, T2>>::
            contract(Tensor<rank_1, dim, T1>(), Tensor<rank_2, dim, T2>()))
      {
        using Contraction_t =
          FullContraction<Tensor<rank_1, dim, T1>, Tensor<rank_2, dim, T2>>;
        return Contraction_t::contract(Tensor<rank_1, dim, T1>(t1), t2);
      }
    };


    // Valid for cell and face assembly
    template <enum AccumulationSign Sign,
              typename ScalarType,
              int dim,
              int spacedim,
              typename ValueTypeTest,
              typename ValueTypeFunctor,
              typename ValueTypeTrial>
    void
    assemble_cell_matrix_contribution(
      FullMatrix<ScalarType> &                        cell_matrix,
      const FEValuesBase<dim, spacedim> &             fe_values_dofs,
      const FEValuesBase<dim, spacedim> &             fe_values_q_points,
      const std::vector<std::vector<ValueTypeTest>> & shapes_test,
      const std::vector<ValueTypeFunctor> &           values_functor,
      const std::vector<std::vector<ValueTypeTrial>> &shapes_trial,
      const std::vector<double> &                     JxW,
      const bool                                      symmetric_contribution)
    {
      Assert(shapes_test.size() == fe_values_dofs.dofs_per_cell,
             ExcDimensionMismatch(shapes_test.size(),
                                  fe_values_dofs.dofs_per_cell));
      Assert(shapes_trial.size() == fe_values_dofs.dofs_per_cell,
             ExcDimensionMismatch(shapes_trial.size(),
                                  fe_values_dofs.dofs_per_cell));
      Assert(values_functor.size() == fe_values_q_points.n_quadrature_points,
             ExcDimensionMismatch(values_functor.size(),
                                  fe_values_q_points.n_quadrature_points));
      Assert(JxW.size() == fe_values_q_points.n_quadrature_points,
             ExcDimensionMismatch(JxW.size(),
                                  fe_values_q_points.n_quadrature_points));
      for (const unsigned int k : fe_values_dofs.dof_indices())
        {
          (void)k;
          Assert(shapes_test[k].size() ==
                   fe_values_q_points.n_quadrature_points,
                 ExcDimensionMismatch(shapes_test[k].size(),
                                      fe_values_q_points.n_quadrature_points));
          Assert(shapes_trial[k].size() ==
                   fe_values_q_points.n_quadrature_points,
                 ExcDimensionMismatch(shapes_trial[k].size(),
                                      fe_values_q_points.n_quadrature_points));
        }

      // This is the equivalent of
      // for (q : q_points)
      //   for (i : dof_indices)
      //     for (j : dof_indices)
      //       cell_matrix(i,j) += shapes_test[i][q] * values_functor[q] *
      //       shapes_trial[j][q]) * JxW[q]
      const auto qp_range = fe_values_q_points.quadrature_point_indices();
      const auto dof_range_j =
        (symmetric_contribution ? fe_values_dofs.dof_indices() :
                                  fe_values_dofs.dof_indices());
      for (const unsigned int q : qp_range)
        {
          for (const unsigned int j : dof_range_j)
            {
              using ContractionType_FS =
                FullContraction<ValueTypeFunctor, ValueTypeTrial>;
              const auto functor_x_shape_trial_x_JxW =
                JxW[q] * ContractionType_FS::contract(values_functor[q],
                                                      shapes_trial[j][q]);
              using ContractionType_FS_t = typename std::decay<decltype(
                functor_x_shape_trial_x_JxW)>::type;

              // Assemble only the diagonal plus upper half of the matrix if
              // the symmetry flag is set.
              const auto dof_range_i =
                (symmetric_contribution ?
                   fe_values_dofs.dof_indices_ending_at(j) :
                   fe_values_dofs.dof_indices());
              for (const unsigned int i : dof_range_i)
                {
                  using ContractionType_SFS_JxW =
                    FullContraction<ValueTypeTest, ContractionType_FS_t>;
                  const ScalarType integrated_contribution =
                    ContractionType_SFS_JxW::contract(
                      shapes_test[i][q], functor_x_shape_trial_x_JxW);

                  if (Sign == AccumulationSign::plus)
                    {
                      cell_matrix(i, j) += integrated_contribution;
                    }
                  else
                    {
                      Assert(Sign == AccumulationSign::minus,
                             ExcInternalError());
                      cell_matrix(i, j) -= integrated_contribution;
                    }
                }
            }
        }
    }


    // Valid only for cell assembly
    template <enum AccumulationSign Sign,
              typename ScalarType,
              int dim,
              int spacedim,
              typename ValueTypeTest,
              typename ValueTypeFunctor,
              typename ValueTypeTrial>
    void
    assemble_cell_matrix_contribution(
      FullMatrix<ScalarType> &                        cell_matrix,
      const FEValuesBase<dim, spacedim> &             fe_values,
      const std::vector<std::vector<ValueTypeTest>> & shapes_test,
      const std::vector<ValueTypeFunctor> &           values_functor,
      const std::vector<std::vector<ValueTypeTrial>> &shapes_trial,
      const std::vector<double> &                     JxW,
      const bool                                      symmetric_contribution)
    {
      assemble_cell_matrix_contribution<Sign>(cell_matrix,
                                              fe_values,
                                              fe_values,
                                              shapes_test,
                                              values_functor,
                                              shapes_trial,
                                              JxW,
                                              symmetric_contribution);
    }


    // Valid for cell and face assembly
    template <enum AccumulationSign Sign,
              typename ScalarType,
              int dim,
              int spacedim,
              typename ValueTypeTest,
              typename ValueTypeFunctor>
    void
    assemble_cell_vector_contribution(
      Vector<ScalarType> &                           cell_vector,
      const FEValuesBase<dim, spacedim> &            fe_values_dofs,
      const FEValuesBase<dim, spacedim> &            fe_values_q_points,
      const std::vector<std::vector<ValueTypeTest>> &shapes_test,
      const std::vector<ValueTypeFunctor> &          values_functor,
      const std::vector<double> &                    JxW)
    {
      Assert(shapes_test.size() == fe_values_dofs.dofs_per_cell,
             ExcDimensionMismatch(shapes_test.size(),
                                  fe_values_dofs.dofs_per_cell));
      Assert(values_functor.size() == fe_values_q_points.n_quadrature_points,
             ExcDimensionMismatch(values_functor.size(),
                                  fe_values_q_points.n_quadrature_points));
      Assert(JxW.size() == fe_values_q_points.n_quadrature_points,
             ExcDimensionMismatch(JxW.size(),
                                  fe_values_q_points.n_quadrature_points));
      for (const unsigned int k : fe_values_dofs.dof_indices())
        {
          (void)k;
          Assert(shapes_test[k].size() ==
                   fe_values_q_points.n_quadrature_points,
                 ExcDimensionMismatch(shapes_test[k].size(),
                                      fe_values_q_points.n_quadrature_points));
        }

      for (const unsigned int i : fe_values_dofs.dof_indices())
        for (const unsigned int q :
             fe_values_q_points.quadrature_point_indices())
          {
            using ContractionType_SF =
              FullContraction<ValueTypeTest, ValueTypeFunctor>;
            const ScalarType integrated_contribution =
              JxW[q] * ContractionType_SF::contract(shapes_test[i][q],
                                                    values_functor[q]);
            // const auto contribution =
            //   (shapes_test[i][q] * values_functor[q]) * JxW[q];

            if (Sign == AccumulationSign::plus)
              {
                cell_vector(i) += integrated_contribution;
              }
            else
              {
                Assert(Sign == AccumulationSign::minus, ExcInternalError());
                cell_vector(i) -= integrated_contribution;
              }
          }
    }

    // Valid only for cell assembly
    template <enum AccumulationSign Sign,
              typename ScalarType,
              int dim,
              int spacedim,
              typename ValueTypeTest,
              typename ValueTypeFunctor>
    void
    assemble_cell_vector_contribution(
      Vector<ScalarType> &                           cell_vector,
      const FEValuesBase<dim, spacedim> &            fe_values,
      const std::vector<std::vector<ValueTypeTest>> &shapes_test,
      const std::vector<ValueTypeFunctor> &          values_functor,
      const std::vector<double> &                    JxW)
    {
      assemble_cell_vector_contribution<Sign>(
        cell_vector, fe_values, fe_values, shapes_test, values_functor, JxW);
    }

    // ====================================
    // Vectorized counterparts of the above
    // ====================================

#if DEAL_II_VECTORIZATION_WIDTH_IN_BITS > 0
    struct UseVectorization : std::true_type
    {};
#else
    struct UseVectorization : std::false_type
    {};
#endif

    template <typename ScalarType,
              std::size_t width,
              typename = typename std::enable_if<
                std::is_arithmetic<ScalarType>::value>::type>
    void
    set_vectorized_values(VectorizedArray<ScalarType, width> &out,
                          const unsigned int                  v,
                          const ScalarType &                  in)
    {
      Assert(v < width, ExcIndexRange(v, 0, width));
      out[v] = in;
    }


    template <typename ScalarType,
              std::size_t width,
              typename = typename std::enable_if<
                std::is_arithmetic<ScalarType>::value>::type>
    void
    set_vectorized_values(VectorizedArray<std::complex<ScalarType>, width> &out,
                          const unsigned int                                v,
                          const std::complex<ScalarType> &                  in)
    {
      set_vectorized_values(out.real, v, in.real);
      set_vectorized_values(out.imag, v, in.imag);
    }


    template <int dim, typename ScalarType, std::size_t width>
    void set_vectorized_values(
      Tensor<0, dim, VectorizedArray<ScalarType, width>> &out,
      const unsigned int                                  v,
      const Tensor<0, dim, ScalarType> &                  in)
    {
      VectorizedArray<ScalarType, width> &out_val = out;
      const ScalarType &                  in_val  = in;

      set_vectorized_values(out_val, v, in_val);
    }


    template <int rank, int dim, typename ScalarType, std::size_t width>
    void
    set_vectorized_values(
      Tensor<rank, dim, VectorizedArray<ScalarType, width>> &out,
      const unsigned int                                     v,
      const Tensor<rank, dim, ScalarType> &                  in)
    {
      for (unsigned int i = 0; i < out.n_independent_components; ++i)
        {
          const TableIndices<rank> indices(
            out.unrolled_to_component_indices(i));
          set_vectorized_values(out[indices], v, in[indices]);
        }
    }


    template <int dim, typename ScalarType, std::size_t width>
    void set_vectorized_values(
      SymmetricTensor<2, dim, VectorizedArray<ScalarType, width>> &out,
      const unsigned int                                           v,
      const SymmetricTensor<2, dim, ScalarType> &                  in)
    {
      for (unsigned int i = 0; i < out.n_independent_components; ++i)
        {
          const TableIndices<2> indices(out.unrolled_to_component_indices(i));
          set_vectorized_values(out[indices], v, in[indices]);
        }
    }


    // TODO: Reused from differentiation/sd/symengine_tensor_operations.h
    // Add to some common location?
    template <int dim>
    TableIndices<4>
    make_rank_4_tensor_indices(const unsigned int idx_i,
                               const unsigned int idx_j)
    {
      const TableIndices<2> indices_i(
        SymmetricTensor<2, dim>::unrolled_to_component_indices(idx_i));
      const TableIndices<2> indices_j(
        SymmetricTensor<2, dim>::unrolled_to_component_indices(idx_j));
      return TableIndices<4>(indices_i[0],
                             indices_i[1],
                             indices_j[0],
                             indices_j[1]);
    }


    template <int dim, typename ScalarType, std::size_t width>
    void set_vectorized_values(
      SymmetricTensor<4, dim, VectorizedArray<ScalarType, width>> &out,
      const unsigned int                                           v,
      const SymmetricTensor<4, dim, ScalarType> &                  in)
    {
      for (unsigned int i = 0;
           i < SymmetricTensor<2, dim>::n_independent_components;
           ++i)
        for (unsigned int j = 0;
             j < SymmetricTensor<2, dim>::n_independent_components;
             ++j)
          {
            const TableIndices<4> indices =
              make_rank_4_tensor_indices<dim>(i, j);
            set_vectorized_values(out[indices], v, in[indices]);
          }
    }


    // Valid for cell and face assembly
    template <enum AccumulationSign Sign,
              typename ScalarType,
              int dim,
              int spacedim,
              typename VectorizedValueTypeTest,
              typename VectorizedValueTypeFunctor,
              typename VectorizedValueTypeTrial,
              std::size_t width>
    void
    assemble_cell_matrix_vectorized_qp_batch_contribution(
      FullMatrix<ScalarType> &                       cell_matrix,
      const FEValuesBase<dim, spacedim> &            fe_values_dofs,
      const AlignedVector<VectorizedValueTypeTest> & shapes_test,
      const VectorizedValueTypeFunctor &             values_functor,
      const AlignedVector<VectorizedValueTypeTrial> &shapes_trial,
      const VectorizedArray<double, width> &         JxW,
      const bool                                     symmetric_contribution)
    {
      // This is the equivalent of
      // for (q : q_points) --> vectorized
      //   for (i : dof_indices)
      //     for (j : dof_indices)
      //       cell_matrix(i,j) += shapes_test[i][q] * values_functor[q] *
      //       shapes_trial[j][q]) * JxW[q]
      const auto dof_range_j =
        (symmetric_contribution ? fe_values_dofs.dof_indices() :
                                  fe_values_dofs.dof_indices());
      for (const unsigned int j : dof_range_j)
        {
          using ContractionType_FS = FullContraction<VectorizedValueTypeFunctor,
                                                     VectorizedValueTypeTrial>;
          const auto functor_x_shape_trial_x_JxW =
            JxW * ContractionType_FS::contract(values_functor, shapes_trial[j]);
          using ContractionType_FS_t =
            typename std::decay<decltype(functor_x_shape_trial_x_JxW)>::type;

          // Assemble only the diagonal plus upper half of the matrix if
          // the symmetry flag is set.
          const auto dof_range_i =
            (symmetric_contribution ? fe_values_dofs.dof_indices_ending_at(j) :
                                      fe_values_dofs.dof_indices());
          for (const unsigned int i : dof_range_i)
            {
              using ContractionType_SFS_JxW =
                FullContraction<VectorizedValueTypeTest, ContractionType_FS_t>;
              const VectorizedArray<ScalarType, width>
                vectorized_integrated_contribution =
                  ContractionType_SFS_JxW::contract(
                    shapes_test[i], functor_x_shape_trial_x_JxW);

              // Reduce all QP contributions
              ScalarType integrated_contribution =
                dealii::internal::NumberType<ScalarType>::value(0.0);
              for (unsigned int v = 0; v < width; v++)
                integrated_contribution +=
                  vectorized_integrated_contribution[v];

              if (Sign == AccumulationSign::plus)
                {
                  cell_matrix(i, j) += integrated_contribution;
                }
              else
                {
                  Assert(Sign == AccumulationSign::minus, ExcInternalError());
                  cell_matrix(i, j) -= integrated_contribution;
                }
            }
        }
    }


    // Valid for cell and face assembly
    template <enum AccumulationSign Sign,
              typename ScalarType,
              int dim,
              int spacedim,
              typename VectorizedValueTypeTest,
              typename VectorizedValueTypeFunctor,
              std::size_t width>
    void
    assemble_cell_vector_vectorized_qp_batch_contribution(
      Vector<ScalarType> &                          cell_vector,
      const FEValuesBase<dim, spacedim> &           fe_values_dofs,
      const AlignedVector<VectorizedValueTypeTest> &shapes_test,
      const VectorizedValueTypeFunctor &            values_functor,
      const VectorizedArray<double, width> &        JxW)
    {
      for (const unsigned int i : fe_values_dofs.dof_indices())
        {
          using ContractionType_SF =
            FullContraction<VectorizedValueTypeTest,
                            VectorizedValueTypeFunctor>;
          const VectorizedArray<ScalarType, width>
            vectorized_integrated_contribution =
              JxW *
              ContractionType_SF::contract(shapes_test[i], values_functor);

          // Reduce all QP contributions
          ScalarType integrated_contribution =
            dealii::internal::NumberType<ScalarType>::value(0.0);
          for (unsigned int v = 0; v < width; v++)
            integrated_contribution += vectorized_integrated_contribution[v];

          if (Sign == AccumulationSign::plus)
            {
              cell_vector(i) += integrated_contribution;
            }
          else
            {
              Assert(Sign == AccumulationSign::minus, ExcInternalError());
              cell_vector(i) -= integrated_contribution;
            }
        }
    }

    /**
     * Exception denoting that a class requires some specialization
     * in order to be used.
     */
    DeclExceptionMsg(ExcUnexpectedFunctionCall,
                     "This function should never be called, as it is "
                     "expected to be bypassed though the lack of availability "
                     "of a pointer at the calling site.");



    // Utility functions to help with template arguments of the
    // assemble_system() method being void / std::null_ptr_t.

    template <typename ScratchDataType, typename VectorType>
    typename std::enable_if<std::is_same<typename std::decay<VectorType>::type,
                                         std::nullptr_t>::value>::type
    extract_solution_local_dof_values(
      ScratchDataType &                  scratch_data,
      const SolutionStorage<VectorType> &solution_storage)
    {
      (void)scratch_data;
      (void)solution_storage;

      // Void pointer; do nothing.
      AssertThrow(false, ExcUnexpectedFunctionCall());
    }

    template <typename ScratchDataType, typename VectorType>
    typename std::enable_if<!std::is_same<typename std::decay<VectorType>::type,
                                          std::nullptr_t>::value>::type
    extract_solution_local_dof_values(
      ScratchDataType &                  scratch_data,
      const SolutionStorage<VectorType> &solution_storage)
    {
      solution_storage.extract_local_dof_values(scratch_data);
    }



    template <typename ScratchDataType,
              typename FaceQuadratureType,
              typename FiniteElementType,
              typename CellQuadratureType>
    typename std::enable_if<
      std::is_same<typename std::decay<FaceQuadratureType>::type,
                   std::nullptr_t>::value,
      ScratchDataType>::type
    construct_scratch_data(const FiniteElementType &       finite_element,
                           const CellQuadratureType &      cell_quadrature,
                           const UpdateFlags &             cell_update_flags,
                           const FaceQuadratureType *const face_quadrature,
                           const UpdateFlags &             face_update_flags)
    {
      (void)face_quadrature;
      (void)face_update_flags;
      AssertThrow(false, ExcUnexpectedFunctionCall());
      return ScratchDataType(finite_element,
                             cell_quadrature,
                             cell_update_flags);
    }

    template <typename ScratchDataType,
              typename FaceQuadratureType,
              typename FiniteElementType,
              typename CellQuadratureType>
    typename std::enable_if<
      !std::is_same<typename std::decay<FaceQuadratureType>::type,
                    std::nullptr_t>::value,
      ScratchDataType>::type
    construct_scratch_data(const FiniteElementType &       finite_element,
                           const CellQuadratureType &      cell_quadrature,
                           const UpdateFlags &             cell_update_flags,
                           const FaceQuadratureType *const face_quadrature,
                           const UpdateFlags &             face_update_flags)
    {
      return ScratchDataType(finite_element,
                             cell_quadrature,
                             cell_update_flags,
                             *face_quadrature,
                             face_update_flags);
    }


    template <typename ScalarType,
              typename TestOrTrialSpaceOp,
              typename FEValuesDofsType,
              typename FEValuesOpType,
              typename ScratchDataType>
    typename std::enable_if<
      !WeakForms::has_evaluated_with_scratch_data<TestOrTrialSpaceOp>::value,
      std::vector<std::vector<
        typename TestOrTrialSpaceOp::template value_type<ScalarType>>>>::type
    evaluate_fe_space(const TestOrTrialSpaceOp &      test_or_trial_space_op,
                      const FEValuesDofsType &        fe_values_dofs,
                      const FEValuesOpType &          fe_values_op,
                      ScratchDataType &               scratch_data,
                      const std::vector<std::string> &solution_names)
    {
      static_assert(
        is_or_has_test_function_or_trial_solution_op<TestOrTrialSpaceOp>::value,
        "Expected a test function or trial solution.");
      (void)scratch_data;
      (void)solution_names;

      return test_or_trial_space_op.template operator()<ScalarType>(
        fe_values_dofs, fe_values_op);
    }


    template <typename ScalarType,
              typename TestOrTrialSpaceOp,
              typename FEValuesDofsType,
              typename FEValuesOpType,
              typename ScratchDataType>
    typename std::enable_if<
      WeakForms::has_evaluated_with_scratch_data<TestOrTrialSpaceOp>::value,
      std::vector<std::vector<
        typename TestOrTrialSpaceOp::template value_type<ScalarType>>>>::type
    evaluate_fe_space(const TestOrTrialSpaceOp &      test_or_trial_space_op,
                      const FEValuesDofsType &        fe_values_dofs,
                      const FEValuesOpType &          fe_values_op,
                      ScratchDataType &               scratch_data,
                      const std::vector<std::string> &solution_names)
    {
      static_assert(
        is_or_has_test_function_or_trial_solution_op<TestOrTrialSpaceOp>::value,
        "Expected a test function or trial solution.");

      return test_or_trial_space_op.template operator()<ScalarType>(
        fe_values_dofs, fe_values_op, scratch_data, solution_names);
    }


    template <typename ScalarType,
              typename FunctorType,
              typename FEValuesType,
              typename ScratchDataType>
    typename std::enable_if<
      !WeakForms::is_or_has_evaluated_with_scratch_data<FunctorType>::value,
      std::vector<typename FunctorType::template value_type<ScalarType>>>::type
    evaluate_functor(const FunctorType &             functor,
                     const FEValuesType &            fe_values,
                     ScratchDataType &               scratch_data,
                     const std::vector<std::string> &solution_names)
    {
      (void)scratch_data;
      (void)solution_names;
      return functor.template operator()<ScalarType>(fe_values);
    }


    template <typename ScalarType,
              typename FunctorType,
              typename FEValuesType,
              typename ScratchDataType>
    typename std::enable_if<
      WeakForms::is_or_has_evaluated_with_scratch_data<FunctorType>::value &&
        !WeakForms::is_binary_op<FunctorType>::value,
      std::vector<typename FunctorType::template value_type<ScalarType>>>::type
    evaluate_functor(const FunctorType &             functor,
                     const FEValuesType &            fe_values,
                     ScratchDataType &               scratch_data,
                     const std::vector<std::string> &solution_names)
    {
      (void)fe_values;
      return functor.template operator()<ScalarType>(scratch_data,
                                                     solution_names);
    }


    template <typename ScalarType,
              typename FunctorType,
              typename FEValuesType,
              typename ScratchDataType>
    typename std::enable_if<
      WeakForms::is_or_has_evaluated_with_scratch_data<FunctorType>::value &&
        WeakForms::is_binary_op<FunctorType>::value,
      std::vector<typename FunctorType::template value_type<ScalarType>>>::type
    evaluate_functor(const FunctorType &             functor,
                     const FEValuesType &            fe_values,
                     ScratchDataType &               scratch_data,
                     const std::vector<std::string> &solution_names)
    {
      return functor.template operator()<ScalarType>(fe_values,
                                                     scratch_data,
                                                     solution_names);
    }


    template <typename MatrixType, typename VectorType, typename ScalarType>
    typename std::enable_if<std::is_same<typename std::decay<MatrixType>::type,
                                         std::nullptr_t>::value ||
                            std::is_same<typename std::decay<VectorType>::type,
                                         std::nullptr_t>::value>::type
    distribute_local_to_global(
      const AffineConstraints<ScalarType> &               constraints,
      const FullMatrix<ScalarType> &                      cell_matrix,
      const Vector<ScalarType> &                          cell_vector,
      const std::vector<dealii::types::global_dof_index> &local_dof_indices,
      MatrixType *const                                   system_matrix,
      VectorType *const                                   system_vector)
    {
      (void)constraints;
      (void)cell_matrix;
      (void)cell_vector;
      (void)local_dof_indices;
      (void)system_matrix;
      (void)system_vector;

      // Void pointer (either matrix or vector); do nothing.
      AssertThrow(false, ExcUnexpectedFunctionCall());
    }

    template <typename MatrixType, typename VectorType, typename ScalarType>
    typename std::enable_if<!std::is_same<typename std::decay<MatrixType>::type,
                                          std::nullptr_t>::value &&
                            !std::is_same<typename std::decay<VectorType>::type,
                                          std::nullptr_t>::value>::type
    distribute_local_to_global(
      const AffineConstraints<ScalarType> &               constraints,
      const FullMatrix<ScalarType> &                      cell_matrix,
      const Vector<ScalarType> &                          cell_vector,
      const std::vector<dealii::types::global_dof_index> &local_dof_indices,
      MatrixType *const                                   system_matrix,
      VectorType *const                                   system_vector)
    {
      Assert(system_matrix, ExcInternalError());
      Assert(system_vector, ExcInternalError());
      constraints.distribute_local_to_global(cell_matrix,
                                             cell_vector,
                                             local_dof_indices,
                                             *system_matrix,
                                             *system_vector);
    }

    template <typename MatrixType, typename ScalarType>
    typename std::enable_if<std::is_same<typename std::decay<MatrixType>::type,
                                         std::nullptr_t>::value>::type
    distribute_local_to_global(
      const AffineConstraints<ScalarType> &               constraints,
      const FullMatrix<ScalarType> &                      cell_matrix,
      const std::vector<dealii::types::global_dof_index> &local_dof_indices,
      MatrixType *const                                   system_matrix)
    {
      (void)constraints;
      (void)cell_matrix;
      (void)local_dof_indices;
      (void)system_matrix;

      // Void pointer; do nothing.
      AssertThrow(false, ExcUnexpectedFunctionCall());
    }

    template <typename MatrixType, typename ScalarType>
    typename std::enable_if<!std::is_same<typename std::decay<MatrixType>::type,
                                          std::nullptr_t>::value>::type
    distribute_local_to_global(
      const AffineConstraints<ScalarType> &               constraints,
      const FullMatrix<ScalarType> &                      cell_matrix,
      const std::vector<dealii::types::global_dof_index> &local_dof_indices,
      MatrixType *const                                   system_matrix)
    {
      Assert(system_matrix, ExcInternalError());
      constraints.distribute_local_to_global(cell_matrix,
                                             local_dof_indices,
                                             *system_matrix);
    }

    template <typename VectorType, typename ScalarType>
    typename std::enable_if<std::is_same<typename std::decay<VectorType>::type,
                                         std::nullptr_t>::value>::type
    distribute_local_to_global(
      const AffineConstraints<ScalarType> &               constraints,
      const Vector<ScalarType> &                          cell_vector,
      const std::vector<dealii::types::global_dof_index> &local_dof_indices,
      VectorType *const                                   system_vector)
    {
      (void)constraints;
      (void)cell_vector;
      (void)local_dof_indices;
      (void)system_vector;

      // Void pointer; do nothing.
      AssertThrow(false, ExcUnexpectedFunctionCall());
    }

    template <typename VectorType, typename ScalarType>
    typename std::enable_if<!std::is_same<typename std::decay<VectorType>::type,
                                          std::nullptr_t>::value>::type
    distribute_local_to_global(
      const AffineConstraints<ScalarType> &               constraints,
      const Vector<ScalarType> &                          cell_vector,
      const std::vector<dealii::types::global_dof_index> &local_dof_indices,
      VectorType *const                                   system_vector)
    {
      Assert(system_vector, ExcInternalError());
      constraints.distribute_local_to_global(cell_vector,
                                             local_dof_indices,
                                             *system_vector);
    }

    template <typename MatrixOrVectorType>
    typename std::enable_if<
      std::is_same<typename std::decay<MatrixOrVectorType>::type,
                   std::nullptr_t>::value>::type
    compress(MatrixOrVectorType *const system_matrix_or_vector)
    {
      (void)system_matrix_or_vector;

      // Void pointer; do nothing.
      AssertThrow(false, ExcUnexpectedFunctionCall());
    }

    template <typename MatrixOrVectorType>
    typename std::enable_if<
      !std::is_same<typename std::decay<MatrixOrVectorType>::type,
                    std::nullptr_t>::value>::type
    compress(MatrixOrVectorType *const system_matrix_or_vector)
    {
      Assert(system_matrix_or_vector, ExcInternalError());
      system_matrix_or_vector->compress(VectorOperation::add);
    }

  } // namespace internal



  template <int dim, int spacedim, typename ScalarType, bool use_vectorization>
  class AssemblerBase
  {
  public:
    using scalar_type = ScalarType;

    using AsciiLatexOperation =
      std::function<std::string(const SymbolicDecorations &decorator)>;
    using StringOperation = std::function<
      std::pair<AsciiLatexOperation, enum internal::AccumulationSign>(void)>;

    using CellSolutionUpdateOperation =
      std::function<void(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const std::vector<std::string> &solution_names)>;

    using BoundaryFaceSolutionUpdateOperation =
      std::function<void(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const std::vector<std::string> &solution_names)>;

    using CellADSDOperation =
      std::function<void(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const std::vector<std::string> &solution_names)>;

    using BoundaryADSDOperation =
      std::function<void(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const std::vector<std::string> &solution_names)>;

    using CellMatrixOperation =
      std::function<void(FullMatrix<ScalarType> &                cell_matrix,
                         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const std::vector<std::string> &        solution_names,
                         const FEValuesBase<dim, spacedim> &     fe_values)>;
    using CellVectorOperation =
      std::function<void(Vector<ScalarType> &                    cell_vector,
                         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const std::vector<std::string> &        solution_names,
                         const FEValuesBase<dim, spacedim> &     fe_values)>;

    using BoundaryMatrixOperation =
      std::function<void(FullMatrix<ScalarType> &                cell_matrix,
                         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const std::vector<std::string> &        solution_names,
                         const FEValuesBase<dim, spacedim> &     fe_values,
                         const FEFaceValuesBase<dim, spacedim> & fe_face_values,
                         const unsigned int                      face)>;
    using BoundaryVectorOperation =
      std::function<void(Vector<ScalarType> &                    cell_vector,
                         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const std::vector<std::string> &        solution_names,
                         const FEValuesBase<dim, spacedim> &     fe_values,
                         const FEFaceValuesBase<dim, spacedim> & fe_face_values,
                         const unsigned int                      face)>;

    using InterfaceMatrixOperation =
      std::function<void(FullMatrix<ScalarType> &                cell_matrix,
                         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const std::vector<std::string> &        solution_names,
                         const FEValuesBase<dim, spacedim> &     fe_values,
                         const FEFaceValuesBase<dim, spacedim> & fe_face_values,
                         const unsigned int                      face)>;
    using InterfaceVectorOperation =
      std::function<void(Vector<ScalarType> &                    cell_vector,
                         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const std::vector<std::string> &        solution_names,
                         const FEValuesBase<dim, spacedim> &     fe_values,
                         const FEFaceValuesBase<dim, spacedim> & fe_face_values,
                         const unsigned int                      face)>;


    virtual ~AssemblerBase() = default;

    // For the cases:
    //  assembler += ().dV + ().dV + ...
    //  assembler += ().dV - ().dV + ...
    //  assembler += ().dV + ().dA + ...
    //  ... etc.
    template <typename BinaryOpType,
              typename std::enable_if<
                // We don't know what the branches of this binary operation
                // are (it might be a composite operation formed of many
                // binary operations), so we cannot query any further about
                // the LHS and RHS operand types. We may assume that the
                // other operators that are called will filter out the
                // leaves at the end, which should all be symbolic integrals.
                is_binary_integral_op<BinaryOpType>::value
                // &&
                // is_unary_integral_op<typename BinaryOpType::LhsOpType>::value
                // && is_unary_integral_op<typename
                // BinaryOpType::RhsOpType>::value
                >::type * = nullptr>
    AssemblerBase &
    operator+=(const BinaryOpType &composite_integral)
    {
      // TODO: Or need a composite integral op?!?
      *this += composite_integral.get_lhs_operand();

      // For addition, the RHS of the composite operation retains its sign.
      if (BinaryOpType::op_code == Operators::BinaryOpCodes::add)
        *this += composite_integral.get_rhs_operand();
      else if (BinaryOpType::op_code == Operators::BinaryOpCodes::subtract)
        *this -= composite_integral.get_rhs_operand();
      else
        {
          AssertThrow(BinaryOpType::op_code == Operators::BinaryOpCodes::add ||
                        BinaryOpType::op_code ==
                          Operators::BinaryOpCodes::subtract,
                      ExcNotImplemented());
        }

      return *this;
    }


    // For the cases:
    //  assembler -= ().dV + ().dV + ...
    //  assembler -= ().dV - ().dV + ...
    //  assembler -= ().dV + ().dA + ...
    //  ... etc.
    template <typename BinaryOpType,
              typename std::enable_if<
                // We don't know what the branches of this binary operation
                // are (it might be a composite operation formed of many
                // binary operations), so we cannot query any further about
                // the LHS and RHS operand types. We may assume that the
                // other operators that are called will filter out the
                // leaves at the end, which should all be symbolic integrals.
                is_binary_integral_op<BinaryOpType>::value
                // &&
                // is_unary_integral_op<typename BinaryOpType::LhsOpType>::value
                // && is_unary_integral_op<typename
                // BinaryOpType::RhsOpType>::value
                >::type * = nullptr>
    AssemblerBase &
    operator-=(const BinaryOpType &composite_integral)
    {
      *this -= composite_integral.get_lhs_operand();

      // For subtraction, the RHS of the composite operation swaps its sign.
      if (BinaryOpType::op_code == Operators::BinaryOpCodes::add)
        *this -= composite_integral.get_rhs_operand();
      else if (BinaryOpType::op_code == Operators::BinaryOpCodes::subtract)
        *this += composite_integral.get_rhs_operand();
      else
        {
          AssertThrow(BinaryOpType::op_code == Operators::BinaryOpCodes::add ||
                        BinaryOpType::op_code ==
                          Operators::BinaryOpCodes::subtract,
                      ExcNotImplemented());
        }

      return *this;
    }

    // For the cases:
    //  assembler += -().dV + ().dV + ...
    //  assembler += -(().dV - ().dV) + ...
    //  assembler += -().dV - ().dA + ...
    //  ... etc.
    template <typename UnaryOpType,
              typename std::enable_if<
                // We don't know what the branches of this unary operation
                // are (it might be a composite operation formed of many
                // binary operations), so we cannot query any further about
                // the operand types. We may assume that the other operators
                // that are called will filter out the leaves at the end, which
                // should all be symbolic integrals.
                is_unary_integral_op<UnaryOpType>::value>::type * = nullptr>
    AssemblerBase &
    operator+=(const UnaryOpType &unary_integral)
    {
      if (UnaryOpType::op_code == Operators::UnaryOpCodes::negate)
        *this -= unary_integral.get_operand();
      else
        {
          AssertThrow(UnaryOpType::op_code == Operators::UnaryOpCodes::negate,
                      ExcNotImplemented());
        }

      return *this;
    }

    // For the cases:
    //  assembler -= -().dV + ().dV + ...
    //  assembler -= -(().dV - ().dV) + ...
    //  assembler -= -().dV - ().dA + ...
    //  ... etc.
    template <typename UnaryOpType,
              typename std::enable_if<
                // We don't know what the branches of this unary operation
                // are (it might be a composite operation formed of many
                // binary operations), so we cannot query any further about
                // the operand types. We may assume that the other operators
                // that are called will filter out the leaves at the end, which
                // should all be symbolic integrals.
                is_unary_integral_op<UnaryOpType>::value>::type * = nullptr>
    AssemblerBase &
    operator-=(const UnaryOpType &unary_integral)
    {
      // For subtraction, the value of the operation swaps its sign.
      if (UnaryOpType::op_code == Operators::UnaryOpCodes::negate)
        *this += unary_integral.get_operand();
      else
        {
          AssertThrow(UnaryOpType::op_code == Operators::UnaryOpCodes::negate,
                      ExcNotImplemented());
        }

      return *this;
    }


    template <
      typename SymbolicOpType,
      typename std::enable_if<
        is_symbolic_integral_op<SymbolicOpType>::value &&
        is_self_linearizing_form<
          typename SymbolicOpType::IntegrandType>::value>::type * = nullptr>
    AssemblerBase &
    operator+=(const SymbolicOpType &integral)
    {
      constexpr auto op_sign = internal::AccumulationSign::plus;

      const auto &form    = integral.get_integrand();
      const auto &functor = form.get_functor();

      // We don't care about the sign of the AD operation, because it is
      // layer corrected in the accumulate_into() operation.
      auto f = [functor](MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const std::vector<std::string> &solution_names) {
        functor.template operator()<ScalarType>(scratch_data, solution_names);
      };
      if (is_volume_integral_op<SymbolicOpType>::value)
        {
          cell_update_flags |= functor.get_update_flags();
          cell_ad_sd_operations.emplace_back(f);
        }
      else if (is_boundary_integral_op<SymbolicOpType>::value)
        {
          boundary_face_update_flags |= functor.get_update_flags();
          boundary_face_ad_sd_operations.emplace_back(f);
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }

      // The form is self-linearizing, so the assembler doesn't know what
      // contributions it will form. So we just get the form to submit its
      // own linear and bilinear form contributions that stem from the
      // self-linearization process. To achieve this, we also need to inform
      // the form over which domain it is integrated.
      form.template accumulate_into<op_sign>(*this,
                                             integral.get_integral_operation());

      return *this;
    }


    template <
      typename SymbolicOpType,
      typename std::enable_if<
        is_symbolic_integral_op<SymbolicOpType>::value &&
        is_volume_integral_op<SymbolicOpType>::value &&
        !is_self_linearizing_form<
          typename SymbolicOpType::IntegrandType>::value>::type * = nullptr>
    AssemblerBase &
    operator+=(const SymbolicOpType &volume_integral)
    {
      // TODO: Detect if the Test+Trial combo is the same as one that has
      // already been added. If so, augment the functor rather than repeating
      // the loop?
      // Potential problem: One functor is scalar valued, and the other is
      // tensor valued...

      // Linear forms go on the RHS, bilinear forms go on the LHS.
      // So we switch the sign based on this.
      using IntegrandType         = typename SymbolicOpType::IntegrandType;
      constexpr bool keep_op_sign = is_bilinear_form<IntegrandType>::value;
      constexpr auto print_sign   = internal::AccumulationSign::plus;
      constexpr auto op_sign =
        (keep_op_sign ? internal::AccumulationSign::plus :
                        internal::AccumulationSign::minus);

      add_ascii_latex_operations<print_sign>(volume_integral);
      add_cell_operation<op_sign>(volume_integral);

      // const auto &form    = volume_integral.get_integrand();
      // const auto &functor = form.get_functor();
      // add_solution_update_operation(functor);

      return *this;
    }


    template <
      typename SymbolicOpType,
      typename std::enable_if<
        is_symbolic_integral_op<SymbolicOpType>::value &&
        is_boundary_integral_op<SymbolicOpType>::value &&
        !is_self_linearizing_form<
          typename SymbolicOpType::IntegrandType>::value>::type * = nullptr>
    AssemblerBase &
    operator+=(const SymbolicOpType &boundary_integral)
    {
      // TODO: Detect if the Test+Trial combo is the same as one that has
      // already been added. If so, augment the functor rather than repeating
      // the loop?
      // Potential problem: One functor is scalar valued, and the other is
      // tensor valued...

      // Linear forms go on the RHS, bilinear forms go on the LHS.
      // So we switch the sign based on this.
      using IntegrandType         = typename SymbolicOpType::IntegrandType;
      constexpr bool keep_op_sign = is_bilinear_form<IntegrandType>::value;
      constexpr auto print_sign   = internal::AccumulationSign::plus;
      constexpr auto op_sign =
        (keep_op_sign ? internal::AccumulationSign::plus :
                        internal::AccumulationSign::minus);

      add_ascii_latex_operations<print_sign>(boundary_integral);
      add_boundary_face_operation<op_sign>(boundary_integral);

      // const auto &form    = boundary_integral.get_integrand();
      // const auto &functor = form.get_functor();
      // add_solution_update_operation(functor);

      return *this;
    }


    template <
      typename SymbolicOpType,
      typename std::enable_if<
        is_symbolic_integral_op<SymbolicOpType>::value &&
        is_interface_integral_op<SymbolicOpType>::value &&
        !is_self_linearizing_form<
          typename SymbolicOpType::IntegrandType>::value>::type * = nullptr>
    AssemblerBase &
    operator+=(const SymbolicOpType &interface_integral)
    {
      (void)interface_integral;

      AssertThrow(false, ExcNotImplemented());

      // static_assert(false, "Assembler: operator+= not yet implemented for
      // interface integrals");

      // TODO: Detect if the Test+Trial combo is the same as one that has
      // already been added. If so, augment the functor rather than repeating
      // the loop?
      // Potential problem: One functor is scalar valued, and the other is
      // tensor valued...

      // constexpr auto sign = internal::AccumulationSign::plus;
      // add_ascii_latex_operations<sign>(boundary_integral);
      // add_cell_operation<sign>(boundary_integral);

      // const auto &form    = boundary_integral.get_integrand();
      // const auto &functor = form.get_functor();
      // add_solution_update_operation(functor);

      return *this;
    }


    template <
      typename SymbolicOpType,
      typename std::enable_if<
        is_symbolic_integral_op<SymbolicOpType>::value &&
        is_self_linearizing_form<
          typename SymbolicOpType::IntegrandType>::value>::type * = nullptr>
    AssemblerBase &
    operator-=(const SymbolicOpType &integral)
    {
      constexpr auto op_sign = internal::AccumulationSign::minus;

      const auto &form    = integral.get_integrand();
      const auto &functor = form.get_functor();

      // We don't care about the sign of the AD operation, because it is
      // layer corrected in the accumulate_into() operation.
      auto f = [functor](MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const std::vector<std::string> &solution_names) {
        functor.template operator()<ScalarType>(scratch_data, solution_names);
      };
      if (is_volume_integral_op<SymbolicOpType>::value)
        {
          cell_update_flags |= functor.get_update_flags();
          cell_ad_sd_operations.emplace_back(f);
        }
      else if (is_boundary_integral_op<SymbolicOpType>::value)
        {
          boundary_face_update_flags |= functor.get_update_flags();
          boundary_face_ad_sd_operations.emplace_back(f);
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }

      // The form is self-linearizing, so the assembler doesn't know what
      // contributions it will form. So we just get the form to submit its
      // own linear and bilinear form contributions that stem from the
      // self-linearization process. To achieve this, we also need to inform
      // the form over which domain it is integrated.
      form.template accumulate_into<op_sign>(*this,
                                             integral.get_integral_operation());

      return *this;
    }


    template <
      typename SymbolicOpType,
      typename std::enable_if<
        is_symbolic_integral_op<SymbolicOpType>::value &&
        is_volume_integral_op<SymbolicOpType>::value &&
        !is_self_linearizing_form<
          typename SymbolicOpType::IntegrandType>::value>::type * = nullptr>
    AssemblerBase &
    operator-=(const SymbolicOpType &volume_integral)
    {
      // TODO: Detect if the Test+Trial combo is the same as one that has
      // already been added. If so, augment the functor rather than repeating
      // the loop?
      // Potential problem: One functor is scalar valued, and the other is
      // tensor valued...

      // Linear forms go on the RHS, bilinear forms go on the LHS.
      // So we switch the sign based on this.
      using IntegrandType         = typename SymbolicOpType::IntegrandType;
      constexpr bool keep_op_sign = is_bilinear_form<IntegrandType>::value;
      constexpr auto print_sign   = internal::AccumulationSign::minus;
      constexpr auto op_sign =
        (keep_op_sign ? internal::AccumulationSign::minus :
                        internal::AccumulationSign::plus);

      add_ascii_latex_operations<print_sign>(volume_integral);
      add_cell_operation<op_sign>(volume_integral);

      // const auto &form    = volume_integral.get_integrand();
      // const auto &functor = form.get_functor();
      // add_solution_update_operation(functor);

      return *this;
    }


    template <
      typename SymbolicOpType,
      typename std::enable_if<
        is_symbolic_integral_op<SymbolicOpType>::value &&
        is_boundary_integral_op<SymbolicOpType>::value &&
        !is_self_linearizing_form<
          typename SymbolicOpType::IntegrandType>::value>::type * = nullptr>
    AssemblerBase &
    operator-=(const SymbolicOpType &boundary_integral)
    {
      // TODO: Detect if the Test+Trial combo is the same as one that has
      // already been added. If so, augment the functor rather than repeating
      // the loop?
      // Potential problem: One functor is scalar valued, and the other is
      // tensor valued...

      // Linear forms go on the RHS, bilinear forms go on the LHS.
      // So we switch the sign based on this.
      using IntegrandType         = typename SymbolicOpType::IntegrandType;
      constexpr bool keep_op_sign = is_bilinear_form<IntegrandType>::value;
      constexpr auto print_sign   = internal::AccumulationSign::minus;
      constexpr auto op_sign =
        (keep_op_sign ? internal::AccumulationSign::minus :
                        internal::AccumulationSign::plus);

      add_ascii_latex_operations<print_sign>(boundary_integral);
      add_boundary_face_operation<op_sign>(boundary_integral);

      // const auto &form    = boundary_integral.get_integrand();
      // const auto &functor = form.get_functor();
      // add_solution_update_operation(functor);

      return *this;
    }


    template <
      typename SymbolicOpType,
      typename std::enable_if<
        is_symbolic_integral_op<SymbolicOpType>::value &&
        is_interface_integral_op<SymbolicOpType>::value &&
        !is_self_linearizing_form<
          typename SymbolicOpType::IntegrandType>::value>::type * = nullptr>
    AssemblerBase &
    operator-=(const SymbolicOpType &interface_integral)
    {
      (void)interface_integral;
      AssertThrow(false, ExcNotImplemented());

      // static_assert(false, "Assembler: operator-= not yet implemented for
      // interface integrals");

      // TODO: Detect if the Test+Trial combo is the same as one that has
      // already been added. If so, augment the functor rather than repeating
      // the loop?
      // Potential problem: One functor is scalar valued, and the other is
      // tensor valued...

      // constexpr auto sign = internal::AccumulationSign::minus;
      // add_ascii_latex_operations<sign>(boundary_integral);
      // add_cell_operation<sign>(boundary_integral);

      // const auto &form    = boundary_integral.get_integrand();
      // const auto &functor = form.get_functor();
      // add_solution_update_operation(functor);

      return *this;
    }


    // TODO:
    std::string
    as_ascii(const SymbolicDecorations &decorator) const
    {
      std::string output = "0 = ";
      for (unsigned int i = 0; i < as_ascii_operations.size(); ++i)
        {
          Assert(as_ascii_operations[i], ExcNotInitialized());
          const auto &current_term_function = as_ascii_operations[i];

          // If first term is negative, then we need to make sure that
          // this is shown.
          if (i == 0 && current_term_function().second ==
                          internal::AccumulationSign::minus)
            output += "- ";

          const AsciiLatexOperation &string_op = current_term_function().first;
          output += string_op(decorator);

          if (i + 1 < as_ascii_operations.size())
            {
              Assert(as_ascii_operations[i + 1], ExcNotInitialized());
              const auto &next_term_function = as_ascii_operations[i + 1];

              if (next_term_function().second ==
                  internal::AccumulationSign::plus)
                {
                  output += " + ";
                }
              else
                {
                  Assert(next_term_function().second ==
                           internal::AccumulationSign::minus,
                         ExcInternalError());
                  output += " - ";
                }
            }
        }
      return output;
    }

    std::string
    as_latex(const SymbolicDecorations &decorator) const
    {
      std::string output = "0 = ";
      for (unsigned int i = 0; i < as_latex_operations.size(); ++i)
        {
          Assert(as_latex_operations[i], ExcNotInitialized());
          const auto &current_term_function = as_latex_operations[i];

          // If first term is negative, then we need to make sure that
          // this is shown.
          if (i == 0 && current_term_function().second ==
                          internal::AccumulationSign::minus)
            output += "- ";

          const AsciiLatexOperation &string_op = current_term_function().first;
          output += string_op(decorator);

          if (i + 1 < as_latex_operations.size())
            {
              Assert(as_latex_operations[i + 1], ExcNotInitialized());
              const auto &next_term_function = as_latex_operations[i + 1];

              if (next_term_function().second ==
                  internal::AccumulationSign::plus)
                {
                  output += " + ";
                }
              else
                {
                  Assert(next_term_function().second ==
                           internal::AccumulationSign::minus,
                         ExcInternalError());
                  output += " - ";
                }
            }
        }
      return output;
    }

    bool
    is_symmetric() const
    {
      return global_system_symmetry_flag;
    }

    void
    set_global_system_symmetry_flag(const bool flag)
    {
      global_system_symmetry_flag = flag;
    }

    void
    symmetrize()
    {
      set_global_system_symmetry_flag(true);
    }


    // template <typename VectorType,
    //           typename DoFHandlerType,
    //           typename CellQuadratureType>
    // void
    // update_solution(const VectorType &        solution_vector,
    //                 const DoFHandlerType &    dof_handler,
    //                 const CellQuadratureType &cell_quadrature)
    // {
    //   static_assert(DoFHandlerType::dimension == dim,
    //                 "Dimension is incompatible");
    //   static_assert(DoFHandlerType::space_dimension == spacedim,
    //                 "Space dimension is incompatible");

    //   using CellIteratorType = typename DoFHandlerType::active_cell_iterator;
    //   using ScratchData      = MeshWorker::ScratchData<dim, spacedim>;
    //   using CopyData         = MeshWorker::CopyData<0, 0, 0>; // Empty copier

    //   // Define a cell worker
    //   const auto &cell_solution_update_operations =
    //     this->cell_solution_update_operations;
    //   auto cell_worker = [&cell_solution_update_operations,
    //                       &solution_vector](const CellIteratorType &cell,
    //                                         ScratchData &scratch_data,
    //                                         CopyData &   copy_data) {
    //     (void)copy_data;
    //     const auto &fe_values          = scratch_data.reinit(cell);
    //     copy_data                      = CopyData(fe_values.dofs_per_cell);
    //     copy_data.local_dof_indices[0] =
    //     scratch_data.get_local_dof_indices();

    //     std::vector<ScalarType> solution_local_dof_values;
    //     Assert(copy_data.local_dof_indices[0].size() ==
    //     fe_values.dofs_per_cell,
    //            ExcDimensionMismatch(copy_data.local_dof_indices[0].size(),
    //                                 fe_values.dofs_per_cell));

    //     solution_local_dof_values.resize(fe_values.dofs_per_cell);
    //     internal::extract_solution_local_dof_values(
    //       solution_local_dof_values,
    //       copy_data.local_dof_indices[0],
    //       &solution_vector);

    //     for (const auto &cell_solution_update_op :
    //          cell_solution_update_operations)
    //       {
    //         cell_solution_update_op(solution_local_dof_values, fe_values);
    //       }

    //     // TODO:
    //     // boundary_matrix_operations
    //     // interface_matrix_operations
    //   };

    //   auto dummy_copier = [](const CopyData &copy_data) { (void)copy_data; };

    //   const ScratchData sample_scratch_data(dof_handler.get_fe(),
    //                                         cell_quadrature,
    //                                         this->get_cell_update_flags());
    //   const CopyData    sample_copy_data(dof_handler.get_fe().dofs_per_cell);

    //   MeshWorker::mesh_loop(dof_handler.active_cell_iterators(),
    //                         cell_worker,
    //                         dummy_copier,
    //                         sample_scratch_data,
    //                         sample_copy_data,
    //                         MeshWorker::assemble_own_cells);
    // }

  protected:
    /**
     * A flag to indicate whether or not the global system is to be assembled
     * in symmetric form, or not.
     *
     * If so, then we only ever assemble the upper half plus diagonal
     * contributions from any form, and then mirror the upper half contribution
     * into the lower half of the local matrix.
     */
    bool global_system_symmetry_flag;

    explicit AssemblerBase()
      : cell_update_flags(update_default)
      // , cell_solution_update_flags(update_default)
      , boundary_face_update_flags(update_default)
      // , boundary_face_solution_update_flags(update_default)
      , interface_face_update_flags(update_default)
      // , interface_face_solution_update_flags(update_default)
      , global_system_symmetry_flag(false)
    {}


    template <enum internal::AccumulationSign Sign, typename IntegralType>
    typename std::enable_if<is_symbolic_integral_op<IntegralType>::value>::type
    add_ascii_latex_operations(const IntegralType &integral)
    {
      // Augment the composition of the operation
      // Important note: All operations must be captured by copy!
      as_ascii_operations.push_back([integral]() {
        return std::make_pair(
          [integral](const SymbolicDecorations &decorator) {
            return integral.as_ascii(decorator);
          },
          Sign);
      });
      as_latex_operations.push_back([integral]() {
        return std::make_pair(
          [integral](const SymbolicDecorations &decorator) {
            return integral.as_latex(decorator);
          },
          Sign);
      });
    }

    /**
     * Cell operations for bilinear forms
     *
     * @tparam SymbolicOpVolumeIntegral
     * @tparam std::enable_if<is_bilinear_form<
     * typename SymbolicOpVolumeIntegral::IntegrandType>::value>::type
     * @param volume_integral
     *
     * Providing the @p solution solution pointer is optional, as we might
     * be assembling with a functor that does not require it. But if it
     * does then we'll check that the @p VectorType is valid and that it
     * points to something valid.
     *
     *   typename VectorType = void
     *   const VectorType *const solution = nullptr
     *
     */
    template <enum internal::AccumulationSign Sign,
              typename SymbolicOpVolumeIntegral,
              typename std::enable_if<is_bilinear_form<
                typename SymbolicOpVolumeIntegral::IntegrandType>::value>::type
                * = nullptr>
    void
    add_cell_operation(const SymbolicOpVolumeIntegral &volume_integral)
    {
      static_assert(is_volume_integral_op<SymbolicOpVolumeIntegral>::value,
                    "Expected a volume integral type.");

      // We need to update the flags that need to be set for
      // cell operations. The flags from the composite operation
      // that composes the integrand will be bubbled down to the
      // integral itself.
      cell_update_flags |= volume_integral.get_update_flags();

      // Extract some information about the form that we'll be
      // constructing and integrating
      const auto &form = volume_integral.get_integrand();
      static_assert(
        is_bilinear_form<typename std::decay<decltype(form)>::type>::value,
        "Incompatible integrand type.");

      const auto &test_space_op  = form.get_test_space_operation();
      const auto &functor        = form.get_functor();
      const auto &trial_space_op = form.get_trial_space_operation();

      using TestSpaceOp  = typename std::decay<decltype(test_space_op)>::type;
      using Functor      = typename std::decay<decltype(functor)>::type;
      using TrialSpaceOp = typename std::decay<decltype(trial_space_op)>::type;

      // // Improve the error message that might stem from misuse of a templated
      // function. static_assert(!is_field_solution_op<Functor>::value,
      //               "This add_cell_operation() can only work with functors
      //               that are not " "field solutions. This is because we do
      //               not provide the solution vector " "to the functor to
      //               perform is operation.");

      using ValueTypeTest =
        typename TestSpaceOp::template value_type<ScalarType>;
      using ValueTypeFunctor =
        typename Functor::template value_type<ScalarType>;
      using ValueTypeTrial =
        typename TrialSpaceOp::template value_type<ScalarType>;

      // Contribution symmetry
      // We consider the case that the user adjust the symmetry at any point
      // before the actual operation. We therefore capture the local symmetry
      // flag by copy, but the global symmetry flag refers back to that stored
      // in the assembler itself.
      const bool local_contribution_symmetry_flag =
        false; // form.local_contribution_symmetry_flag()
      const bool &global_system_symmetry_flag =
        this->global_system_symmetry_flag;

      // Now, compose all of this into a bespoke operation for this
      // contribution.
      //
      // Important note: All operations must be captured by copy!
      // We do this in case someone inlines a call to bilinear_form()
      // with operator+= , e.g.
      //   MatrixBasedAssembler<dim, spacedim> assembler;
      //   assembler += bilinear_form(test_val, coeff_func, trial_val).dV();
      auto f = [volume_integral,
                test_space_op,
                functor,
                trial_space_op,
                local_contribution_symmetry_flag,
                &global_system_symmetry_flag](
                 FullMatrix<ScalarType> &                cell_matrix,
                 MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                 const std::vector<std::string> &        solution_names,
                 const FEValuesBase<dim, spacedim> &     fe_values) {
        // Skip this cell if it doesn't match the criteria set for the
        // integration domain.
        if (!volume_integral.get_integral_operation().integrate_on_cell(
              fe_values.get_cell()))
          {
            return;
          }

        const unsigned int n_dofs_per_cell = fe_values.dofs_per_cell;
        const unsigned int n_q_points      = fe_values.n_quadrature_points;

        // Decide whether or not to assemble in symmetry mode, i.e. Only
        // assemble the lower half of the matrix plus the diagonal.
        const bool symmetric_contribution =
          local_contribution_symmetry_flag | global_system_symmetry_flag;

        // If the local contribution is symmetric, but the global system is not,
        // then we need to write our contributions into an intermediate data
        // structure.
        // TODO[JPP]: Put this somewhere reuseable, e.g. ScratchData?
        const bool use_scratch_cell_matrix =
          local_contribution_symmetry_flag && !global_system_symmetry_flag;
        FullMatrix<ScalarType> scratch_cell_matrix;
        if (use_scratch_cell_matrix)
          scratch_cell_matrix.reinit({cell_matrix.m(), cell_matrix.n()});

        // Decide whether or not to contribute directly into the overall system
        // matrix, or own own scratch data.
        auto &assembly_cell_matrix =
          (use_scratch_cell_matrix ? scratch_cell_matrix : cell_matrix);

        // Get the shape function data (value, gradients, curls, etc.)
        // for all quadrature points at all DoFs. We construct it in this
        // manner (with the q_point indices fast) so that we can perform
        // contractions in an optimal manner.
        const std::vector<std::vector<ValueTypeTest>> shapes_test =
          internal::evaluate_fe_space<ScalarType>(
            test_space_op, fe_values, fe_values, scratch_data, solution_names);
        const std::vector<std::vector<ValueTypeTrial>> shapes_trial =
          internal::evaluate_fe_space<ScalarType>(
            trial_space_op, fe_values, fe_values, scratch_data, solution_names);

        if (use_vectorization)
          {
            // Get all functor values at the quadrature points
            const std::vector<ValueTypeFunctor> all_values_functor =
              internal::evaluate_functor<ScalarType>(functor,
                                                     fe_values,
                                                     scratch_data,
                                                     solution_names);

            // We wish to vectorize the quadrature point data / indices.
            // Determine the quadrature point batch size for all vectorized
            // operations.
            constexpr std::size_t width =
              dealii::internal::VectorizedArrayWidthSpecifier<
                ScalarType>::max_width;
            using Vector_t = VectorizedArray<ScalarType, width>;
            using VectorizedValueTypeTest =
              typename TestSpaceOp::template value_type<Vector_t>;
            using VectorizedValueTypeFunctor =
              typename Functor::template value_type<Vector_t>;
            using VectorizedValueTypeTrial =
              typename TrialSpaceOp::template value_type<Vector_t>;

            // Alias some reused names
            const auto &all_shapes_test  = shapes_test;
            const auto &all_shapes_trial = shapes_trial;

            // To fill: the integration constant and functor values, as well as
            // the all DoF shape function data (value, gradients, curls, etc.)
            // for all batch quadrature points.
            VectorizedArray<double, width>         JxW(0.0);
            VectorizedValueTypeFunctor             values_functor{};
            AlignedVector<VectorizedValueTypeTest> shapes_test(n_dofs_per_cell);
            AlignedVector<VectorizedValueTypeTrial> shapes_trial(
              n_dofs_per_cell);

            using QPRange_t =
              std_cxx20::ranges::iota_view<unsigned int, unsigned int>;
            for (unsigned int batch_start = 0; batch_start < n_q_points;
                 batch_start += width)
              {
                // Make sure that the range doesn't go out of bounds if we
                // cannot divide up the work evenly.
                const unsigned int batch_end =
                  std::min(batch_start + static_cast<unsigned int>(width),
                           n_q_points);
                const QPRange_t q_range{batch_start, batch_end};

                // Assign values for each entry in vectorized arrays.
                for (unsigned int v = 0; v < width; v++)
                  {
                    // The entire vectorization lane might not be filled, so
                    // we need an early exit for this condition.
                    // These elements still participate in the assembly through,
                    // so we need to make sure that their contributions
                    // integrate to zero.
                    if (v >= q_range.size())
                      {
                        internal::set_vectorized_values(JxW, v, 0.0);
                        continue;
                      }

                    // Quadrature point index corresponding to the
                    // vectorization index.
                    const unsigned int q = q_range[v];

                    // Copy non-vectorized data into the vectorized
                    // counterparts.
                    internal::set_vectorized_values(
                      JxW,
                      v,
                      volume_integral.template operator()<ScalarType>(fe_values,
                                                                      q));
                    internal::set_vectorized_values(values_functor,
                                                    v,
                                                    all_values_functor[q]);

                    for (const unsigned int k : fe_values.dof_indices())
                      {
                        internal::set_vectorized_values(shapes_test[k],
                                                        v,
                                                        all_shapes_test[k][q]);
                        internal::set_vectorized_values(shapes_trial[k],
                                                        v,
                                                        all_shapes_trial[k][q]);
                      }
                  }

                // Do the assembly for the current batch of quadrature points
                internal::assemble_cell_matrix_vectorized_qp_batch_contribution<
                  Sign>(assembly_cell_matrix,
                        fe_values,
                        shapes_test,
                        values_functor,
                        shapes_trial,
                        JxW,
                        symmetric_contribution);
              }
          }
        else
          {
            // Get all values at the quadrature points
            const std::vector<double> &JxW =
              volume_integral.template operator()<ScalarType>(fe_values);
            const std::vector<ValueTypeFunctor> values_functor =
              internal::evaluate_functor<ScalarType>(functor,
                                                     fe_values,
                                                     scratch_data,
                                                     solution_names);

            // Assemble for all DoFs and quadrature points
            internal::assemble_cell_matrix_contribution<Sign>(
              assembly_cell_matrix,
              fe_values,
              shapes_test,
              values_functor,
              shapes_trial,
              JxW,
              symmetric_contribution);
          }

        if (use_scratch_cell_matrix)
          {
            Assert(!global_system_symmetry_flag,
                   ExcMessage("Expect global symmetry flag to be false."));
            Assert(&assembly_cell_matrix == &scratch_cell_matrix,
                   ExcMessage(
                     "Expected to be working with scratch cell matrix object"));

            // Symmetrize this contribution
            for (const unsigned int i : fe_values.dof_indices())
              {
                // Accumulate into diagonal
                cell_matrix(i, i) += scratch_cell_matrix(i, i);
                for (const unsigned int j :
                     fe_values.dof_indices_starting_at(i + 1))
                  {
                    // Accumulate into lower and upper halves from the lower
                    // half contribution that we've just assembled.
                    cell_matrix(i, j) += scratch_cell_matrix(j, i);
                    cell_matrix(j, i) += scratch_cell_matrix(j, i);
                  }
              }
          }
      };
      cell_matrix_operations.emplace_back(f);
    }


    template <
      enum internal::AccumulationSign Sign,
      typename SymbolicOpBoundaryIntegral,
      typename std::enable_if<is_bilinear_form<
        typename SymbolicOpBoundaryIntegral::IntegrandType>::value>::type * =
        nullptr>
    void
    add_boundary_face_operation(
      const SymbolicOpBoundaryIntegral &boundary_integral)
    {
      (void)boundary_integral;
      static_assert(is_boundary_integral_op<SymbolicOpBoundaryIntegral>::value,
                    "Expected a boundary integral type.");
      // static_assert(false, "Assembler: Boundary face operations not yet
      // implemented for bilinear forms.")
    }


    template <
      enum internal::AccumulationSign Sign,
      typename SymbolicOpInterfaceIntegral,
      typename std::enable_if<is_bilinear_form<
        typename SymbolicOpInterfaceIntegral::IntegrandType>::value>::type * =
        nullptr>
    void
    add_internal_face_operation(
      const SymbolicOpInterfaceIntegral &interface_integral)
    {
      (void)interface_integral;
      static_assert(
        is_interface_integral_op<SymbolicOpInterfaceIntegral>::value,
        "Expected an interface integral type.");
      // static_assert(false, "Assembler: Internal face operations not yet
      // implemented for bilinear forms.")
    }


    /**
     * Cell operations for linear forms
     *
     * @tparam SymbolicOpVolumeIntegral
     * @tparam std::enable_if<is_linear_form<
     * typename SymbolicOpVolumeIntegral::IntegrandType>::value>::type
     * @param volume_integral
     */
    template <enum internal::AccumulationSign Sign,
              typename SymbolicOpVolumeIntegral,
              typename std::enable_if<is_linear_form<
                typename SymbolicOpVolumeIntegral::IntegrandType>::value>::type
                * = nullptr>
    void
    add_cell_operation(const SymbolicOpVolumeIntegral &volume_integral)
    {
      static_assert(is_volume_integral_op<SymbolicOpVolumeIntegral>::value,
                    "Expected a volume integral type.");

      // We need to update the flags that need to be set for
      // cell operations. The flags from the composite operation
      // that composes the integrand will be bubbled down to the
      // integral itself.
      cell_update_flags |= volume_integral.get_update_flags();

      // Extract some information about the form that we'll be
      // constructing and integrating
      const auto &form = volume_integral.get_integrand();
      static_assert(
        is_linear_form<typename std::decay<decltype(form)>::type>::value,
        "Incompatible integrand type.");

      const auto &test_space_op = form.get_test_space_operation();
      const auto &functor       = form.get_functor();

      using TestSpaceOp = typename std::decay<decltype(test_space_op)>::type;
      using Functor     = typename std::decay<decltype(functor)>::type;

      // Improve the error message that might stem from misuse of a templated
      // function. static_assert(!is_field_solution_op<Functor>::value,
      //               "This add_cell_operation() can only work with functors
      //               that are not " "field solutions. This is because we do
      //               not provide the solution vector " "to the functor to
      //               perform is operation.");

      using ValueTypeTest =
        typename TestSpaceOp::template value_type<ScalarType>;
      using ValueTypeFunctor =
        typename Functor::template value_type<ScalarType>;

      // Now, compose all of this into a bespoke operation for this
      // contribution.
      //
      // Important note: All operations must be captured by copy!
      // We do this in case someone inlines a call to bilinear_form()
      // with operator+= , e.g.
      //   MatrixBasedAssembler<dim, spacedim> assembler;
      //   assembler += linear_form(test_val, coeff_func).dV();
      auto f = [volume_integral,
                test_space_op,
                functor](Vector<ScalarType> &                    cell_vector,
                         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const std::vector<std::string> &        solution_names,
                         const FEValuesBase<dim, spacedim> &     fe_values) {
        // Skip this cell if it doesn't match the criteria set for the
        // integration domain.
        if (!volume_integral.get_integral_operation().integrate_on_cell(
              fe_values.get_cell()))
          {
            return;
          }

        const unsigned int n_dofs_per_cell = fe_values.dofs_per_cell;
        const unsigned int n_q_points      = fe_values.n_quadrature_points;

        // Get the shape function data (value, gradients, curls, etc.)
        // for all quadrature points at all DoFs. We construct it in this
        // manner (with the q_point indices fast) so that we can perform
        // contractions in an optimal manner.
        const std::vector<std::vector<ValueTypeTest>> shapes_test =
          internal::evaluate_fe_space<ScalarType>(
            test_space_op, fe_values, fe_values, scratch_data, solution_names);

        if (use_vectorization)
          {
            // Get all functor values at the quadrature points
            const std::vector<ValueTypeFunctor> all_values_functor =
              internal::evaluate_functor<ScalarType>(functor,
                                                     fe_values,
                                                     scratch_data,
                                                     solution_names);

            // We wish to vectorize the quadrature point data / indices.
            // Determine the quadrature point batch size for all vectorized
            // operations.
            constexpr std::size_t width =
              dealii::internal::VectorizedArrayWidthSpecifier<
                ScalarType>::max_width;
            using Vector_t = VectorizedArray<ScalarType, width>;
            using VectorizedValueTypeTest =
              typename TestSpaceOp::template value_type<Vector_t>;
            using VectorizedValueTypeFunctor =
              typename Functor::template value_type<Vector_t>;

            // Alias some reused names
            const auto &all_shapes_test = shapes_test;

            // To fill: the integration constant and functor values, as well as
            // the all DoF shape function data (value, gradients, curls, etc.)
            // for all batch quadrature points.
            VectorizedArray<double, width>         JxW(0.0);
            VectorizedValueTypeFunctor             values_functor{};
            AlignedVector<VectorizedValueTypeTest> shapes_test(n_dofs_per_cell);

            using QPRange_t =
              std_cxx20::ranges::iota_view<unsigned int, unsigned int>;
            for (unsigned int batch_start = 0; batch_start < n_q_points;
                 batch_start += width)
              {
                // Make sure that the range doesn't go out of bounds if we
                // cannot divide up the work evenly.
                const unsigned int batch_end =
                  std::min(batch_start + static_cast<unsigned int>(width),
                           n_q_points);
                const QPRange_t q_range{batch_start, batch_end};

                // Assign values for each entry in vectorized arrays.
                for (unsigned int v = 0; v < width; v++)
                  {
                    // The entire vectorization lane might not be filled, so
                    // we need an early exit for this condition.
                    // These elements still participate in the assembly through,
                    // so we need to make sure that their contributions
                    // integrate to zero.
                    if (v >= q_range.size())
                      {
                        internal::set_vectorized_values(JxW, v, 0.0);
                        continue;
                      }

                    // Quadrature point index corresponding to the
                    // vectorization index.
                    const unsigned int q = q_range[v];

                    // Copy non-vectorized data into the vectorized
                    // counterparts.
                    internal::set_vectorized_values(
                      JxW,
                      v,
                      volume_integral.template operator()<ScalarType>(fe_values,
                                                                      q));
                    internal::set_vectorized_values(values_functor,
                                                    v,
                                                    all_values_functor[q]);

                    for (const unsigned int k : fe_values.dof_indices())
                      internal::set_vectorized_values(shapes_test[k],
                                                      v,
                                                      all_shapes_test[k][q]);
                  }

                // Do the assembly for the current batch of quadrature points
                internal::assemble_cell_vector_vectorized_qp_batch_contribution<
                  Sign>(
                  cell_vector, fe_values, shapes_test, values_functor, JxW);
              }
          }
        else
          {
            // Get all values at the quadrature points
            const std::vector<double> &JxW =
              volume_integral.template operator()<ScalarType>(fe_values);
            const std::vector<ValueTypeFunctor> values_functor =
              internal::evaluate_functor<ScalarType>(functor,
                                                     fe_values,
                                                     scratch_data,
                                                     solution_names);

            // Assemble for all DoFs and quadrature points
            internal::assemble_cell_vector_contribution<Sign>(
              cell_vector, fe_values, shapes_test, values_functor, JxW);
          }
      };
      cell_vector_operations.emplace_back(f);
    }


    template <
      enum internal::AccumulationSign Sign,
      typename SymbolicOpBoundaryIntegral,
      typename std::enable_if<is_linear_form<
        typename SymbolicOpBoundaryIntegral::IntegrandType>::value>::type * =
        nullptr>
    void
    add_boundary_face_operation(
      const SymbolicOpBoundaryIntegral &boundary_integral)
    {
      static_assert(is_boundary_integral_op<SymbolicOpBoundaryIntegral>::value,
                    "Expected a boundary integral type.");
      // static_assert(false, "Assembler: Boundary face operations not yet
      // implemented for linear forms.")

      // We need to update the flags that need to be set for
      // cell operations. The flags from the composite operation
      // that composes the integrand will be bubbled down to the
      // integral itself.
      boundary_face_update_flags |= boundary_integral.get_update_flags();

      // Extract some information about the form that we'll be
      // constructing and integrating
      const auto &form = boundary_integral.get_integrand();
      static_assert(
        is_linear_form<typename std::decay<decltype(form)>::type>::value,
        "Incompatible integrand type.");

      const auto &test_space_op = form.get_test_space_operation();
      const auto &functor       = form.get_functor();

      using TestSpaceOp = typename std::decay<decltype(test_space_op)>::type;
      using Functor     = typename std::decay<decltype(functor)>::type;

      // Improve the error message that might stem from misuse of a templated
      // function. static_assert(!is_field_solution_op<Functor>::value,
      //               "This add_cell_operation() can only work with functors
      //               that are not " "field solutions. This is because we do
      //               not provide the solution vector " "to the functor to
      //               perform is operation.");

      using ValueTypeTest =
        typename TestSpaceOp::template value_type<ScalarType>;
      using ValueTypeFunctor =
        typename Functor::template value_type<ScalarType>;

      // Now, compose all of this into a bespoke operation for this
      // contribution.
      //
      // Important note: All operations must be captured by copy!
      // We do this in case someone inlines a call to bilinear_form()
      // with operator+= , e.g.
      //   MatrixBasedAssembler<dim, spacedim> assembler;
      //   assembler += linear_form(test_val, boundary_func).dA();
      auto f = [boundary_integral,
                test_space_op,
                functor](Vector<ScalarType> &                    cell_vector,
                         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                         const std::vector<std::string> &        solution_names,
                         const FEValuesBase<dim, spacedim> &     fe_values,
                         const FEFaceValuesBase<dim, spacedim> & fe_face_values,
                         const unsigned int                      face) {
        // Skip this cell face if it doesn't match the criteria set for the
        // integration domain.
        if (!boundary_integral.get_integral_operation().integrate_on_face(
              fe_values.get_cell(), face))
          {
            return;
          }

        const unsigned int n_dofs_per_cell = fe_values.dofs_per_cell;
        const unsigned int n_q_points      = fe_face_values.n_quadrature_points;

        // Get the shape function data (value, gradients, curls, etc.)
        // for all quadrature points at all DoFs. We construct it in this
        // manner (with the q_point indices fast) so that we can perform
        // contractions in an optimal manner.
        const std::vector<std::vector<ValueTypeTest>> shapes_test =
          internal::evaluate_fe_space<ScalarType>(test_space_op,
                                                  fe_values,
                                                  fe_face_values,
                                                  scratch_data,
                                                  solution_names);

        if (use_vectorization)
          {
            // Get all functor values at the quadrature points
            const std::vector<ValueTypeFunctor> all_values_functor =
              internal::evaluate_functor<ScalarType>(functor,
                                                     fe_face_values,
                                                     scratch_data,
                                                     solution_names);

            // We wish to vectorize the quadrature point data / indices.
            // Determine the quadrature point batch size for all vectorized
            // operations.
            constexpr std::size_t width =
              dealii::internal::VectorizedArrayWidthSpecifier<
                ScalarType>::max_width;
            using Vector_t = VectorizedArray<ScalarType, width>;
            using VectorizedValueTypeTest =
              typename TestSpaceOp::template value_type<Vector_t>;
            using VectorizedValueTypeFunctor =
              typename Functor::template value_type<Vector_t>;

            // Alias some reused names
            const auto &all_shapes_test = shapes_test;

            // To fill: the integration constant and functor values, as well as
            // the all DoF shape function data (value, gradients, curls, etc.)
            // for all batch quadrature points.
            VectorizedArray<double, width>         JxW(0.0);
            VectorizedValueTypeFunctor             values_functor{};
            AlignedVector<VectorizedValueTypeTest> shapes_test(n_dofs_per_cell);

            using QPRange_t =
              std_cxx20::ranges::iota_view<unsigned int, unsigned int>;
            for (unsigned int batch_start = 0; batch_start < n_q_points;
                 batch_start += width)
              {
                // Make sure that the range doesn't go out of bounds if we
                // cannot divide up the work evenly.
                const unsigned int batch_end =
                  std::min(batch_start + static_cast<unsigned int>(width),
                           n_q_points);
                const QPRange_t q_range{batch_start, batch_end};

                // Assign values for each entry in vectorized arrays.
                for (unsigned int v = 0; v < width; v++)
                  {
                    // The entire vectorization lane might not be filled, so
                    // we need an early exit for this condition.
                    // These elements still participate in the assembly through,
                    // so we need to make sure that their contributions
                    // integrate to zero.
                    if (v >= q_range.size())
                      {
                        internal::set_vectorized_values(JxW, v, 0.0);
                        continue;
                      }

                    // Quadrature point index corresponding to the
                    // vectorization index.
                    const unsigned int q = q_range[v];

                    // Copy non-vectorized data into the vectorized
                    // counterparts.
                    internal::set_vectorized_values(
                      JxW,
                      v,
                      boundary_integral.template operator()<ScalarType>(
                        fe_face_values, q));
                    internal::set_vectorized_values(values_functor,
                                                    v,
                                                    all_values_functor[q]);

                    for (const unsigned int k : fe_values.dof_indices())
                      internal::set_vectorized_values(shapes_test[k],
                                                      v,
                                                      all_shapes_test[k][q]);
                  }

                // Do the assembly for the current batch of quadrature points
                internal::assemble_cell_vector_vectorized_qp_batch_contribution<
                  Sign>(
                  cell_vector, fe_values, shapes_test, values_functor, JxW);
              }
          }
        else
          {
            // Get all values at the quadrature points
            const std::vector<double> &  JxW =
              boundary_integral.template operator()<ScalarType>(fe_face_values);
            const std::vector<ValueTypeFunctor> values_functor =
              internal::evaluate_functor<ScalarType>(functor,
                                                     fe_face_values,
                                                     scratch_data,
                                                     solution_names);

            // Assemble for all DoFs and quadrature points
            internal::assemble_cell_vector_contribution<Sign>(cell_vector,
                                                              fe_values,
                                                              fe_face_values,
                                                              shapes_test,
                                                              values_functor,
                                                              JxW);
          }
      };
      boundary_face_vector_operations.emplace_back(f);
    }


    template <enum internal::AccumulationSign Sign,
              typename SymbolicOpInterfaceIntegral,
              typename std::enable_if<
                is_interface_integral_op<SymbolicOpInterfaceIntegral>::value &&
                is_linear_form<typename SymbolicOpInterfaceIntegral::
                                 IntegrandType>::value>::type * = nullptr>
    void
    add_internal_face_operation(
      const SymbolicOpInterfaceIntegral &interface_integral)
    {
      (void)interface_integral;
      static_assert(
        is_interface_integral_op<SymbolicOpInterfaceIntegral>::value,
        "Expected an interface integral type.");
      // static_assert(false, "Assembler: Internal face operations not yet
      // implemented for linear forms.")
    }


    // template <typename FunctorType>
    // typename std::enable_if<!is_ad_functor_op<FunctorType>::value>::type
    // add_solution_update_operation(FunctorType &functor)
    // {
    //   // Do nothing
    //   (void)functor;
    // }


    // template <typename FunctorType>
    // typename std::enable_if<is_ad_functor_op<FunctorType>::value>::type
    // add_solution_update_operation(FunctorType &functor)
    // {
    //   cell_solution_update_flags |= functor.get_update_flags();

    //   auto f =
    //     [&functor](const std::vector<ScalarType> & solution_local_dof_values,
    //                const FEValuesBase<dim, spacedim> &fe_values) {
    //       functor.update_from_solution(fe_values, solution_local_dof_values);
    //     };
    //   cell_solution_update_operations.emplace_back(f);
    // }

    UpdateFlags
    get_cell_update_flags() const
    {
      return cell_update_flags /*| cell_solution_update_flags*/;
    }

    UpdateFlags
    get_face_update_flags() const
    {
      return boundary_face_update_flags /*|
                                           boundary_face_solution_update_flags*/
             |
             interface_face_update_flags /*|
                                            interface_face_solution_update_flags*/
        ;
    }

    std::vector<StringOperation> as_ascii_operations;
    std::vector<StringOperation> as_latex_operations;

    // Do I actually need this? The caching ScratchData might be enough,
    // because it should be guarenteed that any field functors and field
    // solutions extracted by the user directly have the same stored
    // variable in scratch.
    UpdateFlags                              cell_solution_update_flags;
    std::vector<CellSolutionUpdateOperation> cell_field_solution_operations;
    // Ditto here
    UpdateFlags boundary_face_solution_update_flags;
    std::vector<BoundaryFaceSolutionUpdateOperation>
      boundary_face_field_solution_operations;

    std::vector<CellADSDOperation>     cell_ad_sd_operations;
    std::vector<BoundaryADSDOperation> boundary_face_ad_sd_operations;

    UpdateFlags                      cell_update_flags;
    std::vector<CellMatrixOperation> cell_matrix_operations;
    std::vector<CellVectorOperation> cell_vector_operations;

    UpdateFlags                          boundary_face_update_flags;
    std::vector<BoundaryMatrixOperation> boundary_face_matrix_operations;
    std::vector<BoundaryVectorOperation> boundary_face_vector_operations;

    // UpdateFlags boundary_face_solution_update_flags;
    // std::vector<CellSolutionUpdateOperation>
    //   boundary_face_solution_update_operations;

    UpdateFlags                           interface_face_update_flags;
    std::vector<InterfaceMatrixOperation> interface_face_matrix_operations;
    std::vector<InterfaceVectorOperation> interface_face_vector_operations;

    // UpdateFlags interface_face_solution_update_flags;
    // std::vector<CellSolutionUpdateOperation>
    //   interface_face_solution_update_operations;

    // --- AD/SD support ---

    // An object that stores pointers to ADHelpers / SDBatchOptimizers.
    // These items need to be kept alive for as long as the assembler
    // is in scope, but the object that creates the AD/SD based forms
    // may only be temporary. To we allow
    // GeneralDataStorage ad_sd_cache;

    // Expose the cache to the AD forms
    // template <int                                   dim2,
    //           enum Differentiation::NumberTypes ADScalarTypeCode,
    //           typename ScalarType>
    // friend class AutoDifferentiation::EnergyFunctional;
  };

} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_assembler_base_h
