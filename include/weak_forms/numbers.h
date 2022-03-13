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

#ifndef dealii_weakforms_numbers_h
#define dealii_weakforms_numbers_h

#include <deal.II/base/config.h>

#include <deal.II/base/numbers.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <weak_forms/config.h>
#include <weak_forms/types.h>


WEAK_FORMS_NAMESPACE_OPEN

namespace WeakForms
{
  namespace numbers
  {
    const types::field_index invalid_field_index =
      static_cast<types::field_index>(-1);

    const types::solution_index invalid_solution_index =
      static_cast<types::solution_index>(-1);

    const types::solution_index linearizable_solution_index = 0;


#if DEAL_II_VECTORIZATION_WIDTH_IN_BITS > 0
    template <typename ScalarType>
    struct VectorizationDefaults
    {
      static constexpr std::size_t width =
        dealii::internal::VectorizedArrayWidthSpecifier<ScalarType>::max_width;
    };

    struct UseVectorization : std::true_type
    {};
#else
    template <typename ScalarType>
    struct VectorizationDefaults
    {
      static constexpr std::size_t width = 1;
    };

    struct UseVectorization : std::false_type
    {};
#endif


    // Promote a type to a vectorized type
    template <typename T, typename U = void>
    struct VectorizedValue;


    template <typename ScalarType>
    struct VectorizedValue<
      ScalarType,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      template <std::size_t width>
      using type = VectorizedArray<ScalarType, width>;
    };


    template <int rank, int dim, typename ScalarType>
    struct VectorizedValue<
      Tensor<rank, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      template <std::size_t width>
      using type = Tensor<rank, dim, VectorizedArray<ScalarType, width>>;
    };


    template <int rank, int dim, typename ScalarType>
    struct VectorizedValue<
      SymmetricTensor<rank, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      template <std::size_t width>
      using type =
        SymmetricTensor<rank, dim, VectorizedArray<ScalarType, width>>;
    };


    template <typename ScalarType>
    struct VectorizedValue<
      std::complex<ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      template <std::size_t width>
      using type = VectorizedArray<std::complex<ScalarType>, width>;
    };


    template <int rank, int dim, typename ScalarType>
    struct VectorizedValue<
      Tensor<rank, dim, std::complex<ScalarType>>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      template <std::size_t width>
      using type =
        Tensor<rank, dim, VectorizedArray<std::complex<ScalarType>, width>>;
    };


    template <int rank, int dim, typename ScalarType>
    struct VectorizedValue<
      SymmetricTensor<rank, dim, std::complex<ScalarType>>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      template <std::size_t width>
      using type =
        SymmetricTensor<rank,
                        dim,
                        VectorizedArray<std::complex<ScalarType>, width>>;
    };


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
    void
    set_vectorized_values(
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
    void
    set_vectorized_values(
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
    void
    set_vectorized_values(
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

  } // namespace numbers

} // namespace WeakForms

WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_numbers_h